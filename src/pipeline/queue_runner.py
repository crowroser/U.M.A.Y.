"""
src/pipeline/queue_runner.py
Streaming Prefetch Pipeline — v3 (Telemetry + Cache).

Mimari:
  OCR metin → sentence_splitter → Processor Thread →
    Cache lookup → [hit: skip TTS] / [miss: TTS (numpy) → RVC (numpy) → cache store]
    → Audio Queue → Player Thread → sounddevice

Iyilestirmeler (v2'ye gore):
  - PipelineTimer ile hassas adim olcumu
  - PipelineStats ile rolling istatistikler (avg/min/max/p95)
  - TTSCache entegrasyonu (ayni replik tekrarlarina 0ms yanit)
  - on_stats callback ile UI'a canli performans verisi
"""

import itertools
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

from src.pipeline.sentence_splitter import split_sentences
from src.pipeline.telemetry import PipelineTimer, PipelineStats
from src.pipeline.tts_cache import TTSCache
from src.pipeline.rvc_cache import RVCAudioCache

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Dongusel dosya tamponu (fallback icin)
_SLOT_COUNT = 4
_slot_counter = itertools.cycle(range(_SLOT_COUNT))
_slot_lock = threading.Lock()


def _next_slot() -> int:
    with _slot_lock:
        return next(_slot_counter)


class AudioItem:
    """Oynatici thread'e gecirilen hazir ses birimi (numpy tabanli)."""
    __slots__ = ("sr", "data", "wav_path", "label")

    def __init__(self, sr: int = 0, data=None, wav_path: str = "", label: str = ""):
        self.sr = sr
        self.data = data  # numpy array (int16)
        self.wav_path = wav_path  # fallback dosya yolu
        self.label = label


class PipelineItem:
    """OCR'dan gelen ham altyazi birimi."""
    __slots__ = ("speaker", "text", "timestamp")

    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text
        self.timestamp = time.monotonic()


# Sentinel: pipeline durdurma sinyali
_SENTINEL = object()


class QueueRunner:
    """
    Streaming prefetch pipeline v3:
    - processor_thread: metin → cumle bol → cache → TTS(numpy) → RVC(numpy) → audio_queue
    - player_thread:    audio_queue → sounddevice.play

    Telemetri: her adimin suresi olculur, istatistikler toplanir.
    Cache: ayni replik tekrarlarinda TTS atlanir (~0ms).
    """

    # Eski bir altyazinin "bayatlama" suresi (saniye)
    STALE_THRESHOLD = 6.0

    def __init__(
        self,
        tts,
        rvc,
        translator=None,
        analyzer=None,
        ducker=None,
        on_log: Optional[Callable[[str, str], None]] = None,
        on_timing: Optional[Callable[[dict], None]] = None,
        on_stats: Optional[Callable[[dict], None]] = None,
        text_maxsize: int = 4,
        audio_maxsize: int = 4,
        cache_mb: int = 200,
    ):
        self._tts = tts
        self._rvc = rvc
        self._translator = translator
        self._analyzer = analyzer
        self._ducker = ducker
        self._on_log = on_log or (lambda msg, tag: None)
        self._on_timing = on_timing or (lambda d: None)
        self._on_stats = on_stats or (lambda d: None)

        self._text_queue: queue.Queue = queue.Queue(maxsize=text_maxsize)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=audio_maxsize)

        self._processor_thread: Optional[threading.Thread] = None
        self._player_thread: Optional[threading.Thread] = None
        self._running = False

        # Stale detection
        self._generation = 0
        self._gen_lock = threading.Lock()

        # Telemetri
        self._stats = PipelineStats(window_size=100)

        # TTS Cache
        self._cache = TTSCache(max_memory_mb=cache_mb)

        # RVC Cache
        self._rvc_cache = RVCAudioCache(max_memory_mb=200, enabled=True)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    @property
    def cache(self) -> TTSCache:
        return self._cache

    def update_cache_settings(self, config: dict):
        """Çalışma zamanında TTS ve RVC önbellek ayarlarını günceller."""
        cache_cfg = config.get("cache", {})
        
        # TTS Cache güncelle
        max_mem = cache_cfg.get("tts_max_memory_mb", 200)
        disk_enabled = cache_cfg.get("tts_disk_enabled", False)
        disk_limit = cache_cfg.get("tts_disk_limit_mb", 500)
        
        self._cache.update_settings(
            max_memory_mb=max_mem,
            disk_enabled=disk_enabled,
            disk_limit_mb=disk_limit
        )
        
        # RVC Cache güncelle
        rvc_enabled = cache_cfg.get("rvc_enabled", True)
        rvc_max_mem = cache_cfg.get("rvc_max_memory_mb", 200)
        rvc_disk_enabled = cache_cfg.get("rvc_disk_enabled", False)
        rvc_disk_limit = cache_cfg.get("rvc_disk_limit_mb", 500)
        
        if self._rvc_cache:
            self._rvc_cache.update_settings(
                enabled=rvc_enabled,
                max_memory_mb=rvc_max_mem,
                disk_enabled=rvc_disk_enabled,
                disk_limit_mb=rvc_disk_limit
            )

    def start(self):
        if self._running:
            return
        self._running = True
        self._generation = 0

        self._processor_thread = threading.Thread(
            target=self._processor_loop,
            daemon=True,
            name="PipelineProcessor",
        )
        self._player_thread = threading.Thread(
            target=self._player_loop,
            daemon=True,
            name="PipelinePlayer",
        )
        self._processor_thread.start()
        self._player_thread.start()
        self._log("Pipeline baslatildi (v3: streaming + cache + telemetri).", "info")

    def stop(self):
        if not self._running:
            return
        self._running = False

        for q in (self._text_queue, self._audio_queue):
            try:
                q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

        for t in (self._processor_thread, self._player_thread):
            if t:
                t.join(timeout=5)

        self._processor_thread = None
        self._player_thread = None

        # Pipeline durma ozeti
        stats = self._stats.get_stats()
        if stats["total_processed"] > 0:
            self._log(
                f"[OZET] {stats['total_processed']} islem | "
                f"Ort: {stats['avg_latency_ms']}ms | "
                f"Cache: %{int(stats['cache']['hit_rate'] * 100)} "
                f"({stats['cache']['hits']} hit)",
                "info",
            )
        self._log("Pipeline durduruldu.", "info")

    def push(self, speaker: str, text: str):
        with self._gen_lock:
            self._generation += 1

        item = PipelineItem(speaker, text)
        self._flush_audio_queue()

        while True:
            try:
                self._text_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._text_queue.get_nowait()
                except queue.Empty:
                    pass

    def get_stats(self) -> dict:
        """Pipeline + cache istatistiklerini doner."""
        result = self._stats.get_stats()
        result["cache"] = self._cache.get_stats()
        if self._rvc_cache:
            result["rvc_cache"] = self._rvc_cache.get_stats()
        return result

    def get_stats_summary(self) -> str:
        """Tek satirlik ozet (status bar icin)."""
        return self._stats.format_summary()

    # ── Stale Detection ────────────────────────────────────────────────

    def _flush_audio_queue(self):
        flushed = 0
        while True:
            try:
                self._audio_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        if flushed:
            self._log(f"[FLUSH] {flushed} eski ses parcasi atildi.", "info")

    def _is_stale(self, item: PipelineItem) -> bool:
        age = time.monotonic() - item.timestamp
        return age > self.STALE_THRESHOLD

    # ── Stage 1: Processor Thread ──────────────────────────────────────

    def _processor_loop(self):
        while self._running:
            try:
                item = self._text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                try:
                    self._audio_queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
                break
            self._process(item)

    def _process(self, item: PipelineItem):
        speaker, text = item.speaker, item.text
        timer = PipelineTimer()
        timer.start_total()

        # Bayatlama kontrolu
        if self._is_stale(item):
            self._log(f"[STALE] Atiliyor: {text[:40]}...", "info")
            return

        self._log(f"[PROC] {speaker}: {text}", "info")

        # 1. Ceviri
        with timer.measure("translate"):
            if self._translator and self._translator.enabled:
                translated = self._translator.translate(text)
                if translated != text:
                    self._log(f"[CEV] {text[:40]} -> {translated[:40]}", "tts")
                text = translated

        # 2. Duygu analizi
        emotion = "neutral"
        pitch_delta = 0
        speed_delta = 0.0
        with timer.measure("sentiment"):
            if self._analyzer and self._analyzer.enabled:
                result = self._analyzer.analyze(speaker, text)
                emotion = result.get("emotion", "neutral")
                params = result.get("params", {})
                pitch_delta = params.get("pitch_delta", 0)
                speed_delta = params.get("speed_delta", 0.0)
                self._log(f"[DUYGU] {speaker}: {emotion}", "info")

        # 3. Cumle bolme
        chunks = split_sentences(text)
        if not chunks:
            chunks = [text]
        self._log(f"[SPLIT] {len(chunks)} parca", "info")

        # Her parca icin: Cache → TTS → RVC → audio_queue
        for i, chunk in enumerate(chunks):
            if not self._running:
                break
            if self._is_stale(item):
                self._log(f"[STALE] Parca {i+1}/{len(chunks)} atiliyor.", "info")
                break

            self._process_chunk(
                chunk, speaker, emotion, speed_delta, pitch_delta,
                i, len(chunks), timer
            )

        timer.stop_total()
        timing = timer.get_durations()
        timing["speaker"] = speaker
        timing["text_len"] = len(text)
        timing["chunk_count"] = len(chunks)

        # Istatistikleri kaydet
        self._stats.record(timing)
        self._on_timing(timing)
        self._on_stats(self._stats.get_stats())

        # Karakter veritabanina kaydet
        try:
            from src.pipeline.character_db import CharacterDatabase
            latency_sec = timing.get("total", 0) / 1000.0
            # Eger TTS suresi 5ms'den kucukse cache hit sayalim
            cache_hit = timing.get("translate", 0) + timing.get("sentiment", 0) >= timing.get("total", 0) - 10
            CharacterDatabase.get_instance().record_line(
                name=speaker,
                text=text,
                latency=latency_sec,
                cache_hit=cache_hit
            )
        except Exception as e:
            self._log(f"Karakter istatistik kayit hatasi: {e}", "error")

    def _process_chunk(
        self,
        text: str,
        speaker: str,
        emotion: str,
        speed_delta: float,
        pitch_delta: int,
        chunk_idx: int,
        total_chunks: int,
        timer: PipelineTimer,
    ):
        """Tek bir metin parcasini isler. Cache → TTS → RVC → audio_queue."""
        label = f"{speaker}: {text[:30]}..." if len(text) > 30 else f"{speaker}: {text}"
        step_prefix = f"c{chunk_idx}_"

        # ── Cache Lookup ──
        cached = self._cache.get(text, speaker, emotion)
        if cached:
            sr, audio_data = cached
            self._stats.record_cache_hit()
            self._log(
                f"[CACHE HIT] Parca {chunk_idx+1}/{total_chunks}: {text[:40]}", "tts"
            )
            timer.record(f"{step_prefix}tts", 0)
            # Cache hit'te RVC zaten uygulanmis, dogrudan oynat
            audio_item = AudioItem(sr=sr, data=audio_data, label=label)
            self._enqueue_audio(audio_item)
            return

        self._stats.record_cache_miss()

        # ── TTS ──
        tts_ms = 0
        with timer.measure(f"{step_prefix}tts"):
            tts_result = self._tts.synthesize_to_array(
                text, speaker=speaker, emotion=emotion, speed_delta=speed_delta,
            )

        tts_ms = timer.get_durations().get(f"{step_prefix}tts", 0)

        if tts_result is None:
            # Fallback: dosya tabanli
            self._process_chunk_file_fallback(
                text, speaker, emotion, speed_delta, pitch_delta,
                chunk_idx, total_chunks, timer,
            )
            return

        sr, audio_data = tts_result
        self._log(
            f"[TTS] Parca {chunk_idx+1}/{total_chunks} ({tts_ms}ms)", "tts"
        )

        # ── RVC ──
        with timer.measure(f"{step_prefix}rvc"):
            if self._rvc:
                # RVC Cache lookup
                cached_rvc = None
                input_audio_data = audio_data.copy() if self._rvc_cache else None
                if self._rvc_cache:
                    cached_rvc = self._rvc_cache.get(audio_data, speaker, pitch_delta)
                
                if cached_rvc:
                    sr, audio_data = cached_rvc
                    self._log(f"[RVC CACHE HIT] Parca {chunk_idx+1}/{total_chunks} önbellekten yüklendi.", "rvc")
                else:
                    rvc_result = self._rvc.convert_array(
                        audio_data, sr,
                        character=speaker,
                        pitch_override_delta=pitch_delta,
                    )
                    if rvc_result:
                        sr, audio_data = rvc_result
                        if self._rvc_cache:
                            self._rvc_cache.put(input_audio_data, speaker, pitch_delta, sr, audio_data)

        rvc_ms = timer.get_durations().get(f"{step_prefix}rvc", 0)
        if self._rvc:
            self._log(
                f"[RVC] Parca {chunk_idx+1}/{total_chunks} ({rvc_ms}ms)", "rvc"
            )

        # ── Cache Store (RVC sonrasi sonucu sakla) ──
        self._cache.put(text, speaker, emotion, sr, audio_data, tts_duration_ms=tts_ms)

        # ── Audio Queue ──
        audio_item = AudioItem(sr=sr, data=audio_data, label=label)
        self._enqueue_audio(audio_item)

    def _process_chunk_file_fallback(
        self,
        text: str,
        speaker: str,
        emotion: str,
        speed_delta: float,
        pitch_delta: int,
        chunk_idx: int,
        total_chunks: int,
        timer: PipelineTimer,
    ):
        """numpy basarisiz olursa dosya tabanli fallback."""
        step_prefix = f"c{chunk_idx}_"
        slot = _next_slot()
        tts_out = OUTPUT_DIR / f"tts_{slot}.wav"
        rvc_out = OUTPUT_DIR / f"rvc_{slot}.wav"

        with timer.measure(f"{step_prefix}tts_fb"):
            wav_path = self._tts.synthesize(
                text, speaker=speaker, emotion=emotion,
                speed_delta=speed_delta, output_path=str(tts_out),
            )

        if not wav_path:
            self._log("[HATA] TTS cikisi alinamadi.", "error")
            return

        tts_ms = timer.get_durations().get(f"{step_prefix}tts_fb", 0)
        self._log(f"[TTS-FB] Parca {chunk_idx+1}/{total_chunks} ({tts_ms}ms)", "tts")

        final_path = wav_path
        if self._rvc:
            with timer.measure(f"{step_prefix}rvc_fb"):
                rvc_path = self._rvc.convert_for_character(
                    wav_path, character=speaker,
                    output_path=str(rvc_out),
                    pitch_override_delta=pitch_delta,
                )
            final_path = rvc_path if rvc_path else wav_path

        label = f"{speaker}: {text[:30]}..."
        audio_item = AudioItem(wav_path=final_path, label=label)
        self._enqueue_audio(audio_item)

    def _enqueue_audio(self, item: AudioItem):
        """Audio kuyruğuna ekle (doluysa bekle)."""
        while self._running:
            try:
                self._audio_queue.put(item, timeout=0.5)
                break
            except queue.Full:
                pass

    # ── Stage 2: Player Thread ─────────────────────────────────────────

    def _player_loop(self):
        while self._running:
            try:
                item = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            self._play(item)

    def _play(self, item: AudioItem):
        try:
            if item.data is not None and item.sr > 0:
                sr, data = item.sr, item.data
            elif item.wav_path:
                sr, data = wavfile.read(item.wav_path)
            else:
                return

            self._log(f"[SES] {item.label}", "audio")

            if self._ducker:
                self._ducker.duck()

            sd.play(data, sr)
            sd.wait()

            if self._ducker:
                self._ducker.restore()

        except Exception as e:
            if self._ducker:
                try:
                    self._ducker.restore()
                except Exception:
                    pass
            self._log(f"[HATA] Ses oynatma: {e}", "error")

    def _log(self, msg: str, tag: str):
        self._on_log(msg, tag)
