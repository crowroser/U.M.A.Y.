"""
src/pipeline/queue_runner.py
Iki asamali Prefetch Pipeline.

Asama 1 — Islemci Thread (processor):
  text_queue  ->  Translate -> Sentiment -> TTS -> RVC  ->  audio_queue

Asama 2 — Oynatici Thread (player):
  audio_queue  ->  Ducking -> sounddevice.play -> sd.wait()

Iki thread paralel calisir: oynatma surerken bir sonraki cumlenin
TTS+RVC islemi arka planda tamamlanir. Bu sayede surekli diyalogda
efektif gecikme ~0-500ms'ye duser.

Her islemci adimi benzersiz cikti dosyasi kullanir (dongusel sayac).
"""

import itertools
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd
import scipy.io.wavfile as wavfile

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 4 yuvalik dongusel tampon: ayni anda en fazla 2 islemci + 1 oynatici = 3 dosya aktif
_SLOT_COUNT = 4
_slot_counter = itertools.cycle(range(_SLOT_COUNT))
_slot_lock = threading.Lock()


def _next_slot() -> int:
    with _slot_lock:
        return next(_slot_counter)


class AudioItem:
    """Oynatici thread'e gecirilen hazir ses birimi."""
    __slots__ = ("wav_path",)

    def __init__(self, wav_path: str):
        self.wav_path = wav_path


class PipelineItem:
    """OCR'dan gelen ham altyazi birimi."""
    __slots__ = ("speaker", "text")

    def __init__(self, speaker: str, text: str):
        self.speaker = speaker
        self.text = text


class QueueRunner:
    """
    Iki asamali prefetch pipeline:
    - processor_thread: metin -> ses donusumu (TTS+RVC)
    - player_thread: hazir sesi aninda oynatma

    Oynatma surerken bir sonraki cumlenin TTS+RVC'si paralel calismaya devam eder.
    """

    def __init__(
        self,
        tts,
        rvc,
        translator=None,
        analyzer=None,
        ducker=None,
        on_log: Optional[Callable[[str, str], None]] = None,
        text_maxsize: int = 3,
        audio_maxsize: int = 2,
    ):
        """
        tts        : TTSGenerator instance
        rvc        : RVCConverter instance  (None = bypass)
        translator : Translator instance    (None = bypass)
        analyzer   : SentimentAnalyzer      (None = bypass)
        ducker     : AudioDucker instance   (None = bypass)
        on_log     : UI log callback (msg, tag)
        """
        self._tts = tts
        self._rvc = rvc
        self._translator = translator
        self._analyzer = analyzer
        self._ducker = ducker
        self._on_log = on_log or (lambda msg, tag: None)

        # Asama 1: OCR metnini alir, Asama 2 icin ses uretir
        self._text_queue: queue.Queue[Optional[PipelineItem]] = queue.Queue(maxsize=text_maxsize)
        # Asama 2: hazir ses dosyalarini oynatir
        self._audio_queue: queue.Queue[Optional[AudioItem]] = queue.Queue(maxsize=audio_maxsize)

        self._processor_thread: Optional[threading.Thread] = None
        self._player_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
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
        self._log("Pipeline baslatildi (2 thread).", "info")

    def stop(self):
        if not self._running:
            return
        self._running = False

        # Sentinel gonder, iki thread de uyansın
        for q in (self._text_queue, self._audio_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        for t in (self._processor_thread, self._player_thread):
            if t:
                t.join(timeout=5)

        self._processor_thread = None
        self._player_thread = None
        self._log("Pipeline durduruldu.", "info")

    def push(self, speaker: str, text: str):
        """
        Yeni altyaziyi kuyruga ekler.
        Metin kuyrugu doluysa en eski ogeyi at, yenisini ekle.
        """
        item = PipelineItem(speaker, text)
        while True:
            try:
                self._text_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._text_queue.get_nowait()
                except queue.Empty:
                    pass

    # ── Asama 1: Islemci Thread ────────────────────────────────────────

    def _processor_loop(self):
        while self._running:
            try:
                item: Optional[PipelineItem] = self._text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                # Oynatici icin de sentinel
                try:
                    self._audio_queue.put_nowait(None)
                except queue.Full:
                    pass
                break
            self._process(item)

    def _process(self, item: PipelineItem):
        speaker, text = item.speaker, item.text
        self._log(f"[PROC] {speaker}: {text}", "info")

        slot = _next_slot()
        tts_out = OUTPUT_DIR / f"tts_{slot}.wav"
        rvc_out = OUTPUT_DIR / f"rvc_{slot}.wav"

        # 1. Ceviri
        if self._translator and self._translator.enabled:
            translated = self._translator.translate(text)
            if translated != text:
                self._log(f"[CEV] {text[:40]} -> {translated[:40]}", "tts")
            text = translated

        # 2. Duygu analizi
        emotion = "neutral"
        pitch_delta = 0
        speed_delta = 0.0
        if self._analyzer and self._analyzer.enabled:
            result = self._analyzer.analyze(speaker, text)
            emotion = result.get("emotion", "neutral")
            params  = result.get("params", {})
            pitch_delta = params.get("pitch_delta", 0)
            speed_delta = params.get("speed_delta", 0.0)
            self._log(f"[DUYGU] {speaker}: {emotion}", "info")

        # 3. TTS
        wav_path = self._tts.synthesize(
            text,
            speaker=speaker,
            emotion=emotion,
            speed_delta=speed_delta,
            output_path=str(tts_out),
        )
        if not wav_path:
            self._log("[HATA] TTS cikisi alinamadi.", "error")
            return
        self._log(f"[TTS] Slot {slot} hazir.", "tts")

        # 4. RVC
        if self._rvc:
            rvc_path = self._rvc.convert_for_character(
                wav_path,
                character=speaker,
                output_path=str(rvc_out),
                pitch_override_delta=pitch_delta,
            )
            final_path = rvc_path if rvc_path else wav_path
            tag = "rvc" if rvc_path else "rvc"
            self._log(f"[RVC] Slot {slot} {'tamam' if rvc_path else 'bypass'}.", tag)
        else:
            final_path = wav_path

        # Ses kuyruğuna ekle (oynatici thread alir)
        audio_item = AudioItem(final_path)
        while self._running:
            try:
                self._audio_queue.put(audio_item, timeout=0.5)
                break
            except queue.Full:
                # Oynatici yetisemiyorsa bekle
                pass

    # ── Asama 2: Oynatici Thread ───────────────────────────────────────

    def _player_loop(self):
        while self._running:
            try:
                item: Optional[AudioItem] = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            self._play(item.wav_path)

    def _play(self, wav_path: str):
        try:
            sr, data = wavfile.read(wav_path)
            self._log(f"[SES] {Path(wav_path).name}", "audio")

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
