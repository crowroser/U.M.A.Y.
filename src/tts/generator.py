"""
src/tts/generator.py
GPU VRAM singleton TTS modulu.

Ozellikler:
- Standart Coqui model (tts_models/...) veya yerel HF modeli (local_model_dir)
- Multi-reference WAV: karakter + duygu kombinasyonuna gore speaker_wav secimi
- Duygu tabanli hiz (speed) override
- Model bir kez VRAM'e yuklenir, program boyunca bellekte kalir
"""

import os
import threading
from pathlib import Path
from typing import Callable, Optional

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_TTS_DIR = Path(__file__).parent.parent.parent / "models" / "tts"
MODELS_TTS_DIR.mkdir(parents=True, exist_ok=True)

TTS_OUTPUT_PATH = OUTPUT_DIR / "tts_output.wav"

_instance: Optional["TTSGenerator"] = None
_instance_lock = threading.Lock()


def get_tts(config: dict, on_status: Optional[Callable[[str], None]] = None) -> "TTSGenerator":
    """Global TTSGenerator singleton'ini doner."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TTSGenerator(config, on_status=on_status)
    return _instance


def scan_local_tts_models() -> list[dict]:
    """models/tts/ altindaki indirilen HF modellerini tarar."""
    results = []
    if not MODELS_TTS_DIR.exists():
        return results
    for d in MODELS_TTS_DIR.iterdir():
        if not d.is_dir():
            continue
        model_pth = d / "model.pth"
        config_json = d / "config.json"
        if model_pth.exists() and config_json.exists():
            results.append({
                "name": d.name,
                "dir": str(d),
                "model_path": str(model_pth),
                "config_path": str(config_json),
            })
    return results


class TTSGenerator:
    """
    Coqui TTS singleton motoru.
    - Standart model: TTS(model_name)
    - Yerel HF modeli: TTS(model_path=..., config_path=...)
    - synthesize(text, speaker, emotion, speed_delta) ile cagrilir
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        config: dict,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        tts_cfg = config.get("tts", {})
        self.model_name: str = tts_cfg.get("model", self.MODEL_NAME)
        self.local_model_dir: Optional[str] = tts_cfg.get("local_model_dir")
        self.language: str = tts_cfg.get("language", "tr")
        self.speed: float = float(tts_cfg.get("speed", 1.0))
        self.speaker_wav: Optional[str] = tts_cfg.get("speaker_wav")
        self._on_status = on_status or (lambda _: None)

        self._character_refs: dict[str, dict[str, str]] = dict(
            config.get("character_refs", {})
        )

        self._tts = None
        self._loaded = False
        self._lock = threading.Lock()
        self._device = self._detect_device()
        self._notify(f"TTS cihazi: {self._device}")

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _notify(self, msg: str):
        self._on_status(msg)

    def load(self) -> bool:
        """Modeli VRAM'e yukler (bloklayan)."""
        with self._lock:
            if self._loaded:
                return True
            try:
                from TTS.api import TTS

                if self.local_model_dir:
                    local = Path(self.local_model_dir)
                    model_pth = local / "model.pth"
                    config_pth = local / "config.json"
                    self._notify(f"Yerel model yukleniyor: {local.name}")
                    self._tts = TTS(
                        model_path=str(model_pth),
                        config_path=str(config_pth),
                    ).to(self._device)
                else:
                    self._notify(f"TTS modeli yukleniyor ({self._device}): {self.model_name}")
                    self._tts = TTS(self.model_name).to(self._device)

                self._loaded = True
                self._notify("TTS hazir.")
                return True
            except ImportError:
                self._notify("HATA: TTS paketi bulunamadi. pip install TTS")
                return False
            except Exception as e:
                self._notify(f"TTS yukleme hatasi: {e}")
                return False

    def load_async(self, on_done: Optional[Callable[[bool], None]] = None):
        def _run():
            ok = self.load()
            if on_done:
                on_done(ok)
        threading.Thread(target=_run, daemon=True).start()

    def _resolve_ref(
        self, speaker: Optional[str], emotion: Optional[str]
    ) -> Optional[str]:
        """
        Karakter ve duygu icin uygun referans WAV yolunu doner.
        Oncelik sirasi:
        1. character_refs[speaker][emotion]
        2. character_refs[speaker]["default"]
        3. genel speaker_wav
        4. None (varsayilan XTTS sesi kullanilir)
        """
        if speaker:
            key = speaker.strip().lower()
            refs = None
            for k, v in self._character_refs.items():
                if k.strip().lower() == key:
                    refs = v
                    break

            if refs:
                emo_wav = refs.get(emotion or "neutral") or refs.get("default")
                if emo_wav and os.path.isfile(emo_wav):
                    return emo_wav

        if self.speaker_wav and os.path.isfile(self.speaker_wav):
            return self.speaker_wav

        return None

    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        emotion: Optional[str] = None,
        speed_delta: float = 0.0,
        output_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[str]:
        """
        Metni sese cevirir.
        speaker / emotion  -> multi-ref WAV secimi
        speed_delta        -> duygu tabanli hiz ayari
        """
        text = text.strip()
        if not text:
            return None

        if not self._loaded:
            if not self.load():
                return None

        out = output_path or str(TTS_OUTPUT_PATH)
        lang = language or self.language
        effective_speed = max(0.5, min(2.5, self.speed + speed_delta))
        ref_wav = self._resolve_ref(speaker, emotion)

        with self._lock:
            try:
                self._notify(
                    f"TTS [{emotion or 'neutral'}] {speaker or ''}: "
                    f"'{text[:50]}{'...' if len(text) > 50 else ''}'"
                )
                kwargs: dict = {
                    "text": text,
                    "language": lang,
                    "file_path": out,
                    "speed": effective_speed,
                }
                if ref_wav:
                    kwargs["speaker_wav"] = ref_wav
                else:
                    kwargs["speaker"] = "Claribel Dervla"

                self._tts.tts_to_file(**kwargs)
                return out
            except Exception as e:
                self._notify(f"TTS sentez hatasi: {e}")
                return None

    def update_model(self, local_model_dir: str) -> bool:
        """Yerel HF modelini degistirir; mevcut modeli bellekten atar ve yeniden yukler."""
        with self._lock:
            self._tts = None
            self._loaded = False
            self.local_model_dir = local_model_dir
        return self.load()

    def update_model_async(
        self,
        local_model_dir: str,
        on_done: Optional[Callable[[bool], None]] = None,
    ):
        def _run():
            ok = self.update_model(local_model_dir)
            if on_done:
                on_done(ok)
        threading.Thread(target=_run, daemon=True).start()

    def update_character_refs(self, refs: dict):
        """character_refs config'ini gunceller."""
        self._character_refs = dict(refs)

    def is_ready(self) -> bool:
        return self._loaded

    def update_settings(
        self,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        speaker_wav: Optional[str] = None,
    ):
        if language:
            self.language = language
        if speed is not None:
            self.speed = max(0.5, min(2.0, speed))
        if speaker_wav is not None:
            self.speaker_wav = speaker_wav
