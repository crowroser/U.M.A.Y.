import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional


OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TTS_OUTPUT_PATH = OUTPUT_DIR / "tts_output.wav"


class TTSEngine:
    """
    Coqui XTTS-v2 kullanarak Turkce TTS motoru.
    Model ilk kullanımda otomatik indirilir (~1.8 GB).
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        language: str = "tr",
        speed: float = 1.0,
        speaker_wav: Optional[str] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.language = language
        self.speed = speed
        self.speaker_wav = speaker_wav
        self._on_status = on_status or (lambda msg: None)
        self._tts = None
        self._lock = threading.Lock()
        self._loaded = False

    def _notify(self, msg: str):
        self._on_status(msg)

    def load(self):
        """TTS modelini yukler (bloklar, arka planda cagir)."""
        with self._lock:
            if self._loaded:
                return
            try:
                self._notify("TTS modeli yukleniyor...")
                from TTS.api import TTS
                self._tts = TTS(self.model_name)
                self._loaded = True
                self._notify("TTS modeli hazir.")
            except ImportError:
                self._notify("HATA: TTS paketi yuklu degil. 'pip install TTS' calistirin.")
                raise
            except Exception as e:
                self._notify(f"TTS yukleme hatasi: {e}")
                raise

    def load_async(self, on_done: Optional[Callable[[], None]] = None):
        """Modeli arka planda yukler."""
        def _run():
            try:
                self.load()
            finally:
                if on_done:
                    on_done()
        threading.Thread(target=_run, daemon=True).start()

    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Metni sese cevir ve wav dosyasına yaz.
        output_path belirtilmezse varsayilan output/tts_output.wav kullanilir.
        Basarili olursa dosya yolunu, hata olursa None doner.
        """
        if not text.strip():
            return None

        if not self._loaded:
            try:
                self.load()
            except Exception:
                return None

        out = output_path or str(TTS_OUTPUT_PATH)

        with self._lock:
            try:
                self._notify(f"TTS isleniyor: {text[:50]}...")

                kwargs = {
                    "text": text,
                    "language": self.language,
                    "file_path": out,
                    "speed": self.speed,
                }

                if self.speaker_wav and os.path.isfile(self.speaker_wav):
                    kwargs["speaker_wav"] = self.speaker_wav
                else:
                    kwargs["speaker"] = "Claribel Dervla"

                self._tts.tts_to_file(**kwargs)
                self._notify("TTS tamamlandi.")
                return out
            except Exception as e:
                self._notify(f"TTS hatasi: {e}")
                return None

    def is_ready(self) -> bool:
        return self._loaded

    def update_settings(self, language: str = None, speed: float = None, speaker_wav: str = None):
        if language:
            self.language = language
        if speed is not None:
            self.speed = max(0.5, min(2.0, speed))
        if speaker_wav is not None:
            self.speaker_wav = speaker_wav
