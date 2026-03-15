"""
src/llm/analyzer.py
Duygu analizi modulu — singleton, GPU/CPU, rolling context window.

Model: j-hartmann/emotion-english-distilroberta-base  (~66 MB)
7 sinif: anger / disgust / fear / joy / neutral / sadness / surprise

Rolling context: son N cumle birlestirilip modele gonderilir;
bu sayede "Harika..." sarkastik mi yoksa gercekten sevinc mi?
sorusunu daha iyi cevaplar.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

EMOTION_PARAMS: dict[str, dict] = {
    "anger":    {"speed_delta": +0.30, "pitch_delta": +3},
    "fear":     {"speed_delta": +0.20, "pitch_delta": +2},
    "sadness":  {"speed_delta": -0.20, "pitch_delta": -2},
    "joy":      {"speed_delta": +0.15, "pitch_delta": +1},
    "surprise": {"speed_delta": +0.20, "pitch_delta": +2},
    "disgust":  {"speed_delta": -0.10, "pitch_delta": -1},
    "neutral":  {"speed_delta":  0.00, "pitch_delta":  0},
}

DEFAULT_EMOTION = "neutral"

_instance: Optional["SentimentAnalyzer"] = None
_instance_lock = threading.Lock()


def get_analyzer(
    config: dict,
    on_status: Optional[Callable[[str], None]] = None,
) -> "SentimentAnalyzer":
    """Global SentimentAnalyzer singleton'ini doner."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = SentimentAnalyzer(config, on_status=on_status)
    return _instance


class SentimentAnalyzer:
    """
    Transformers text-classification pipeline tabanli duygu analistoru.
    enabled=False ise analyze() her zaman {"emotion": "neutral", "confidence": 1.0} doner.
    """

    DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(
        self,
        config: dict,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        sent_cfg = config.get("sentiment", {})
        self.enabled: bool = sent_cfg.get("enabled", False)
        self.model_name: str = sent_cfg.get("model", self.DEFAULT_MODEL)
        context_size: int = int(sent_cfg.get("context_window", 5))

        self._on_status = on_status or (lambda _: None)
        self._pipe = None
        self._loaded = False
        self._lock = threading.Lock()
        self._device = self._detect_device()
        self._context: deque[str] = deque(maxlen=context_size)
        self._notify(f"Duygu analizi cihazi: {self._device}")

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    def _notify(self, msg: str):
        self._on_status(msg)

    def load(self) -> bool:
        """Modeli yukler (bloklayan)."""
        with self._lock:
            if self._loaded:
                return True
            try:
                self._notify(f"Duygu modeli yukleniyor: {self.model_name}")
                from transformers import pipeline
                from src.utils.download_progress import download_progress_context

                with download_progress_context(self._notify):
                    self._pipe = pipeline(
                        "text-classification",
                        model=self.model_name,
                        top_k=1,
                        device=self._device,
                    )
                self._loaded = True
                self._notify("Duygu modeli hazir.")
                return True
            except ImportError:
                self._notify("HATA: transformers yuklu degil.")
                return False
            except Exception as e:
                self._notify(f"Duygu modeli yukleme hatasi: {e}")
                return False

    def load_async(self, on_done: Optional[Callable[[bool], None]] = None):
        def _run():
            ok = self.load()
            if on_done:
                on_done(ok)
        threading.Thread(target=_run, daemon=True).start()

    def analyze(self, speaker: str, text: str) -> dict:
        """
        Konusmaci + metni rolling context'e ekler ve duygu analizi yapar.
        Doner: {"emotion": str, "confidence": float, "params": dict}
        """
        entry = f"{speaker}: {text}"
        self._context.append(entry)

        if not self.enabled:
            return self._neutral_result()

        if not self._loaded:
            if not self.load():
                return self._neutral_result()

        context_text = " ".join(self._context)

        with self._lock:
            try:
                result = self._pipe(context_text[:512])
                label = result[0][0]["label"].lower()
                score = result[0][0]["score"]
                params = EMOTION_PARAMS.get(label, EMOTION_PARAMS["neutral"])
                self._notify(f"[DUYGU] {speaker}: {label} ({score:.2f})")
                return {"emotion": label, "confidence": score, "params": params}
            except Exception as e:
                self._notify(f"Duygu analiz hatasi: {e}")
                return self._neutral_result()

    def clear_context(self):
        """Bağlamsal hafızayı temizler (yeni oyun sahnesi için)."""
        self._context.clear()

    def unload(self):
        """Modeli bellekten kaldirir, VRAM'i serbest birakir."""
        with self._lock:
            self._pipe = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._notify("Duygu modeli bellekten kaldirildi.")

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        context_window: Optional[int] = None,
    ):
        if enabled is not None:
            prev = self.enabled
            self.enabled = enabled
            if enabled and not self._loaded:
                self.load_async()
            elif not enabled and prev and self._loaded:
                self.unload()
        if context_window is not None:
            new_size = max(1, min(20, context_window))
            old_items = list(self._context)
            self._context = deque(old_items, maxlen=new_size)

    def is_ready(self) -> bool:
        return self._loaded

    @staticmethod
    def _neutral_result() -> dict:
        return {
            "emotion": DEFAULT_EMOTION,
            "confidence": 1.0,
            "params": EMOTION_PARAMS["neutral"],
        }
