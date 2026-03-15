"""
src/translate/translator.py
Yerel Opus-MT ile otomatik ceviri modulu.
Model CUDA/CPU'ya bir kez yuklenir ve bellekte kalir (singleton).
enabled=False ise translate() metni aynen geri doner.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

_instance: Optional["Translator"] = None
_instance_lock = threading.Lock()


def get_translator(
    config: dict,
    on_status: Optional[Callable[[str], None]] = None,
) -> "Translator":
    """Global Translator singleton'ini doner."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Translator(config, on_status=on_status)
    return _instance


class Translator:
    """
    Helsinki-NLP/opus-mt-tc-big-en-tr modeli ile Ingilizce → Turkce ceviri.
    enabled=False ise ceviri atlanir, girdi aynen doner.
    """

    DEFAULT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-tr"

    def __init__(
        self,
        config: dict,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        tr_cfg = config.get("translate", {})
        self.enabled: bool = tr_cfg.get("enabled", False)
        self.source_lang: str = tr_cfg.get("source_lang", "eng")
        self.model_name: str = tr_cfg.get("model", self.DEFAULT_MODEL)
        self._on_status = on_status or (lambda _: None)

        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._lock = threading.Lock()
        self._device = self._detect_device()
        self._notify(f"Ceviri cihazi: {self._device}")

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
        """Opus-MT modelini yukler (bloklayan)."""
        with self._lock:
            if self._loaded:
                return True
            try:
                self._notify(f"Opus-MT yukleniyor: {self.model_name}")
                from transformers import MarianMTModel, MarianTokenizer
                from src.utils.download_progress import download_progress_context

                def _do_load(force_download: bool = False):
                    with download_progress_context(self._notify):
                        self._tokenizer = MarianTokenizer.from_pretrained(
                            self.model_name, force_download=force_download
                        )
                        self._model = MarianMTModel.from_pretrained(
                            self.model_name, force_download=force_download
                        )
                    if self._device == "cuda":
                        self._model = self._model.to("cuda")

                try:
                    _do_load(force_download=False)
                except (OSError, ImportError) as e:
                    err_msg = str(e).lower()
                    if any(x in err_msg for x in ("vocabulary", "source.spm", "not found", "no such file")):
                        self._notify("Onbellek bozuk, yeniden indiriliyor...")
                        _do_load(force_download=True)
                    else:
                        raise

                self._loaded = True
                self._notify("Ceviri modeli hazir.")
                return True
            except ImportError:
                self._notify("HATA: transformers yuklu degil. pip install transformers sentencepiece")
                return False
            except Exception as e:
                self._notify(f"Ceviri modeli yukleme hatasi: {e}")
                return False

    def load_async(self, on_done: Optional[Callable[[bool], None]] = None):
        def _run():
            ok = self.load()
            if on_done:
                on_done(ok)
        threading.Thread(target=_run, daemon=True).start()

    def translate(self, text: str) -> str:
        """
        Metni Turkceye cevir.
        - enabled=False ise metni aynen doner
        - Model yuklu degilse ilk cagride otomatik yukler
        """
        if not self.enabled or not text.strip():
            return text

        if not self._loaded:
            if not self.load():
                return text

        with self._lock:
            try:
                inputs = self._tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                if self._device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                translated = self._model.generate(**inputs)
                result = self._tokenizer.decode(translated[0], skip_special_tokens=True)
                self._notify(f"[CEV] {text[:40]} → {result[:40]}")
                return result
            except Exception as e:
                self._notify(f"Ceviri hatasi: {e}")
                return text

    def unload(self):
        """Modeli bellekten kaldirir, VRAM'i serbest birakir."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._notify("Ceviri modeli bellekten kaldirildi.")

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        source_lang: Optional[str] = None,
    ):
        if enabled is not None:
            prev = self.enabled
            self.enabled = enabled
            if enabled and not self._loaded:
                self.load_async()
            elif not enabled and prev and self._loaded:
                self.unload()
        if source_lang is not None:
            self.source_lang = source_lang

    def is_ready(self) -> bool:
        return self._loaded
