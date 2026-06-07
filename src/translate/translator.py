"""
src/translate/translator.py
Gelişmiş Çeviri Modülü.
Helsinki-NLP (Yerel), Google Translate (Ücretsiz) ve DeepL (API) motorlarını destekler.
Karakter isimlerini koruma ve çeviri önbelleği (Cache) mekanizması içerir.
"""

from __future__ import annotations

import re
import urllib.parse
import threading
import logging
from typing import Callable, Optional, Dict
import requests

logger = logging.getLogger("UMAY.Translator")

_instance: Optional["Translator"] = None
_instance_lock = threading.Lock()


def get_translator(
    config: dict,
    on_status: Optional[Callable[[str], None]] = None,
) -> "Translator":
    """Global Translator singleton'ini döner."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Translator(config, on_status=on_status)
    return _instance


class Translator:
    """
    Çoklu motor destekli çeviri modülü.
    İsim koruması ve çeviri önbelleği barındırır.
    """

    DEFAULT_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-tr"

    def __init__(
        self,
        config: dict,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self._on_status = on_status or (lambda _: None)
        self._lock = threading.Lock()
        
        # Sınıf değişkenleri
        self._tokenizer = None
        self._model = None
        self._loaded = False
        self._device = self._detect_device()

        # Önbellek (Cache)
        self.cache: Dict[str, str] = {}
        self.cache_lock = threading.Lock()

        # Konfigürasyonu yükle
        self.update_settings_from_config(config)

    def update_settings_from_config(self, config: dict):
        """Uygulama ayarlarını günceller."""
        tr_cfg = config.get("translate", {})
        self.enabled: bool = tr_cfg.get("enabled", False)
        self.engine: str = tr_cfg.get("engine", "google").lower()  # google, opus, deepl
        self.api_key: str = tr_cfg.get("api_key", "")
        self.source_lang: str = tr_cfg.get("source_lang", "eng")
        self.model_name: str = tr_cfg.get("model", self.DEFAULT_MODEL)

        # Temizlik
        if not self.enabled:
            self.unload()

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
        """Opus-MT modeli seçilmişse yerel olarak yükler (bloklayan)."""
        if self.engine != "opus":
            self._loaded = True
            return True

        with self._lock:
            if self._loaded and self._model is not None:
                return True
            try:
                self._notify(f"Opus-MT yükleniyor: {self.model_name}")
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
                        self._notify("Önbellek bozuk, yeniden indiriliyor...")
                        _do_load(force_download=True)
                    else:
                        raise

                self._loaded = True
                self._notify("Opus-MT çeviri modeli hazır.")
                return True
            except ImportError:
                self._notify("HATA: transformers yüklü değil. pip install transformers sentencepiece")
                return False
            except Exception as e:
                self._notify(f"Çeviri modeli yükleme hatası: {e}")
                return False

    def load_async(self, on_done: Optional[Callable[[bool], None]] = None):
        def _run():
            ok = self.load()
            if on_done:
                on_done(ok)
        threading.Thread(target=_run, daemon=True).start()

    def _protect_speaker(self, text: str) -> tuple[Optional[str], str]:
        """
        Karakter isimlerini korumak için altyazıyı ayırır.
        'GLaDOS: Hello' -> ('GLaDOS: ', 'Hello')
        """
        match = re.match(r"^([A-ZÇĞİÖŞÜa-zçğışöü\d\s\-\_]+):\s*(.*)", text)
        if match:
            speaker = match.group(1).strip()
            body = match.group(2).strip()
            return f"{speaker}: ", body
        return None, text

    def translate(self, text: str) -> str:
        """Metni Türkçeye çevirir (Cache + İsim Korumalı)."""
        if not self.enabled or not text.strip():
            return text

        # 1. Konuşmacı ismini koru
        prefix, body = self._protect_speaker(text)
        
        # 2. Önbellek kontrolü
        with self.cache_lock:
            if body in self.cache:
                cached_translated = self.cache[body]
                return f"{prefix or ''}{cached_translated}"

        # 3. Model yükleme kontrolü (Opus için)
        if self.engine == "opus" and not self._loaded:
            if not self.load():
                return text

        # 4. İlgili motor ile çevir
        translated_body = body
        try:
            if self.engine == "google":
                translated_body = self._translate_google(body)
            elif self.engine == "deepl":
                translated_body = self._translate_deepl(body)
            elif self.engine == "opus":
                translated_body = self._translate_opus(body)
        except Exception as e:
            logger.error(f"Çeviri hatası ({self.engine}): {e}")
            translated_body = body

        # 5. Önbelleğe kaydet (Limit 1000 cümle)
        with self.cache_lock:
            if len(self.cache) < 1000:
                self.cache[body] = translated_body
            else:
                self.cache.clear()
                self.cache[body] = translated_body

        return f"{prefix or ''}{translated_body}"

    def _translate_google(self, text: str) -> str:
        """Ücretsiz Google Translate API kullanarak çevirir."""
        sl = "en" if "eng" in self.source_lang or "en" in self.source_lang else "auto"
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl=tr&dt=t&q={urllib.parse.quote(text)}"
        
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data and data[0]:
                translated_parts = [part[0] for part in data[0] if part[0]]
                return "".join(translated_parts).strip()
        raise RuntimeError(f"Google Translate API hatası: {res.status_code}")

    def _translate_deepl(self, text: str) -> str:
        """DeepL API kullanarak çevirir."""
        if not self.api_key:
            self._notify("HATA: DeepL API anahtarı boş.")
            return text

        url = "https://api-free.deepl.com/v2/translate" if self.api_key.endswith(":fx") else "https://api.deepl.com/v2/translate"
        headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
        data = {
            "text": [text],
            "target_lang": "TR"
        }
        
        res = requests.post(url, headers=headers, json=data, timeout=5)
        if res.status_code == 200:
            result = res.json()
            if result and "translations" in result:
                return result["translations"][0]["text"].strip()
        raise RuntimeError(f"DeepL API hatası: {res.status_code}")

    def _translate_opus(self, text: str) -> str:
        """Opus-MT (transformers) yerel modeli ile çevirir."""
        with self._lock:
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
            return self._tokenizer.decode(translated[0], skip_special_tokens=True).strip()

    def unload(self):
        """Opus-MT modelini bellekten kaldırır."""
        with self._lock:
            if self._model is not None:
                self._model = None
                self._tokenizer = None
                self._loaded = False
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                self._notify("Opus-MT çeviri modeli bellekten kaldırıldı.")

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        source_lang: Optional[str] = None,
        engine: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Ayarları günceller ve gerekirse modeli yükler/boşaltır."""
        with self._lock:
            if enabled is not None:
                prev = self.enabled
                self.enabled = enabled
                if enabled and self.engine == "opus" and not self._loaded:
                    self.load_async()
                elif not enabled and prev:
                    self.unload()
            
            if engine is not None:
                prev_engine = self.engine
                self.engine = engine.lower()
                if self.enabled:
                    if self.engine == "opus" and not self._loaded:
                        self.load_async()
                    elif prev_engine == "opus" and self.engine != "opus":
                        self.unload()
            
            if source_lang is not None:
                self.source_lang = source_lang
            if api_key is not None:
                self.api_key = api_key

    def is_ready(self) -> bool:
        if self.engine == "opus":
            return self._loaded
        return self.enabled
