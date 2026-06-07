"""
src/pipeline/tts_cache.py
Akilli TTS cache — ayni altyazi tekrar geldiginde TTS'i atlar.

Mimari:
- RAM cache: dict[hash -> (sr, numpy_array)] — anlik erisim (~0ms)
- LRU eviction: bellek limiti asildiginda en eski oge silinir
- Text hash: md5(text + speaker + emotion) -> benzersiz anahtar
- Thread-safe: threading.Lock ile esanli erisim

Oyunlarda ayni karakter ayni repligi sik tekrarlar (savas cigliklari,
selamlamalar, NPC tekrar diyaloglari). Cache tek basina ortalama
gecikmeyi %30-50 azaltabilir.
"""

import hashlib
import threading
import time
import sys
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


class TTSCache:
    """
    LRU RAM cache — tekrarlayan TTS cagrilarini atlamak icin.

    Kullanim:
        cache = TTSCache(max_memory_mb=200)
        
        # Ara
        result = cache.get("Merhaba!", "GLaDOS", "neutral")
        if result:
            sr, data = result  # ~0ms
        else:
            sr, data = tts.synthesize_to_array(...)  # ~1-3s
            cache.put("Merhaba!", "GLaDOS", "neutral", sr, data)
    """

    def __init__(self, max_memory_mb: int = 200, enabled: bool = True):
        self._enabled = enabled
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._current_bytes = 0
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Istatistikler
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_saved_ms = 0  # Tahmini tasarruf (cache hit basina avg TTS suresi)
        self._avg_tts_ms = 1500  # Baslangic tahmini

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @staticmethod
    def _make_key(text: str, speaker: str, emotion: str) -> str:
        """Metin + konusmaci + duygu icin benzersiz hash olusturur."""
        raw = f"{text.strip().lower()}|{(speaker or '').strip().lower()}|{(emotion or 'neutral').strip().lower()}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(
        self, text: str, speaker: str = "", emotion: str = "neutral"
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        Cache'den ses verisini arar.
        Bulursa (sr, numpy_array) doner ve entry'yi LRU basina tasir.
        Bulamazsa None doner.
        """
        if not self._enabled:
            return None

        key = self._make_key(text, speaker, emotion)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # LRU: en son kullanilani sona tasir
            self._cache.move_to_end(key)
            entry.hit_count += 1
            entry.last_access = time.monotonic()
            self._hits += 1
            self._total_saved_ms += self._avg_tts_ms
            return (entry.sr, entry.data.copy())  # copy: thread safety

    def put(
        self,
        text: str,
        speaker: str,
        emotion: str,
        sr: int,
        data: np.ndarray,
        tts_duration_ms: int = 0,
    ):
        """
        Ses verisini cache'e ekler.
        Bellek limiti asilirsa en eski ogeler silinir (LRU eviction).
        """
        if not self._enabled:
            return

        key = self._make_key(text, speaker, emotion)
        entry_size = data.nbytes + 128  # numpy array + overhead

        # Cok buyuk tek parca: cache'e almaya degmez
        if entry_size > self._max_bytes * 0.5:
            return

        # Ortalama TTS suresini guncelle
        if tts_duration_ms > 0:
            self._avg_tts_ms = int(
                self._avg_tts_ms * 0.8 + tts_duration_ms * 0.2
            )

        with self._lock:
            # Zaten varsa guncelle
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_bytes -= old_entry.data.nbytes + 128

            # Yer ac (LRU eviction)
            while self._current_bytes + entry_size > self._max_bytes and self._cache:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._current_bytes -= evicted_entry.data.nbytes + 128
                self._evictions += 1

            # Ekle
            self._cache[key] = _CacheEntry(
                sr=sr,
                data=data.copy(),
                text_preview=text[:60],
                speaker=speaker,
            )
            self._current_bytes += entry_size

    def clear(self):
        """Tum cache'i temizler."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def get_stats(self) -> dict:
        """Cache istatistiklerini doner."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "memory_mb": round(self._current_bytes / (1024 * 1024), 1),
                "max_memory_mb": round(self._max_bytes / (1024 * 1024), 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 2) if total > 0 else 0.0,
                "evictions": self._evictions,
                "saved_ms": self._total_saved_ms,
                "avg_tts_ms": self._avg_tts_ms,
            }

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        max_memory_mb: Optional[int] = None,
    ):
        """Cache ayarlarini gunceller."""
        if enabled is not None:
            self._enabled = enabled
        if max_memory_mb is not None:
            self._max_bytes = max_memory_mb * 1024 * 1024
            # Yeni limitin altina dusur
            with self._lock:
                while self._current_bytes > self._max_bytes and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._current_bytes -= evicted.data.nbytes + 128
                    self._evictions += 1


class _CacheEntry:
    """Cache'deki tek bir ses verisi ogesi."""

    __slots__ = ("sr", "data", "text_preview", "speaker", "hit_count", "created", "last_access")

    def __init__(self, sr: int, data: np.ndarray, text_preview: str, speaker: str):
        self.sr = sr
        self.data = data
        self.text_preview = text_preview
        self.speaker = speaker
        self.hit_count = 0
        self.created = time.monotonic()
        self.last_access = time.monotonic()
