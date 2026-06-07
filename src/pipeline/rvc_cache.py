"""
src/pipeline/rvc_cache.py
RVC Ses Dönüşüm Önbellek Modülü.
"""

import hashlib
import threading
import time
import logging
from collections import OrderedDict
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("UMAY.RVCCache")


class RVCAudioCache:
    """
    RVC Ses Dönüşüm sonuçlarını (numpy) RAM ve disk üzerinde saklayan sınıf.
    Anahtar: md5(input_audio_bytes) + character + pitch_override
    """

    def __init__(
        self,
        max_memory_mb: int = 200,
        enabled: bool = True,
        disk_enabled: bool = False,
        disk_limit_mb: int = 500,
    ):
        self._enabled = enabled
        self._max_bytes = max_memory_mb * 1024 * 1024
        self._current_bytes = 0
        self._cache: OrderedDict[str, _RVCCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Disk cache ayarları
        self._disk_enabled = disk_enabled
        self._disk_limit_bytes = disk_limit_mb * 1024 * 1024
        self._cache_dir = Path(__file__).parent.parent.parent / "models" / "cache" / "rvc"

        if self._enabled and self._disk_enabled:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"RVC Disk Cache klasoru olusturulamadi: {e}")

        # İstatistikler
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def disk_enabled(self) -> bool:
        return self._disk_enabled

    @disk_enabled.setter
    def disk_enabled(self, value: bool):
        self._disk_enabled = value
        if value:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _make_key(self, audio_data: np.ndarray, character: str, pitch: int) -> str:
        """Giriş ses verisi, karakter adı ve pitch ayarına göre benzersiz md5 üretir."""
        try:
            audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        except Exception:
            # Fallback (veride hata varsa)
            audio_hash = str(hash(audio_data.data.tobytes() if hasattr(audio_data, 'data') else audio_data))
        raw = f"{audio_hash}|{(character or '').strip().lower()}|{pitch}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(
        self, audio_data: np.ndarray, character: str = "", pitch: int = 0
    ) -> Optional[Tuple[int, np.ndarray]]:
        """Önbellekten RVC sonucunu arar."""
        if not self._enabled or audio_data is None:
            return None

        key = self._make_key(audio_data, character, pitch)

        # 1. RAM Cache kontrolü
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                entry.hit_count += 1
                entry.last_access = time.monotonic()
                self._hits += 1
                return (entry.sr, entry.data.copy())

        # 2. Disk Cache kontrolü
        if self._disk_enabled:
            filepath = self._cache_dir / f"{key}.wav"
            if filepath.exists():
                try:
                    import scipy.io.wavfile as wavfile
                    sr, data = wavfile.read(str(filepath))
                    
                    # RAM cache'e yaz (yazarken tekrar diske yazmasın diye write_disk=False)
                    self.put(audio_data, character, pitch, sr, data, write_disk=False)
                    
                    try:
                        filepath.touch()
                    except OSError:
                        pass
                        
                    self._hits += 1
                    return sr, data.copy()
                except Exception as e:
                    logger.error(f"RVC Disk cache okuma hatasi: {e}")

        self._misses += 1
        return None

    def put(
        self,
        audio_data: np.ndarray,
        character: str,
        pitch: int,
        sr: int,
        output_data: np.ndarray,
        write_disk: bool = True,
    ):
        """Sonucu önbelleğe ekler."""
        if not self._enabled or audio_data is None or output_data is None:
            return

        key = self._make_key(audio_data, character, pitch)
        entry_size = output_data.nbytes + 128

        if entry_size > self._max_bytes * 0.5:
            return

        # 1. RAM Cache'e ekleme
        with self._lock:
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._current_bytes -= old_entry.data.nbytes + 128

            while self._current_bytes + entry_size > self._max_bytes and self._cache:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._current_bytes -= evicted_entry.data.nbytes + 128
                self._evictions += 1

            self._cache[key] = _RVCCacheEntry(
                sr=sr,
                data=output_data.copy(),
                character=character,
            )
            self._current_bytes += entry_size

        # 2. Disk Cache'e ekleme
        if self._disk_enabled and write_disk:
            try:
                filepath = self._cache_dir / f"{key}.wav"
                import scipy.io.wavfile as wavfile
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                wavfile.write(str(filepath), sr, output_data)
                self._manage_disk_eviction()
            except Exception as e:
                logger.error(f"RVC Disk cache yazma hatasi: {e}")

    def _manage_disk_eviction(self):
        """Disk limit aşımında LRU temizliği."""
        if not self._cache_dir.exists():
            return
        try:
            files = list(self._cache_dir.glob("*.wav"))
            file_stats = []
            for f in files:
                try:
                    stat = f.stat()
                    file_stats.append((f, stat.st_size, stat.st_mtime))
                except OSError:
                    pass

            total_bytes = sum(s for _, s, _ in file_stats)
            if total_bytes <= self._disk_limit_bytes:
                return

            file_stats.sort(key=lambda x: x[2])

            for f, size, _ in file_stats:
                if total_bytes <= self._disk_limit_bytes:
                    break
                try:
                    f.unlink()
                    total_bytes -= size
                    self._evictions += 1
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"RVC Disk cache temizleme hatasi: {e}")

    def clear(self):
        """RAM ve Disk önbelleği temizler."""
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
            
        if self._disk_enabled and self._cache_dir.exists():
            try:
                for f in self._cache_dir.glob("*.wav"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
            except Exception as e:
                logger.error(f"RVC Disk cache temizlenirken hata: {e}")

    def get_stats(self) -> dict:
        """İstatistikleri döner."""
        with self._lock:
            total = self._hits + self._misses
            
            disk_bytes = 0
            disk_entries = 0
            if self._disk_enabled and self._cache_dir.exists():
                try:
                    for f in self._cache_dir.glob("*.wav"):
                        disk_bytes += f.stat().st_size
                        disk_entries += 1
                except Exception:
                    pass
                    
            return {
                "entries": len(self._cache),
                "memory_mb": round(self._current_bytes / (1024 * 1024), 1),
                "max_memory_mb": round(self._max_bytes / (1024 * 1024), 1),
                "disk_entries": disk_entries,
                "disk_size_mb": round(disk_bytes / (1024 * 1024), 1),
                "max_disk_mb": round(self._disk_limit_bytes / (1024 * 1024), 1),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 2) if total > 0 else 0.0,
                "evictions": self._evictions,
            }

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        max_memory_mb: Optional[int] = None,
        disk_enabled: Optional[bool] = None,
        disk_limit_mb: Optional[int] = None,
    ):
        if enabled is not None:
            self._enabled = enabled
        if max_memory_mb is not None:
            self._max_bytes = max_memory_mb * 1024 * 1024
            with self._lock:
                while self._current_bytes > self._max_bytes and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._current_bytes -= evicted.data.nbytes + 128
                    self._evictions += 1
        if disk_enabled is not None:
            self._disk_enabled = disk_enabled
            if disk_enabled:
                try:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
        if disk_limit_mb is not None:
            self._disk_limit_bytes = disk_limit_mb * 1024 * 1024
            self._manage_disk_eviction()


class _RVCCacheEntry:
    __slots__ = ("sr", "data", "character", "hit_count", "created", "last_access")

    def __init__(self, sr: int, data: np.ndarray, character: str):
        self.sr = sr
        self.data = data
        self.character = character
        self.hit_count = 0
        self.created = time.monotonic()
        self.last_access = time.monotonic()
