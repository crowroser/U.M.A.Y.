"""
src/pipeline/ocr_cache.py
OCR Önbelleği ve Filtreleme Modülü.
Gelişmiş duplicate detection ve temporal (zamansal) filtreleme.
"""

import time
import logging
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

logger = logging.getLogger("UMAY.OCRCache")


def calculate_similarity(a: str, b: str) -> float:
    """İki metin arasındaki SequenceMatcher benzerlik oranını döner."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


class OCRCache:
    """
    OCR okumalarını hafızada tutan, gürültü ve tekrar filtrelemesi yapan sınıf.
    """

    def __init__(
        self,
        history_size: int = 10,
        similarity_threshold: float = 0.90,
        stale_time_sec: float = 8.0,
        stabilize_delay_sec: float = 0.15,
    ):
        self.history_size = history_size
        self.similarity_threshold = similarity_threshold
        self.stale_time_sec = stale_time_sec
        self.stabilize_delay_sec = stabilize_delay_sec

        # Tarihçe: list of (speaker, text, timestamp)
        self._history: List[Tuple[str, str, float]] = []

        # Aday stabilizasyon takibi: key -> (first_seen_time, count)
        self._candidates: dict = {}

    def is_duplicate(self, speaker: str, text: str) -> bool:
        """
        Yeni okunan metnin yakın geçmiştekilerle çakışıp çakışmadığını (fuzzy duplicate) denetler.
        """
        now = time.monotonic()
        # Bayat kayıtları temizle
        self._history = [h for h in self._history if now - h[2] < self.stale_time_sec]

        for hist_sp, hist_tx, _ in self._history:
            # Aynı konuşmacı ise veya konuşmacı belirtilmemişse metin benzerliğini kontrol et
            if not hist_sp or not speaker or hist_sp.lower() == speaker.lower():
                sim = calculate_similarity(text, hist_tx)
                if sim >= self.similarity_threshold:
                    return True
        return False

    def add(self, speaker: str, text: str):
        """Başarıyla işlenen altyazıyı tarihçeye ekler."""
        now = time.monotonic()
        self._history.append((speaker, text, now))
        if len(self._history) > self.history_size:
            self._history.pop(0)

    def process_temporal(self, speaker: str, text: str) -> Optional[Tuple[str, str]]:
        """
        Temporal subtitle filtering:
        Metnin en az stabilize_delay_sec (örneğin 150ms) boyunca istikrarlı şekilde
        göründüğünden emin olur. İlk kez görünen metinler için None döner,
        süre tamamlanınca (stabilize olunca) (speaker, text) döner.
        """
        now = time.monotonic()
        key = f"{speaker}:{text}"

        # Eski adayları temizle (>2 saniye geçmiş olanlar)
        stale_keys = [k for k, v in self._candidates.items() if now - v[0] > 2.0]
        for k in stale_keys:
            self._candidates.pop(k, None)

        if key not in self._candidates:
            self._candidates[key] = (now, 1)
            return None

        first_seen, count = self._candidates[key]
        self._candidates[key] = (first_seen, count + 1)

        if now - first_seen >= self.stabilize_delay_sec:
            # Stabilize oldu, adayı temizle ve döndür
            self._candidates.pop(key, None)
            return speaker, text

        return None

    def clear(self):
        """Önbelleği temizler."""
        self._history.clear()
        self._candidates.clear()
