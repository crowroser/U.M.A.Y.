"""
src/pipeline/telemetry.py
Merkezi performans olcum ve istatistik toplama modulu.

Ozellikler:
- PipelineTimer: context manager ile adim suresi olcumu
- PipelineStats: son N islem icin ortalama/min/max/p95 hesaplama
- Rolling window: son 100 islem saklama
- Thread-safe tum operasyonlar
- SPR (Subtitle Processing Rate): saniyede islenen altyazi sayisi
"""

import time
import threading
from collections import deque, defaultdict
from typing import Optional


class PipelineTimer:
    """
    Pipeline adimlarinin suresini olcer.

    Kullanim:
        timer = PipelineTimer()
        with timer.measure("tts"):
            tts_result = tts.synthesize(...)
        print(timer.get_durations())  # {"tts": 1234}
    """

    def __init__(self):
        self._durations: dict[str, int] = {}
        self._start_time: float = 0.0
        self._total_start: float = 0.0

    def start_total(self):
        """Toplam pipeline suresini baslatir."""
        self._total_start = time.perf_counter()
        self._durations.clear()

    def stop_total(self):
        """Toplam pipeline suresini durdurur."""
        if self._total_start > 0:
            self._durations["total"] = int(
                (time.perf_counter() - self._total_start) * 1000
            )

    def measure(self, step_name: str):
        """Context manager: adim suresini olcer."""
        return _TimerContext(self, step_name)

    def record(self, step_name: str, duration_ms: int):
        """Manuel sure kaydeder."""
        self._durations[step_name] = duration_ms

    def get_durations(self) -> dict[str, int]:
        """Tum olculen sureleri doner (ms cinsinden)."""
        return dict(self._durations)


class _TimerContext:
    """PipelineTimer icin context manager yardimcisi."""

    __slots__ = ("_timer", "_name", "_t0")

    def __init__(self, timer: PipelineTimer, name: str):
        self._timer = timer
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        ms = int((time.perf_counter() - self._t0) * 1000)
        self._timer.record(self._name, ms)


class PipelineStats:
    """
    Son N islem icin istatistik toplama ve raporlama.

    Her islem bir timing dict'i olarak kaydedilir:
    {"tts": 1200, "rvc": 300, "total": 1800, ...}

    Raporlar:
    - Adim bazli ortalama/min/max/p95
    - Toplam ortalama gecikme
    - SPR (Subtitle Processing Rate)
    - Cache hit rate
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._records: deque[dict] = deque(maxlen=window_size)
        self._lock = threading.Lock()

        # Cache istatistikleri
        self._cache_hits = 0
        self._cache_misses = 0

        # SPR hesaplama icin
        self._spr_timestamps: deque[float] = deque(maxlen=60)

        # Toplam islem sayaci
        self._total_processed = 0

    def record(self, timing: dict):
        """Yeni bir pipeline isleminin zamanlama verilerini kaydeder."""
        with self._lock:
            self._records.append(timing)
            self._spr_timestamps.append(time.monotonic())
            self._total_processed += 1

    def record_cache_hit(self):
        """Cache hit kaydeder."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self):
        """Cache miss kaydeder."""
        with self._lock:
            self._cache_misses += 1

    def get_stats(self) -> dict:
        """
        Tum istatistikleri doner.

        Donus formati:
        {
            "count": 42,
            "total_processed": 142,
            "steps": {
                "tts": {"avg": 1200, "min": 800, "max": 2000, "p95": 1800},
                "rvc": {"avg": 300, "min": 200, "max": 500, "p95": 450},
                "total": {"avg": 1800, "min": 1200, "max": 2800, "p95": 2500},
            },
            "spr": 0.8,  # subtitle per second
            "cache": {"hits": 10, "misses": 32, "hit_rate": 0.24},
            "avg_latency_ms": 1800,
        }
        """
        with self._lock:
            if not self._records:
                return self._empty_stats()

            # Adim bazli istatistikler
            step_values: dict[str, list[int]] = defaultdict(list)
            for rec in self._records:
                for key, val in rec.items():
                    if isinstance(val, (int, float)) and not key.startswith("chunk_"):
                        step_values[key].append(int(val))

            steps = {}
            for step_name, values in step_values.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                steps[step_name] = {
                    "avg": int(sum(sorted_vals) / n),
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "p95": sorted_vals[int(n * 0.95)] if n >= 2 else sorted_vals[-1],
                    "count": n,
                }

            # SPR hesaplama (son 60 saniyedeki islem sayisi)
            now = time.monotonic()
            recent = [t for t in self._spr_timestamps if now - t <= 60.0]
            spr = len(recent) / 60.0 if recent else 0.0

            # Cache
            total_cache = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache if total_cache > 0 else 0.0
            )

            # Ortalama gecikme
            total_vals = step_values.get("total", [])
            avg_latency = int(sum(total_vals) / len(total_vals)) if total_vals else 0

            return {
                "count": len(self._records),
                "total_processed": self._total_processed,
                "steps": steps,
                "spr": round(spr, 2),
                "cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate": round(cache_hit_rate, 2),
                },
                "avg_latency_ms": avg_latency,
            }

    def get_last_timing(self) -> Optional[dict]:
        """Son islenen ogemin zamanlama verisini doner."""
        with self._lock:
            return dict(self._records[-1]) if self._records else None

    def reset(self):
        """Tum istatistikleri sifirlar."""
        with self._lock:
            self._records.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._spr_timestamps.clear()
            self._total_processed = 0

    def _empty_stats(self) -> dict:
        return {
            "count": 0,
            "total_processed": 0,
            "steps": {},
            "spr": 0.0,
            "cache": {"hits": 0, "misses": 0, "hit_rate": 0.0},
            "avg_latency_ms": 0,
        }

    def format_summary(self) -> str:
        """Okunabilir tek satirlik ozet."""
        s = self.get_stats()
        parts = []
        for step in ("tts", "rvc", "total"):
            if step in s["steps"]:
                parts.append(f"{step.upper()}: {s['steps'][step]['avg']}ms")
        if s["cache"]["hits"] + s["cache"]["misses"] > 0:
            parts.append(f"Cache: %{int(s['cache']['hit_rate'] * 100)}")
        if s["spr"] > 0:
            parts.append(f"SPR: {s['spr']}/s")
        return " | ".join(parts) if parts else "Veri yok"
