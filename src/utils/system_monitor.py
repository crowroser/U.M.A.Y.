"""
src/utils/system_monitor.py
Canlı sistem kaynak izleme modülü.
CPU, RAM, GPU ve VRAM metriklerini periyodik olarak toplar.
"""

import subprocess
import threading
import time
import logging
from collections import deque
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger("UMAY.SystemMonitor")


class SystemMonitor:
    """
    Sistem kaynaklarını (CPU, RAM, GPU, VRAM) izleyen ve UI'a raporlayan sınıf.
    """

    def __init__(
        self,
        interval: float = 2.0,
        max_history: int = 60,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.interval = interval
        self.max_history = max_history
        self.callback = callback

        self.history: deque = deque(maxlen=max_history)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """İzleme işlemini arka planda başlatır."""
        with self._lock:
            if not self.running:
                self.running = True
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="SystemMonitorThread"
                )
                self._thread.start()
                logger.info("SystemMonitor başlatıldı.")

    def stop(self):
        """İzleme işlemini durdurur."""
        with self._lock:
            self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            logger.info("SystemMonitor durduruldu.")

    def _query_gpu(self) -> Dict[str, float]:
        """
        nvidia-smi kullanarak GPU ve VRAM durumunu sorgular.
        Başarısız olursa PyTorch fallback kullanır.
        """
        stats = {
            "gpu_util": 0.0,
            "vram_total": 0.0,
            "vram_used": 0.0,
            "vram_percent": 0.0,
        }
        try:
            # nvidia-smi sorgusu (Toplam bellek, Kullanılan bellek, GPU Kullanım yüzdesi)
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                shell=True,
            )
            out = res.stdout.strip()
            if out:
                parts = [x.strip() for x in out.split(",")]
                if len(parts) >= 3:
                    total = float(parts[0])
                    used = float(parts[1])
                    gpu_util = float(parts[2])
                    percent = (used / total) * 100.0 if total > 0 else 0.0
                    stats["gpu_util"] = gpu_util
                    stats["vram_total"] = total
                    stats["vram_used"] = used
                    stats["vram_percent"] = percent
                    return stats
        except Exception:
            # nvidia-smi çalışmazsa PyTorch fallback
            pass

        try:
            import torch

            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / (
                    1024 * 1024
                )
                used = torch.cuda.memory_allocated(0) / (1024 * 1024)
                percent = (used / total) * 100.0 if total > 0 else 0.0
                stats["vram_total"] = total
                stats["vram_used"] = used
                stats["vram_percent"] = percent
        except Exception:
            pass

        return stats

    def _run(self):
        import psutil

        while True:
            with self._lock:
                if not self.running:
                    break

            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                ram_percent = ram.percent
                ram_used_gb = ram.used / (1024**3)
                ram_total_gb = ram.total / (1024**3)

                gpu_stats = self._query_gpu()

                stats = {
                    "timestamp": time.time(),
                    "cpu": cpu,
                    "ram_percent": ram_percent,
                    "ram_used": ram_used_gb,
                    "ram_total": ram_total_gb,
                    "gpu": gpu_stats["gpu_util"],
                    "vram_percent": gpu_stats["vram_percent"],
                    "vram_used": gpu_stats["vram_used"],
                    "vram_total": gpu_stats["vram_total"],
                }

                self.history.append(stats)

                if self.callback:
                    try:
                        self.callback(stats)
                    except Exception as ce:
                        logger.error(f"SystemMonitor callback hatası: {ce}")

            except Exception as e:
                logger.error(f"SystemMonitor veri toplama hatası: {e}")

            time.sleep(self.interval)
