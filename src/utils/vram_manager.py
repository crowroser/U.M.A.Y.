"""
src/utils/vram_manager.py
Merkezi VRAM Yöneticisi modülü.
GPU VRAM kullanım bütçesini kontrol eder ve gerekirse en eski (LRU) modeli bellekten atar.
"""

import gc
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger("UMAY.VRAMManager")


class VRAMManager:
    """
    Model bazlı VRAM bütçesini yöneten Singleton sınıfı.
    """

    _instance: Optional["VRAMManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VRAMManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        # Sadece ilk kez çağrıldığında çalışır
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.enabled = True
        self.budget_mb = 3000.0  # Varsayılan limit 3 GB
        self.auto_detect = True
        self.device_str = "cuda:0"
        self.device_id = 0
        self.models: Dict[str, Dict[str, Any]] = {}  # model_id -> metadata
        self.lock = threading.Lock()
        self.load_lock = threading.RLock()

    def configure(self, config: Dict[str, Any]):
        """Konfigürasyona göre VRAM yöneticisini ayarlar."""
        with self.lock:
            vram_cfg = config.get("vram", {})
            self.enabled = vram_cfg.get("enabled", True)
            self.budget_mb = float(vram_cfg.get("budget_mb", 3000.0))
            self.auto_detect = vram_cfg.get("auto_detect", True)
            
            # Cihaz belirleme
            device_str = vram_cfg.get("device", "cuda:0")
            self.device_str = device_str
            if "cuda" in device_str:
                try:
                    parts = device_str.split(":")
                    self.device_id = int(parts[1]) if len(parts) > 1 else 0
                except Exception:
                    self.device_id = 0
            else:
                self.device_id = -1  # CPU

            if self.auto_detect and self.device_id >= 0:
                try:
                    import torch
                    if torch.cuda.is_available():
                        total_vram = torch.cuda.get_device_properties(self.device_id).total_memory / (1024 * 1024)
                        # Toplam VRAM'in %70'ini bütçe olarak belirle
                        self.budget_mb = total_vram * 0.70
                        logger.info(f"VRAM Yöneticisi: Otomatik bütçe belirlendi: {self.budget_mb:.1f} MB ({device_str} için Toplam VRAM'in %70'i)")
                except Exception as e:
                    logger.warning(f"VRAM Yöneticisi: CUDA sorgusu başarısız, varsayılan limit kullanılacak: {e}")

    def register_model(
        self,
        model_id: str,
        model_type: str,
        unload_cb: Callable[[], None],
        estimated_vram_mb: float = 500.0,
    ):
        """Yöneticide bir model kaydı oluşturur."""
        with self.lock:
            if model_id not in self.models:
                self.models[model_id] = {
                    "id": model_id,
                    "type": model_type,
                    "unload_cb": unload_cb,
                    "estimated_vram_mb": estimated_vram_mb,
                    "last_used": 0.0,
                    "is_loaded": False,
                    "use_count": 0,
                }
                logger.info(f"VRAM Yöneticisi: Model kaydedildi: {model_id} ({model_type}, Est: {estimated_vram_mb} MB)")

    def touch_model(self, model_id: str):
        """Modelin son kullanım zaman damgasını ve kullanım sıklığını günceller."""
        with self.lock:
            if model_id in self.models:
                self.models[model_id]["last_used"] = time.time()
                self.models[model_id]["is_loaded"] = True
                self.models[model_id]["use_count"] = self.models[model_id].get("use_count", 0) + 1

    def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Yüklü olan modellerin listesini döner."""
        with self.lock:
            return [m for m in self.models.values() if m["is_loaded"]]

    def get_vram_usage(self) -> float:
        """PyTorch CUDA tarafından seçilen aygıtta ayrılan VRAM miktarını (MB) döner."""
        try:
            import torch
            if torch.cuda.is_available() and self.device_id >= 0:
                return torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def request_load(
        self,
        model_id: str,
        load_fn: Callable[[], bool],
        estimated_vram_mb: float = 500.0,
    ) -> bool:
        """
        Bütçeye göre modeli yüklemeyi talep eder.
        Gerekirse bütçe açmak için en soğuk modelleri bellekten boşaltır (LRU + Sıcaklık).
        Yüklemeleri sıralı ve thread-safe yapmak için load_lock kullanır.
        """
        if not self.enabled:
            return load_fn()

        # Model yükleme kuyruğu kilidi
        with self.load_lock:
            # Son tahmini güncelle veya yoksa kaydet
            with self.lock:
                if model_id in self.models:
                    self.models[model_id]["estimated_vram_mb"] = estimated_vram_mb
                    self.models[model_id]["use_count"] = self.models[model_id].get("use_count", 0) + 1
                else:
                    self.models[model_id] = {
                        "id": model_id,
                        "type": "Model",
                        "unload_cb": lambda: None,
                        "estimated_vram_mb": estimated_vram_mb,
                        "last_used": time.time(),
                        "is_loaded": False,
                        "use_count": 1,
                    }

            # Bütçe kontrolü yap ve gerekirse LRU / Sıcaklık bazlı boşalt
            self._ensure_vram_budget(model_id, estimated_vram_mb)

            start_vram = self.get_vram_usage()
            logger.info(f"VRAM Yöneticisi: Yükleme başlatılıyor: {model_id}")
            success = load_fn()
            end_vram = self.get_vram_usage()

            if success:
                actual_vram = max(0.0, end_vram - start_vram)
                with self.lock:
                    self.models[model_id]["is_loaded"] = True
                    self.models[model_id]["last_used"] = time.time()
                    # Eğer gerçek VRAM ölçülebildiyse tahmini güncelle
                    if actual_vram > 50.0:
                        self.models[model_id]["estimated_vram_mb"] = actual_vram
                        logger.info(f"VRAM Yöneticisi: {model_id} gerçek VRAM yükü ölçüldü: {actual_vram:.1f} MB")
                return True
            else:
                with self.lock:
                    if model_id in self.models:
                        self.models[model_id]["is_loaded"] = False
                return False

    def _ensure_vram_budget(self, prospective_model_id: str, required_mb: float):
        """Bütçeyi aşmamak için en soğuk modelleri serbest bırakır (LRU + Sıcaklık)."""
        has_torch = False
        try:
            import torch
            has_torch = True
        except ImportError:
            pass

        while True:
            loaded_models = []
            with self.lock:
                for mid, m in self.models.items():
                    if m["is_loaded"] and mid != prospective_model_id:
                        loaded_models.append(m)

            total_estimated = sum(m["estimated_vram_mb"] for m in loaded_models)
            
            # Eğer yeni modelle birlikte bütçe aşılmıyorsa dur
            if total_estimated + required_mb <= self.budget_mb:
                break

            if not loaded_models:
                # Boşaltılacak başka model kalmadıysa çık
                break

            # Sıcaklık skoru hesaplama
            def get_temperature(m):
                last_used = m.get("last_used", 0.0)
                if last_used == 0.0:
                    return 0.0
                idle_time = time.time() - last_used
                use_count = m.get("use_count", 1)
                # Sıcaklık skoru: Sık kullanılan ve yeni kullanılan modeller daha sıcaktır
                return use_count / (idle_time + 60.0)

            # Modelleri sıcaklık skoruna göre sırala (en soğuk en başta)
            loaded_models.sort(key=get_temperature)
            lru_model = loaded_models[0]

            logger.info(
                f"VRAM Yöneticisi: Bütçe yetersiz. Soğuk {lru_model['id']} modeli bellekten atılıyor "
                f"(Sıcaklık Skoru: {get_temperature(lru_model):.4f}, Est: {lru_model['estimated_vram_mb']:.1f} MB)"
            )

            try:
                lru_model["unload_cb"]()
            except Exception as e:
                logger.error(f"VRAM Yöneticisi: Model boşaltma hatası ({lru_model['id']}): {e}")

            with self.lock:
                if lru_model["id"] in self.models:
                    self.models[lru_model["id"]]["is_loaded"] = False

            # Çöp toplama ve CUDA hafıza temizliği
            try:
                gc.collect()
                if has_torch and torch.cuda.is_available() and self.device_id >= 0:
                    torch.cuda.empty_cache()
            except Exception:
                pass
