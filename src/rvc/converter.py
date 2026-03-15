"""
src/rvc/converter.py
GPU VRAM cok-model onbellekli RVC ses donusum modulu.

Temel ozellikler:
- Her karakter modeli bir kez VRAM'e yuklenir, hic kaldirilmaz (_model_cache)
- preload_all(): uygulama basinda tum haritalanmis modelleri arka planda yukler
- realtime_mode=True: f0_method otomatik 'pm'ye donusur (~0.3s, rmvpe yerine ~1-2s)
- Per-model lock: farkli karakter seslerini paralel islemeye izin verir
"""

import os
import threading
from pathlib import Path
from typing import Callable, Optional

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
RVC_OUTPUT_PATH = OUTPUT_DIR / "rvc_output.wav"


def _detect_model_version(pth: str) -> str:
    """enc_p.emb_phone.weight boyutuna bakarak model versiyonunu (v1/v2) tespit eder."""
    try:
        import torch
        cpt = torch.load(pth, map_location="cpu", weights_only=False)
        state = cpt.get("weight", cpt)
        if isinstance(state, dict):
            emb = state.get("enc_p.emb_phone.weight")
            if emb is not None:
                return "v2" if emb.shape[1] == 768 else "v1"
    except Exception:
        pass
    return "v2"

_instance: Optional["RVCConverter"] = None
_instance_lock = threading.Lock()


def get_rvc(
    config: dict,
    on_status: Optional[Callable[[str], None]] = None,
) -> "RVCConverter":
    """Global RVCConverter singleton'ini doner."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = RVCConverter(config, on_status=on_status)
    return _instance


def scan_models(models_dir: Optional[Path] = None) -> list[dict]:
    """models/ altindaki .pth dosyalarini tarar."""
    base = models_dir or MODELS_DIR
    results = []
    if not base.exists():
        return results
    for pth in base.glob("**/*.pth"):
        name = pth.stem
        idx_candidates = list(pth.parent.glob(f"{name}*.index")) + list(
            pth.parent.glob("*.index")
        )
        results.append(
            {
                "name": name,
                "pth": str(pth),
                "index": str(idx_candidates[0]) if idx_candidates else None,
            }
        )
    return results


class RVCConverter:
    """
    rvc-python tabanli ses donusturucu singleton.
    Tum karakter modelleri VRAM'de ayri nesneler olarak tutulur;
    karakter gecisi sifir yukleme gecikmesiyle gerceklesir.
    """

    def __init__(
        self,
        config: dict,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        rvc_cfg = config.get("rvc", {})
        self._on_status = on_status or (lambda _: None)

        self.pitch: int = rvc_cfg.get("pitch", 0)
        self.filter_radius: int = rvc_cfg.get("filter_radius", 3)
        self.index_rate: float = rvc_cfg.get("index_rate", 0.75)
        self.rms_mix_rate: float = rvc_cfg.get("rms_mix_rate", 0.25)
        self.protect: float = rvc_cfg.get("protect", 0.33)
        self.f0_method: str = rvc_cfg.get("f0_method", "rmvpe")
        self.realtime_mode: bool = rvc_cfg.get("realtime_mode", True)

        self._device = self._detect_device()
        self._notify(f"RVC cihazi: {self._device} | realtime_mode={self.realtime_mode}")

        self._character_map: dict[str, str] = {}
        self._rebuild_character_map(config.get("characters", {}))

        self._default_model: Optional[str] = rvc_cfg.get("model_path")
        self._default_index: Optional[str] = rvc_cfg.get("index_path")

        # Cok-model onbellegi: pth_yolu -> RVCInference
        self._model_cache: dict[str, object] = {}
        # Per-model yukleme kilidi (ayni modeli iki thread birden yuklemesin)
        self._load_locks: dict[str, threading.Lock] = {}
        # Per-model cikarim kilidi (ayni RVCInference nesnesi thread-safe degil)
        self._infer_locks: dict[str, threading.Lock] = {}
        # Genel kilit sadece lock-dict'lerini korur
        self._meta_lock = threading.Lock()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _notify(self, msg: str):
        self._on_status(msg)

    def _get_locks(self, pth: str) -> tuple[threading.Lock, threading.Lock]:
        """pth icin (load_lock, infer_lock) cifti doner; yoksa olusturur."""
        with self._meta_lock:
            if pth not in self._load_locks:
                self._load_locks[pth] = threading.Lock()
                self._infer_locks[pth] = threading.Lock()
            return self._load_locks[pth], self._infer_locks[pth]

    def _rebuild_character_map(self, characters: dict):
        self._character_map = {
            k.strip().lower(): v for k, v in characters.items() if v
        }

    def update_character_map(self, characters: dict):
        self._rebuild_character_map(characters)

    def _resolve_model(self, character: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Karakter adina gore (pth_yolu, index_yolu) doner."""
        if character:
            key = character.strip().lower()
            pth = self._character_map.get(key)
            if pth and os.path.isfile(pth):
                pth_path = Path(pth)
                idx_candidates = list(pth_path.parent.glob(f"{pth_path.stem}*.index"))
                index = str(idx_candidates[0]) if idx_candidates else None
                return pth, index

        if self._default_model and os.path.isfile(self._default_model):
            return self._default_model, self._default_index

        return None, None

    def _load_model(self, pth: str, index: Optional[str]) -> Optional[object]:
        """
        pth icin RVCInference nesnesini doner.
        Onbellekte varsa hemen, yoksa yukleyip onbellege ekler.
        Thread-safe: ayni model icin yalnizca tek yukleme yapilir.
        """
        if pth in self._model_cache:
            return self._model_cache[pth]

        load_lock, _ = self._get_locks(pth)
        with load_lock:
            # Double-checked locking: lock alindiktan sonra tekrar kontrol
            if pth in self._model_cache:
                return self._model_cache[pth]
            try:
                self._notify(f"RVC yukleniyor: {Path(pth).name}")
                from rvc_python.infer import RVCInference
                version = _detect_model_version(pth)
                self._notify(f"RVC model versiyonu: {version}")
                rvc = RVCInference(device=self._device)
                rvc.load_model(pth, version=version, index_path=index or "")
                self._model_cache[pth] = rvc
                self._notify(f"RVC hazir: {Path(pth).name}")
                return rvc
            except ImportError:
                self._notify("HATA: rvc-python bulunamadi. pip install rvc-python")
                return None
            except Exception as e:
                self._notify(f"RVC yukleme hatasi ({Path(pth).name}): {e}")
                return None

    def preload_all(self):
        """
        Haritalanmis tum karakter modellerini arka planda yukler.
        Uygulama basinda cagrildigi zaman pipeline baslatildiginda tum
        modeller hazir olur ve ilk geciste 0ms bekleme saglanir.
        """
        models_to_load: list[tuple[str, Optional[str]]] = []

        for pth in self._character_map.values():
            if pth and os.path.isfile(pth):
                pth_path = Path(pth)
                idx = next(iter(pth_path.parent.glob(f"{pth_path.stem}*.index")), None)
                models_to_load.append((pth, str(idx) if idx else None))

        if self._default_model and os.path.isfile(self._default_model):
            models_to_load.append((self._default_model, self._default_index))

        if not models_to_load:
            return

        self._notify(f"RVC on-yukleme: {len(models_to_load)} model...")
        for pth, index in models_to_load:
            threading.Thread(
                target=self._load_model,
                args=(pth, index),
                daemon=True,
                name=f"RVCPreload-{Path(pth).stem}",
            ).start()

    def _effective_f0_method(self) -> str:
        """realtime_mode aktifse 'pm', degilse konfigurasyondaki yontemi doner."""
        if self.realtime_mode:
            return "pm"
        return self.f0_method

    def convert_for_character(
        self,
        input_wav: str,
        character: Optional[str] = None,
        output_path: Optional[str] = None,
        pitch_override_delta: int = 0,
    ) -> Optional[str]:
        """
        Karakter ismine gore onbellekteki modeli kullanip ses donusumu yapar.
        Model VRAM'de zaten yukluyse yukleme gecikmesi 0ms'dir.
        pitch_override_delta: duygu analizinden gelen ek pitch (yari ton).
        """
        if not os.path.isfile(input_wav):
            self._notify(f"RVC: Giris dosyasi yok: {input_wav}")
            return None

        pth, index = self._resolve_model(character)
        if not pth:
            self._notify("RVC: Model yok, TTS sesi gecildi.")
            return None

        rvc_obj = self._load_model(pth, index)
        if rvc_obj is None:
            return None

        out = output_path or str(RVC_OUTPUT_PATH)
        _, infer_lock = self._get_locks(pth)
        effective_pitch = self.pitch + pitch_override_delta
        f0 = self._effective_f0_method()

        with infer_lock:
            try:
                self._notify(
                    f"RVC [{character or 'varsayilan'}] f0={f0} pitch={effective_pitch}..."
                )
                rvc_obj.set_params(
                    f0up_key=effective_pitch,
                    filter_radius=self.filter_radius,
                    index_rate=self.index_rate,
                    rms_mix_rate=self.rms_mix_rate,
                    protect=self.protect,
                    f0method=f0,
                )
                rvc_obj.infer_file(
                    input_path=input_wav,
                    output_path=out,
                )
                return out
            except Exception as e:
                self._notify(f"RVC donusum hatasi: {e}")
                return None

    def set_default_model(self, model_path: str, index_path: Optional[str] = None):
        """Varsayilan modeli degistirir; once VRAM'e yukler."""
        self._default_model = model_path
        self._default_index = index_path
        threading.Thread(
            target=self._load_model,
            args=(model_path, index_path),
            daemon=True,
        ).start()

    def set_default_model_async(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        on_done: Optional[Callable[[bool], None]] = None,
    ):
        def _run():
            self._default_model = model_path
            self._default_index = index_path
            rvc_obj = self._load_model(model_path, index_path)
            if on_done:
                on_done(rvc_obj is not None)
        threading.Thread(target=_run, daemon=True).start()

    def update_settings(
        self,
        pitch: Optional[int] = None,
        filter_radius: Optional[int] = None,
        index_rate: Optional[float] = None,
        rms_mix_rate: Optional[float] = None,
        protect: Optional[float] = None,
        f0_method: Optional[str] = None,
        realtime_mode: Optional[bool] = None,
    ):
        if pitch is not None:
            self.pitch = pitch
        if filter_radius is not None:
            self.filter_radius = filter_radius
        if index_rate is not None:
            self.index_rate = max(0.0, min(1.0, index_rate))
        if rms_mix_rate is not None:
            self.rms_mix_rate = max(0.0, min(1.0, rms_mix_rate))
        if protect is not None:
            self.protect = max(0.0, min(0.5, protect))
        if f0_method is not None:
            self.f0_method = f0_method
        if realtime_mode is not None:
            self.realtime_mode = realtime_mode

    def unload_all(self):
        """Tum onbellekteki RVC modellerini bellekten kaldirir."""
        with self._meta_lock:
            self._model_cache.clear()
            self._load_locks.clear()
            self._infer_locks.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._notify("Tum RVC modelleri bellekten kaldirildi.")

    def is_ready(self) -> bool:
        return bool(self._model_cache) or bool(self._default_model)

    def cached_model_count(self) -> int:
        return len(self._model_cache)

    def has_model_for(self, character: str) -> bool:
        pth, _ = self._resolve_model(character)
        return pth is not None
