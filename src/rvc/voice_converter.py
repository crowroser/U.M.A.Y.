import os
import threading
from pathlib import Path
from typing import Callable, Optional


MODELS_DIR = Path(__file__).parent.parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

RVC_OUTPUT_PATH = OUTPUT_DIR / "rvc_output.wav"


def scan_models(models_dir: Optional[Path] = None) -> list[dict]:
    """
    models/ klasorundeki .pth dosyalarini tarar.
    Her model icin {'name': str, 'pth': str, 'index': str|None} sozlugu doner.
    """
    base = models_dir or MODELS_DIR
    results = []
    if not base.exists():
        return results

    pth_files = list(base.glob("**/*.pth"))
    for pth in pth_files:
        name = pth.stem
        index_candidates = list(pth.parent.glob(f"{name}*.index")) + list(pth.parent.glob("*.index"))
        index_path = str(index_candidates[0]) if index_candidates else None
        results.append({
            "name": name,
            "pth": str(pth),
            "index": index_path,
        })
    return results


class RVCConverter:
    """
    rvc-python kutuphanesi ile TTS ciktisini RVC ses modeline donusturur.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        pitch: int = 0,
        filter_radius: int = 3,
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        f0_method: str = "rmvpe",
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.model_path = model_path
        self.index_path = index_path
        self.pitch = pitch
        self.filter_radius = filter_radius
        self.index_rate = index_rate
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.f0_method = f0_method
        self._on_status = on_status or (lambda msg: None)
        self._rvc = None
        self._loaded = False
        self._lock = threading.Lock()

    def _notify(self, msg: str):
        self._on_status(msg)

    def load(self):
        """RVC modelini yukler."""
        if not self.model_path or not os.path.isfile(self.model_path):
            self._notify("RVC: Model dosyasi bulunamadi.")
            return False

        with self._lock:
            try:
                self._notify("RVC modeli yukleniyor...")
                from rvc_python.infer import RVCInference
                self._rvc = RVCInference(device="cpu")
                self._rvc.load_model(self.model_path, self.index_path or "")
                self._loaded = True
                self._notify("RVC modeli hazir.")
                return True
            except ImportError:
                self._notify("HATA: rvc-python yuklu degil. 'pip install rvc-python' calistirin.")
                return False
            except Exception as e:
                self._notify(f"RVC yukleme hatasi: {e}")
                return False

    def load_async(self, on_done: Optional[Callable[[bool], None]] = None):
        def _run():
            success = self.load()
            if on_done:
                on_done(success)
        threading.Thread(target=_run, daemon=True).start()

    def convert(self, input_wav: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        TTS cikti dosyasini RVC ile donusturur.
        Basarili olursa cikti dosya yolunu, hata olursa None doner.
        """
        if not os.path.isfile(input_wav):
            self._notify(f"RVC: Giris dosyasi bulunamadi: {input_wav}")
            return None

        if not self._loaded:
            success = self.load()
            if not success:
                return None

        out = output_path or str(RVC_OUTPUT_PATH)

        with self._lock:
            try:
                self._notify("RVC isleniyor...")
                self._rvc.infer_file(
                    input_path=input_wav,
                    output_path=out,
                    f0_up_key=self.pitch,
                    filter_radius=self.filter_radius,
                    index_rate=self.index_rate,
                    rms_mix_rate=self.rms_mix_rate,
                    protect=self.protect,
                    f0_method=self.f0_method,
                )
                self._notify("RVC tamamlandi.")
                return out
            except Exception as e:
                self._notify(f"RVC hatasi: {e}")
                return None

    def set_model(self, model_path: str, index_path: Optional[str] = None):
        """Modeli degistir ve yeniden yukle (bloklayan)."""
        with self._lock:
            self.model_path = model_path
            self.index_path = index_path
            self._loaded = False
            self._rvc = None
        self.load()

    def set_model_async(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        on_done: Optional[Callable[[bool], None]] = None,
    ):
        """Modeli arka planda degistir ve yukle."""
        def _run():
            with self._lock:
                self.model_path = model_path
                self.index_path = index_path
                self._loaded = False
                self._rvc = None
            success = self.load()
            if on_done:
                on_done(success)
        threading.Thread(target=_run, daemon=True).start()

    def update_settings(
        self,
        pitch: int = None,
        filter_radius: int = None,
        index_rate: float = None,
        rms_mix_rate: float = None,
        protect: float = None,
        f0_method: str = None,
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

    def is_ready(self) -> bool:
        return self._loaded
