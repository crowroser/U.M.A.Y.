"""
src/utils/download_progress.py
HuggingFace model indirmelerinde ilerleme gosterimi icin yardimci.
- Custom tqdm ile indirme ilerlemesi callback'e iletilir
- Uygulama log/status alaninda gorunur
"""

from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Callable, Optional

# Thread-safe callback: (mesaj: str) -> None
_download_progress_callback: ContextVar[Optional[Callable[[str], None]]] = ContextVar(
    "download_progress_callback", default=None
)


def _format_size(n: float) -> str:
    """Byte sayisini okunabilir formata cevirir."""
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def get_progress_callback() -> Optional[Callable[[str], None]]:
    """Aktif progress callback'i dondurur."""
    return _download_progress_callback.get()


def _make_progress_tqdm():
    """HuggingFace tqdm'den tureyen, ilerlemeyi callback'e ileten sinif."""
    from tqdm.auto import tqdm as _base_tqdm
    try:
        from huggingface_hub.utils import tqdm as hf_tqdm
    except ImportError:
        hf_tqdm = _base_tqdm

    class ProgressTqdm(hf_tqdm):
        """Ilerlemeyi callback'e ileten tqdm."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_pct = -1

        def update(self, n: int = 1) -> bool:
            ret = super().update(n)
            cb = get_progress_callback()
            if cb and self.total and self.total > 0:
                pct = int(100 * self.n / self.total)
                if pct != self._last_pct and (pct % 5 == 0 or pct >= 99):
                    self._last_pct = pct
                    desc = (self.desc or "Indiriliyor").strip()
                    msg = f"{desc}: {pct}% ({_format_size(self.n)} / {_format_size(self.total)})"
                    cb(msg)
            return ret

    # tqdm.tqdm(...) seklinde kullanan kutuphaneler icin (orn. Coqui TTS)
    # ProgressTqdm.tqdm kendi kendine isaretlenmeli ki tum cagrilar progress raporu versin
    ProgressTqdm.tqdm = ProgressTqdm

    return ProgressTqdm


@contextmanager
def download_progress_context(callback: Callable[[str], None]):
    """
    Indirme sirasinda ilerleme mesajlarini callback'e yonlendirir.
    huggingface_hub tqdm gecici olarak progress raporlayan sinifla degistirilir.
    """
    from tqdm.auto import tqdm as _base_tqdm

    token = _download_progress_callback.set(callback)
    ProgressTqdm = _make_progress_tqdm()

    # Modul seviyesinde import edilmis hf_tqdm siniflarina da .tqdm ekle
    # (bazi paketler class.tqdm(...) seklinde kullanir)
    _patched_classes: list = []

    def _patch_class(cls):
        if cls is not None and not hasattr(cls, "tqdm"):
            try:
                cls.tqdm = ProgressTqdm  # tqdm.tqdm(...) cagrilarinda da progress raporlansin
                _patched_classes.append(cls)
            except (AttributeError, TypeError):
                pass

    try:
        import sys
        import importlib
        import huggingface_hub.utils.tqdm  # modülün sys.modules'a yüklenmesini garantile
        import huggingface_hub.utils as utils_mod
        import huggingface_hub.file_download as fd_mod
        import huggingface_hub._snapshot_download as snap_mod

        # huggingface_hub/utils/__init__.py tqdm CLASS'ını re-export ettiğinden
        # doğrudan import CLASS döndürür; gerçek modülü sys.modules'dan alıyoruz.
        tqdm_mod = sys.modules.get("huggingface_hub.utils.tqdm")
        if tqdm_mod is None or not hasattr(tqdm_mod, "tqdm"):
            tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")

        orig_tqdm = tqdm_mod.tqdm

        # tqdm.tqdm(...) kullanan paketler icin: sinifa .tqdm ekle
        _patch_class(orig_tqdm)
        try:
            import tqdm.auto as tqdm_auto_mod
            _patch_class(getattr(tqdm_auto_mod, "tqdm", None))
        except ImportError:
            pass
        try:
            import tqdm.std as tqdm_std_mod
            _patch_class(getattr(tqdm_std_mod, "tqdm", None))
        except ImportError:
            pass

        tqdm_mod.tqdm = utils_mod.tqdm = ProgressTqdm
        if hasattr(fd_mod, "tqdm"):
            orig_fd_tqdm = fd_mod.tqdm
            _patch_class(orig_fd_tqdm)
            fd_mod.tqdm = ProgressTqdm
        if hasattr(snap_mod, "hf_tqdm"):
            orig_snap_tqdm = snap_mod.hf_tqdm
            _patch_class(orig_snap_tqdm)
            snap_mod.hf_tqdm = ProgressTqdm
        try:
            yield
        finally:
            tqdm_mod.tqdm = utils_mod.tqdm = orig_tqdm
            if hasattr(fd_mod, "tqdm"):
                fd_mod.tqdm = orig_fd_tqdm
            if hasattr(snap_mod, "hf_tqdm"):
                snap_mod.hf_tqdm = orig_snap_tqdm
            # Eklenen .tqdm attribute'larini temizle
            for cls in _patched_classes:
                try:
                    del cls.tqdm
                except (AttributeError, TypeError):
                    pass
    finally:
        _download_progress_callback.reset(token)
