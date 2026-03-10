"""
src/ui/hf_downloader.py
HuggingFace TTS modeli indirme penceresi.
- HF model ID girisinden sonra huggingface_hub.snapshot_download() ile indirir
- Hedef klasor: models/tts/<sanitized_id>/
- Indirilen modeller listelenir ve secilip etkinlestirilebilir
Ornek kullanim: WarriorMama777/GLaDOS_TTS
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk

MODELS_TTS_DIR = Path(__file__).parent.parent.parent / "models" / "tts"


def _sanitize(repo_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", repo_id)


class HFDownloader(ctk.CTkToplevel):
    """HuggingFace'den TTS modeli indirme ve secme penceresi."""

    def __init__(
        self,
        master,
        on_downloaded: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._on_downloaded = on_downloaded
        self._download_thread: Optional[threading.Thread] = None

        self.title("HuggingFace Model Indir — U.M.A.Y")
        self.geometry("560x480")
        self.resizable(False, False)
        self.grab_set()

        self._build()
        self._refresh_local_list()

    def _build(self):
        ctk.CTkLabel(
            self,
            text="HuggingFace TTS Model Indir",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            self,
            text="Model ID'si girin (ornek: WarriorMama777/GLaDOS_TTS).\n"
                 "Indirilen model models/tts/ klasorune kaydedilir.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        input_row = ctk.CTkFrame(self, fg_color="transparent")
        input_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(input_row, text="HF Model ID:").pack(side="left", padx=(0, 8))
        self._model_id = ctk.CTkEntry(
            input_row, placeholder_text="KullanicıAdi/ModelAdi", width=320
        )
        self._model_id.pack(side="left", fill="x", expand=True)
        self._dl_btn = ctk.CTkButton(
            input_row, text="Indir", width=70,
            fg_color=("#1a6aaf", "#2980b9"),
            command=self._start_download,
        )
        self._dl_btn.pack(side="left", padx=(8, 0))

        self._progress_bar = ctk.CTkProgressBar(self)
        self._progress_bar.set(0)
        self._progress_bar.pack(fill="x", padx=16, pady=(0, 4))

        self._log = ctk.CTkTextbox(self, height=110, state="disabled",
                                   font=ctk.CTkFont(family="Consolas", size=11))
        self._log.pack(fill="x", padx=16, pady=(0, 10))

        ctk.CTkLabel(
            self, text="Mevcut Yerel Modeller:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=16)

        self._local_scroll = ctk.CTkScrollableFrame(self, height=120)
        self._local_scroll.pack(fill="both", expand=True, padx=16, pady=(4, 8))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(footer, text="Kapat", width=80, command=self.destroy).pack(side="right")
        ctk.CTkButton(footer, text="Yenile", width=80, command=self._refresh_local_list).pack(
            side="right", padx=(0, 8)
        )

    def _log_write(self, msg: str):
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _start_download(self):
        repo_id = self._model_id.get().strip()
        if not repo_id or "/" not in repo_id:
            self._log_write("HATA: Gecerli bir HF model ID'si girin (KullanicıAdi/ModelAdi).")
            return

        if self._download_thread and self._download_thread.is_alive():
            self._log_write("Zaten bir indirme devam ediyor...")
            return

        self._dl_btn.configure(state="disabled", text="Indiriliyor...")
        self._progress_bar.set(0)
        self._progress_bar.start()
        self._log_write(f"Indiriliyor: {repo_id} ...")

        def _run():
            try:
                from huggingface_hub import snapshot_download

                dest = MODELS_TTS_DIR / _sanitize(repo_id)
                dest.mkdir(parents=True, exist_ok=True)

                local_dir = snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(dest),
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                )
                self._log_write(f"Tamamlandi: {local_dir}")

                self.after(0, self._progress_bar.stop)
                self.after(0, lambda: self._progress_bar.set(1))
                self.after(0, lambda: self._dl_btn.configure(state="normal", text="Indir"))
                self.after(0, self._refresh_local_list)

                if self._on_downloaded:
                    self.after(0, lambda: self._on_downloaded(local_dir))

            except ImportError:
                self._log_write("HATA: huggingface_hub yuklu degil.\n'pip install huggingface_hub' calistirin.")
                self.after(0, self._progress_bar.stop)
                self.after(0, lambda: self._dl_btn.configure(state="normal", text="Indir"))
            except Exception as e:
                self._log_write(f"HATA: {e}")
                self.after(0, self._progress_bar.stop)
                self.after(0, lambda: self._dl_btn.configure(state="normal", text="Indir"))

        self._download_thread = threading.Thread(target=_run, daemon=True)
        self._download_thread.start()

    def _refresh_local_list(self):
        for w in self._local_scroll.winfo_children():
            w.destroy()

        models = []
        if MODELS_TTS_DIR.exists():
            for d in sorted(MODELS_TTS_DIR.iterdir()):
                if d.is_dir():
                    has_model = (d / "model.pth").exists() or any(d.glob("*.pth"))
                    has_config = (d / "config.json").exists()
                    if has_model and has_config:
                        models.append(d)

        if not models:
            ctk.CTkLabel(
                self._local_scroll, text="Henuz indirilmis model yok.",
                text_color="gray",
            ).pack(pady=12)
            return

        for model_dir in models:
            row = ctk.CTkFrame(self._local_scroll, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=model_dir.name, anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkButton(
                row, text="Sec", width=60,
                command=lambda d=str(model_dir): self._select_model(d),
            ).pack(side="right")

    def _select_model(self, model_dir: str):
        self._log_write(f"Secildi: {model_dir}")
        if self._on_downloaded:
            self._on_downloaded(model_dir)
        self.destroy()
