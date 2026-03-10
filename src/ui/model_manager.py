from __future__ import annotations

import os
import shutil
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from src.rvc.voice_converter import scan_models, MODELS_DIR


class ModelManager(ctk.CTkToplevel):
    """
    models/ klasorundeki RVC modellerini yoneten pencere.
    Kullanici .pth dosyalarini ekleyip seçebilir.
    """

    def __init__(
        self,
        master,
        on_model_selected: Callable[[str, Optional[str]], None],
        current_model: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.title("Model Yoneticisi - U.M.A.Y")
        self.geometry("600x450")
        self.resizable(False, False)
        self._on_model_selected = on_model_selected
        self._current_model = current_model
        self._models: list[dict] = []
        self._build()
        self.refresh()
        self.grab_set()

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(16, 8))

        ctk.CTkLabel(
            header, text="RVC Ses Modelleri", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left")

        ctk.CTkButton(header, text="Model Ekle (.pth)", width=130, command=self._add_model).pack(
            side="right"
        )
        ctk.CTkButton(header, text="Yenile", width=70, command=self.refresh).pack(
            side="right", padx=(0, 8)
        )

        ctk.CTkLabel(
            self,
            text="models/ klasorune .pth ve opsiyonel .index dosyalari ekleyin.",
            text_color="gray",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=16)

        self._scrollframe = ctk.CTkScrollableFrame(self, label_text="")
        self._scrollframe.pack(fill="both", expand=True, padx=16, pady=8)

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 16))

        self._status_label = ctk.CTkLabel(footer, text="", text_color="gray")
        self._status_label.pack(side="left")

        ctk.CTkButton(footer, text="Kapat", width=80, command=self.destroy).pack(side="right")

    def refresh(self):
        for widget in self._scrollframe.winfo_children():
            widget.destroy()

        self._models = scan_models()

        if not self._models:
            ctk.CTkLabel(
                self._scrollframe,
                text="Henuz model yok.\nmodels/ klasorune .pth dosyasi ekleyin.",
                text_color="gray",
            ).pack(pady=32)
            return

        for model in self._models:
            self._add_model_row(model)

    def _add_model_row(self, model: dict):
        row = ctk.CTkFrame(self._scrollframe)
        row.pack(fill="x", pady=4, padx=4)

        is_selected = (
            self._current_model and
            os.path.normpath(model["pth"]) == os.path.normpath(self._current_model)
        )

        indicator = "●" if is_selected else "○"
        color = "#4CAF50" if is_selected else "gray"

        ctk.CTkLabel(row, text=indicator, text_color=color, width=20).pack(side="left", padx=4)
        ctk.CTkLabel(
            row,
            text=model["name"],
            font=ctk.CTkFont(weight="bold" if is_selected else "normal"),
        ).pack(side="left", padx=4)

        index_text = "✓ Index" if model["index"] else "Index yok"
        index_color = "#4CAF50" if model["index"] else "gray"
        ctk.CTkLabel(row, text=index_text, text_color=index_color, font=ctk.CTkFont(size=11)).pack(
            side="left", padx=8
        )

        ctk.CTkButton(
            row,
            text="Sil",
            width=50,
            fg_color="#c0392b",
            hover_color="#922b21",
            command=lambda m=model: self._delete_model(m),
        ).pack(side="right", padx=(0, 4))

        ctk.CTkButton(
            row,
            text="Sec",
            width=60,
            command=lambda m=model: self._select_model(m),
        ).pack(side="right", padx=(0, 4))

        ctk.CTkLabel(
            row,
            text=os.path.basename(model["pth"]),
            text_color="gray",
            font=ctk.CTkFont(size=11),
        ).pack(side="right", padx=8)

    def _add_model(self):
        paths = filedialog.askopenfilenames(
            title="RVC Model Dosyasi Sec (.pth)",
            filetypes=[("PyTorch modeli", "*.pth"), ("Tum dosyalar", "*.*")],
        )
        if not paths:
            return

        added = 0
        for path in paths:
            dest = MODELS_DIR / os.path.basename(path)
            if not dest.exists():
                shutil.copy2(path, dest)
                added += 1

                index_candidates = [
                    Path(path).parent / (Path(path).stem + ".index"),
                    Path(path).parent / "added.index",
                ]
                for ic in index_candidates:
                    if ic.exists():
                        shutil.copy2(ic, MODELS_DIR / ic.name)
                        break

        self._status_label.configure(text=f"{added} model eklendi.")
        self.refresh()

    def _select_model(self, model: dict):
        self._current_model = model["pth"]
        self._on_model_selected(model["pth"], model["index"])
        self._status_label.configure(text=f"Secildi: {model['name']}")
        self.refresh()

    def _delete_model(self, model: dict):
        import tkinter.messagebox as mb
        if mb.askyesno("Modeli Sil", f"'{model['name']}' silinsin mi?", parent=self):
            try:
                os.remove(model["pth"])
                if model["index"] and os.path.isfile(model["index"]):
                    os.remove(model["index"])
                self._status_label.configure(text=f"Silindi: {model['name']}")
                if self._current_model and os.path.normpath(model["pth"]) == os.path.normpath(self._current_model):
                    self._current_model = None
                    self._on_model_selected(None, None)
            except Exception as e:
                self._status_label.configure(text=f"Hata: {e}")
            self.refresh()
