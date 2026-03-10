"""
src/ui/char_refs_dialog.py
Karakter icin duygu bazli referans WAV dosyasi secme diyalogu.
Her duygu etiketine farkli bir WAV dosyasi atanabilir.
Kaydet -> config["character_refs"][karakter_adi] guncellenir.
"""

from __future__ import annotations

import os
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

EMOTIONS = ["default", "anger", "fear", "sadness", "joy", "surprise", "disgust", "neutral"]


class CharacterRefsDialog(ctk.CTkToplevel):
    """
    Bir karakter icin duygu -> WAV eslemesini duzenleyen popup.
    """

    def __init__(
        self,
        master,
        character_name: str,
        existing_refs: Optional[dict] = None,
        on_save: Optional[Callable[[str, dict], None]] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._character_name = character_name
        self._on_save = on_save
        self._vars: dict[str, ctk.StringVar] = {}

        self.title(f"Referans WAV — {character_name}")
        self.geometry("580x460")
        self.resizable(False, False)
        self.grab_set()

        self._build(existing_refs or {})

    def _build(self, refs: dict):
        ctk.CTkLabel(
            self,
            text=f"Karakter: {self._character_name}",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))

        ctk.CTkLabel(
            self,
            text="Her duygu icin 3 saniyelik referans WAV secin.\n"
                 "Bos birakilirsa 'default' veya genel speaker_wav kullanilir.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(anchor="w", padx=16, pady=(0, 10))

        scroll = ctk.CTkScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        for emotion in EMOTIONS:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=4)

            label_text = emotion.capitalize() + ("  (varsayilan)" if emotion == "default" else "")
            ctk.CTkLabel(row, text=label_text, width=120, anchor="w").pack(side="left", padx=(0, 6))

            var = ctk.StringVar(value=refs.get(emotion, ""))
            self._vars[emotion] = var

            entry = ctk.CTkEntry(row, textvariable=var, placeholder_text=f"{emotion}.wav", width=320)
            entry.pack(side="left", fill="x", expand=True)

            ctk.CTkButton(
                row, text="...", width=32,
                command=lambda v=var, e=emotion: self._browse(v, e),
            ).pack(side="left", padx=(4, 0))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 14))

        ctk.CTkButton(footer, text="Iptal", width=80, command=self.destroy).pack(side="right", padx=(8, 0))
        ctk.CTkButton(
            footer, text="Kaydet", width=90,
            fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            command=self._save,
        ).pack(side="right")

    def _browse(self, var: ctk.StringVar, emotion: str):
        path = filedialog.askopenfilename(
            title=f"'{emotion}' icin WAV sec",
            filetypes=[("WAV dosyalari", "*.wav"), ("Tum dosyalar", "*.*")],
        )
        if path:
            var.set(path)

    def _save(self):
        refs = {
            emotion: v.get().strip()
            for emotion, v in self._vars.items()
            if v.get().strip()
        }
        if self._on_save:
            self._on_save(self._character_name, refs)
        self.destroy()
