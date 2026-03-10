"""
src/ui/preset_panel.py
Preset (oyun profili) secme/kaydetme/silme UI bileşeni.
UMAYApp header'inin altina yerlestirilen ince bir bar.
"""

from __future__ import annotations

import tkinter.simpledialog as simpledialog
import tkinter.messagebox as mb
from typing import Callable, Optional

import customtkinter as ctk

from src.presets.manager import PresetManager


class PresetBar(ctk.CTkFrame):
    """
    Tek satirlik preset yonetim cubugu.
    Layout: [Profil: ▾ <ComboBox>]  [Yukle]  [Mevcut Kaydet]  [Sil]
    """

    def __init__(
        self,
        master,
        preset_manager: PresetManager,
        on_load: Callable[[dict], None],
        **kwargs,
    ):
        super().__init__(master, height=40, **kwargs)
        self._pm = preset_manager
        self._on_load = on_load
        self._build()
        self._refresh_list()

    def _build(self):
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text="Profil:", width=52).grid(
            row=0, column=0, padx=(10, 4), pady=6
        )

        self._preset_var = ctk.StringVar()
        self._combo = ctk.CTkComboBox(
            self,
            variable=self._preset_var,
            values=[],
            width=200,
            state="readonly",
        )
        self._combo.grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        ctk.CTkButton(
            self, text="Yükle", width=70,
            command=self._load,
        ).grid(row=0, column=2, padx=4)

        ctk.CTkButton(
            self, text="Kaydet", width=80,
            fg_color=("#2d7d46", "#27ae60"),
            hover_color=("#235f36", "#1e8449"),
            command=self._save_current,
        ).grid(row=0, column=3, padx=4)

        ctk.CTkButton(
            self, text="Sil", width=50,
            fg_color=("#c0392b", "#e74c3c"),
            hover_color=("#922b21", "#c0392b"),
            command=self._delete,
        ).grid(row=0, column=4, padx=(4, 10))

        self._status = ctk.CTkLabel(
            self, text="", text_color="gray", font=ctk.CTkFont(size=11)
        )
        self._status.grid(row=0, column=5, padx=8)

    def _refresh_list(self):
        names = self._pm.list_names()
        self._combo.configure(values=names if names else [""])
        active = self._pm._config.get("active_preset")
        if active and active in names:
            self._preset_var.set(active)
        elif names:
            self._preset_var.set(names[0])
        else:
            self._preset_var.set("")

    def _load(self):
        name = self._preset_var.get().strip()
        if not name:
            self._set_status("Yuklenecek profil secin.", error=True)
            return
        data = self._pm.load(name)
        if not data:
            self._set_status(f"'{name}' bulunamadi.", error=True)
            return
        self._pm.set_active(name)
        self._on_load(data)
        self._set_status(f"'{name}' yuklendi.")

    def _save_current(self):
        existing = self._preset_var.get().strip()
        name = simpledialog.askstring(
            "Profil Kaydet",
            "Profil adi:",
            initialvalue=existing or "",
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._pm.list_names():
            if not mb.askyesno("Üzerine Yaz", f"'{name}' mevcut. Üzerine yazilsin mi?"):
                return
        snapshot = self._pm.snapshot_from_config(self._pm._config)
        self._pm.save(name, snapshot)
        self._refresh_list()
        self._preset_var.set(name)
        self._set_status(f"'{name}' kaydedildi.")

    def _delete(self):
        name = self._preset_var.get().strip()
        if not name:
            return
        if not mb.askyesno("Profil Sil", f"'{name}' silinsin mi?"):
            return
        self._pm.delete(name)
        self._refresh_list()
        self._set_status(f"'{name}' silindi.")

    def _set_status(self, msg: str, error: bool = False):
        color = "#e74c3c" if error else "#4CAF50"
        self._status.configure(text=msg, text_color=color)
        self.after(3000, lambda: self._status.configure(text=""))

    def refresh(self):
        self._refresh_list()
