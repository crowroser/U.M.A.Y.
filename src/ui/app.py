"""
src/ui/app.py
U.M.A.Y ana penceresi.

Pipeline: OCR (SubtitleMonitor) → QueueRunner → Translate → TTS → RVC → Duck → sounddevice
- Singleton TTS, RVC, Translator; VRAM'de kalir
- Karakter eslestirme paneli
- Preset (oyun profili) cubugu
- Audio ducking (pycaw)
- RegionSelector bagimsiz modulden import edilir
"""

from __future__ import annotations

import os
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from src.ui.settings_panel import SettingsPanel

BASE_DIR = Path(__file__).parent.parent.parent


class UMAYApp(ctk.CTk):

    APP_TITLE = "U.M.A.Y — Unified Model-based Audio Yield"
    ICON_PATH = BASE_DIR / "assets" / "icon.ico"

    def __init__(self, config: dict, save_config_fn, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self._save_config = save_config_fn

        self._capture = None
        self._monitor = None
        self._tts = None
        self._rvc = None
        self._translator = None
        self._analyzer = None
        self._ducker = None
        self._runner = None
        self._preset_mgr = None
        self._pipeline_running = False
        self._region: Optional[tuple] = None

        ui_cfg = config.get("ui", {})
        ctk.set_appearance_mode(ui_cfg.get("theme", "dark"))
        ctk.set_default_color_theme(ui_cfg.get("color_theme", "blue"))

        self.title(self.APP_TITLE)
        self.geometry(f"{ui_cfg.get('window_width', 1100)}x{ui_cfg.get('window_height', 720)}")
        self.minsize(900, 640)

        if self.ICON_PATH.exists():
            self.iconbitmap(str(self.ICON_PATH))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._init_modules()

    # ──────────────────────────── Layout ────────────────────────────

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._main_frame = ctk.CTkFrame(self)
        self._main_frame.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        self._main_frame.grid_columnconfigure(0, weight=1)
        self._main_frame.grid_rowconfigure(5, weight=1)

        self._right_panel = ctk.CTkTabview(self, width=280)
        self._right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        self._right_panel.add("Ayarlar")
        self._right_panel.add("Karakterler")

        self._settings_panel = SettingsPanel(
            self._right_panel.tab("Ayarlar"),
            on_save=self._apply_settings,
        )
        self._settings_panel.pack(fill="both", expand=True)
        self._settings_panel.load_config(self._config)

        self._build_char_panel(self._right_panel.tab("Karakterler"))

        self._build_header()
        self._build_preset_bar()
        self._build_control_bar()
        self._build_subtitle_bar()
        self._build_log_area()
        self._build_status_bar()

    def _build_header(self):
        header = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 0))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="U.M.A.Y",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#1a6aaf", "#4da6ff"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header, text="Unified Model-based Audio Yield",
            font=ctk.CTkFont(size=12), text_color="gray",
        ).grid(row=1, column=0, sticky="w")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=2, rowspan=2, sticky="e")

        ctk.CTkButton(
            btn_frame, text="Ses Modeli", width=110,
            command=self._open_model_manager,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_frame, text="Test Yakala", width=110,
            fg_color=("#6d3d91", "#8e44ad"), hover_color=("#522d6e", "#7d3c98"),
            command=self._test_capture,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_frame, text="Bölge Seç", width=100,
            fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            command=self._select_region,
        ).pack(side="right", padx=(8, 0))

    def _build_preset_bar(self):
        from src.presets.manager import PresetManager
        from src.ui.preset_panel import PresetBar

        self._preset_mgr = PresetManager(self._config, self._save_config)

        self._preset_bar = PresetBar(
            self._main_frame,
            preset_manager=self._preset_mgr,
            on_load=self._apply_preset,
            fg_color=("gray90", "gray17"),
            corner_radius=8,
        )
        self._preset_bar.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 0))

    def _build_control_bar(self):
        bar = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=8)

        self._start_btn = ctk.CTkButton(
            bar, text="▶  Başlat", width=130, height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#1a6aaf", "#2980b9"),
            hover_color=("#145288", "#1f6391"),
            command=self._toggle_pipeline,
        )
        self._start_btn.pack(side="left")

        # TTS ve RVC durum etiketleri (temel modüller, toggle yok)
        self._tts_status = ctk.CTkLabel(bar, text="TTS: Hazırlanıyor…", text_color="orange")
        self._tts_status.pack(side="left", padx=(16, 4))

        self._rvc_status = ctk.CTkLabel(bar, text="RVC: Model Yok", text_color="gray")
        self._rvc_status.pack(side="left", padx=4)

        # Ayırıcı
        ctk.CTkLabel(bar, text="|", text_color="gray").pack(side="left", padx=6)

        # --- Tıklanabilir modül toggle butonları ---
        self._cev_btn = ctk.CTkButton(
            bar, text="CEV: Kapalı", width=110, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            command=self._toggle_translate,
        )
        self._cev_btn.pack(side="left", padx=4)

        self._duygu_btn = ctk.CTkButton(
            bar, text="DUYGU: Kapalı", width=118, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            command=self._toggle_analyzer,
        )
        self._duygu_btn.pack(side="left", padx=4)

        self._rvc_toggle_btn = ctk.CTkButton(
            bar, text="RVC: Acik", width=90, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            command=self._toggle_rvc,
        )
        self._rvc_toggle_btn.pack(side="left", padx=4)

        self._duck_btn = ctk.CTkButton(
            bar, text="DUCK: Kapalı", width=110, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            command=self._toggle_ducking,
        )
        self._duck_btn.pack(side="left", padx=4)

        self._region_label = ctk.CTkLabel(bar, text="Bölge: Tam Ekran", text_color="gray")
        self._region_label.pack(side="right")

    def _build_subtitle_bar(self):
        frame = ctk.CTkFrame(self._main_frame)
        frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))
        ctk.CTkLabel(frame, text="Son Altyazı:", width=110).pack(side="left", padx=8)
        self._subtitle_var = ctk.StringVar(value="—")
        ctk.CTkLabel(
            frame, textvariable=self._subtitle_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#f0e68c",
        ).pack(side="left", padx=4, fill="x", expand=True)

    def _build_log_area(self):
        ctk.CTkLabel(
            self._main_frame, text="Pipeline Kayıtları",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=4, column=0, sticky="w", padx=16, pady=(0, 2))

        self._log_box = ctk.CTkTextbox(
            self._main_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            state="disabled", wrap="word",
        )
        self._log_box.grid(row=5, column=0, sticky="nsew", padx=16, pady=(0, 8))

        for tag, color in [
            ("ocr", "#5dade2"), ("tts", "#a9cce3"), ("rvc", "#a9dfbf"),
            ("audio", "#f9e79f"), ("error", "#f1948a"), ("info", "#d5d8dc"),
        ]:
            self._log_box.tag_config(tag, foreground=color)

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self._main_frame, height=28, fg_color=("gray85", "gray20"))
        bar.grid(row=6, column=0, sticky="ew")
        self._status_var = ctk.StringVar(value="Hazır.")
        ctk.CTkLabel(
            bar, textvariable=self._status_var,
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(side="left", padx=8)

    # ──────────────────────── Karakter Paneli ─────────────────────────

    def _build_char_panel(self, parent):
        ctk.CTkLabel(
            parent, text="Karakter → RVC Model Eşleştirme",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(10, 4))
        ctk.CTkLabel(
            parent,
            text="OCR'dan okunan karakter adı otomatik\nolarak ilgili modele yönlendirilir.",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=8)

        self._char_scroll = ctk.CTkScrollableFrame(parent, height=320)
        self._char_scroll.pack(fill="both", expand=True, padx=8, pady=8)
        self._char_rows: list[dict] = []

        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(footer, text="+ Satır Ekle", command=self._add_char_row).pack(side="left")
        ctk.CTkButton(footer, text="Kaydet", command=self._save_char_map).pack(side="right")

        for name, pth in self._config.get("characters", {}).items():
            self._add_char_row(name, pth)

    def _add_char_row(self, name: str = "", pth: str = ""):
        row_frame = ctk.CTkFrame(self._char_scroll)
        row_frame.pack(fill="x", pady=3)
        name_var = ctk.StringVar(value=name)
        pth_var = ctk.StringVar(value=pth)

        ctk.CTkEntry(row_frame, textvariable=name_var, placeholder_text="Karakter Adi", width=100).pack(side="left", padx=(4, 2))
        ctk.CTkEntry(row_frame, textvariable=pth_var, placeholder_text="model.pth", width=120).pack(side="left", padx=2)
        ctk.CTkButton(row_frame, text="…", width=28, command=lambda v=pth_var: self._browse_pth(v)).pack(side="left", padx=2)

        row = {"frame": row_frame, "name": name_var, "pth": pth_var}

        ctk.CTkButton(
            row_frame, text="Refs", width=44,
            fg_color=("#1a6aaf", "#2980b9"),
            command=lambda r=row: self._open_char_refs(r),
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            row_frame, text="✕", width=28,
            fg_color="#c0392b", hover_color="#922b21",
            command=lambda r=row: self._remove_char_row(r),
        ).pack(side="left", padx=(2, 4))
        self._char_rows.append(row)

    def _remove_char_row(self, row: dict):
        row["frame"].destroy()
        self._char_rows.remove(row)

    def _open_char_refs(self, row: dict):
        from src.ui.char_refs_dialog import CharacterRefsDialog
        name = row["name"].get().strip()
        if not name:
            self._set_status("Once karakter adi girin.")
            return
        existing = self._config.get("character_refs", {}).get(name, {})
        CharacterRefsDialog(
            self,
            character_name=name,
            existing_refs=existing,
            on_save=self._on_char_refs_saved,
        )

    def _browse_pth(self, var: ctk.StringVar):
        path = filedialog.askopenfilename(
            title="RVC Model Seç (.pth)",
            filetypes=[("PyTorch model", "*.pth"), ("Tüm dosyalar", "*.*")],
        )
        if path:
            var.set(path)

    def _save_char_map(self):
        mapping = {
            row["name"].get().strip(): row["pth"].get().strip()
            for row in self._char_rows
            if row["name"].get().strip() and row["pth"].get().strip()
        }
        self._config["characters"] = mapping
        self._save_config(self._config)
        if self._rvc:
            self._rvc.update_character_map(mapping)
            self._rvc.preload_all()
        if self._tts:
            self._tts.update_character_refs(self._config.get("character_refs", {}))
        self._log(f"Karakter haritasi kaydedildi ({len(mapping)} karakter).", "info")
        self._set_status("Karakter haritasi guncellendi.")

    def _on_char_refs_saved(self, name: str, refs: dict):
        self._config.setdefault("character_refs", {})[name] = refs
        self._save_config(self._config)
        if self._tts:
            self._tts.update_character_refs(self._config.get("character_refs", {}))
        self._log(f"[REFS] {name}: {len(refs)} duygu WAV'i kaydedildi.", "info")

    # ──────────────────────── Modül Başlatma ─────────────────────────

    def _init_modules(self):
        from src.ocr.capture import ScreenCapture, SubtitleMonitor
        from src.tts.generator import get_tts
        from src.rvc.converter import get_rvc
        from src.translate.translator import get_translator
        from src.llm.analyzer import get_analyzer
        from src.audio.ducking import AudioDucker
        from src.pipeline.queue_runner import QueueRunner

        ocr_cfg = self._config.get("ocr", {})
        dk_cfg  = self._config.get("ducking", {})

        self._capture = ScreenCapture(
            tesseract_path=ocr_cfg.get("tesseract_path"),
            language=ocr_cfg.get("language", "tur"),
            preprocess=True,
        )

        def _on_status(tag: str, prefix: str):
            def _cb(m: str):
                self._log(f"[{prefix}] {m}", tag)
                if "%" in m and "/" in m:
                    self._set_status(m)
            return _cb

        self._tts = get_tts(self._config, on_status=_on_status("tts", "TTS"))
        self._rvc = get_rvc(self._config, on_status=_on_status("rvc", "RVC"))
        self._rvc.preload_all()
        self._translator = get_translator(
            self._config, on_status=_on_status("tts", "CEV")
        )
        self._analyzer = get_analyzer(
            self._config, on_status=_on_status("info", "DUYGU")
        )

        self._ducker = AudioDucker(
            enabled=dk_cfg.get("enabled", False),
            duck_level=dk_cfg.get("level", 0.35),
            target_process=dk_cfg.get("target_process", ""),
        )

        self._runner = QueueRunner(
            tts=self._tts,
            rvc=self._rvc,
            translator=self._translator,
            analyzer=self._analyzer,
            ducker=self._ducker,
            on_log=self._log,
        )

        self._monitor = SubtitleMonitor(
            capture=self._capture,
            on_new_subtitle=self._on_subtitle_detected,
            interval=ocr_cfg.get("interval", 0.4),
            on_log=self._log,
        )

        self._tts.load_async(on_done=lambda ok: self.after(0, lambda: self._on_tts_ready(ok)))

        if self._config.get("translate", {}).get("enabled"):
            self._translator.load_async(
                on_done=lambda ok: self.after(0, lambda: self._on_translate_ready(ok))
            )

        if self._config.get("sentiment", {}).get("enabled"):
            self._analyzer.load_async(
                on_done=lambda ok: self.after(0, lambda: self._on_analyzer_ready(ok))
            )

        self._refresh_indicators()

    def _on_tts_ready(self, ok: bool):
        if ok:
            self._tts_status.configure(text="TTS: Hazır ✓", text_color="#4CAF50")
        else:
            self._tts_status.configure(text="TTS: Hata ✗", text_color="#e74c3c")

    def _on_rvc_ready(self, ok: bool):
        if ok:
            n = self._rvc.cached_model_count() if self._rvc else 0
            label = f"RVC: {n} model ✓" if n > 0 else "RVC: Hazir ✓"
            self._rvc_status.configure(text=label, text_color="#4CAF50")
        else:
            self._rvc_status.configure(text="RVC: Hata ✗", text_color="#e74c3c")

    def _on_translate_ready(self, ok: bool):
        if ok:
            self._cev_btn.configure(
                text="CEV: Acik ✓",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        else:
            self._cev_btn.configure(
                text="CEV: Hata ✗",
                fg_color=("#c0392b", "#e74c3c"), hover_color=("#922b21", "#c0392b"),
            )

    def _on_analyzer_ready(self, ok: bool):
        if ok:
            self._duygu_btn.configure(
                text="DUYGU: Acik ✓",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        else:
            self._duygu_btn.configure(
                text="DUYGU: Hata ✗",
                fg_color=("#c0392b", "#e74c3c"), hover_color=("#922b21", "#c0392b"),
            )
        self._log(f"Duygu modeli {'yuklu' if ok else 'hata'}.", "info")

    # ─────────────────── Modül Toggle'ları ───────────────────────────

    def _toggle_translate(self):
        """Çeviri modülünü açar/kapatır; kapatınca modeli bellekten siler."""
        enabled = self._config.get("translate", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("translate", {})["enabled"] = new_state
        self._save_config(self._config)

        if new_state:
            self._cev_btn.configure(
                text="CEV: Yukleniyor…",
                fg_color=("orange", "#e67e22"), hover_color=("orange", "#e67e22"),
            )
            if self._translator:
                self._translator.update_settings(enabled=True)
                self._translator.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_translate_ready(ok))
                )
        else:
            if self._translator:
                self._translator.update_settings(enabled=False)
            self._cev_btn.configure(
                text="CEV: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )
            self._log("Ceviri kapatildi, bellek serbest.", "info")

    def _toggle_analyzer(self):
        """Duygu analizi modülünü açar/kapatır; kapatınca modeli bellekten siler."""
        enabled = self._config.get("sentiment", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("sentiment", {})["enabled"] = new_state
        self._save_config(self._config)

        if new_state:
            self._duygu_btn.configure(
                text="DUYGU: Yukleniyor…",
                fg_color=("orange", "#e67e22"), hover_color=("orange", "#e67e22"),
            )
            if self._analyzer:
                self._analyzer.update_settings(enabled=True)
                self._analyzer.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_analyzer_ready(ok))
                )
        else:
            if self._analyzer:
                self._analyzer.update_settings(enabled=False)
            self._duygu_btn.configure(
                text="DUYGU: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )
            self._log("Duygu analizi kapatildi, bellek serbest.", "info")

    def _toggle_rvc(self):
        """RVC modülünü açar/kapatır; kapatınca tüm model önbelleğini temizler."""
        currently_open = bool(self._rvc and self._rvc.cached_model_count() > 0)
        if currently_open:
            if self._rvc:
                self._rvc.unload_all()
            self._rvc_toggle_btn.configure(
                text="RVC: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )
            self._rvc_status.configure(text="RVC: Bellek Bos", text_color="gray")
            self._log("RVC onbellegi temizlendi, VRAM serbest.", "info")
        else:
            self._rvc_toggle_btn.configure(
                text="RVC: Yukleniyor…",
                fg_color=("orange", "#e67e22"), hover_color=("orange", "#e67e22"),
            )
            if self._rvc:
                def _on_preload_done():
                    n = self._rvc.cached_model_count()
                    label = f"RVC: {n} model" if n else "RVC: Hazir"
                    self.after(0, lambda: self._rvc_toggle_btn.configure(
                        text=label + " ✓",
                        fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
                    ))
                import threading as _t
                _t.Thread(target=lambda: (self._rvc.preload_all(), _on_preload_done()), daemon=True).start()

    def _toggle_ducking(self):
        """Audio Ducking'i açar/kapatır."""
        enabled = self._config.get("ducking", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("ducking", {})["enabled"] = new_state
        self._save_config(self._config)
        if self._ducker:
            self._ducker.update_settings(enabled=new_state)
        if new_state:
            self._duck_btn.configure(
                text="DUCK: Acik",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        else:
            self._duck_btn.configure(
                text="DUCK: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )

    def _refresh_indicators(self):
        """Uygulama baslangindan veya preset yuklemesinden sonra tum toglelari gunceller."""
        tr_on = self._config.get("translate", {}).get("enabled", False)
        if tr_on and self._translator and self._translator.is_ready():
            self._cev_btn.configure(
                text="CEV: Acik ✓",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        elif tr_on:
            self._cev_btn.configure(
                text="CEV: Yukleniyor…",
                fg_color=("orange", "#e67e22"), hover_color=("orange", "#e67e22"),
            )
        else:
            self._cev_btn.configure(
                text="CEV: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )

        sent_on = self._config.get("sentiment", {}).get("enabled", False)
        if sent_on and self._analyzer and self._analyzer.is_ready():
            self._duygu_btn.configure(
                text="DUYGU: Acik ✓",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        elif sent_on:
            self._duygu_btn.configure(
                text="DUYGU: Yukleniyor…",
                fg_color=("orange", "#e67e22"), hover_color=("orange", "#e67e22"),
            )
        else:
            self._duygu_btn.configure(
                text="DUYGU: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )

        dk_on = self._config.get("ducking", {}).get("enabled", False)
        if dk_on:
            self._duck_btn.configure(
                text="DUCK: Acik",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        else:
            self._duck_btn.configure(
                text="DUCK: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )

        rvc_n = self._rvc.cached_model_count() if self._rvc else 0
        if rvc_n > 0:
            self._rvc_toggle_btn.configure(
                text=f"RVC: {rvc_n} model ✓",
                fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            )
        else:
            self._rvc_toggle_btn.configure(
                text="RVC: Kapalı",
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray25"),
            )

    # ──────────────────────── Pipeline Kontrolü ──────────────────────

    def _toggle_pipeline(self):
        if self._pipeline_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self):
        self._pipeline_running = True
        self._start_btn.configure(
            text="■  Durdur",
            fg_color=("#c0392b", "#e74c3c"), hover_color=("#922b21", "#c0392b"),
        )
        self._runner.start()
        self._monitor.start()
        self._log("Pipeline başlatıldı.", "info")
        if not self._region:
            self._log(
                "[UYARI] Bölge seçilmedi — tam ekran taranıyor. "
                "Altyazı bölgesini seçmek OCR doğruluğunu artırır.",
                "error",
            )
        self._log(
            f"[OCR] Dil: {self._config.get('ocr', {}).get('language', 'tur')}  "
            f"Aralık: {self._config.get('ocr', {}).get('interval', 0.4)}s",
            "info",
        )
        self._set_status("Pipeline çalışıyor…")

    def _stop_pipeline(self):
        self._pipeline_running = False
        self._start_btn.configure(
            text="▶  Başlat",
            fg_color=("#1a6aaf", "#2980b9"), hover_color=("#145288", "#1f6391"),
        )
        if self._monitor:
            self._monitor.stop()
        if self._runner:
            self._runner.stop()
        self._log("Pipeline durduruldu.", "info")
        self._set_status("Durduruldu.")

    def _on_subtitle_detected(self, speaker: str, text: str):
        self.after(0, lambda: self._subtitle_var.set(f"{speaker}: {text}"))
        self._log(f"[OCR] {speaker}: {text}", "ocr")
        if self._runner:
            self._runner.push(speaker, text)

    # ─────────────────── Preset Yükleme ──────────────────────────────

    def _apply_preset(self, data: dict):
        """Preset verisini mevcut oturuma uygular."""
        region = data.get("region")
        if region:
            self._on_region_selected(tuple(region), from_selector=False)
        else:
            self._on_region_selected(None, from_selector=False)

        chars = data.get("characters", {})
        self._config["characters"] = chars
        if self._rvc:
            self._rvc.update_character_map(chars)
        self._rebuild_char_rows(chars)

        ocr_s = data.get("ocr", {})
        tts_s = data.get("tts", {})
        tr_s  = data.get("translate", {})

        if ocr_s and self._capture:
            if ocr_s.get("language"):
                self._capture.language = ocr_s["language"]
            if ocr_s.get("interval") and self._monitor:
                self._monitor.interval = ocr_s["interval"]
        if tts_s and self._tts:
            self._tts.update_settings(
                language=tts_s.get("language"),
                speed=tts_s.get("speed"),
            )
        if tr_s and self._translator:
            self._translator.update_settings(
                enabled=tr_s.get("enabled"),
                source_lang=tr_s.get("source_lang"),
            )

        for section, vals in data.items():
            if isinstance(vals, dict):
                self._config.setdefault(section, {}).update(vals)

        self._save_config(self._config)
        self._settings_panel.load_config(self._config)
        self._refresh_indicators()
        self._log(f"Profil yüklendi.", "info")
        self._set_status("Profil uygulandı.")

    def _rebuild_char_rows(self, chars: dict):
        for row in self._char_rows:
            row["frame"].destroy()
        self._char_rows.clear()
        for name, pth in chars.items():
            self._add_char_row(name, pth)

    # ─────────────────── Test Yakalama ───────────────────────────────

    def _test_capture(self):
        """
        Mevcut bölgeden ekran görüntüsü alır; ham ve işlenmiş görselleri
        output/ klasörüne kaydeder, OCR sonucunu log'a yazar.
        """
        import threading as _t

        def _run():
            if not self._capture:
                self._log("[TEST] Capture modülü henüz hazır değil.", "error")
                return

            self._log("[TEST] Ekran yakalanıyor…", "info")
            from src.ocr.capture import preprocess_image
            from pathlib import Path

            out_dir = Path(__file__).parent.parent.parent / "output"
            out_dir.mkdir(exist_ok=True)

            img = self._capture.capture()
            if img is None:
                err = getattr(self._capture, "_last_capture_error", None)
                self._log("[TEST] Ekran yakalanamadı.", "error")
                if err:
                    for line in err.strip().splitlines():
                        self._log(f"[TEST] {line}", "error")
                return

            raw_path = out_dir / "debug_raw.png"
            pre_path = out_dir / "debug_preprocessed.png"
            img.save(str(raw_path))
            self._log(f"[TEST] Ham görüntü kaydedildi: {raw_path}", "info")

            pre = preprocess_image(img.copy())
            pre.save(str(pre_path))
            self._log(f"[TEST] İşlenmiş görüntü kaydedildi: {pre_path}", "info")

            text = self._capture.extract_text(img)
            if text:
                self._log(f"[TEST] OCR sonucu: {text[:200].replace(chr(10), ' | ')}", "ocr")
            else:
                self._log(
                    "[TEST] OCR hiç metin bulamadı. "
                    "output/debug_raw.png ve debug_preprocessed.png dosyalarını inceleyin.",
                    "error",
                )

        _t.Thread(target=_run, daemon=True).start()

    # ─────────────────── Bölge Seçimi ────────────────────────────────

    def _select_region(self):
        from src.ui.region_selector import RegionSelector
        RegionSelector(self, callback=self._on_region_selected)

    def _scale_region_for_mss(self, region: tuple) -> tuple:
        """
        Tkinter (mantıksal piksel) koordinatlarını mss (fiziksel piksel) koordinatlarına
        dönüştürür. Windows DPI ölçeklemesinde tkinter ile mss arasındaki uyuşmazlığı giderir.
        """
        if not self._capture or not region:
            return region
        try:
            monitors = self._capture._sct.monitors
            if len(monitors) < 2:
                return region
            mon = monitors[1]  # primary monitor
            tk_w = self.winfo_screenwidth()
            tk_h = self.winfo_screenheight()
            if tk_w <= 0 or tk_h <= 0:
                return region
            mss_w = mon["width"]
            mss_h = mon["height"]
            scale_x = mss_w / tk_w
            scale_y = mss_h / tk_h
            if abs(scale_x - 1.0) < 0.01 and abs(scale_y - 1.0) < 0.01:
                return region
            x1, y1, w, h = region
            return (
                int(mon["left"] + x1 * scale_x),
                int(mon["top"] + y1 * scale_y),
                int(w * scale_x),
                int(h * scale_y),
            )
        except Exception:
            return region

    def _on_region_selected(self, region: Optional[tuple], from_selector: bool = True):
        if region:
            # Sadece RegionSelector'dan gelen tkinter koordinatlarını mss'e dönüştür;
            # config/preset'ten gelenler zaten mss formatında
            mss_region = self._scale_region_for_mss(region) if from_selector else region
            self._region = mss_region
            l, t, w, h = mss_region
            self._region_label.configure(text=f"Bölge: {w}×{h} @ ({l},{t})")
            if self._capture:
                self._capture.set_region_from_tuple(mss_region)
            self._config.setdefault("ocr", {})["region"] = list(mss_region)
        else:
            self._region = None
            self._region_label.configure(text="Bölge: Tam Ekran")
            if self._capture:
                self._capture.set_region_from_tuple(None)
            self._config.setdefault("ocr", {})["region"] = None
        self._log(f"[OCR] Bölge: {region}", "info")

    # ─────────────────── Model Yöneticisi ────────────────────────────

    def _open_model_manager(self):
        from src.ui.model_manager import ModelManager
        ModelManager(
            self,
            on_model_selected=self._on_default_model_selected,
            current_model=self._config.get("rvc", {}).get("model_path"),
        )

    def _on_default_model_selected(self, model_path: Optional[str], index_path: Optional[str]):
        if not model_path:
            self._rvc_status.configure(text="RVC: Model Yok", text_color="gray")
            return
        self._config.setdefault("rvc", {})["model_path"] = model_path
        self._config.setdefault("rvc", {})["index_path"] = index_path
        self._save_config(self._config)
        self._rvc_status.configure(text="RVC: Yükleniyor…", text_color="orange")
        if self._rvc:
            self._rvc.set_default_model_async(
                model_path, index_path,
                on_done=lambda ok: self.after(0, lambda: self._on_rvc_ready(ok)),
            )

    # ─────────────────── Ayarlar ─────────────────────────────────────

    def _apply_settings(self, settings: dict):
        ocr_s  = settings.get("ocr", {})
        tts_s  = settings.get("tts", {})
        rvc_s  = settings.get("rvc", {})
        tr_s   = settings.get("translate", {})
        dk_s   = settings.get("ducking", {})

        if self._capture:
            if ocr_s.get("tesseract_path"):
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = ocr_s["tesseract_path"]
            self._capture.language = ocr_s.get("language", "tur")
        if self._monitor:
            self._monitor.interval = ocr_s.get("interval", 0.4)

        if self._tts:
            self._tts.update_settings(
                language=tts_s.get("language"),
                speed=tts_s.get("speed"),
                speaker_wav=tts_s.get("speaker_wav"),
            )
        if self._rvc:
            self._rvc.update_settings(
                pitch=rvc_s.get("pitch"),
                index_rate=rvc_s.get("index_rate"),
                f0_method=rvc_s.get("f0_method"),
                protect=rvc_s.get("protect"),
                realtime_mode=rvc_s.get("realtime_mode"),
            )
        if self._translator:
            self._translator.update_settings(
                enabled=tr_s.get("enabled"),
                source_lang=tr_s.get("source_lang"),
            )
            if tr_s.get("enabled") and not self._translator.is_ready():
                self._translator.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_translate_ready(ok))
                )

        sent_s = settings.get("sentiment", {})
        if self._analyzer:
            self._analyzer.update_settings(
                enabled=sent_s.get("enabled"),
                context_window=sent_s.get("context_window"),
            )
            if sent_s.get("enabled") and not self._analyzer.is_ready():
                self._analyzer.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_analyzer_ready(ok))
                )

        tts_local = tts_s.get("local_model_dir")
        if tts_local and self._tts:
            self._tts_status.configure(text="TTS: Yukleniyor...", text_color="orange")
            self._tts.update_model_async(
                tts_local,
                on_done=lambda ok: self.after(0, lambda: self._on_tts_ready(ok)),
            )

        if self._ducker:
            self._ducker.update_settings(
                enabled=dk_s.get("enabled"),
                duck_level=dk_s.get("level"),
                target_process=dk_s.get("target_process"),
            )

        for section, data in settings.items():
            self._config.setdefault(section, {}).update(data)
        self._save_config(self._config)
        self._refresh_indicators()
        self._log("Ayarlar kaydedildi.", "info")
        self._set_status("Ayarlar uygulandı.")

    # ─────────────────── Yardımcılar ─────────────────────────────────

    def _log(self, msg: str, tag: str = "info"):
        def _write():
            self._log_box.configure(state="normal")
            self._log_box.insert("end", msg + "\n", tag)
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
        self.after(0, _write)

    def _set_status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    def _on_close(self):
        self._stop_pipeline()
        if self._capture:
            self._capture.close()
        self.destroy()
