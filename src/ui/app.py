"""
src/ui/app.py
U.M.A.Y ana penceresi — Premium UI v2.

Ozellikler:
- Canli pipeline performans dashboard (TTS/RVC/Toplam sure barlari)
- Modern modul durum kartlari (TTS/RVC/CEV/DUYGU/DUCK)
- Animasyonlu pipeline durumu (pulsating dot)
- Gelismis altyazi ekrani (son 3 altyazi, karakter renkleri)
- Cache hit gostergesi ve istatistikler
- Pipeline zamanlama cubugu
- Log filtresi ve temizle butonu
- Premium koyu tema renk paleti
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from src.ui.settings_panel import SettingsPanel

BASE_DIR = Path(__file__).parent.parent.parent

# ── Premium Renk Paleti ───────────────────────────────────────────────
# Koyu tema icin ozenle secilmis HSL tabanli renkler
COLORS = {
    "bg_dark":       "#0d1117",
    "bg_card":       "#161b22",
    "bg_card_hover": "#1c2333",
    "bg_elevated":   "#21262d",
    "border":        "#30363d",
    "text_primary":  "#e6edf3",
    "text_secondary":"#8b949e",
    "text_muted":    "#484f58",
    "accent_blue":   "#58a6ff",
    "accent_cyan":   "#56d4dd",
    "accent_green":  "#3fb950",
    "accent_orange": "#d29922",
    "accent_red":    "#f85149",
    "accent_purple": "#bc8cff",
    "accent_pink":   "#f778ba",
    "gradient_start":"#1a3a5c",
    "gradient_end":  "#0d1117",
    "subtitle_bg":   "#1c2333",
    "bar_tts":       "#58a6ff",
    "bar_rvc":       "#3fb950",
    "bar_total":     "#bc8cff",
    "bar_cache":     "#d29922",
}


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
        self._pulse_active = False
        self._overlay = None
        self._subtitle_history: list[tuple[str, str]] = []  # (speaker, text) son 3

        ui_cfg = config.get("ui", {})
        ctk.set_appearance_mode(ui_cfg.get("theme", "dark"))
        ctk.set_default_color_theme(ui_cfg.get("color_theme", "blue"))

        self.title(self.APP_TITLE)
        self.geometry(f"{ui_cfg.get('window_width', 1200)}x{ui_cfg.get('window_height', 780)}")
        self.minsize(1000, 680)
        self.configure(fg_color=COLORS["bg_dark"])

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

        # Sol: Ana panel
        self._main_frame = ctk.CTkFrame(self, fg_color=COLORS["bg_card"], corner_radius=12)
        self._main_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self._main_frame.grid_columnconfigure(0, weight=1)
        self._main_frame.grid_rowconfigure(6, weight=1)  # Log alani genisler

        # Sag: Tab paneli
        self._right_panel = ctk.CTkTabview(
            self, width=300,
            fg_color=COLORS["bg_card"],
            segmented_button_fg_color=COLORS["bg_elevated"],
            segmented_button_selected_color=COLORS["accent_blue"],
            segmented_button_selected_hover_color="#4a90d9",
            segmented_button_unselected_color=COLORS["bg_elevated"],
            segmented_button_unselected_hover_color=COLORS["bg_card_hover"],
        )
        self._right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self._right_panel.add("⚙ Ayarlar")
        self._right_panel.add("👤 Karakterler")
        self._right_panel.add("📊 Performans")
        self._right_panel.add("⚡ Benchmark")

        self._settings_panel = SettingsPanel(
            self._right_panel.tab("⚙ Ayarlar"),
            on_save=self._apply_settings,
        )
        self._settings_panel.pack(fill="both", expand=True)
        self._settings_panel.load_config(self._config)

        self._build_char_panel(self._right_panel.tab("👤 Karakterler"))
        self._build_perf_panel(self._right_panel.tab("📊 Performans"))
        self._build_benchmark_panel(self._right_panel.tab("⚡ Benchmark"))

        self._build_header()
        self._build_preset_bar()
        self._build_module_cards()
        self._build_control_bar()
        self._build_subtitle_display()
        self._build_pipeline_timing_bar()
        self._build_log_area()
        self._build_status_bar()

    # ── Header ─────────────────────────────────────────────────────────

    def _build_header(self):
        header = ctk.CTkFrame(
            self._main_frame,
            fg_color=COLORS["gradient_start"],
            corner_radius=10,
            height=70,
        )
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 0))
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        # Logo ve baslik
        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=16, pady=12, sticky="w")

        ctk.CTkLabel(
            logo_frame, text="U.M.A.Y",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=COLORS["accent_cyan"],
        ).pack(side="left")

        ctk.CTkLabel(
            logo_frame, text="  v2.0",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        ).pack(side="left", pady=(6, 0))

        # Pipeline durum gostergesi (animasyonlu)
        self._pulse_frame = ctk.CTkFrame(header, fg_color="transparent")
        self._pulse_frame.grid(row=0, column=1, sticky="e", padx=8)

        self._pulse_dot = ctk.CTkLabel(
            self._pulse_frame, text="●",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["text_muted"],
        )
        self._pulse_dot.pack(side="left", padx=(0, 6))

        self._pipeline_status_label = ctk.CTkLabel(
            self._pulse_frame, text="Beklemede",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"],
        )
        self._pipeline_status_label.pack(side="left")

        # Sag ust butonlar
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=12, sticky="e")

        for text, color, hover, cmd in [
            ("🎙 Ses Modeli", COLORS["accent_purple"], "#9b6fd9", self._open_model_manager),
            ("📸 Test Yakala", COLORS["accent_orange"], "#b5841c", self._test_capture),
            ("🖼 Bölge Seç", COLORS["accent_green"], "#2ea043", self._select_region),
        ]:
            ctk.CTkButton(
                btn_frame, text=text, width=115, height=32,
                font=ctk.CTkFont(size=12),
                fg_color=color, hover_color=hover,
                corner_radius=8,
                command=cmd,
            ).pack(side="right", padx=3)

        self._overlay_btn = ctk.CTkButton(
            btn_frame, text="💬 Overlay Aç", width=115, height=32,
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["accent_blue"], hover_color="#4a90d9",
            corner_radius=8,
            command=self._toggle_overlay_window,
        )
        self._overlay_btn.pack(side="right", padx=3)

    # ── Modul Durum Kartlari ───────────────────────────────────────────

    def _build_module_cards(self):
        """Her modul icin kucuk durum karti."""
        cards_frame = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        cards_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 0))
        for i in range(5):
            cards_frame.grid_columnconfigure(i, weight=1)

        self._module_cards = {}
        modules = [
            ("TTS", "🔊", COLORS["bar_tts"], "Hazırlanıyor…"),
            ("RVC", "🎤", COLORS["bar_rvc"], "Model Yok"),
            ("CEV", "🌐", COLORS["text_muted"], "Kapalı"),
            ("DUYGU", "😊", COLORS["text_muted"], "Kapalı"),
            ("DUCK", "🔉", COLORS["text_muted"], "Kapalı"),
        ]

        for col, (name, icon, color, status) in enumerate(modules):
            card = ctk.CTkFrame(
                cards_frame,
                fg_color=COLORS["bg_elevated"],
                corner_radius=8,
                height=54,
            )
            card.grid(row=0, column=col, padx=3, pady=2, sticky="ew")
            card.grid_propagate(False)
            card.grid_columnconfigure(1, weight=1)

            # Ikon
            ctk.CTkLabel(
                card, text=icon, font=ctk.CTkFont(size=18),
                width=30,
            ).grid(row=0, column=0, padx=(8, 2), pady=6, rowspan=2)

            # Baslik
            ctk.CTkLabel(
                card, text=name,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["text_primary"],
            ).grid(row=0, column=1, sticky="w", padx=2, pady=(6, 0))

            # Durum etiketi
            status_label = ctk.CTkLabel(
                card, text=status,
                font=ctk.CTkFont(size=10),
                text_color=color,
            )
            status_label.grid(row=1, column=1, sticky="w", padx=2, pady=(0, 6))

            # Tiklama toggle
            card.bind("<Button-1>", lambda e, n=name: self._toggle_module(n))
            # Tum child'lar da tiklanasin
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, n=name: self._toggle_module(n))

            self._module_cards[name] = {
                "card": card,
                "status": status_label,
                "color": color,
            }

    def _update_card(self, name: str, status: str, color: str):
        """Modul kartini gunceller."""
        if name in self._module_cards:
            mc = self._module_cards[name]
            mc["status"].configure(text=status, text_color=color)
            mc["color"] = color

    def _toggle_module(self, name: str):
        """Modul karti tiklandiginda toggle."""
        toggles = {
            "CEV": self._toggle_translate,
            "DUYGU": self._toggle_analyzer,
            "RVC": self._toggle_rvc,
            "DUCK": self._toggle_ducking,
        }
        fn = toggles.get(name)
        if fn:
            fn()

    # ── Kontrol Cubugu ─────────────────────────────────────────────────

    def _build_control_bar(self):
        bar = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=12, pady=6)

        # Baslat / Durdur butonu
        self._start_btn = ctk.CTkButton(
            bar, text="▶  Başlat", width=140, height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLORS["accent_blue"],
            hover_color="#4a90d9",
            corner_radius=10,
            command=self._toggle_pipeline,
        )
        self._start_btn.pack(side="left")

        # Bolge etiketi
        self._region_label = ctk.CTkLabel(
            bar, text="📍 Bölge: Tam Ekran",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(size=11),
        )
        self._region_label.pack(side="right", padx=8)

    # ── Altyazi Ekrani ─────────────────────────────────────────────────

    def _build_subtitle_display(self):
        """Son 3 altyaziyi gosteren premium gorunum."""
        sub_frame = ctk.CTkFrame(
            self._main_frame,
            fg_color=COLORS["subtitle_bg"],
            corner_radius=10,
            height=90,
        )
        sub_frame.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 4))
        sub_frame.grid_propagate(False)
        sub_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(sub_frame, fg_color="transparent", height=20)
        header.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(
            header, text="💬 Altyazılar",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(side="left")

        self._sub_lines: list[ctk.CTkLabel] = []
        for i in range(3):
            alpha = 1.0 - i * 0.3  # Eski satirlar soluk
            lbl = ctk.CTkLabel(
                sub_frame, text="",
                font=ctk.CTkFont(
                    family="Segoe UI",
                    size=14 - i * 1,
                    weight="bold" if i == 0 else "normal",
                ),
                text_color=COLORS["text_primary"] if i == 0 else COLORS["text_secondary"],
                anchor="w",
            )
            lbl.pack(fill="x", padx=14, pady=(2, 0))
            self._sub_lines.append(lbl)

    def _update_subtitle_display(self, speaker: str, text: str):
        """Yeni altyazi geldiginde ekrani gunceller."""
        self._subtitle_history.insert(0, (speaker, text))
        if len(self._subtitle_history) > 3:
            self._subtitle_history = self._subtitle_history[:3]

        from src.pipeline.character_db import CharacterDatabase
        db = CharacterDatabase.get_instance()

        for i, lbl in enumerate(self._sub_lines):
            if i < len(self._subtitle_history):
                sp, tx = self._subtitle_history[i]
                display = f"{sp}: {tx}" if sp else tx
                lbl.configure(text=display)
                
                # Karakter rengini uygula
                char_info = db.get_character(sp) if sp else None
                if char_info and char_info.get("color"):
                    lbl.configure(text_color=char_info["color"])
                else:
                    lbl.configure(text_color=COLORS["text_primary"] if i == 0 else COLORS["text_secondary"])
            else:
                lbl.configure(text="")

    # ── Pipeline Zamanlama Cubugu ───────────────────────────────────────

    def _build_pipeline_timing_bar(self):
        """Canli zamanlama gostergesi — TTS/RVC/Toplam sure barlari."""
        timing_frame = ctk.CTkFrame(
            self._main_frame,
            fg_color=COLORS["bg_elevated"],
            corner_radius=8,
            height=40,
        )
        timing_frame.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 4))
        timing_frame.grid_propagate(False)
        timing_frame.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(timing_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=6)

        self._timing_labels = {}
        items = [
            ("⏱ TTS:", "tts", COLORS["bar_tts"]),
            ("⏱ RVC:", "rvc", COLORS["bar_rvc"]),
            ("⏱ Toplam:", "total", COLORS["bar_total"]),
            ("📦 Cache:", "cache", COLORS["bar_cache"]),
        ]

        for text, key, color in items:
            f = ctk.CTkFrame(inner, fg_color="transparent")
            f.pack(side="left", padx=(0, 16))
            ctk.CTkLabel(
                f, text=text,
                font=ctk.CTkFont(size=10),
                text_color=COLORS["text_muted"],
            ).pack(side="left")
            lbl = ctk.CTkLabel(
                f, text="—",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=color,
            )
            lbl.pack(side="left", padx=(4, 0))
            self._timing_labels[key] = lbl

    def _update_timing_bar(self, stats: dict):
        """Zamanlama cubuğunu istatistiklerle günceller."""
        def _update():
            steps = stats.get("steps", {})
            for key in ("tts", "rvc", "total"):
                if key in steps:
                    avg = steps[key].get("avg", 0)
                    if key in self._timing_labels:
                        if avg < 1000:
                            self._timing_labels[key].configure(text=f"{avg}ms")
                        else:
                            self._timing_labels[key].configure(text=f"{avg/1000:.1f}s")

            cache = stats.get("cache", {})
            if isinstance(cache, dict):
                hits = cache.get("hits", 0)
                hit_rate = cache.get("hit_rate", 0)
                if "cache" in self._timing_labels:
                    self._timing_labels["cache"].configure(
                        text=f"%{int(hit_rate * 100)} ({hits} hit)"
                    )
        self.after(0, _update)

    # ── Log Alani ──────────────────────────────────────────────────────

    def _build_log_area(self):
        log_header = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        log_header.grid(row=6, column=0, sticky="ew", padx=12, pady=(0, 2))

        ctk.CTkLabel(
            log_header, text="📋 Pipeline Kayıtları",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_secondary"],
        ).pack(side="left")

        ctk.CTkButton(
            log_header, text="🗑 Temizle", width=80, height=24,
            font=ctk.CTkFont(size=10),
            fg_color=COLORS["bg_elevated"],
            hover_color=COLORS["bg_card_hover"],
            corner_radius=6,
            command=self._clear_log,
        ).pack(side="right")

        self._log_box = ctk.CTkTextbox(
            self._main_frame,
            font=ctk.CTkFont(family="Cascadia Code,Consolas", size=11),
            state="disabled", wrap="word",
            fg_color=COLORS["bg_dark"],
            border_color=COLORS["border"],
            border_width=1,
            corner_radius=8,
        )
        self._log_box.grid(row=7, column=0, sticky="nsew", padx=12, pady=(0, 6))
        self._main_frame.grid_rowconfigure(7, weight=1)

        for tag, color in [
            ("ocr", "#5dade2"), ("tts", COLORS["bar_tts"]), ("rvc", COLORS["bar_rvc"]),
            ("audio", "#f9e79f"), ("error", COLORS["accent_red"]),
            ("info", COLORS["text_secondary"]),
        ]:
            self._log_box.tag_config(tag, foreground=color)

    def _clear_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    # ── Durum Cubugu ───────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = ctk.CTkFrame(
            self._main_frame, height=28,
            fg_color=COLORS["bg_elevated"],
            corner_radius=0,
        )
        bar.grid(row=8, column=0, sticky="ew")
        self._status_var = ctk.StringVar(value="✓ Hazır")
        ctk.CTkLabel(
            bar, textvariable=self._status_var,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=10)

        self._perf_var = ctk.StringVar(value="")
        ctk.CTkLabel(
            bar, textvariable=self._perf_var,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["accent_cyan"],
        ).pack(side="right", padx=10)

    # ── Preset Bar ─────────────────────────────────────────────────────

    def _build_preset_bar(self):
        from src.presets.manager import PresetManager
        from src.ui.preset_panel import PresetBar

        self._preset_mgr = PresetManager(self._config, self._save_config)

        self._preset_bar = PresetBar(
            self._main_frame,
            preset_manager=self._preset_mgr,
            on_load=self._apply_preset,
            fg_color=COLORS["bg_elevated"],
            corner_radius=8,
        )
        self._preset_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(6, 0))

    # ── Performans Paneli (Sag Tab) ────────────────────────────────────

    def _build_perf_panel(self, parent):
        """Performans istatistikleri paneli."""
        ctk.CTkLabel(
            parent, text="Pipeline Performans",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(12, 8))

        self._perf_info = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Cascadia Code,Consolas", size=11),
            state="disabled",
            height=400,
            fg_color=COLORS["bg_dark"],
            corner_radius=8,
        )
        self._perf_info.pack(fill="both", expand=True, padx=8, pady=4)

        ctk.CTkButton(
            parent, text="🔄 Yenile", width=100,
            fg_color=COLORS["accent_blue"],
            command=self._refresh_perf_panel,
        ).pack(pady=8)

    def _refresh_perf_panel(self):
        """Performans panelini güncel verilerle doldurur."""
        if not self._runner:
            return
        stats = self._runner.get_stats()
        cache_stats = self._runner.cache.get_stats()

        lines = []
        lines.append("═══ Pipeline İstatistikleri ═══\n")
        lines.append(f"  Toplam İşlem : {stats.get('total_processed', 0)}")
        lines.append(f"  SPR          : {stats.get('spr', 0)}/s")
        lines.append(f"  Ort Gecikme  : {stats.get('avg_latency_ms', 0)}ms\n")

        # Canlı Sistem Kaynakları
        if hasattr(self, "_sys_monitor") and self._sys_monitor.history:
            sys_stats = self._sys_monitor.history[-1]
            lines.append("═══ Sistem Kaynakları ═══\n")
            lines.append(f"  CPU Kullanımı : %{sys_stats.get('cpu', 0):.0f}")
            lines.append(f"  RAM Kullanımı : %{sys_stats.get('ram_percent', 0):.0f} ({sys_stats.get('ram_used', 0.0):.1f}/{sys_stats.get('ram_total', 0.0):.1f} GB)")
            lines.append(f"  GPU Kullanımı : %{sys_stats.get('gpu', 0):.0f}")
            lines.append(f"  VRAM          : %{sys_stats.get('vram_percent', 0):.0f} ({sys_stats.get('vram_used', 0.0):.0f}/{sys_stats.get('vram_total', 0.0):.0f} MB)\n")

        # VRAM Yöneticisi Envanteri
        from src.utils.vram_manager import VRAMManager
        from pathlib import Path
        import time
        mgr = VRAMManager.get_instance()
        loaded_models = mgr.get_loaded_models()
        lines.append("═══ Aktif Modeller (VRAM) ═══\n")
        lines.append(f"  Bütçe         : {mgr.budget_mb:.0f} MB")
        lines.append(f"  Yüklü Model   : {len(loaded_models)}")
        for m in loaded_models:
            lines.append(f"  • [{m['type']}] {Path(m['id']).name}")
            lines.append(f"    VRAM: {m['estimated_vram_mb']:.0f} MB | Son: {time.strftime('%H:%M:%S', time.localtime(m['last_used'])) if m['last_used'] else 'Hiç'}")
        lines.append("")

        steps = stats.get("steps", {})
        if steps:
            lines.append("═══ Adım Süreleri (ms) ═══\n")
            for step in ("tts", "rvc", "translate", "sentiment", "total"):
                if step in steps:
                    s = steps[step]
                    lines.append(
                        f"  {step.upper():10s}  "
                        f"avg={s['avg']:5d}  "
                        f"min={s['min']:5d}  "
                        f"max={s['max']:5d}  "
                        f"p95={s['p95']:5d}"
                    )

        lines.append(f"\n═══ TTS Cache ═══\n")
        lines.append(f"  RAM Öğe S.   : {cache_stats.get('entries', 0)}")
        lines.append(f"  RAM Bellek   : {cache_stats.get('memory_mb', 0)} / {cache_stats.get('max_memory_mb', 0)} MB")
        lines.append(f"  Disk Öğe S.  : {cache_stats.get('disk_entries', 0)}")
        lines.append(f"  Disk Boyutu  : {cache_stats.get('disk_size_mb', 0)} / {cache_stats.get('max_disk_mb', 0)} MB")
        lines.append(f"  Hit          : {cache_stats.get('hits', 0)}")
        lines.append(f"  Miss         : {cache_stats.get('misses', 0)}")
        lines.append(f"  Hit Rate     : %{int(cache_stats.get('hit_rate', 0) * 100)}")
        lines.append(f"  Evictions    : {cache_stats.get('evictions', 0)}")
        saved = cache_stats.get("saved_ms", 0)
        lines.append(f"  Tasarruf     : {saved/1000:.1f}s")

        rvc_stats = stats.get("rvc_cache", {})
        if rvc_stats:
            lines.append(f"\n═══ RVC Cache ═══\n")
            lines.append(f"  RAM Öğe S.   : {rvc_stats.get('entries', 0)}")
            lines.append(f"  RAM Bellek   : {rvc_stats.get('memory_mb', 0)} / {rvc_stats.get('max_memory_mb', 0)} MB")
            lines.append(f"  Disk Öğe S.  : {rvc_stats.get('disk_entries', 0)}")
            lines.append(f"  Disk Boyutu  : {rvc_stats.get('disk_size_mb', 0)} / {rvc_stats.get('max_disk_mb', 0)} MB")
            lines.append(f"  Hit          : {rvc_stats.get('hits', 0)}")
            lines.append(f"  Miss         : {rvc_stats.get('misses', 0)}")
            lines.append(f"  Hit Rate     : %{int(rvc_stats.get('hit_rate', 0) * 100)}")
            lines.append(f"  Evictions    : {rvc_stats.get('evictions', 0)}")

        text = "\n".join(lines)
        self._perf_info.configure(state="normal")
        self._perf_info.delete("1.0", "end")
        self._perf_info.insert("1.0", text)
        self._perf_info.configure(state="disabled")

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
        ctk.CTkButton(footer, text="+ Ekle", width=60, command=self._add_char_row).pack(side="left", padx=2)
        ctk.CTkButton(footer, text="İçe Aktar", width=80, fg_color=COLORS["accent_orange"], command=self._import_char_pack).pack(side="left", padx=2)
        ctk.CTkButton(footer, text="Kaydet", width=70, command=self._save_char_map).pack(side="right", padx=2)

        from src.pipeline.character_db import CharacterDatabase
        db = CharacterDatabase.get_instance()
        
        # Migration from config if database is empty
        if not db.list_characters() and self._config.get("characters"):
            for name, pth in self._config.get("characters", {}).items():
                refs = self._config.get("character_refs", {}).get(name, {})
                db.add_character(name=name, rvc_model=pth, tts_refs=refs)
        
        for char in db.list_characters():
            self._add_char_row(
                name=char["name"],
                pth=char.get("rvc_model") or "",
                color=char.get("color") or "#ffffff"
            )

    def _add_char_row(self, name: str = "", pth: str = "", color: str = "#ffffff"):
        row_frame = ctk.CTkFrame(self._char_scroll)
        row_frame.pack(fill="x", pady=3)
        name_var = ctk.StringVar(value=name)
        pth_var = ctk.StringVar(value=pth)
        color_var = ctk.StringVar(value=color)

        ctk.CTkEntry(row_frame, textvariable=name_var, placeholder_text="Ad", width=70).pack(side="left", padx=(4, 2))
        ctk.CTkEntry(row_frame, textvariable=pth_var, placeholder_text="model.pth", width=85).pack(side="left", padx=2)
        ctk.CTkButton(row_frame, text="…", width=24, command=lambda v=pth_var: self._browse_pth(v)).pack(side="left", padx=2)
        
        # Renk secici butonu
        color_btn = ctk.CTkButton(
            row_frame, text="", width=24, height=24,
            fg_color=color, hover_color=color,
            corner_radius=4,
            command=lambda v=color_var: self._pick_color(v)
        )
        color_btn.pack(side="left", padx=2)
        
        def update_btn_color(*args):
            c = color_var.get()
            color_btn.configure(fg_color=c, hover_color=c)
        color_var.trace_add("write", update_btn_color)

        row = {"frame": row_frame, "name": name_var, "pth": pth_var, "color": color_var, "color_btn": color_btn}

        ctk.CTkButton(
            row_frame, text="Refs", width=40,
            fg_color=COLORS["accent_blue"],
            command=lambda r=row: self._open_char_refs(r),
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            row_frame, text="📤", width=28,
            fg_color="#884ea0", hover_color="#7d3c98",
            command=lambda r=row: self._export_char_pack(r),
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            row_frame, text="✕", width=24,
            fg_color=COLORS["accent_red"], hover_color="#d73a3a",
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
            self._set_status("Önce karakter adı girin.")
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

    def _pick_color(self, var: ctk.StringVar):
        from tkinter import colorchooser
        init_color = var.get() or "#ffffff"
        color = colorchooser.askcolor(initialcolor=init_color, title="Karakter Subtitle Rengi Seç")
        if color[1]:
            var.set(color[1])

    def _export_char_pack(self, row: dict):
        name = row["name"].get().strip()
        if not name:
            self._set_status("Önce karakter adı girin.")
            return
        
        path = filedialog.asksaveasfilename(
            title=f"{name} Karakter Paketini İhraç Et",
            initialfile=f"{name.lower()}_pack.zip",
            filetypes=[("ZIP dosyaları", "*.zip"), ("Tüm dosyalar", "*.*")]
        )
        if path:
            self._save_char_map()
            from src.pipeline.character_db import CharacterDatabase
            db = CharacterDatabase.get_instance()
            success = db.export_character_pack(name, path)
            if success:
                self._log(f"[CHAR] {name} paketi ihraç edildi: {Path(path).name}", "info")
                self._set_status(f"{name} paketi ihraç edildi.")
            else:
                self._log(f"[HATA] {name} paketi ihraç edilemedi.", "error")
                self._set_status("İhraç hatası.")

    def _import_char_pack(self):
        path = filedialog.askopenfilename(
            title="Karakter Paketi İçe Aktar (.zip)",
            filetypes=[("ZIP dosyaları", "*.zip"), ("Tüm dosyalar", "*.*")]
        )
        if path:
            from src.pipeline.character_db import CharacterDatabase
            db = CharacterDatabase.get_instance()
            import_name = db.import_character_pack(path)
            if import_name:
                self._log(f"[CHAR] Karakter paketi içe aktarıldı: {import_name}", "info")
                self._set_status(f"{import_name} içe aktarıldı.")
                self._rebuild_char_rows_from_db()
            else:
                self._log("[HATA] Karakter paketi içe aktarılamadı.", "error")
                self._set_status("İthalat hatası.")

    def _rebuild_char_rows_from_db(self):
        for row in self._char_rows:
            row["frame"].destroy()
        self._char_rows.clear()
        
        from src.pipeline.character_db import CharacterDatabase
        db = CharacterDatabase.get_instance()
        for char in db.list_characters():
            self._add_char_row(
                name=char["name"],
                pth=char.get("rvc_model") or "",
                color=char.get("color") or "#ffffff"
            )

    def _save_char_map(self):
        from src.pipeline.character_db import CharacterDatabase
        db = CharacterDatabase.get_instance()

        config_characters = {}
        for row in self._char_rows:
            name = row["name"].get().strip()
            pth = row["pth"].get().strip()
            color = row["color"].get().strip()
            if not name:
                continue

            config_characters[name] = pth
            
            existing = db.get_character(name)
            rvc_index = None
            if pth:
                pth_path = Path(pth)
                idx_candidates = list(pth_path.parent.glob(f"{pth_path.stem}*.index"))
                if idx_candidates:
                    rvc_index = str(idx_candidates[0])
            
            db.add_character(
                name=name,
                rvc_model=pth or None,
                rvc_index=rvc_index,
                color=color,
                avatar=existing.get("avatar") if existing else "assets/avatars/default.png",
                regex_patterns=existing.get("regex_patterns") if existing else None,
                tts_refs=existing.get("tts_refs") if existing else {}
            )

        current_names = {row["name"].get().strip().lower() for row in self._char_rows if row["name"].get().strip()}
        for char in db.list_characters():
            if char["name"].lower() not in current_names:
                db.remove_character(char["name"])

        self._config["characters"] = config_characters
        self._save_config(self._config)

        if self._rvc:
            self._rvc.update_character_map(config_characters)
            self._rvc.preload_all()
        if self._tts:
            self._tts.update_character_refs(self._config.get("character_refs", {}))

        self._log(f"Karakterler veritabanına kaydedildi ({len(current_names)} karakter).", "info")
        self._set_status("Karakterler güncellendi.")

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
        from src.utils.vram_manager import VRAMManager
        from src.utils.system_monitor import SystemMonitor

        # VRAM Yöneticisini yapılandır
        VRAMManager.get_instance().configure(self._config)

        # Sistem Kaynak Monitörünü başlat
        self._sys_monitor = SystemMonitor(
            interval=2.0,
            callback=self._on_sys_monitor_stats
        )
        self._sys_monitor.start()

        ocr_cfg = self._config.get("ocr", {})
        dk_cfg  = self._config.get("ducking", {})

        self._capture = ScreenCapture(
            tesseract_path=ocr_cfg.get("tesseract_path"),
            language=ocr_cfg.get("language", "tur"),
            preprocess=True,
            engine_type=ocr_cfg.get("engine_type", "tesseract"),
        )

        from src.ui.overlay import OverlayWindow
        self._overlay = OverlayWindow(self)
        self._overlay.withdraw()

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
            on_stats=self._update_timing_bar,
        )
        self._runner.update_cache_settings(self._config)

        from src.utils.benchmark import BenchmarkSuite
        self._benchmark_suite = BenchmarkSuite(self)

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

        self._apply_overlay_settings()
        self._refresh_indicators()

    def _on_tts_ready(self, ok: bool):
        if ok:
            self._update_card("TTS", "Hazır ✓", COLORS["accent_green"])
        else:
            self._update_card("TTS", "Hata ✗", COLORS["accent_red"])

    def _on_rvc_ready(self, ok: bool):
        if ok:
            n = self._rvc.cached_model_count() if self._rvc else 0
            label = f"{n} model ✓" if n > 0 else "Hazır ✓"
            self._update_card("RVC", label, COLORS["accent_green"])
        else:
            self._update_card("RVC", "Hata ✗", COLORS["accent_red"])

    def _on_translate_ready(self, ok: bool):
        if ok:
            self._update_card("CEV", "Açık ✓", COLORS["accent_green"])
        else:
            self._update_card("CEV", "Hata ✗", COLORS["accent_red"])

    def _on_analyzer_ready(self, ok: bool):
        if ok:
            self._update_card("DUYGU", "Açık ✓", COLORS["accent_green"])
        else:
            self._update_card("DUYGU", "Hata ✗", COLORS["accent_red"])

    # ─────────────────── Modül Toggle'ları ───────────────────────────

    def _toggle_translate(self):
        enabled = self._config.get("translate", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("translate", {})["enabled"] = new_state
        self._save_config(self._config)

        if new_state:
            self._update_card("CEV", "Yükleniyor…", COLORS["accent_orange"])
            if self._translator:
                self._translator.update_settings(enabled=True)
                self._translator.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_translate_ready(ok))
                )
        else:
            if self._translator:
                self._translator.update_settings(enabled=False)
            self._update_card("CEV", "Kapalı", COLORS["text_muted"])
            self._log("Çeviri kapatıldı, bellek serbest.", "info")

    def _toggle_analyzer(self):
        enabled = self._config.get("sentiment", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("sentiment", {})["enabled"] = new_state
        self._save_config(self._config)

        if new_state:
            self._update_card("DUYGU", "Yükleniyor…", COLORS["accent_orange"])
            if self._analyzer:
                self._analyzer.update_settings(enabled=True)
                self._analyzer.load_async(
                    on_done=lambda ok: self.after(0, lambda: self._on_analyzer_ready(ok))
                )
        else:
            if self._analyzer:
                self._analyzer.update_settings(enabled=False)
            self._update_card("DUYGU", "Kapalı", COLORS["text_muted"])
            self._log("Duygu analizi kapatıldı.", "info")

    def _toggle_rvc(self):
        currently_open = bool(self._rvc and self._rvc.cached_model_count() > 0)
        if currently_open:
            if self._rvc:
                self._rvc.unload_all()
            self._update_card("RVC", "Kapalı", COLORS["text_muted"])
            self._log("RVC önbelleği temizlendi, VRAM serbest.", "info")
        else:
            self._update_card("RVC", "Yükleniyor…", COLORS["accent_orange"])
            if self._rvc:
                def _on_preload_done():
                    n = self._rvc.cached_model_count()
                    label = f"{n} model ✓" if n else "Hazır ✓"
                    self.after(0, lambda: self._update_card("RVC", label, COLORS["accent_green"]))
                threading.Thread(
                    target=lambda: (self._rvc.preload_all(), _on_preload_done()),
                    daemon=True,
                ).start()

    def _toggle_ducking(self):
        enabled = self._config.get("ducking", {}).get("enabled", False)
        new_state = not enabled
        self._config.setdefault("ducking", {})["enabled"] = new_state
        self._save_config(self._config)
        if self._ducker:
            self._ducker.update_settings(enabled=new_state)
        if new_state:
            self._update_card("DUCK", "Açık", COLORS["accent_green"])
        else:
            self._update_card("DUCK", "Kapalı", COLORS["text_muted"])

    def _refresh_indicators(self):
        # TTS
        if self._tts and self._tts.is_ready():
            self._update_card("TTS", "Hazır ✓", COLORS["accent_green"])

        # Translate
        tr_on = self._config.get("translate", {}).get("enabled", False)
        if tr_on and self._translator and self._translator.is_ready():
            self._update_card("CEV", "Açık ✓", COLORS["accent_green"])
        elif tr_on:
            self._update_card("CEV", "Yükleniyor…", COLORS["accent_orange"])
        else:
            self._update_card("CEV", "Kapalı", COLORS["text_muted"])

        # Sentiment
        sent_on = self._config.get("sentiment", {}).get("enabled", False)
        if sent_on and self._analyzer and self._analyzer.is_ready():
            self._update_card("DUYGU", "Açık ✓", COLORS["accent_green"])
        elif sent_on:
            self._update_card("DUYGU", "Yükleniyor…", COLORS["accent_orange"])
        else:
            self._update_card("DUYGU", "Kapalı", COLORS["text_muted"])

        # Ducking
        dk_on = self._config.get("ducking", {}).get("enabled", False)
        if dk_on:
            self._update_card("DUCK", "Açık", COLORS["accent_green"])
        else:
            self._update_card("DUCK", "Kapalı", COLORS["text_muted"])

        # RVC
        rvc_n = self._rvc.cached_model_count() if self._rvc else 0
        if rvc_n > 0:
            self._update_card("RVC", f"{rvc_n} model ✓", COLORS["accent_green"])
        else:
            self._update_card("RVC", "Model Yok", COLORS["text_muted"])

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
            fg_color=COLORS["accent_red"],
            hover_color="#d73a3a",
        )
        self._pipeline_status_label.configure(
            text="Çalışıyor",
            text_color=COLORS["accent_green"],
        )
        self._pulse_active = True
        self._animate_pulse()

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
        self._pulse_active = False
        self._start_btn.configure(
            text="▶  Başlat",
            fg_color=COLORS["accent_blue"],
            hover_color="#4a90d9",
        )
        self._pipeline_status_label.configure(
            text="Durduruldu",
            text_color=COLORS["text_muted"],
        )
        self._pulse_dot.configure(text_color=COLORS["text_muted"])

        if self._monitor:
            self._monitor.stop()
        if self._runner:
            self._runner.stop()
        if self._overlay:
            self._overlay.clear()
        self._log("Pipeline durduruldu.", "info")
        self._set_status("Durduruldu.")

    def _animate_pulse(self):
        """Pulsating yesil dot animasyonu."""
        if not self._pulse_active:
            return
        current = self._pulse_dot.cget("text_color")
        if current == COLORS["accent_green"]:
            self._pulse_dot.configure(text_color="#1a5c2e")
        else:
            self._pulse_dot.configure(text_color=COLORS["accent_green"])
        self.after(800, self._animate_pulse)

    def _on_subtitle_detected(self, speaker: str, text: str):
        from src.pipeline.character_db import CharacterDatabase
        db = CharacterDatabase.get_instance()
        detected_sp, cleaned_tx = db.detect_character(text)
        
        if detected_sp:
            speaker = detected_sp
            text = cleaned_tx
        else:
            char_info = db.get_character(speaker)
            if char_info:
                speaker = char_info["name"]

        # Karakter rengini veritabanından al
        char_info = db.get_character(speaker)
        color = char_info.get("color", "#ffffff") if char_info else "#ffffff"

        # Eğer çeviri aktifse ve translator varsa metni çevir
        translated_text = text
        if self._translator and self._translator.enabled:
            try:
                translated_text = self._translator.translate(text)
            except Exception as e:
                self._log(f"[Çeviri Hatası] {e}", "error")

        # UI güncellemelerini ana thread'e yönlendir
        self.after(0, lambda: self._update_subtitle_display(speaker, translated_text))
        
        # Overlay penceresini güncelle (açıksa)
        if self._overlay and self._config.get("overlay", {}).get("enabled", False):
            self.after(0, lambda: self._overlay.show_subtitle(speaker, translated_text, color))

        if translated_text != text:
            self._log(f"[OCR] {speaker}: {text} (Çeviri: {translated_text})", "ocr")
        else:
            self._log(f"[OCR] {speaker}: {text}", "ocr")

        if self._runner:
            self._runner.push(speaker, text)

    # ─────────────────── Preset Yükleme ──────────────────────────────

    def _apply_preset(self, data: dict):
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
            engine_type = ocr_s.get("engine_type", self._capture.engine_type)
            language = ocr_s.get("language", self._capture.language)
            tesseract_path = ocr_s.get("tesseract_path", self._config.get("ocr", {}).get("tesseract_path"))
            self._capture.update_engine(
                engine_type=engine_type,
                language=language,
                tesseract_path=tesseract_path
            )
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
        self._log("Profil yüklendi.", "info")
        self._set_status("Profil uygulandı.")

    def _rebuild_char_rows(self, chars: dict):
        for row in self._char_rows:
            row["frame"].destroy()
        self._char_rows.clear()
        for name, pth in chars.items():
            self._add_char_row(name, pth)

    # ─────────────────── Test Yakalama ───────────────────────────────

    def _test_capture(self):
        def _run():
            if not self._capture:
                self._log("[TEST] Capture modülü henüz hazır değil.", "error")
                return

            self._log("[TEST] Ekran yakalanıyor…", "info")
            from src.ocr.capture import preprocess_image

            out_dir = BASE_DIR / "output"
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
            self._log(f"[TEST] Ham görüntü: {raw_path}", "info")

            pre = preprocess_image(img.copy())
            pre.save(str(pre_path))
            self._log(f"[TEST] İşlenmiş: {pre_path}", "info")

            text = self._capture.extract_text(img)
            if text:
                self._log(f"[TEST] OCR: {text[:200].replace(chr(10), ' | ')}", "ocr")
            else:
                self._log("[TEST] OCR hiç metin bulamadı.", "error")

        threading.Thread(target=_run, daemon=True).start()

    # ─────────────────── Bölge Seçimi ────────────────────────────────

    def _select_region(self):
        from src.ui.region_selector import RegionSelector
        RegionSelector(self, callback=self._on_region_selected)

    def _scale_region_for_mss(self, region: tuple) -> tuple:
        if not self._capture or not region:
            return region
        try:
            monitors = self._capture._sct.monitors
            if len(monitors) < 2:
                return region
            mon = monitors[1]
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
            mss_region = self._scale_region_for_mss(region) if from_selector else region
            self._region = mss_region
            l, t, w, h = mss_region
            self._region_label.configure(text=f"📍 Bölge: {w}×{h} @ ({l},{t})")
            if self._capture:
                self._capture.set_region_from_tuple(mss_region)
            self._config.setdefault("ocr", {})["region"] = list(mss_region)
        else:
            self._region = None
            self._region_label.configure(text="📍 Bölge: Tam Ekran")
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
            self._update_card("RVC", "Model Yok", COLORS["text_muted"])
            return
        self._config.setdefault("rvc", {})["model_path"] = model_path
        self._config.setdefault("rvc", {})["index_path"] = index_path
        self._save_config(self._config)
        self._update_card("RVC", "Yükleniyor…", COLORS["accent_orange"])
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
            engine_type = ocr_s.get("engine_type", self._capture.engine_type)
            language = ocr_s.get("language", "tur")
            tesseract_path = ocr_s.get("tesseract_path")
            self._capture.update_engine(
                engine_type=engine_type,
                language=language,
                tesseract_path=tesseract_path
            )
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
                engine=tr_s.get("engine"),
                api_key=tr_s.get("api_key"),
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
            self._update_card("TTS", "Yükleniyor…", COLORS["accent_orange"])
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
        
        # VRAM Yöneticisini yeni ayarlarla yapılandır
        from src.utils.vram_manager import VRAMManager
        VRAMManager.get_instance().configure(self._config)

        if self._runner:
            self._runner.update_cache_settings(self._config)

        self._save_config(self._config)
        self._apply_overlay_settings()
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

    def _on_sys_monitor_stats(self, stats: dict):
        cpu = stats.get("cpu", 0.0)
        ram = stats.get("ram_percent", 0.0)
        gpu = stats.get("gpu", 0.0)
        vram_used = stats.get("vram_used", 0.0)
        vram_total = stats.get("vram_total", 0.0)
        perf_text = f"💻 CPU: {cpu:.0f}% | RAM: {ram:.0f}% | 🎮 GPU: {gpu:.0f}% | VRAM: {vram_used:.0f}/{vram_total:.0f}MB"
        self.after(0, lambda: self._perf_var.set(perf_text))

    def _on_close(self):
        self._pulse_active = False
        self._stop_pipeline()
        if hasattr(self, "_sys_monitor") and self._sys_monitor:
            try:
                self._sys_monitor.stop()
            except Exception:
                pass
        if self._overlay:
            try:
                self._overlay.destroy()
            except Exception:
                pass
        if self._capture:
            self._capture.close()
        self.destroy()

    def _toggle_overlay_window(self):
        cfg = self._config.setdefault("overlay", {})
        enabled = cfg.get("enabled", False)
        new_state = not enabled
        cfg["enabled"] = new_state
        self._save_config(self._config)
        self._settings_panel.load_config(self._config)
        self._apply_overlay_settings()

    def _apply_overlay_settings(self):
        if not self._overlay:
            return
        cfg = self._config.get("overlay", {})
        enabled = cfg.get("enabled", False)
        if enabled:
            font_size = cfg.get("font_size", 20)
            position = cfg.get("position", "bottom")
            custom_y = cfg.get("custom_y")
            self._overlay.update_style(
                font_size=font_size,
                text_color="#ffffff",
                position=position,
                custom_y=custom_y
            )
            self._overlay.deiconify()
            self._overlay.wm_attributes("-topmost", True)
            self._overlay_btn.configure(
                text="💬 Overlay Kapat",
                fg_color=COLORS["accent_red"],
                hover_color="#d73a3a",
            )
        else:
            self._overlay.withdraw()
            self._overlay_btn.configure(
                text="💬 Overlay Aç",
                fg_color=COLORS["accent_blue"],
                hover_color="#4a90d9",
            )

    # ── Benchmark Arayüzü ──────────────────────────────────────────────
    
    def _build_benchmark_panel(self, parent):
        """Dahili Benchmark paneli."""
        ctk.CTkLabel(
            parent, text="Dahili Benchmark Sistemi",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(12, 6))
        
        # Butonlar grubu
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", padx=8, pady=4)
        
        # Grid layout for buttons
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        
        self._btn_ocr_bench = ctk.CTkButton(
            btn_frame, text="📸 OCR Testi",
            fg_color=COLORS["bg_elevated"],
            hover_color=COLORS["bg_card_hover"],
            command=self._run_ocr_bench
        )
        self._btn_ocr_bench.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        self._btn_tts_bench = ctk.CTkButton(
            btn_frame, text="🔊 TTS Testi",
            fg_color=COLORS["bg_elevated"],
            hover_color=COLORS["bg_card_hover"],
            command=self._run_tts_bench
        )
        self._btn_tts_bench.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        
        self._btn_rvc_bench = ctk.CTkButton(
            btn_frame, text="🎤 RVC Testi",
            fg_color=COLORS["bg_elevated"],
            hover_color=COLORS["bg_card_hover"],
            command=self._run_rvc_bench
        )
        self._btn_rvc_bench.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        self._btn_all_bench = ctk.CTkButton(
            btn_frame, text="⚡ Tümünü Çalıştır",
            fg_color=COLORS["accent_orange"],
            hover_color="#b5841c",
            command=self._run_all_bench
        )
        self._btn_all_bench.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
        
        # İlerleme Çubuğu ve Durum
        self._bench_progress = ctk.CTkProgressBar(parent)
        self._bench_progress.set(0.0)
        self._bench_progress.pack(fill="x", padx=10, pady=(8, 2))
        
        self._bench_status_label = ctk.CTkLabel(
            parent, text="Benchmark testi başlatılmaya hazır.",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
            anchor="w"
        )
        self._bench_status_label.pack(fill="x", padx=10, pady=(0, 8))
        
        # Rapor kutusu
        self._bench_report_box = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Cascadia Code,Consolas", size=11),
            state="disabled",
            fg_color=COLORS["bg_dark"],
            corner_radius=8,
            height=260
        )
        self._bench_report_box.pack(fill="both", expand=True, padx=8, pady=4)
        
        # Alt buton
        self._copy_btn = ctk.CTkButton(
            parent, text="📋 Panoya Kopyala",
            fg_color=COLORS["accent_blue"],
            hover_color="#4a90d9",
            command=self._copy_bench_report
        )
        self._copy_btn.pack(fill="x", padx=8, pady=8)

    def _toggle_bench_buttons(self, state: str):
        self._btn_ocr_bench.configure(state=state)
        self._btn_tts_bench.configure(state=state)
        self._btn_rvc_bench.configure(state=state)
        self._btn_all_bench.configure(state=state)

    def _run_ocr_bench(self):
        self._toggle_bench_buttons("disabled")
        self._bench_progress.set(0.2)
        self._bench_status_label.configure(text="OCR Testi çalıştırılıyor...", text_color=COLORS["accent_blue"])
        
        def _thread():
            try:
                res = self._benchmark_suite.run_ocr_benchmark()
                from src.utils.benchmark import generate_markdown_report
                report = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ocr": res,
                    "tts": {"error": "Bu test çalıştırılmadı."},
                    "rvc": {"error": "Bu test çalıştırılmadı."}
                }
                md = generate_markdown_report(report)
                self.after(0, lambda: self._show_bench_result(md, "OCR testi tamamlandı.", 1.0))
            except Exception as e:
                self.after(0, lambda e=e: self._show_bench_error(f"OCR Benchmark hatası: {e}"))
                
        threading.Thread(target=_thread, daemon=True).start()

    def _run_tts_bench(self):
        self._toggle_bench_buttons("disabled")
        self._bench_progress.set(0.2)
        self._bench_status_label.configure(text="TTS Sentez Testi çalıştırılıyor...", text_color=COLORS["accent_blue"])
        
        def _thread():
            try:
                res = self._benchmark_suite.run_tts_benchmark()
                from src.utils.benchmark import generate_markdown_report
                report = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ocr": {"error": "Bu test çalıştırılmadı."},
                    "tts": res,
                    "rvc": {"error": "Bu test çalıştırılmadı."}
                }
                md = generate_markdown_report(report)
                self.after(0, lambda: self._show_bench_result(md, "TTS testi tamamlandı.", 1.0))
            except Exception as e:
                self.after(0, lambda e=e: self._show_bench_error(f"TTS Benchmark hatası: {e}"))
                
        threading.Thread(target=_thread, daemon=True).start()

    def _run_rvc_bench(self):
        self._toggle_bench_buttons("disabled")
        self._bench_progress.set(0.2)
        self._bench_status_label.configure(text="RVC Ses Dönüşüm Testi çalıştırılıyor...", text_color=COLORS["accent_blue"])
        
        def _thread():
            try:
                res = self._benchmark_suite.run_rvc_benchmark()
                from src.utils.benchmark import generate_markdown_report
                report = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "ocr": {"error": "Bu test çalıştırılmadı."},
                    "tts": {"error": "Bu test çalıştırılmadı."},
                    "rvc": res
                }
                md = generate_markdown_report(report)
                self.after(0, lambda: self._show_bench_result(md, "RVC testi tamamlandı.", 1.0))
            except Exception as e:
                self.after(0, lambda e=e: self._show_bench_error(f"RVC Benchmark hatası: {e}"))
                
        threading.Thread(target=_thread, daemon=True).start()

    def _run_all_bench(self):
        self._toggle_bench_buttons("disabled")
        self._bench_progress.set(0.0)
        self._bench_status_label.configure(text="Tüm testler başlatılıyor...", text_color=COLORS["accent_blue"])
        
        def on_progress(status_text, progress_val):
            self.after(0, lambda: (
                self._bench_progress.set(progress_val),
                self._bench_status_label.configure(text=status_text)
            ))
            
        def _thread():
            try:
                report = self._benchmark_suite.run_all(on_progress=on_progress)
                from src.utils.benchmark import generate_markdown_report
                md = generate_markdown_report(report)
                self.after(0, lambda: self._show_bench_result(md, "Tüm testler başarıyla tamamlandı.", 1.0))
            except Exception as e:
                self.after(0, lambda e=e: self._show_bench_error(f"Tümünü Çalıştır hatası: {e}"))
                
        threading.Thread(target=_thread, daemon=True).start()

    def _show_bench_result(self, md_report: str, status_msg: str, progress_val: float):
        self._bench_progress.set(progress_val)
        self._bench_status_label.configure(text=status_msg, text_color=COLORS["accent_green"])
        
        self._bench_report_box.configure(state="normal")
        self._bench_report_box.delete("1.0", "end")
        self._bench_report_box.insert("1.0", md_report)
        self._bench_report_box.configure(state="disabled")
        
        self._toggle_bench_buttons("normal")

    def _show_bench_error(self, err_msg: str):
        self._bench_progress.set(0.0)
        self._bench_status_label.configure(text=err_msg, text_color=COLORS["accent_red"])
        
        self._bench_report_box.configure(state="normal")
        self._bench_report_box.delete("1.0", "end")
        self._bench_report_box.insert("1.0", f"Hata Oluştu:\n{err_msg}")
        self._bench_report_box.configure(state="disabled")
        
        self._toggle_bench_buttons("normal")

    def _copy_bench_report(self):
        report_text = self._bench_report_box.get("1.0", "end-1c").strip()
        if not report_text or report_text.startswith("Hata Oluştu:") or report_text == "":
            self._set_status("Kopyalanacak geçerli bir rapor yok.")
            return
        
        try:
            self.clipboard_clear()
            self.clipboard_append(report_text)
            self._set_status("Rapor panoya kopyalandı!")
            self._log("[BENCHMARK] Rapor panoya kopyalandı.", "info")
        except Exception as e:
            self._log(f"[BENCHMARK] Panoya kopyalama hatası: {e}", "error")
            self._set_status("Kopyalama hatası.")
