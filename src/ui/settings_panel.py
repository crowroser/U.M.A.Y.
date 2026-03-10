from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk


class SettingsPanel(ctk.CTkFrame):
    """OCR, TTS, RVC, Ceviri, Duygu ve Ses ayarlari paneli."""

    def __init__(self, master, on_save: Callable[[dict], None], **kwargs):
        super().__init__(master, **kwargs)
        self._on_save = on_save
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Ayarlar", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(16, 8), padx=16, anchor="w"
        )
        self._notebook = ctk.CTkTabview(self)
        self._notebook.pack(fill="both", expand=True, padx=8, pady=4)
        for t in ["OCR", "TTS", "RVC", "Ceviri", "Duygu", "Ses"]:
            self._notebook.add(t)
        self._build_ocr_tab()
        self._build_tts_tab()
        self._build_rvc_tab()
        self._build_translate_tab()
        self._build_sentiment_tab()
        self._build_audio_tab()
        ctk.CTkButton(self, text="Kaydet", command=self._save).pack(pady=12, padx=16, fill="x")

    # ── OCR ──────────────────────────────────────────────────────────

    def _build_ocr_tab(self):
        tab = self._notebook.tab("OCR")
        ctk.CTkLabel(tab, text="Tesseract Yolu:").pack(anchor="w", padx=8, pady=(8, 0))
        self._tess_path = ctk.CTkEntry(tab, placeholder_text="C:\\...\\tesseract.exe")
        self._tess_path.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="OCR Dili:").pack(anchor="w", padx=8)
        self._ocr_lang = ctk.CTkEntry(tab, placeholder_text="tur")
        self._ocr_lang.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="Tarama Araligi (sn):").pack(anchor="w", padx=8)
        self._ocr_interval = ctk.CTkSlider(tab, from_=0.3, to=5.0, number_of_steps=14)
        self._ocr_interval.set(0.4)
        self._ocr_interval.pack(fill="x", padx=8, pady=(0, 4))
        self._interval_label = ctk.CTkLabel(tab, text="0.4 sn")
        self._interval_label.pack(anchor="e", padx=8)
        self._ocr_interval.configure(command=self._update_interval_label)
        ctk.CTkLabel(tab, text="Monitor:").pack(anchor="w", padx=8, pady=(8, 0))
        self._monitor_var = ctk.StringVar(value="1")
        self._monitor_combo = ctk.CTkComboBox(tab, values=["1"], variable=self._monitor_var)
        self._monitor_combo.pack(fill="x", padx=8, pady=(0, 8))

    # ── TTS ──────────────────────────────────────────────────────────

    def _build_tts_tab(self):
        tab = self._notebook.tab("TTS")
        ctk.CTkLabel(tab, text="Dil:").pack(anchor="w", padx=8, pady=(8, 0))
        self._tts_lang = ctk.CTkEntry(tab, placeholder_text="tr")
        self._tts_lang.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="Konusma Hizi:").pack(anchor="w", padx=8)
        self._tts_speed = ctk.CTkSlider(tab, from_=0.5, to=2.0, number_of_steps=6)
        self._tts_speed.set(1.0)
        self._tts_speed.pack(fill="x", padx=8, pady=(0, 4))
        self._speed_label = ctk.CTkLabel(tab, text="1.0x")
        self._speed_label.pack(anchor="e", padx=8)
        self._tts_speed.configure(command=self._update_speed_label)
        ctk.CTkLabel(tab, text="Speaker WAV (Opsiyonel):").pack(anchor="w", padx=8, pady=(8, 0))
        spk_frame = ctk.CTkFrame(tab, fg_color="transparent")
        spk_frame.pack(fill="x", padx=8, pady=(0, 8))
        self._speaker_wav = ctk.CTkEntry(spk_frame, placeholder_text="Ses klonu .wav")
        self._speaker_wav.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(spk_frame, text="...", width=32, command=self._browse_speaker_wav).pack(
            side="left", padx=(4, 0)
        )
        ctk.CTkLabel(
            tab, text="--- HuggingFace / Yerel Model ---",
            text_color="gray", font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=8, pady=(10, 2))
        ctk.CTkLabel(tab, text="Yerel Model Klasoru:").pack(anchor="w", padx=8)
        hf_row = ctk.CTkFrame(tab, fg_color="transparent")
        hf_row.pack(fill="x", padx=8, pady=(0, 4))
        self._local_model_dir = ctk.CTkEntry(hf_row, placeholder_text="models/tts/ModelAdi/")
        self._local_model_dir.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(hf_row, text="...", width=32, command=self._browse_local_model).pack(
            side="left", padx=(4, 0)
        )
        hf_btns = ctk.CTkFrame(tab, fg_color="transparent")
        hf_btns.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkButton(
            hf_btns, text="HF'den Indir", width=115,
            fg_color=("#2d7d46", "#27ae60"), hover_color=("#235f36", "#1e8449"),
            command=self._open_hf_downloader,
        ).pack(side="left")
        ctk.CTkButton(
            hf_btns, text="Etkinlestir", width=90,
            command=self._activate_local_model,
        ).pack(side="left", padx=(8, 0))

    def _browse_local_model(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="Yerel TTS Model Klasoru Sec")
        if path:
            self._local_model_dir.delete(0, "end")
            self._local_model_dir.insert(0, path)

    def _open_hf_downloader(self):
        try:
            from src.ui.hf_downloader import HFDownloader
            HFDownloader(
                self.winfo_toplevel(),
                on_downloaded=lambda d: (
                    self._local_model_dir.delete(0, "end"),
                    self._local_model_dir.insert(0, d),
                ),
            )
        except Exception:
            pass

    def _activate_local_model(self):
        path = self._local_model_dir.get().strip()
        if path:
            self._on_save({"tts": {"local_model_dir": path}})

    # ── RVC ──────────────────────────────────────────────────────────

    def _build_rvc_tab(self):
        tab = self._notebook.tab("RVC")

        self._realtime_mode = ctk.BooleanVar(value=True)
        rt_frame = ctk.CTkFrame(tab, fg_color="transparent")
        rt_frame.pack(fill="x", padx=8, pady=(10, 4))
        ctk.CTkSwitch(
            rt_frame, text="Gercek Zamanli Mod (F0: pm ~0.3s)",
            variable=self._realtime_mode,
        ).pack(side="left")
        ctk.CTkLabel(
            tab,
            text="Kapali: rmvpe (~1-2s, kaliteli)  Acik: pm (~0.3s, hizli)",
            font=ctk.CTkFont(size=10), text_color="gray",
        ).pack(anchor="w", padx=8, pady=(0, 6))

        ctk.CTkLabel(tab, text="Pitch (yari ton):").pack(anchor="w", padx=8, pady=(4, 0))
        self._rvc_pitch = ctk.CTkSlider(tab, from_=-24, to=24, number_of_steps=48)
        self._rvc_pitch.set(0)
        self._rvc_pitch.pack(fill="x", padx=8, pady=(0, 4))
        self._pitch_label = ctk.CTkLabel(tab, text="0")
        self._pitch_label.pack(anchor="e", padx=8)
        self._rvc_pitch.configure(command=self._update_pitch_label)
        ctk.CTkLabel(tab, text="Index Orani:").pack(anchor="w", padx=8, pady=(8, 0))
        self._index_rate = ctk.CTkSlider(tab, from_=0.0, to=1.0, number_of_steps=10)
        self._index_rate.set(0.75)
        self._index_rate.pack(fill="x", padx=8, pady=(0, 4))
        self._index_label = ctk.CTkLabel(tab, text="0.75")
        self._index_label.pack(anchor="e", padx=8)
        self._index_rate.configure(command=self._update_index_label)
        ctk.CTkLabel(tab, text="F0 Yontemi (Gercek Zamanli Mod kapali ise):").pack(anchor="w", padx=8, pady=(8, 0))
        self._f0_method = ctk.CTkComboBox(tab, values=["rmvpe", "harvest", "crepe", "pm"])
        self._f0_method.set("rmvpe")
        self._f0_method.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="Koruma:").pack(anchor="w", padx=8)
        self._protect = ctk.CTkSlider(tab, from_=0.0, to=0.5, number_of_steps=10)
        self._protect.set(0.33)
        self._protect.pack(fill="x", padx=8, pady=(0, 4))
        self._protect_label = ctk.CTkLabel(tab, text="0.33")
        self._protect_label.pack(anchor="e", padx=8)
        self._protect.configure(command=self._update_protect_label)

    # ── Ceviri ───────────────────────────────────────────────────────

    def _build_translate_tab(self):
        tab = self._notebook.tab("Ceviri")
        self._translate_enabled = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            tab, text="Otomatik Ceviriyi Etkinlestir",
            variable=self._translate_enabled,
        ).pack(anchor="w", padx=8, pady=(12, 8))
        ctk.CTkLabel(tab, text="Kaynak Dil:").pack(anchor="w", padx=8)
        self._src_lang = ctk.CTkComboBox(
            tab, values=["eng", "deu", "fra", "jpn", "kor", "zho", "spa", "rus"]
        )
        self._src_lang.set("eng")
        self._src_lang.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="Model (HuggingFace):").pack(anchor="w", padx=8)
        self._translate_model = ctk.CTkEntry(tab, placeholder_text="Helsinki-NLP/opus-mt-en-tr")
        self._translate_model.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(
            tab, text="Ilk etkinlestirmede yuklenir (~300 MB).",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=8)

    # ── Duygu ────────────────────────────────────────────────────────

    def _build_sentiment_tab(self):
        tab = self._notebook.tab("Duygu")
        self._sentiment_enabled = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            tab, text="Duygu Analizini Etkinlestir",
            variable=self._sentiment_enabled,
        ).pack(anchor="w", padx=8, pady=(12, 8))
        ctk.CTkLabel(tab, text="Model (HuggingFace):").pack(anchor="w", padx=8)
        self._sentiment_model = ctk.CTkEntry(
            tab, placeholder_text="j-hartmann/emotion-english-distilroberta-base"
        )
        self._sentiment_model.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(tab, text="Baglamsal Hafiza (cumle):").pack(anchor="w", padx=8)
        self._context_window = ctk.CTkSlider(tab, from_=1, to=10, number_of_steps=9)
        self._context_window.set(5)
        self._context_window.pack(fill="x", padx=8, pady=(0, 4))
        self._context_label = ctk.CTkLabel(tab, text="5 cumle")
        self._context_label.pack(anchor="e", padx=8)
        self._context_window.configure(command=self._update_context_label)
        ctk.CTkLabel(
            tab,
            text="Duygu tespiti TTS hizini ve\nRVC pitch'ini otomatik ayarlar. (~66 MB)",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=8, pady=(8, 0))

    # ── Ses ──────────────────────────────────────────────────────────

    def _build_audio_tab(self):
        tab = self._notebook.tab("Ses")
        self._ducking_enabled = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            tab, text="Audio Ducking (Oyun Sesini Kis)",
            variable=self._ducking_enabled,
        ).pack(anchor="w", padx=8, pady=(12, 8))
        ctk.CTkLabel(tab, text="Kisma Seviyesi (%):").pack(anchor="w", padx=8)
        self._duck_level = ctk.CTkSlider(tab, from_=0.0, to=1.0, number_of_steps=20)
        self._duck_level.set(0.35)
        self._duck_level.pack(fill="x", padx=8, pady=(0, 4))
        self._duck_label = ctk.CTkLabel(tab, text="%35")
        self._duck_label.pack(anchor="e", padx=8)
        self._duck_level.configure(command=self._update_duck_label)
        ctk.CTkLabel(tab, text="Hedef Surec Adi:").pack(anchor="w", padx=8, pady=(8, 0))
        self._duck_process = ctk.CTkEntry(tab, placeholder_text="hades2.exe ...")
        self._duck_process.pack(fill="x", padx=8, pady=(0, 8))
        ctk.CTkLabel(
            tab, text="Bos birakilirsa tum ses oturumlari kisilir.",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).pack(anchor="w", padx=8)

    # ── Slider etiketleri ─────────────────────────────────────────────

    def _update_duck_label(self, v):
        self._duck_label.configure(text=f"%{int(float(v) * 100)}")

    def _update_interval_label(self, v):
        self._interval_label.configure(text=f"{float(v):.1f} sn")

    def _update_speed_label(self, v):
        self._speed_label.configure(text=f"{float(v):.1f}x")

    def _update_pitch_label(self, v):
        self._pitch_label.configure(text=str(int(float(v))))

    def _update_index_label(self, v):
        self._index_label.configure(text=f"{float(v):.2f}")

    def _update_protect_label(self, v):
        self._protect_label.configure(text=f"{float(v):.2f}")

    def _update_context_label(self, v):
        self._context_label.configure(text=f"{int(float(v))} cumle")

    def _browse_speaker_wav(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Speaker WAV sec",
            filetypes=[("WAV dosyalari", "*.wav"), ("Tum dosyalar", "*.*")],
        )
        if path:
            self._speaker_wav.delete(0, "end")
            self._speaker_wav.insert(0, path)

    # ── Kaydet / Yukle ────────────────────────────────────────────────

    def _save(self):
        self._on_save({
            "ocr": {
                "tesseract_path": self._tess_path.get().strip() or None,
                "language": self._ocr_lang.get().strip() or "tur",
                "interval": self._ocr_interval.get(),
                "monitor": int(self._monitor_var.get()),
            },
            "tts": {
                "language": self._tts_lang.get().strip() or "tr",
                "speed": self._tts_speed.get(),
                "speaker_wav": self._speaker_wav.get().strip() or None,
                "local_model_dir": self._local_model_dir.get().strip() or None,
            },
            "rvc": {
                "pitch": int(self._rvc_pitch.get()),
                "index_rate": self._index_rate.get(),
                "f0_method": self._f0_method.get(),
                "protect": self._protect.get(),
                "realtime_mode": self._realtime_mode.get(),
            },
            "translate": {
                "enabled": self._translate_enabled.get(),
                "source_lang": self._src_lang.get(),
                "model": self._translate_model.get().strip() or "Helsinki-NLP/opus-mt-en-tr",
            },
            "sentiment": {
                "enabled": self._sentiment_enabled.get(),
                "model": self._sentiment_model.get().strip() or "j-hartmann/emotion-english-distilroberta-base",
                "context_window": int(self._context_window.get()),
            },
            "ducking": {
                "enabled": self._ducking_enabled.get(),
                "level": self._duck_level.get(),
                "target_process": self._duck_process.get().strip(),
            },
        })

    def load_config(self, config: dict):
        ocr  = config.get("ocr", {})
        tts  = config.get("tts", {})
        rvc  = config.get("rvc", {})
        tr   = config.get("translate", {})
        sent = config.get("sentiment", {})
        dk   = config.get("ducking", {})

        if ocr.get("tesseract_path"):
            self._tess_path.insert(0, ocr["tesseract_path"])
        if ocr.get("language"):
            self._ocr_lang.insert(0, ocr["language"])
        if ocr.get("interval"):
            self._ocr_interval.set(ocr["interval"])
            self._interval_label.configure(text=f"{ocr['interval']:.1f} sn")

        if tts.get("language"):
            self._tts_lang.insert(0, tts["language"])
        if tts.get("speed"):
            self._tts_speed.set(tts["speed"])
            self._speed_label.configure(text=f"{tts['speed']:.1f}x")
        if tts.get("speaker_wav"):
            self._speaker_wav.insert(0, tts["speaker_wav"])
        if tts.get("local_model_dir"):
            self._local_model_dir.insert(0, tts["local_model_dir"])

        if rvc.get("pitch") is not None:
            self._rvc_pitch.set(rvc["pitch"])
            self._pitch_label.configure(text=str(rvc["pitch"]))
        if rvc.get("index_rate") is not None:
            self._index_rate.set(rvc["index_rate"])
            self._index_label.configure(text=f"{rvc['index_rate']:.2f}")
        if rvc.get("f0_method"):
            self._f0_method.set(rvc["f0_method"])
        if rvc.get("protect") is not None:
            self._protect.set(rvc["protect"])
            self._protect_label.configure(text=f"{rvc['protect']:.2f}")
        if rvc.get("realtime_mode") is not None:
            self._realtime_mode.set(rvc["realtime_mode"])

        if tr.get("enabled"):
            self._translate_enabled.set(tr["enabled"])
        if tr.get("source_lang"):
            self._src_lang.set(tr["source_lang"])
        if tr.get("model"):
            self._translate_model.delete(0, "end")
            self._translate_model.insert(0, tr["model"])

        if sent.get("enabled"):
            self._sentiment_enabled.set(sent["enabled"])
        if sent.get("model"):
            self._sentiment_model.delete(0, "end")
            self._sentiment_model.insert(0, sent["model"])
        if sent.get("context_window") is not None:
            self._context_window.set(sent["context_window"])
            self._context_label.configure(text=f"{sent['context_window']} cumle")

        if dk.get("enabled"):
            self._ducking_enabled.set(dk["enabled"])
        if dk.get("level") is not None:
            self._duck_level.set(dk["level"])
            self._duck_label.configure(text=f"%{int(dk['level'] * 100)}")
        if dk.get("target_process"):
            self._duck_process.delete(0, "end")
            self._duck_process.insert(0, dk["target_process"])

    def update_monitor_list(self, count: int):
        self._monitor_combo.configure(values=[str(i) for i in range(1, count + 1)])
