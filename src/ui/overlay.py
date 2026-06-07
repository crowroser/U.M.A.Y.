"""
src/ui/overlay.py
Oyun Üstü Şeffaf Altyazı Penceresi (Overlay).
Windows Win32 API'leri kullanarak fare tıklamalarını geçiren (click-through),
her zaman üstte (always-on-top) duran ve arka planı şeffaf olan pencere oluşturur.
"""

import ctypes
import logging
from typing import Optional
import customtkinter as ctk

logger = logging.getLogger("UMAY.Overlay")

TRANS_COLOR = "#010101"  # Şeffaflık rengi (neredeyse siyah)


class OverlayWindow(ctk.CTkToplevel):
    """
    Oyunların üzerinde altyazı akışını gösteren şeffaf pencere.
    """

    def __init__(self, master=None):
        super().__init__(master)
        
        # Kenarlıksız pencere
        self.overrideredirect(True)
        # Her zaman üstte
        self.wm_attributes("-topmost", True)
        # Şeffaflık rengi ataması
        self.wm_attributes("-transparentcolor", TRANS_COLOR)
        # Arka plan rengi
        self.configure(fg_color=TRANS_COLOR)
        
        # Başlangıç konum ve boyutları (Alt orta)
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        self.overlay_width = 800
        self.overlay_height = 100
        self.x_pos = (self.screen_width - self.overlay_width) // 2
        self.y_pos = self.screen_height - 150
        
        self.geometry(f"{self.overlay_width}x{self.overlay_height}+{self.x_pos}+{self.y_pos}")
        
        # Etiketler (Altyazı gösterim alanları)
        self.subtitle_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#ffffff",
            anchor="center",
            justify="center",
            bg_color=TRANS_COLOR
        )
        self.subtitle_label.pack(fill="both", expand=True, padx=20, pady=10)

        # Win32 click-through stilini yükle
        self.after(100, self._make_click_through)

    def _make_click_through(self):
        """Pencereyi tıklama geçirgen (click-through) yapar."""
        try:
            # Tkinter pencere ID'sini HWND olarak al
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            
            # GWL_EXSTYLE = -20
            # WS_EX_LAYERED = 0x00080000
            # WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            
            # Stilleri uygula
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000 | 0x00000020)
            logger.info("Overlay penceresi tıklama geçirgen hale getirildi.")
        except Exception as e:
            logger.error(f"Overlay tıklama geçirgenlik hatası: {e}")

    def show_subtitle(self, speaker: str, text: str, color: str = "#ffffff"):
        """Yeni altyazıyı ekranda görüntüler."""
        display_text = f"{speaker}: {text}" if speaker else text
        
        def _update():
            self.subtitle_label.configure(text=display_text, text_color=color)
            
        self.after(0, _update)

    def clear(self):
        """Altyazıyı temizler."""
        self.after(0, lambda: self.subtitle_label.configure(text=""))

    def update_style(self, font_size: int, text_color: str, position: str = "bottom", custom_y: Optional[int] = None):
        """Kullanıcı ayarlarına göre font boyutunu ve konumu günceller."""
        def _update():
            # Font boyutu güncelle
            self.subtitle_label.configure(font=ctk.CTkFont(family="Segoe UI", size=font_size, weight="bold"))
            
            # Konum ayarla
            y = self.screen_height - 150
            if position == "top":
                y = 100
            elif position == "custom" and custom_y is not None:
                y = custom_y
                
            self.y_pos = y
            self.geometry(f"{self.overlay_width}x{self.overlay_height}+{self.x_pos}+{self.y_pos}")
            
        self.after(0, _update)
