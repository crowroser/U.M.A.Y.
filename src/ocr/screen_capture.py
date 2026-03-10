import re
import time
import threading
from typing import Callable, Optional, Tuple

import mss
import numpy as np
import pytesseract
from PIL import Image


class SubtitleParser:
    """
    'Kisi: Metin' formatindaki altyazi satirlarini ayristirir.
    Ornek: 'Kisi1 : Merhaba!' -> ('Kisi1', 'Merhaba!')
    """

    PATTERN = re.compile(r"^([^:]+?)\s*:\s*(.+)$", re.MULTILINE)

    @staticmethod
    def parse(text: str) -> list[tuple[str, str]]:
        """OCR metninden (konusmaci, metin) cifti listesi doner."""
        results = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = SubtitleParser.PATTERN.match(line)
            if match:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                results.append((speaker, content))
        return results

    @staticmethod
    def extract_text_only(text: str) -> list[str]:
        """Sadece konusma metnini doner, konusmaci adini atar."""
        return [content for _, content in SubtitleParser.parse(text)]


class ScreenCapture:
    """
    mss ile ekran bolgesini yakalar ve Tesseract OCR ile metin cikarir.
    """

    def __init__(self, tesseract_path: Optional[str] = None, language: str = "tur"):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.language = language
        self.region: Optional[dict] = None
        self._sct = mss.mss()

    def set_region(self, left: int, top: int, width: int, height: int):
        """OCR icin ekran bolgesini ayarlar."""
        self.region = {"left": left, "top": top, "width": width, "height": height}

    def set_region_from_tuple(self, region: Optional[Tuple[int, int, int, int]]):
        """(left, top, width, height) tuple'indan bolge ayarlar."""
        if region:
            self.set_region(*region)
        else:
            self.region = None

    def capture(self) -> Optional[Image.Image]:
        """Belirtilen bolgeyi veya tam ekrani yakalar."""
        try:
            if self.region:
                screenshot = self._sct.grab(self.region)
            else:
                monitor = self._sct.monitors[1]
                screenshot = self._sct.grab(monitor)
            return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception:
            return None

    def extract_text(self, image: Image.Image) -> str:
        """Goruntuden OCR ile metin cikarir."""
        config = "--oem 3 --psm 6"
        try:
            text = pytesseract.image_to_string(image, lang=self.language, config=config)
            return text.strip()
        except Exception:
            return ""

    def capture_and_extract(self) -> str:
        """Ekran yakala ve OCR uygula, ham metni doner."""
        image = self.capture()
        if image is None:
            return ""
        return self.extract_text(image)

    def get_monitor_info(self) -> list[dict]:
        """Bagli monitörlerin bilgilerini doner."""
        return self._sct.monitors[1:]

    def close(self):
        self._sct.close()


class SubtitleMonitor:
    """
    Belirli aralikla ekrani tarar ve yeni altyazi satirlari bulundugunda
    callback cagiran arka plan is parcacigi.
    """

    def __init__(
        self,
        capture: ScreenCapture,
        on_new_subtitle: Callable[[str, str], None],
        interval: float = 1.5,
    ):
        self._capture = capture
        self._on_new_subtitle = on_new_subtitle
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_text = ""

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float):
        self._interval = max(0.5, float(value))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def _loop(self):
        while self._running:
            try:
                raw_text = self._capture.capture_and_extract()
                if raw_text and raw_text != self._last_text:
                    pairs = SubtitleParser.parse(raw_text)
                    for speaker, text in pairs:
                        full_line = f"{speaker}: {text}"
                        if full_line != self._last_text:
                            self._last_text = full_line
                            self._on_new_subtitle(speaker, text)
            except Exception:
                pass
            time.sleep(self._interval)
