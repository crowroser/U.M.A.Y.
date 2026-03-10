"""
src/ocr/capture.py
Gelismis ekran yakalama ve OCR modulu.
- difflib ile %90 benzerlik anti-spam filtresi
- Minimum metin uzunlugu filtresi
- Goruntu on isleme (grayscale + kontrast artirma)
- 0.3s'ye kadar yapilandirilebilir tarama araligi
- Turkce karakterleri destekleyen guclu regex ile karakter/diyalog ayristirma
"""

import re
import time
import threading
from difflib import SequenceMatcher
from typing import Callable, Optional, Tuple

import mss
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


SUBTITLE_PATTERN = re.compile(
    r"^([A-ZÇĞİÖŞÜa-zçğışöü\d][^:\n]{0,40}?)\s*:\s*(.{3,})$",
    re.MULTILINE,
)


def _similarity(a: str, b: str) -> float:
    """Iki metin arasindaki difflib benzerlik oranini doner (0.0–1.0)."""
    return SequenceMatcher(None, a, b).ratio()


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    OCR dogrulugunu artirmak icin goruntu on islemesi:
    grayscale → kontrast 2x → hafif sharpening
    """
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def parse_subtitle(text: str) -> list[tuple[str, str]]:
    """
    OCR metninden (konusmaci, diyalog) cifti listesi cikarir.
    Ornek: 'GLaDOS: Merhaba!' -> [('GLaDOS', 'Merhaba!')]
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 5:
            continue
        m = SUBTITLE_PATTERN.match(line)
        if m:
            speaker = m.group(1).strip()
            dialog = m.group(2).strip()
            results.append((speaker, dialog))
    return results


class ScreenCapture:
    """mss ile ekran bolgesi yakalama ve Tesseract OCR."""

    def __init__(
        self,
        tesseract_path: Optional[str] = None,
        language: str = "tur",
        preprocess: bool = True,
    ):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.language = language
        self.preprocess = preprocess
        self.region: Optional[dict] = None
        self._sct = mss.mss()

    def set_region(self, left: int, top: int, width: int, height: int):
        self.region = {"left": left, "top": top, "width": width, "height": height}

    def set_region_from_tuple(self, region: Optional[Tuple[int, int, int, int]]):
        if region:
            self.set_region(*region)
        else:
            self.region = None

    def capture(self) -> Optional[Image.Image]:
        try:
            area = self.region or self._sct.monitors[1]
            shot = self._sct.grab(area)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception:
            return None

    def extract_text(self, image: Image.Image) -> str:
        if self.preprocess:
            image = preprocess_image(image)
        config = "--oem 3 --psm 6"
        try:
            return pytesseract.image_to_string(
                image, lang=self.language, config=config
            ).strip()
        except Exception:
            return ""

    def capture_and_extract(self) -> str:
        img = self.capture()
        return self.extract_text(img) if img else ""

    def get_monitor_count(self) -> int:
        return len(self._sct.monitors) - 1

    def close(self):
        self._sct.close()


class SubtitleMonitor:
    """
    Arka planda periyodik ekran taramasi yapan uretici (producer).
    Anti-spam: %90 benzerlik esigi, minimum 5 karakter filtresi.
    """

    SIMILARITY_THRESHOLD = 0.90
    MIN_TEXT_LEN = 5

    def __init__(
        self,
        capture: ScreenCapture,
        on_new_subtitle: Callable[[str, str], None],
        interval: float = 0.4,
    ):
        self._capture = capture
        self._on_new_subtitle = on_new_subtitle
        self._interval = max(0.3, float(interval))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_raw = ""

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float):
        self._interval = max(0.3, float(value))

    def start(self):
        if self._running:
            return
        self._running = True
        self._last_raw = ""
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
                raw = self._capture.capture_and_extract()
                if raw and len(raw.strip()) >= self.MIN_TEXT_LEN:
                    sim = _similarity(self._last_raw, raw)
                    if sim < self.SIMILARITY_THRESHOLD:
                        self._last_raw = raw
                        pairs = parse_subtitle(raw)
                        for speaker, dialog in pairs:
                            self._on_new_subtitle(speaker, dialog)
            except Exception:
                pass
            time.sleep(self._interval)
