"""
src/ocr/ocr_engine.py
OCR Motorları Soyutlama Katmanı.
Tesseract, EasyOCR ve PaddleOCR motorlarını sarmalar ve ortak bir API sunar.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

logger = logging.getLogger("UMAY.OCREngine")


class BaseOCREngine(ABC):
    """
    Tüm OCR motorları için temel soyut sınıf.
    """

    def __init__(self, language: str = "tur", preprocess: bool = True):
        self.language = language
        self.preprocess = preprocess

    @abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        """Görselden metin okur ve temizlenmiş metni döner."""
        pass

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Ortak görüntü ön işleme (Tesseract ve piksel tabanlı motorlar için uygun).
        """
        gray = img.convert("L")
        arr = np.array(gray)

        # Koyu zemin tespiti ve ters çevirme
        if arr.mean() < 128:
            gray = Image.fromarray(255 - arr)

        # Kontrast ve keskinleştirme
        gray = ImageEnhance.Contrast(gray).enhance(2.5)
        gray = gray.filter(ImageFilter.SHARPEN)

        # Büyütme (küçük fontları kurtarmak için)
        w, h = gray.size
        if h < 60:
            scale = max(2, 60 // h)
            gray = gray.resize((w * scale, h * scale), Image.LANCZOS)

        return gray


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR motoru entegrasyonu.
    """

    def __init__(self, language: str = "tur", preprocess: bool = True, tesseract_path: Optional[str] = None):
        super().__init__(language, preprocess)
        import pytesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, image: Image.Image) -> str:
        import pytesseract
        if self.preprocess:
            image = self.preprocess_image(image)

        # Farklı PSM modlarını deneyerek en iyi sonucu yakala
        for psm in (6, 11, 3):
            try:
                lang_code = self.language
                # Tesseract Türkçe için 'tur', İngilizce için 'eng' kullanır
                result = pytesseract.image_to_string(
                    image, lang=lang_code, config=f"--oem 3 --psm {psm}"
                ).strip()
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Tesseract PSM {psm} hatası: {e}")
        return ""


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR (Deep Learning tabanlı) motoru entegrasyonu.
    """

    def __init__(self, language: str = "tur", preprocess: bool = False):
        # EasyOCR kendi içinde derin öğrenme modelleri barındırdığı için
        # varsayılan olarak piksel tabanlı ön işlemeyi kapatmak daha iyi sonuç verir.
        super().__init__(language, preprocess)
        import easyocr
        
        # Dil kodunu EasyOCR formatına eşle (tur -> tr, eng -> en)
        self.easyocr_langs = [self._map_lang(self.language)]
        logger.info(f"EasyOCR Reader başlatılıyor: {self.easyocr_langs}...")
        self.reader = easyocr.Reader(self.easyocr_langs, gpu=False)

    def _map_lang(self, lang: str) -> str:
        s = lang.strip().lower()
        if s.startswith("tr") or s == "tur":
            return "tr"
        if s.startswith("en") or s == "eng":
            return "en"
        return "tr"

    def extract_text(self, image: Image.Image) -> str:
        if self.preprocess:
            image = self.preprocess_image(image)
            
        # PIL Image -> numpy array
        arr = np.array(image)
        try:
            results = self.reader.readtext(arr, detail=0)
            if results:
                return " ".join(results).strip()
        except Exception as e:
            logger.error(f"EasyOCR okuma hatası: {e}")
        return ""


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR motoru entegrasyonu (Yedek/Stub).
    """

    def __init__(self, language: str = "tur", preprocess: bool = False):
        super().__init__(language, preprocess)
        try:
            from paddleocr import PaddleOCR
            lang_code = "tr" if "tur" in self.language or "tr" in self.language else "en"
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, show_log=False)
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("PaddleOCR kütüphanesi yüklü değil. Stub modunda çalışacak.")

    def extract_text(self, image: Image.Image) -> str:
        if not self.available:
            logger.error("PaddleOCR kullanılabilir değil. Lütfen 'pip install paddleocr' komutunu çalıştırın.")
            return ""

        if self.preprocess:
            image = self.preprocess_image(image)

        arr = np.array(image)
        try:
            result = self.ocr.ocr(arr, cls=True)
            if result and result[0]:
                lines = [line[1][0] for line in result[0]]
                return " ".join(lines).strip()
        except Exception as e:
            logger.error(f"PaddleOCR okuma hatası: {e}")
        return ""


class OCREngineFactory:
    """
    İstenen motor tipine göre OCR nesnesi üreten fabrika sınıfı.
    """

    @staticmethod
    def create_engine(
        engine_type: str,
        language: str = "tur",
        preprocess: bool = True,
        tesseract_path: Optional[str] = None
    ) -> BaseOCREngine:
        t = engine_type.strip().lower()
        if t == "easyocr":
            try:
                return EasyOCREngine(language=language, preprocess=False)
            except Exception as e:
                logger.error(f"EasyOCR başlatılamadı, Tesseract fallback yapılacak: {e}")
                return TesseractEngine(language=language, preprocess=preprocess, tesseract_path=tesseract_path)
        elif t == "paddleocr":
            engine = PaddleOCREngine(language=language, preprocess=preprocess)
            if engine.available:
                return engine
            else:
                logger.warning("PaddleOCR bulunamadı, Tesseract fallback yapılıyor.")
                return TesseractEngine(language=language, preprocess=preprocess, tesseract_path=tesseract_path)
        else:
            return TesseractEngine(language=language, preprocess=preprocess, tesseract_path=tesseract_path)
