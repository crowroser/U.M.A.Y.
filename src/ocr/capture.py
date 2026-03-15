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
import traceback
from difflib import SequenceMatcher
from typing import Callable, Optional, Tuple

import mss
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


SUBTITLE_PATTERN = re.compile(
    r"^([A-ZÇĞİÖŞÜa-zçğışöü\d][^:\n]{0,40}?)\s*:\s*(.{3,})$",
    re.MULTILINE,
)

# Gürültü satırlarını atmak için: yalnızca özel karakter içeren kısa satırlar
_NOISE_PATTERN = re.compile(r"^[\W\d_]+$")

# Kod/debug/teknik gürültü kalıpları
_GARBAGE_CODE = re.compile(
    r"torch|nn\.|parametri|python\.|Loading:|modules\.|\.pth|\.py|dim\s*:|conv\s|kernel|bias|embed\s*dim",
    re.I,
)

# UI/uygulama/log metinleri — TTS'e gönderme
_UI_BLACKLIST = re.compile(
    r"tesseract|tocr|ocr\b|rvc\b|tts\b|ceviri|duygu|monitor\b|slot\s|program\s*files|"
    r"model\s*versiyonu|on[- ]?yükleme|on[- ]?yukleme|tarama\s*aral|"
    r"son\s*aliyazı|son\s*altyazı|unified\s*model|\.exe|\.pth|"
    r"c:\\|c:/|xtts|multilingual|dataset|proci\b|proc\b",
    re.I,
)

# Türkçe ünlüler — anlamlı kelime kontrolü
_VOWELS = set("aeıioöuüAEIİOÖUÜ")


def _looks_like_real_word(w: str) -> bool:
    """En az 3 harfli ve ünlü içeren kelime anlamlı sayılır."""
    if len(w) < 3:
        return False
    return any(c in _VOWELS for c in w)


def _is_garbage_text(text: str) -> bool:
    """
    OCR gürültüsünü tespit eder. Sadece mantıklı diyaloglar TTS'e gider.
    True = gürültü, TTS'e gönderme.
    """
    if not text or len(text.strip()) < 8:
        return True
    t = text.strip()
    words = re.findall(r"\b[^\W\d_]+\b", t)

    # UI/log/uygulama metinleri
    if _UI_BLACKLIST.search(t):
        return True

    # Kod/teknik kalıpları
    if _GARBAGE_CODE.search(t):
        return True

    # Rakam oranı çok yüksek
    digit_count = sum(1 for c in t if c.isdigit())
    if len(t) > 0 and digit_count / len(t) > 0.3:
        return True

    # Özel karakter yoğunluğu
    special = sum(1 for c in t if c in "|></#©@$%^&*{}[]\\")
    if len(t) > 0 and special / len(t) > 0.2:
        return True

    # Kelime yoksa
    if not words:
        return True

    # En az 2 anlamlı kelime (3+ harf, ünlü içeren)
    real_words = [w for w in words if _looks_like_real_word(w)]
    if len(real_words) < 2:
        return True

    # En az 1 kelime 4+ karakter (tam kelime)
    long_words = [w for w in words if len(w) >= 4]
    if not long_words:
        return True

    # Kesik son: "şanslar. >", "nslar. j" — nokta + 1–2 karakter
    if re.search(r"\.\s*[>\}\]]?\s*[a-zA-ZğüşıöçĞÜŞİÖÇ]?\s*$", t) and len(t) < 25:
        return True

    # Çoğunluk 2 karakterden kısa kelimeler
    short = sum(1 for w in words if len(w) <= 2)
    if len(words) >= 3 and short / len(words) > 0.6:
        return True

    # Başta 2+ adet 2 harfli kelime + rakam: "ER 4 Vİ çaya"
    two_char = [w for w in words if len(w) == 2]
    if len(two_char) >= 2 and any(c.isdigit() for c in t[:20]):
        return True

    # "1k", "2 model" gibi rakam+harf karışımı (OCR hatası)
    if re.search(r"\d[kKmM]\s|\d\s*model", t):
        return True

    # Türkçe OCR artefaktları: bozuk kelime sonları (protoj, palğgsl, MEİN, dei h)
    if re.search(
        r"protoj\b|palğgsl|MEİN\b|dei\s+h\b|pap\s+Dlarak|fazl\s+r\s+",
        t,
        re.I,
    ):
        return True

    # En az 1 kelime 5+ karakter (tam Türkçe kelime göstergesi)
    very_long = [w for w in words if len(w) >= 5]
    if not very_long:
        return True

    # Ardışık 2+ büyük harf + boşluk: "ZE LİGİ", "Jünün bir pap" (OCR parçalanması)
    if re.search(r"\b[A-ZÇĞİÖŞÜ]{2,}\s+[A-ZÇĞİÖŞÜ]", t) and len(long_words) < 3:
        return True

    return False

# mss 9.x thread-safe değil: her thread kendi instance'ını kullanmalı
_tls = threading.local()


def _get_mss():
    """Thread-local mss instance (mss 9.x ana thread dışında grab yapamaz)."""
    if not getattr(_tls, "mss", None):
        _tls.mss = mss.mss()
    return _tls.mss


def _similarity(a: str, b: str) -> float:
    """Iki metin arasindaki difflib benzerlik oranini doner (0.0–1.0)."""
    return SequenceMatcher(None, a, b).ratio()


def _normalize_for_dedup(text: str) -> str:
    """Tekrar kontrolu icin metni normalize eder."""
    t = re.sub(r"[©®™\*\"\'\`]", "", text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _word_overlap_ratio(a: str, b: str) -> float:
    """İki metnin kelime örtüşme oranı (0.0–1.0). OCR bozulmalarında yardımcı."""
    wa = set(re.findall(r"\b[^\W\d_]{3,}\b", a.lower()))
    wb = set(re.findall(r"\b[^\W\d_]{3,}\b", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def _is_duplicate_of_recent(dialog: str, recent: list[str]) -> bool:
    """
    Yeni dialog, son gonderilenlerden biriyle ayni/benzer mi?
    - %85+ benzerlik -> tekrar
    - %75+ kelime örtüşmesi -> tekrar (OCR bozuk varyant)
    - Kisa parca, uzun gonderilenin on eki -> tekrar (kesik cümle)
    """
    d = _normalize_for_dedup(dialog)
    if len(d) < 10:
        return False
    for r in recent:
        rn = _normalize_for_dedup(r)
        if _similarity(d, rn) >= 0.85:
            return True
        # Kelime örtüşmesi: aynı altyazının OCR bozuk versiyonu
        if _word_overlap_ratio(dialog, r) >= 0.75:
            return True
        # Kesik cümle: dialog, daha önce gönderilen uzun metnin ön eki
        if len(rn) > len(d) * 1.2 and (rn.startswith(d[:20]) or d[:20] in rn):
            return True
    return False


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Oyun altyazilari icin akilli goruntu on islemesi.
    Koyu zemin uzerine acik metin (oyunlarda yaygin) otomatik algilanir ve
    Tesseract'in bekledigi koyu-uzerine-beyaz formata cevirilir.
    """
    import numpy as np

    gray = img.convert("L")
    arr = np.array(gray)

    # Görüntünün ortalama parlaklığına göre zemin rengini tahmin et:
    # Ortalama < 128 → koyu zemin (beyaz metin) → ters çevir
    if arr.mean() < 128:
        gray = Image.fromarray(255 - arr)

    # Kontrast artır ve keskinleştir
    gray = ImageEnhance.Contrast(gray).enhance(2.5)
    gray = gray.filter(ImageFilter.SHARPEN)

    # Büyüt: Tesseract küçük fontlarda daha iyi çalışır
    w, h = gray.size
    if h < 60:
        scale = max(2, 60 // h)
        gray = gray.resize((w * scale, h * scale), Image.LANCZOS)

    return gray


def parse_subtitle(text: str) -> list[tuple[str, str]]:
    """
    OCR metninden (konusmaci, diyalog) cifti listesi cikarir.
    Ornek: 'GLaDOS: Merhaba!' -> [('GLaDOS', 'Merhaba!')]
    Eslesme bulunamazsa düz metin satirlarini ('', metin) olarak doner (fallback).
    """
    results = []
    plain_lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 5:
            continue
        m = SUBTITLE_PATTERN.match(line)
        if m:
            results.append((m.group(1).strip(), m.group(2).strip()))
        elif not _NOISE_PATTERN.match(line):
            plain_lines.append(line)

    if results:
        return results

    # Konusmaci:Metin formati bulunamazsa duz metin satirlarini fallback olarak dondur
    return [("", line) for line in plain_lines]


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
        self._last_capture_error: Optional[str] = None

    @property
    def _sct(self):
        """Thread-local mss (app.py _scale_region_for_mss uyumluluğu)."""
        return _get_mss()

    def set_region(self, left: int, top: int, width: int, height: int):
        self.region = {"left": left, "top": top, "width": width, "height": height}

    def set_region_from_tuple(self, region: Optional[Tuple[int, int, int, int]]):
        if region:
            self.set_region(*region)
        else:
            self.region = None

    def capture(self) -> Optional[Image.Image]:
        self._last_capture_error = None
        try:
            sct = _get_mss()
            area = self.region or sct.monitors[1]
            shot = sct.grab(area)
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        except Exception as e:
            self._last_capture_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return None

    def extract_text(self, image: Image.Image) -> str:
        if self.preprocess:
            image = preprocess_image(image)
        # PSM 6 (düzgün metin bloğu) ve PSM 11 (dağınık metin) sırayla dene;
        # ilk sonuç veren kazanır
        for psm in (6, 11, 3):
            try:
                result = pytesseract.image_to_string(
                    image, lang=self.language, config=f"--oem 3 --psm {psm}"
                ).strip()
                if result:
                    return result
            except Exception:
                pass
        return ""

    def capture_and_extract(self) -> str:
        img = self.capture()
        return self.extract_text(img) if img else ""

    def get_monitor_count(self) -> int:
        return len(_get_mss().monitors) - 1

    def close(self):
        if getattr(_tls, "mss", None):
            try:
                _tls.mss.close()
            except Exception:
                pass
            _tls.mss = None


class SubtitleMonitor:
    """
    Arka planda periyodik ekran taramasi yapan uretici (producer).
    Anti-spam: %90 benzerlik esigi, minimum 5 karakter filtresi.
    Stabilizasyon: 2 ardışık benzer okuma gerekir (geçiş anı gürültüsünü azaltır).
    """

    SIMILARITY_THRESHOLD = 0.90
    STABILIZE_SIMILARITY = 0.80  # 2 okuma bu oranda benzer olmalı
    STABILIZE_REQUIRED = 2  # Kaç ardışık benzer okuma gerekli
    MIN_TEXT_LEN = 5

    def __init__(
        self,
        capture: ScreenCapture,
        on_new_subtitle: Callable[[str, str], None],
        interval: float = 0.4,
        on_log: Optional[Callable[[str, str], None]] = None,
    ):
        self._capture = capture
        self._on_new_subtitle = on_new_subtitle
        self._interval = max(0.3, float(interval))
        self._on_log = on_log or (lambda msg, tag: None)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_raw = ""
        self._candidate_raw = ""
        self._candidate_count = 0
        self._error_count = 0
        self._last_sent_dialogs: list[str] = []  # Son 5 gonderilen (tekrar onleme)
        self._MAX_RECENT = 5

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
        self._candidate_raw = ""
        self._candidate_count = 0
        self._last_sent_dialogs.clear()
        self._error_count = 0
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
                if not raw or len(raw.strip()) < self.MIN_TEXT_LEN:
                    self._candidate_raw = ""
                    self._candidate_count = 0
                    self._error_count += 1
                    if self._error_count % 10 == 0:
                        self._on_log(
                            f"[OCR] Ekranda metin bulunamadı (bölge ayarlandı mı? Dil: {self._capture.language})",
                            "error",
                        )
                    time.sleep(self._interval)
                    continue

                self._error_count = 0
                sim_to_last = _similarity(self._last_raw, raw)

                # Aynı kare tekrar: stabilizasyon için say (2 benzer okuma = işle)
                if sim_to_last >= self.SIMILARITY_THRESHOLD:
                    if (
                        self._candidate_raw
                        and _similarity(raw, self._candidate_raw) >= self.STABILIZE_SIMILARITY
                    ):
                        self._candidate_count += 1
                        if self._candidate_count >= self.STABILIZE_REQUIRED:
                            self._process_raw(self._candidate_raw)
                            self._candidate_raw = ""
                            self._candidate_count = 0
                    time.sleep(self._interval)
                    continue

                # Yeni içerik: aday güncelle veya doğrula
                self._last_raw = raw
                self._on_log(f"[OCR-RAW] {raw[:120].replace(chr(10), ' | ')}", "ocr")

                if (
                    self._candidate_raw
                    and _similarity(raw, self._candidate_raw) >= self.STABILIZE_SIMILARITY
                ):
                    self._candidate_count += 1
                    if self._candidate_count >= self.STABILIZE_REQUIRED:
                        self._process_raw(self._candidate_raw)
                        self._candidate_raw = ""
                        self._candidate_count = 0
                else:
                    self._candidate_raw = raw
                    self._candidate_count = 1

            except Exception as exc:
                self._on_log(f"[OCR-HATA] {exc}", "error")

            time.sleep(self._interval)

    def _process_raw(self, raw: str):
        """Ham OCR metnini parse edip TTS'e gönderir."""
        pairs = parse_subtitle(raw)
        if not pairs:
            self._on_log("[OCR] Altyazı formatı eşleşmedi, metin atlandı.", "error")
            return
        sent = 0
        for speaker, dialog in pairs:
            if _is_garbage_text(dialog):
                continue
            if _is_duplicate_of_recent(dialog, self._last_sent_dialogs):
                continue
            self._on_new_subtitle(speaker, dialog)
            self._last_sent_dialogs.append(dialog)
            if len(self._last_sent_dialogs) > self._MAX_RECENT:
                self._last_sent_dialogs.pop(0)
            sent += 1
        if sent == 0:
            self._on_log("[OCR] Tüm parçalar gürültü olarak filtrelendi.", "error")
