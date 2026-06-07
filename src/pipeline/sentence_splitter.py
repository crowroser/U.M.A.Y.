"""
src/pipeline/sentence_splitter.py
Uzun metinleri kisa, TTS-dostu parcalara boler.

Amac: TTS surecini hizlandirmak.  200 karakterlik bir cumleyi tek seferde
sentezlemek ~3-4 saniye surer; ama 3×60 karakter parca olarak sentezlemek
her biri ~0.8s surer ve ilk parca hemen oynatilabilir.

Kurallar:
- Nokta, unlem, soru isareti sonrasi bol
- Virgul, noktali virgul sonrasi bol (cumle yeterince uzunsa)
- Turkce kisaltmalari bolme (Dr., vb., vs., vd., Bkz., No.)
- Minimum 10, maksimum 100 karakter parca uzunlugu
"""

import re
from typing import List

# Turkce kisaltmalar — bunlarin ardindaki noktadan BOLME
_ABBREVIATIONS = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Doç|Yrd|Öğr|Gör|Bkz|bkz|vb|vs|vd|No|Jr|Sr|St|Sn|Bn|Hz)\.",
    re.IGNORECASE,
)

# Cumleler arasi noktalama (son sinir)
_SENTENCE_END = re.compile(r"([.!?…]+)\s+")

# Virgul ve noktali virgul (yumusak sinir)
_CLAUSE_SEP = re.compile(r"([,;:—–\-])\s+")

# Minimum ve maksimum parca uzunlugu
MIN_CHUNK = 10
MAX_CHUNK = 100
SOFT_TARGET = 60  # Ideal parca uzunlugu


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Kisaltmalardaki noktalari gecici olarak korur."""
    replacements = {}
    counter = 0

    def _replace(m):
        nonlocal counter
        key = f"\x00ABBR{counter}\x00"
        replacements[key] = m.group(0)
        counter += 1
        return key

    protected = _ABBREVIATIONS.sub(_replace, text)
    return protected, replacements


def _restore_abbreviations(text: str, replacements: dict[str, str]) -> str:
    """Korunan kisaltmalari geri yukler."""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def split_sentences(text: str) -> List[str]:
    """
    Metni TTS icin uygun parcalara boler.

    Oncelik sirasi:
    1. Cumle sonu noktalamasi (. ! ?)
    2. Virgul/noktali virgul (parca MAX_CHUNK'i asiyorsa)
    3. Bosluk (son care)

    Her parca MIN_CHUNK–MAX_CHUNK arasinda tutulur.
    """
    text = text.strip()
    if not text:
        return []

    # Zaten kisa metin: bolmeye gerek yok
    if len(text) <= MAX_CHUNK:
        return [text]

    # Kisaltmalari koru
    protected, abbr_map = _protect_abbreviations(text)

    # 1. Cumle sonlarindan bol
    parts = _split_by_pattern(protected, _SENTENCE_END)

    # 2. Hala uzun parcalari virgulden bol
    refined = []
    for part in parts:
        if len(part) > MAX_CHUNK:
            refined.extend(_split_by_pattern(part, _CLAUSE_SEP))
        else:
            refined.append(part)

    # 3. Hala uzun parcalari bosluktan bol
    final = []
    for part in refined:
        if len(part) > MAX_CHUNK:
            final.extend(_split_by_whitespace(part, SOFT_TARGET))
        else:
            final.append(part)

    # Kisa parcalari birbirine birlestir
    merged = _merge_short_chunks(final)

    # Kisaltmalari geri yukle
    result = [_restore_abbreviations(p, abbr_map) for p in merged]

    # Son temizlik
    return [p.strip() for p in result if p.strip() and len(p.strip()) >= 3]


def _split_by_pattern(text: str, pattern: re.Pattern) -> List[str]:
    """Regex patternine gore metni boler, ayiriciyi onceki parcaya ekler."""
    parts = pattern.split(text)
    result = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        # Eger sonraki eleman ayirici ise, bu parcaya ekle
        if i + 1 < len(parts) and len(parts[i + 1]) <= 3:
            chunk += parts[i + 1]
            i += 2
        else:
            i += 1
        if chunk.strip():
            result.append(chunk.strip())
    return result if result else [text]


def _split_by_whitespace(text: str, target: int) -> List[str]:
    """Metni bosluk sinirlarindan target uzunluga yakin parcalara boler."""
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current else 0)
        if current_len + word_len > target and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _merge_short_chunks(chunks: List[str]) -> List[str]:
    """MIN_CHUNK'ten kisa parcalari bir onceki veya sonraki ile birlestirir."""
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) < MIN_CHUNK:
            merged[-1] = merged[-1] + " " + chunk
        elif len(chunk) < MIN_CHUNK and len(merged[-1]) + len(chunk) + 1 <= MAX_CHUNK:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged
