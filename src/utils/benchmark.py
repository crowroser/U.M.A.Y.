"""
src/utils/benchmark.py
U.M.A.Y Performans Test ve Benchmark Modülü.
"""

import time
import json
import logging
from pathlib import Path
from typing import Callable, Optional
from PIL import Image, ImageDraw

import numpy as np

logger = logging.getLogger("UMAY.Benchmark")


class BenchmarkSuite:
    """OCR, TTS ve RVC adımlarının performansını ölçen benchmark araçları."""

    def __init__(self, app_instance):
        self.app = app_instance
        self.results_dir = Path(__file__).parent.parent.parent / "output"
        self.results_dir.mkdir(exist_ok=True)

    def run_ocr_benchmark(self, iterations: int = 3) -> dict:
        """OCR motorunun tanıma hızını ölçer."""
        capture = self.app._capture
        if not capture:
            return {"error": "OCR modülü yüklenmemiş."}

        # Test için üzerinde metin olan sanal bir görsel oluştur
        img = Image.new("RGB", (600, 150), color=(10, 10, 10))
        d = ImageDraw.Draw(img)
        # Basit çizgiler ve kutular çizerek metin taklidi yapalım (OCR motorunun çalışması için)
        d.rectangle([(20, 20), (580, 130)], outline=(255, 255, 255), width=2)
        d.text((50, 60), "TEST SPEAKER: Bu bir performans ve hiz testidir.", fill=(255, 255, 255))

        durations = []
        texts = []

        for i in range(iterations):
            t0 = time.perf_counter()
            text = capture.extract_text(img)
            durations.append((time.perf_counter() - t0) * 1000)
            texts.append(text)

        avg_latency = sum(durations) / len(durations)
        return {
            "engine": capture.engine_type,
            "language": capture.language,
            "iterations": iterations,
            "avg_latency_ms": round(avg_latency, 1),
            "min_latency_ms": round(min(durations), 1),
            "max_latency_ms": round(max(durations), 1),
            "sample_output": texts[0][:100] if texts else ""
        }

    def run_tts_benchmark(self, text: str = "Bu bir yapay zeka ses sentezleme testidir. U.M.A.Y sistemi benchmark testi.") -> dict:
        """TTS motorunun sentez hızını, CPS ve RTF değerlerini ölçer."""
        tts = self.app._tts
        if not tts or not tts.is_ready():
            return {"error": "TTS modülü hazır değil."}

        t0 = time.perf_counter()
        result = tts.synthesize_to_array(text, speaker="Claribel Dervla", emotion="neutral")
        latency = (time.perf_counter() - t0) * 1000

        if result is None:
            return {"error": "TTS sentezi başarısız oldu."}

        sr, data = result
        audio_duration_sec = len(data) / sr
        rtf = (latency / 1000) / audio_duration_sec if audio_duration_sec > 0 else 0.0
        cps = len(text) / (latency / 1000) if latency > 0 else 0.0

        return {
            "text_len": len(text),
            "latency_ms": round(latency, 1),
            "audio_duration_sec": round(audio_duration_sec, 2),
            "rtf": round(rtf, 3),
            "cps": round(cps, 1),
            "sample_rate": sr,
        }

    def run_rvc_benchmark(self) -> dict:
        """RVC dönüştürücü hızını ve RTF değerini ölçer."""
        rvc = self.app._rvc
        if not rvc:
            return {"error": "RVC modülü yüklenmemiş."}

        character = None
        # İlk geçerli karakteri bul
        for char in self.app._config.get("characters", {}).keys():
            if rvc.has_model_for(char):
                character = char
                break

        if not character:
            # Fallback to default model if any
            if rvc._default_model:
                character = "default"
            else:
                return {"error": "Aktif veya atanmış RVC modeli bulunamadı. Lütfen Karakterler sekmesinden model atayın."}

        # 2 saniyelik sahte ses verisi (16kHz)
        sr_in = 16000
        dummy_audio = np.sin(2 * np.pi * 440 * np.arange(sr_in * 2) / sr_in).astype(np.float32)

        t0 = time.perf_counter()
        try:
            rvc_result = rvc.convert_array(dummy_audio, sr_in, character=character, pitch_override_delta=0)
        except Exception as e:
            return {"error": f"RVC dönüşüm hatası: {e}"}

        latency = (time.perf_counter() - t0) * 1000

        if not rvc_result:
            return {"error": "RVC dönüşümü başarısız oldu."}

        sr_out, data_out = rvc_result
        audio_duration_sec = len(dummy_audio) / sr_in
        rtf = (latency / 1000) / audio_duration_sec

        return {
            "character_used": character,
            "input_duration_sec": round(audio_duration_sec, 2),
            "latency_ms": round(latency, 1),
            "rtf": round(rtf, 3),
            "sample_rate_out": sr_out,
            "realtime_mode": rvc.realtime_mode,
            "f0_method": rvc.f0_method,
        }

    def run_all(self, on_progress: Callable[[str, float], None]) -> dict:
        """Tüm benchmark testlerini sırasıyla çalıştırır."""
        on_progress("OCR Hız Testi Başlatılıyor...", 0.1)
        ocr_res = self.run_ocr_benchmark()

        on_progress("TTS Sentez Testi Başlatılıyor...", 0.4)
        tts_res = self.run_tts_benchmark()

        on_progress("RVC Dönüşüm Testi Başlatılıyor...", 0.7)
        rvc_res = self.run_rvc_benchmark()

        on_progress("Testler Tamamlandı.", 1.0)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ocr": ocr_res,
            "tts": tts_res,
            "rvc": rvc_res,
        }

        # Sonuçları JSON dosyasına kaydet
        filename = f"benchmark_results_{int(time.time())}.json"
        try:
            with open(self.results_dir / filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
        except OSError as e:
            logger.error(f"Benchmark sonuçları dosyaya yazılamadı: {e}")

        return report


def generate_markdown_report(report: dict) -> str:
    """Benchmark raporunu okunabilir bir markdown tablosu olarak formatlar."""
    lines = []
    lines.append(f"# U.M.A.Y Performans Raporu — {report['timestamp']}\n")

    # 1. OCR Raporu
    lines.append("## 📸 OCR Hız Performansı")
    ocr = report.get("ocr", {})
    if "error" in ocr:
        lines.append(f"  ❌ Hata: {ocr['error']}\n")
    else:
        lines.append(f"- **OCR Motoru:** {ocr.get('engine', 'Bilinmiyor').upper()}")
        lines.append(f"- **Dil:** {ocr.get('language', 'Bilinmiyor')}")
        lines.append(f"- **Ortalama Gecikme:** {ocr.get('avg_latency_ms', 0)} ms")
        lines.append(f"- **Min / Max Gecikme:** {ocr.get('min_latency_ms', 0)} ms / {ocr.get('max_latency_ms', 0)} ms\n")

    # 2. TTS Raporu
    lines.append("## 🔊 TTS Sentez Performansı")
    tts = report.get("tts", {})
    if "error" in tts:
        lines.append(f"  ❌ Hata: {tts['error']}\n")
    else:
        lines.append(f"- **Ortalama Gecikme:** {tts.get('latency_ms', 0)} ms")
        lines.append(f"- **Ses Uzunluğu:** {tts.get('audio_duration_sec', 0)} s")
        lines.append(f"- **Karakter Hızı (CPS):** {tts.get('cps', 0)} harf/sn")
        rtf_color = "🟢 (Çok Hızlı)" if tts.get('rtf', 99) < 0.5 else "🟡 (Yavaş)"
        lines.append(f"- **Real-Time Factor (RTF):** {tts.get('rtf', 0)} {rtf_color}\n")

    # 3. RVC Raporu
    lines.append("## 🎤 RVC Ses Dönüşüm Performansı")
    rvc = report.get("rvc", {})
    if "error" in rvc:
        lines.append(f"  ❌ Hata: {rvc['error']}\n")
    else:
        lines.append(f"- **Kullanılan Karakter:** {rvc.get('character_used', 'Bilinmiyor')}")
        lines.append(f"- **Dönüşüm Gecikmesi:** {rvc.get('latency_ms', 0)} ms")
        lines.append(f"- **RVC Realtime Modu:** {'Açık' if rvc.get('realtime_mode') else 'Kapalı'}")
        lines.append(f"- **F0 Metodu:** {rvc.get('f0_method', 'Bilinmiyor')}")
        rtf_color = "🟢 (Çok Hızlı)" if rvc.get('rtf', 99) < 0.5 else "🟡 (Yavaş)"
        lines.append(f"- **Real-Time Factor (RTF):** {rvc.get('rtf', 0)} {rtf_color}\n")

    # 4. Genel Değerlendirme
    lines.append("## ⏱ Toplam Pipeline Gecikme Değerlendirmesi")
    if "error" not in tts and "error" not in rvc:
        total_latency = tts.get("latency_ms", 0) + rvc.get("latency_ms", 0)
        lines.append(f"- **Uçtan Uca Sentez Süresi (İlk Kelime Oynatılana Kadar):** **{total_latency:.1f} ms**")
        if total_latency < 1000:
            lines.append("- **Durum:** 🚀 **Mükemmel (Gerçek Zamanlı Kullanıma Tam Uygun)**")
        elif total_latency < 2000:
            lines.append("- **Durum:** 👍 **İyi (Küçük bir gecikme hissedilebilir)**")
        else:
            lines.append("- **Durum:** ⚠️ **Geliştirilmeli (RVC Realtime modunu açmayı veya GPU kullanmayı deneyin)**")
    else:
        lines.append("  Hata oluştuğu için toplam gecikme hesaplanamadı.")

    return "\n".join(lines)
