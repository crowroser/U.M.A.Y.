# U.M.A.Y — Unified Model-based Audio Yield

**U.M.A.Y** (Unified Model-based Audio Yield), oyunlardaki ve uygulamalardaki alt yazıları gerçek zamanlı olarak okuyup, metni konuşmaya çeviren (TTS), ardından karakter sesine dönüştüren (RVC) bir masaüstü uygulamasıdır. Ekrandan OCR ile alınan "Karakter: Diyalog" formatındaki metinler, seçilen referans sese ve RVC modeline göre hoparlörden oynatılır.

---

## Proje İsmi ve Açılımı

| Harf | Açılım |
|------|--------|
| **U** | Unified |
| **M** | Model-based |
| **A** | Audio |
| **Y** | Yield |

**Türkçe:** *Birleşik Model Tabanlı Ses Üretimi*

---

## Özellikler

### OCR (Ekran Okuma)
- **Bölge seçimi:** Ekrandan sadece ilgili alt yazı alanını seçin; gereksiz bölgeler taranmaz.
- **Anti-spam filtresi:** %90 benzerlik (difflib) ile tekrar eden alt yazılar engellenir.
- **Ön işleme:** Grayscale, kontrast artırma ve sharpening ile OCR doğruluğu yükseltilir.
- **Ayarlanabilir aralık:** 0.3 saniye ve üzeri tarama aralığı.
- **Türkçe desteği:** Tesseract Türkçe (`tur`) dil paketi ile karakter/diyalog ayrıştırma (regex: `Karakter: Diyalog`).

### TTS (Metin-Ses Dönüşümü)
- **Coqui XTTS-v2:** Çok dilli, yüksek kaliteli ses sentezi.
- **Duygu bazlı referanslar:** Karakter başına birden fazla referans WAV (nötr, mutlu, kızgın vb.) tanımlanabilir.
- **Hugging Face entegrasyonu:** TTS modelleri uygulama içinden indirilebilir.
- **Ses hızı ayarı:** Konuşma temposu ayarlanabilir.

### RVC (Retrieval-based Voice Conversion)
- **Karakter–model eşleştirme:** Her oyun karakteri için ayrı RVC modeli atanabilir.
- **Çoklu model önbelleği:** Tüm modeller VRAM'de tutulur; geçişler hızlıdır.
- **Realtime modu:** Düşük gecikme için PM F0 metodu otomatik seçilebilir.
- **Ayarlanabilir parametreler:** Pitch, index_rate, protect, f0_method vb.

### İki Aşamalı Pipeline
- **Prefetch mimarisi:** Çalan cümlenin yanında bir sonraki cümle arka planda işlenir.
- **Metin ve ses kuyruğu:** `_text_queue` + `_audio_queue` ile akıcı oynatma.

### Oyun Profilleri (Preset)
- **Bölge + karakter ayarları:** Bölge koordinatları ve karakter–model eşleştirmeleri profil olarak kaydedilir.
- **Hızlı geçiş:** Oyun değişiminde tek tıkla profil yükleme.

### Opsiyonel Modüller
- **Auto-translate (CEV):** Opus-MT EN→TR çeviri; İngilizce alt yazıları Türkçeye çevirir.
- **Duygu analizi:** Rolling context ile cümle duygusu belirlenir; TTS speed ve RVC pitch buna göre ayarlanabilir.
- **Audio ducking:** Oyun sesini konuşma sırasında kısma (pycaw, Windows).
- **Modül açma/kapatma:** CEV, DUYGU, RVC, DUCK bağımsız olarak açılıp kapatılabilir; kapatıldığında modeller VRAM'den boşaltılır.

---

## Teknoloji Yığını

| Bileşen | Kütüphane |
|---------|-----------|
| UI | CustomTkinter |
| Ekran yakalama | mss |
| OCR | pytesseract (Tesseract) |
| TTS | Coqui TTS (XTTS-v2) |
| Ses dönüşümü | rvc-python |
| Çalma | sounddevice |
| Çeviri | transformers (Opus-MT) |
| Duygu analizi | j-hartmann/emotion |
| Ses kısma | pycaw (Windows) |
| Model indirme | huggingface_hub |

---

## Proje Yapısı

```
U.M.A.Y/
├── main.py              # Giriş noktası
├── config.json          # Kullanıcı ayarları (gitignore'da)
├── config.example.json  # Örnek yapılandırma
├── requirements.txt     # Python bağımlılıkları
├── README.md
├── models/              # RVC modelleri ve TTS referansları
│   ├── refs/            # XTTS karakter referans WAV'ları
│   └── tts/             # Hugging Face'den indirilen TTS modelleri
├── output/              # Geçici ses çıktıları
└── src/
    ├── ocr/
    │   ├── capture.py       # Ekran yakalama + OCR
    │   └── screen_capture.py
    ├── tts/
    │   ├── generator.py    # TTS üretimi
    │   └── tts_engine.py
    ├── rvc/
    │   ├── converter.py    # RVC ses dönüşümü
    │   └── voice_converter.py
    ├── translate/
    │   └── translator.py  # Opus-MT çeviri
    ├── llm/
    │   └── analyzer.py    # Duygu analizi
    ├── audio/
    │   └── ducking.py     # Ses kısma
    ├── presets/
    │   └── manager.py     # Profil yönetimi
    ├── pipeline/
    │   └── queue_runner.py # İki aşamalı pipeline
    └── ui/
        ├── app.py            # Ana pencere
        ├── settings_panel.py
        ├── preset_panel.py
        ├── region_selector.py
        ├── char_refs_dialog.py
        ├── hf_downloader.py
        └── model_manager.py
```

---

## Kurulum

### Gereksinimler

- **Python 3.10+**
- **Tesseract OCR** (ayrı kurulum gerekir)
- **CUDA** (RVC için önerilir; CPU da çalışır)

### 1. Depoyu klonlayın

```bash
git clone https://github.com/crowroser/U.M.A.Y..git umay
cd umay
```

### 2. Sanal ortam oluşturun (önerilir)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### 4. Tesseract OCR kurulumu

**Windows:**
- [Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki) indirip kurun.
- Kurulumda **Türkçe** (`tur`) dil paketini seçin.
- Varsayılan yol: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Farklı bir yola kurduysanız `config.json` içinde `ocr.tesseract_path` ile belirtin.

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-tur
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 5. Yapılandırma

`config.example.json` dosyasını kopyalayıp `config.json` olarak kaydedin:

```bash
copy config.example.json config.json   # Windows
# cp config.example.json config.json   # Linux/Mac
```

`config.json` içinde şu ayarları kontrol edin:
- `ocr.tesseract_path`: Tesseract executable yolu
- `tts.model`: XTTS-v2 (varsayılan)
- `rvc.model_path` / `rvc.index_path`: RVC model ve index dosyaları

### 6. İlk çalıştırma

```bash
python main.py
```

İlk açılışta **XTTS-v2** modeli (~1.8 GB) Hugging Face'den otomatik indirilir.

### 7. RVC modelleri (opsiyonel)

`.pth` ve `.index` dosyalarınızı `models/` klasörüne koyun. Uygulama içindeki **Karakterler** panelinden karakter–model eşleştirmesi yapabilirsiniz.

---

## Kullanım

1. **Bölge seç:** Ekrandan alt yazı alanını seçin (ör. oyun penceresinin alt kısmı).
2. **Karakter referansı:** TTS için referans WAV yükleyin; her karakter için farklı ses kullanılabilir.
3. **RVC model:** Karakter başına RVC modeli atayın (opsiyonel).
4. **Pipeline'ı başlat:** "Başlat" ile OCR taraması ve ses üretimi başlar.
5. **Profil kaydet:** Oyun + bölge + karakter ayarlarını preset olarak kaydedip sonra hızlıca yükleyebilirsiniz.

---

## Klavye Kısayolları

| Tuş | İşlev |
|-----|-------|
| `F9` | Pipeline başlat/durdur |
| `F10` | Bölge seçim modu |

---

umayyazilim.com

