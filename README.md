# U.M.A.Y — Unified Model-based Audio Yield

**U.M.A.Y** is a real-time desktop application that reads on-screen subtitles from games or applications via OCR, converts the text to speech (TTS), and then transforms the voice using an RVC (Retrieval-based Voice Conversion) model. Subtitles captured from the screen in `Character: Dialogue` format are synthesized with the selected reference audio and played through the speakers.

---

## What does U.M.A.Y stand for?

| Letter | Meaning |
|--------|---------|
| **U** | Unified |
| **M** | Model-based |
| **A** | Audio |
| **Y** | Yield |

---

## Features

### OCR (Screen Reading)
- **Region selection:** Select only the subtitle area on screen — no unnecessary scanning.
- **Anti-spam filter:** Repeated subtitles blocked via 90% similarity threshold (difflib).
- **Pre-processing:** Grayscale, contrast enhancement and sharpening for higher OCR accuracy.
- **Adjustable interval:** Scan interval of 0.3 seconds or more.
- **Multi-language support:** Tesseract with `tur`/`eng` language packs; parses `Character: Dialogue` format via regex.

### TTS (Text-to-Speech)
- **Coqui XTTS-v2:** Multilingual, high-quality speech synthesis.
- **Emotion-based references:** Multiple reference WAVs per character (neutral, happy, angry, etc.).
- **Hugging Face integration:** TTS models downloadable from within the app.
- **Speed control:** Adjustable speech tempo.

### RVC (Voice Conversion)
- **Character–model mapping:** Assign a different RVC model to each game character.
- **Multi-model cache:** All models stay loaded in VRAM; switching between characters has zero loading delay.
- **Auto version detection:** Automatically detects RVC v1 (256-dim) vs v2 (768-dim) models from the checkpoint.
- **Realtime mode:** PM F0 method auto-selected for low-latency (~0.3s vs ~1-2s with rmvpe).
- **Adjustable parameters:** Pitch, index_rate, protect, f0_method, etc.

### Two-Stage Pipeline
- **Prefetch architecture:** While the current sentence plays, the next sentence is processed in the background.
- **Text and audio queues:** `_text_queue` + `_audio_queue` for smooth, uninterrupted playback.

### Game Profiles (Presets)
- **Region + character settings:** Region coordinates and character–model mappings saved as profiles.
- **Quick switch:** Load a profile in one click when switching games.

### Optional Modules
- **Auto-translate:** Helsinki-NLP Opus-MT EN→TR translation; converts English subtitles to Turkish in real time.
- **Emotion analysis:** Sentence emotion detected with rolling context; TTS speed and RVC pitch adjusted accordingly.
- **Audio ducking:** Lowers game audio while speech is playing (pycaw, Windows only).
- **Module toggle:** CEV, EMOTION, RVC, DUCK can each be independently enabled/disabled; disabled modules are unloaded from VRAM.

---

## Technology Stack

| Component | Library |
|-----------|---------|
| UI | CustomTkinter |
| Screen capture | mss |
| OCR | pytesseract (Tesseract) |
| TTS | Coqui TTS (XTTS-v2) |
| Voice conversion | rvc-python |
| Playback | sounddevice |
| Translation | transformers (Opus-MT) |
| Emotion analysis | j-hartmann/emotion-english-distilroberta-base |
| Audio ducking | pycaw (Windows) |
| Model download | huggingface_hub |

---

## Project Structure

```
umay/
├── main.py                  # Entry point
├── config.json              # User settings (gitignored)
├── config.example.json      # Example configuration
├── requirements.txt         # Python dependencies
├── README.md
├── models/                  # RVC models and TTS references
│   ├── refs/                # XTTS character reference WAVs
│   └── tts/                 # HuggingFace-downloaded TTS models
├── output/                  # Temporary audio outputs
└── src/
    ├── ocr/
    │   ├── capture.py           # Screen capture + OCR
    │   └── screen_capture.py
    ├── tts/
    │   └── generator.py         # TTS synthesis engine
    ├── rvc/
    │   ├── converter.py         # RVC voice conversion
    │   └── voice_converter.py
    ├── translate/
    │   └── translator.py        # Opus-MT translation
    ├── llm/
    │   └── analyzer.py          # Emotion analysis
    ├── audio/
    │   └── ducking.py           # Audio ducking
    ├── presets/
    │   └── manager.py           # Profile management
    ├── pipeline/
    │   └── queue_runner.py      # Two-stage pipeline
    ├── utils/
    │   └── download_progress.py # HuggingFace download progress
    └── ui/
        ├── app.py               # Main window
        ├── settings_panel.py
        ├── preset_panel.py
        ├── region_selector.py
        ├── char_refs_dialog.py
        ├── hf_downloader.py
        └── model_manager.py
```

---

## Installation

### Requirements

- **Python 3.10**
- **Tesseract OCR** (separate installation required)
- **NVIDIA GPU with CUDA** (strongly recommended for TTS + RVC)
  - RTX 5000 series (Blackwell / sm_120): requires PyTorch with CUDA 12.8
  - RTX 3000/4000 series (Ampere/Ada): CUDA 12.6 is sufficient

### 1. Clone the repository

```bash
git clone https://github.com/crowroser/U.M.A.Y..git umay
cd umay
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

**For most GPUs (RTX 3000 / 4000 series, CUDA 12.6):**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**For RTX 5000 series / Blackwell GPUs (sm_120, CUDA 12.8):**
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps --no-cache-dir
```

> **Note:** After installing, you may also need to reinstall binary-compiled packages to match the numpy 2.x ABI:
> ```bash
> pip install pandas scipy faiss-cpu --upgrade --force-reinstall --no-deps
> ```

### 4. Install Tesseract OCR

**Windows:**
- Download from [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and install.
- During setup, select the **Turkish** (`tur`) and/or **English** (`eng`) language packs.
- Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- If installed elsewhere, set `ocr.tesseract_path` in `config.json`.

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 5. Configuration

Copy `config.example.json` to `config.json`:

```bash
copy config.example.json config.json   # Windows
# cp config.example.json config.json   # Linux/Mac
```

Key settings to review in `config.json`:
- `ocr.tesseract_path`: Path to Tesseract executable
- `tts.model`: TTS model name (default: `tts_models/multilingual/multi-dataset/xtts_v2`)
- `rvc.model_path` / `rvc.index_path`: Your RVC `.pth` and `.index` files
- `translate.enabled`: Set to `true` to enable EN→TR auto-translation

### 6. First run

```bash
python main.py
```

On first launch, the **XTTS-v2** model (~1.8 GB) is automatically downloaded from Hugging Face.

### 7. RVC models (optional)

Place your `.pth` and `.index` files in the `models/` folder. Use the **Characters** panel inside the app to map character names to their RVC models.

---

## Usage

1. **Select region:** Draw a rectangle over the subtitle area on screen (e.g. bottom of the game window).
2. **Character references:** Load reference WAV files for TTS; different voices can be used per character.
3. **RVC model:** Assign an RVC model per character (optional).
4. **Start pipeline:** Click **Start** — OCR scanning and audio synthesis begin.
5. **Save preset:** Save region + character settings as a game profile for quick recall.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F9` | Start / stop pipeline |
| `F10` | Open region selection mode |

---

## Known Compatibility Notes

### PyTorch 2.6+ and `torch.load`
PyTorch 2.6 changed the default of `weights_only` from `False` to `True` in `torch.load`. This breaks loading of Coqui TTS models which contain custom classes. The application automatically patches `torch.load` before loading TTS to restore the previous behavior safely.

### transformers 5.x and TTS
Coqui TTS 0.22.0 uses `BeamSearchScorer` from the `transformers` library, which was removed in version 5.0. This project pins `transformers==4.46.3` for compatibility, with the `huggingface_hub` version check bypassed since 1.x works at runtime.

### numpy 2.x binary compatibility
numpy 2.0 changed the internal `dtype` struct size (88 → 96 bytes). Packages compiled against numpy 1.x will crash on import with `numpy.dtype size changed`. Required versions with numpy 2.x support:
- `pandas >= 2.3`
- `scipy >= 1.15`
- `faiss-cpu >= 1.13`

### RVC model version detection
`rvc-python` defaults to v2 architecture but some models are v1. U.M.A.Y automatically detects the model version by inspecting `enc_p.emb_phone.weight` shape before loading:
- `[192, 256]` → v1
- `[192, 768]` → v2

---

## License

MIT

---

umayyazilim.com
