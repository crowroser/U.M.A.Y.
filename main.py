"""
U.M.A.Y - Unified Model-based Audio Yield
Giris noktasi: config.json yukler, UMAYApp'i baslatir.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"

sys.path.insert(0, str(BASE_DIR))

DEFAULT_CONFIG = {
    "ocr": {
        "region": None,
        "interval": 1.5,
        "language": "tur",
        "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    },
    "tts": {
        "model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "language": "tr",
        "speed": 1.0,
        "speaker_wav": None,
    },
    "rvc": {
        "model_path": None,
        "index_path": None,
        "pitch": 0,
        "filter_radius": 3,
        "index_rate": 0.75,
        "rms_mix_rate": 0.25,
        "protect": 0.33,
        "f0_method": "rmvpe",
        "realtime_mode": True,
    },
    "translate": {
        "enabled": False,
        "source_lang": "eng",
        "model": "Helsinki-NLP/opus-mt-tc-big-en-tr",
    },
    "sentiment": {
        "enabled": False,
        "model": "j-hartmann/emotion-english-distilroberta-base",
        "context_window": 5,
    },
    "ducking": {
        "enabled": False,
        "level": 0.35,
        "target_process": "",
    },
    "audio": {
        "output_device": None,
        "sample_rate": 40000,
    },
    "ui": {
        "theme": "dark",
        "color_theme": "blue",
        "window_width": 1100,
        "window_height": 720,
    },
    "pipeline": {
        "enabled": False,
        "last_subtitle": "",
    },
    "characters": {},
    "character_refs": {},
    "presets": {},
    "active_preset": None,
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = _deep_merge(DEFAULT_CONFIG, data)
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except OSError as e:
        print(f"Config kaydedilemedi: {e}")


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main():
    config = load_config()

    from src.ui.app import UMAYApp

    app = UMAYApp(config=config, save_config_fn=save_config)
    app.mainloop()


if __name__ == "__main__":
    main()
