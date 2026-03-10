"""
src/presets/manager.py
Oyun bazli profil (preset) yonetimi.

Preset yapisi:
{
    "region":     [left, top, width, height] | null,
    "characters": {"KarakterAdi": "model.pth", ...},
    "ocr":        {"interval": 0.4, "language": "tur"},
    "tts":        {"language": "tr", "speed": 1.0},
    "translate":  {"enabled": false, "source_lang": "eng"}
}

Tum presetler config["presets"] sozlugunde saklanir.
"""

from __future__ import annotations

from typing import Callable, Optional


class PresetManager:
    """
    config dict'i uzerinde calisan CRUD yoneticisi.
    Degisiklikler aninda save_fn() ile kalici hale getirilir.
    """

    def __init__(self, config: dict, save_fn: Callable[[dict], None]):
        self._config = config
        self._save = save_fn
        self._config.setdefault("presets", {})
        self._config.setdefault("active_preset", None)

    def list_names(self) -> list[str]:
        return sorted(self._config["presets"].keys())

    def load(self, name: str) -> Optional[dict]:
        return self._config["presets"].get(name)

    def save(self, name: str, data: dict):
        self._config["presets"][name] = data
        self._config["active_preset"] = name
        self._save(self._config)

    def delete(self, name: str):
        self._config["presets"].pop(name, None)
        if self._config.get("active_preset") == name:
            self._config["active_preset"] = None
        self._save(self._config)

    def rename(self, old_name: str, new_name: str):
        if old_name not in self._config["presets"]:
            return
        self._config["presets"][new_name] = self._config["presets"].pop(old_name)
        if self._config.get("active_preset") == old_name:
            self._config["active_preset"] = new_name
        self._save(self._config)

    def set_active(self, name: Optional[str]):
        self._config["active_preset"] = name
        self._save(self._config)

    def snapshot_from_config(self, config: dict) -> dict:
        """Mevcut config'ten bir preset anlık goruntusü cikarir."""
        return {
            "region": config.get("ocr", {}).get("region"),
            "characters": dict(config.get("characters", {})),
            "ocr": {
                "interval": config.get("ocr", {}).get("interval", 0.4),
                "language": config.get("ocr", {}).get("language", "tur"),
            },
            "tts": {
                "language": config.get("tts", {}).get("language", "tr"),
                "speed": config.get("tts", {}).get("speed", 1.0),
            },
            "translate": {
                "enabled": config.get("translate", {}).get("enabled", False),
                "source_lang": config.get("translate", {}).get("source_lang", "eng"),
            },
        }
