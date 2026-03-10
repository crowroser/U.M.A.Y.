"""
src/audio/ducking.py
Windows Volume Mixer (pycaw) tabanli Audio Ducking.

Karakter konusmaya basladiginda hedef surecin (oyunun) ses seviyesi
duck_level'a dusurulur; ses bittikten sonra eski seviyeye geri alinir.
Hedef surec bulunamazsa veya pycaw yuklu degilse sessizce atlar.
"""

from __future__ import annotations

import os
from typing import Optional


class AudioDucker:
    """
    Windows ses mikseri uzerinden belirli bir surec icin ses dusurme/geri alma.
    enabled=False veya pycaw yuklu degilse tum operasyonlar no-op'tur.
    """

    def __init__(
        self,
        enabled: bool = False,
        duck_level: float = 0.35,
        target_process: str = "",
    ):
        self.enabled = enabled
        self.duck_level = max(0.0, min(1.0, duck_level))
        self.target_process = target_process.strip().lower()
        self._original_levels: dict[object, float] = {}
        self._sessions: list = []
        self._pycaw_available = self._check_pycaw()

    @staticmethod
    def _check_pycaw() -> bool:
        try:
            import pycaw  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_matching_sessions(self) -> list:
        """Hedef surec adiyla eslesen ses oturumlarini doner."""
        if not self._pycaw_available:
            return []
        try:
            from pycaw.pycaw import AudioUtilities
            sessions = AudioUtilities.GetAllSessions()
            if not self.target_process:
                return list(sessions)
            matched = []
            for s in sessions:
                if s.Process and self.target_process in s.Process.name().lower():
                    matched.append(s)
            return matched
        except Exception:
            return []

    def duck(self, level: Optional[float] = None):
        """
        Hedef surecin ses seviyesini duck_level'a dusurir.
        Mevcut seviye geri donus icin saklanir.
        """
        if not self.enabled or not self._pycaw_available:
            return

        target_level = level if level is not None else self.duck_level
        sessions = self._get_matching_sessions()
        self._original_levels.clear()
        self._sessions = sessions

        for session in sessions:
            try:
                volume = session.SimpleAudioVolume
                if volume:
                    current = volume.GetMasterVolume()
                    self._original_levels[id(session)] = current
                    volume.SetMasterVolume(target_level, None)
            except Exception:
                pass

    def restore(self):
        """Tum oturumlarin ses seviyesini orijinal degerine geri alir."""
        if not self.enabled or not self._pycaw_available:
            return

        for session in self._sessions:
            try:
                volume = session.SimpleAudioVolume
                if volume:
                    original = self._original_levels.get(id(session), 1.0)
                    volume.SetMasterVolume(original, None)
            except Exception:
                pass

        self._original_levels.clear()
        self._sessions = []

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        duck_level: Optional[float] = None,
        target_process: Optional[str] = None,
    ):
        if enabled is not None:
            self.enabled = enabled
        if duck_level is not None:
            self.duck_level = max(0.0, min(1.0, duck_level))
        if target_process is not None:
            self.target_process = target_process.strip().lower()
