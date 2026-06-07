"""
src/pipeline/character_db.py
Gelişmiş Karakter Veritabanı modülü.
Karakter profillerini, regex kurallarını, istatistikleri ve konuşma geçmişini yönetir.
Ayrıca karakter paketlerinin ZIP formatında import/export edilmesini sağlar.
"""

import os
import re
import json
import time
import zipfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("UMAY.CharacterDatabase")

CHARACTERS_DB_PATH = Path(__file__).parent.parent.parent / "characters_db.json"
DEFAULT_AVATAR = "assets/avatars/default.png"


class CharacterDatabase:
    """
    Karakter profillerini yöneten, regex tabanlı tespit yapan
    ve ZIP paket desteği (RVC + TTS + Avatar) sunan veritabanı sınıfı.
    """

    _instance: Optional["CharacterDatabase"] = None

    @classmethod
    def get_instance(cls) -> "CharacterDatabase":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.characters: Dict[str, Dict[str, Any]] = {}
        self.db_path = CHARACTERS_DB_PATH
        self.load()

    def load(self):
        """Karakter veritabanını JSON dosyasından yükler."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.characters = json.load(f)
                logger.info(f"Karakter veritabanı yüklendi. Toplam karakter: {len(self.characters)}")
            except Exception as e:
                logger.error(f"Karakter veritabanı yüklenemedi: {e}")
                self.characters = {}
        else:
            self.characters = {}

    def save(self):
        """Karakter veritabanını JSON dosyasına kaydeder."""
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.characters, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Karakter veritabanı kaydedilemedi: {e}")

    def get_character(self, name: str) -> Optional[Dict[str, Any]]:
        """İsme göre karakter detaylarını döner."""
        key = name.strip().lower()
        # Case-insensitive eşleme
        for k, v in self.characters.items():
            if k.lower() == key:
                return v
        return None

    def add_character(
        self,
        name: str,
        rvc_model: Optional[str] = None,
        rvc_index: Optional[str] = None,
        color: str = "#ffffff",
        avatar: str = DEFAULT_AVATAR,
        regex_patterns: Optional[List[str]] = None,
        tts_refs: Optional[Dict[str, str]] = None,
    ):
        """Yeni karakter ekler veya mevcut olanı günceller."""
        if not name:
            return

        # Eğer regex_patterns boşsa varsayılan olarak isim tabanlı desenler oluştur
        if not regex_patterns:
            regex_patterns = [
                rf"^{re.escape(name)}:\s*(.*)",
                rf"^\[{re.escape(name)}\]\s*(.*)",
                rf"^\({re.escape(name)}\)\s*(.*)",
            ]

        # Konuşma geçmişi ve istatistikleri koru
        existing = self.get_character(name)
        stats = existing.get("stats", {"lines_spoken": 0, "total_latency": 0.0, "cache_hits": 0}) if existing else {
            "lines_spoken": 0,
            "total_latency": 0.0,
            "cache_hits": 0
        }
        history = existing.get("history", []) if existing else []

        self.characters[name] = {
            "name": name,
            "rvc_model": rvc_model,
            "rvc_index": rvc_index,
            "color": color,
            "avatar": avatar,
            "regex_patterns": regex_patterns,
            "tts_refs": tts_refs or {},
            "stats": stats,
            "history": history,
        }
        self.save()
        logger.info(f"Karakter eklendi/güncellendi: {name}")

    def remove_character(self, name: str):
        """Karakter veritabanından bir karakteri siler."""
        # Case-insensitive silme
        key = name.strip().lower()
        to_delete = None
        for k in self.characters.keys():
            if k.lower() == key:
                to_delete = k
                break
        if to_delete:
            del self.characters[to_delete]
            self.save()
            logger.info(f"Karakter silindi: {to_delete}")

    def list_characters(self) -> List[Dict[str, Any]]:
        """Veritabanındaki tüm karakterlerin listesini döner."""
        return list(self.characters.values())

    def detect_character(self, text: str) -> Tuple[Optional[str], str]:
        """
        OCR metninden regex yardımıyla konuşan karakteri ve temiz altyazı metnini tespit eder.
        Eğer eşleşen karakter yoksa (None, temizlenmemiş_metin) döner.
        """
        for char_name, info in self.characters.items():
            patterns = info.get("regex_patterns", [])
            for pattern in patterns:
                try:
                    match = re.match(pattern, text, re.IGNORECASE)
                    if match:
                        cleaned_text = match.group(1).strip()
                        return char_name, cleaned_text
                except Exception as e:
                    logger.error(f"Regex eşleşme hatası ({pattern}): {e}")

        # Eğer hiçbir regex eşleşmezse ve metinde ':' varsa sol tarafı karakter ismi olarak deneyebiliriz
        if ":" in text:
            parts = text.split(":", 1)
            left = parts[0].strip()
            right = parts[1].strip()
            # Eğer sol taraf veritabanındaki bir karakterle tam eşleşiyorsa
            char_info = self.get_character(left)
            if char_info:
                return char_info["name"], right

        return None, text

    def record_line(self, name: str, text: str, latency: float = 0.0, cache_hit: bool = False):
        """Karakterin konuştuğu satırı geçmişe ekler ve istatistiklerini günceller."""
        char_info = self.get_character(name)
        if not char_info:
            return

        # İstatistikleri güncelle
        stats = char_info.setdefault("stats", {"lines_spoken": 0, "total_latency": 0.0, "cache_hits": 0})
        stats["lines_spoken"] = stats.get("lines_spoken", 0) + 1
        stats["total_latency"] = stats.get("total_latency", 0.0) + latency
        if cache_hit:
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1

        # Geçmişe ekle (maksimum son 20 satır)
        history = char_info.setdefault("history", [])
        history.append({
            "timestamp": time.time(),
            "text": text,
            "latency": latency,
            "cache_hit": cache_hit
        })
        if len(history) > 20:
            history.pop(0)

        self.save()

    def export_character_pack(self, name: str, export_zip_path: str) -> bool:
        """
        Karakter verilerini, model pth/index dosyalarını ve TTS ses referanslarını
        ZIP paketi olarak ihraç eder.
        """
        char_info = self.get_character(name)
        if not char_info:
            logger.error(f"İhraç edilecek karakter bulunamadı: {name}")
            return False

        try:
            export_path = Path(export_zip_path)
            temp_dir = export_path.parent / f"_temp_export_{name}"
            temp_dir.mkdir(exist_ok=True)

            # Metadata JSON'u hazırla
            meta = {
                "name": char_info["name"],
                "color": char_info["color"],
                "regex_patterns": char_info["regex_patterns"],
                "rvc_model_filename": None,
                "rvc_index_filename": None,
                "avatar_filename": None,
                "tts_refs": {}
            }

            # RVC Model dosyasını kopyala
            rvc_model = char_info.get("rvc_model")
            if rvc_model and os.path.isfile(rvc_model):
                shutil.copy(rvc_model, temp_dir)
                meta["rvc_model_filename"] = Path(rvc_model).name

            # RVC Index dosyasını kopyala
            rvc_index = char_info.get("rvc_index")
            if rvc_index and os.path.isfile(rvc_index):
                shutil.copy(rvc_index, temp_dir)
                meta["rvc_index_filename"] = Path(rvc_index).name

            # Avatar görselini kopyala
            avatar = char_info.get("avatar")
            if avatar and os.path.isfile(avatar) and avatar != DEFAULT_AVATAR:
                shutil.copy(avatar, temp_dir)
                meta["avatar_filename"] = Path(avatar).name

            # TTS referans ses dosyalarını kopyala
            for emotion, wav_path in char_info.get("tts_refs", {}).items():
                if wav_path and os.path.isfile(wav_path):
                    shutil.copy(wav_path, temp_dir)
                    meta["tts_refs"][emotion] = Path(wav_path).name

            # Metadata JSON dosyasını yaz
            with open(temp_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            # ZIP paketini oluştur
            with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as z:
                for file in temp_dir.iterdir():
                    z.write(file, arcname=file.name)

            # Geçici klasörü temizle
            shutil.rmtree(temp_dir)
            logger.info(f"Karakter paketi başarıyla ihraç edildi: {export_path.name}")
            return True

        except Exception as e:
            logger.error(f"Karakter ihraç etme hatası: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return False

    def import_character_pack(self, import_zip_path: str, extract_base_dir: Optional[str] = None) -> Optional[str]:
        """
        ZIP paketini çözer, model/ses dosyalarını ilgili klasörlere taşır
        ve karakteri veritabanına ekler. Karakter ismini döner.
        """
        if not os.path.isfile(import_zip_path):
            logger.error(f"İthal edilecek ZIP dosyası bulunamadı: {import_zip_path}")
            return None

        try:
            zip_path = Path(import_zip_path)
            base_dir = Path(extract_base_dir) if extract_base_dir else Path(__file__).parent.parent.parent
            
            # Hedef klasörleri belirle
            models_dir = base_dir / "models"
            tts_ref_dir = base_dir / "assets" / "references"
            avatar_dir = base_dir / "assets" / "avatars"

            models_dir.mkdir(parents=True, exist_ok=True)
            tts_ref_dir.mkdir(parents=True, exist_ok=True)
            avatar_dir.mkdir(parents=True, exist_ok=True)

            temp_dir = zip_path.parent / f"_temp_import_{zip_path.stem}"
            temp_dir.mkdir(exist_ok=True)

            # ZIP dosyasını geçici klasöre çıkar
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)

            meta_file = temp_dir / "meta.json"
            if not meta_file.exists():
                raise ValueError("Geçersiz paket: meta.json bulunamadı.")

            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            char_name = meta.get("name")
            if not char_name:
                raise ValueError("Geçersiz paket: Karakter ismi meta.json içinde bulunamadı.")

            # RVC model dosyasını kopyala
            rvc_model_filename = meta.get("rvc_model_filename")
            dest_model_path = None
            if rvc_model_filename:
                src_model = temp_dir / rvc_model_filename
                if src_model.exists():
                    dest_model_path = models_dir / rvc_model_filename
                    shutil.copy(src_model, dest_model_path)

            # RVC index dosyasını kopyala
            rvc_index_filename = meta.get("rvc_index_filename")
            dest_index_path = None
            if rvc_index_filename:
                src_index = temp_dir / rvc_index_filename
                if src_index.exists():
                    dest_index_path = models_dir / rvc_index_filename
                    shutil.copy(src_index, dest_index_path)

            # Avatar dosyasını kopyala
            avatar_filename = meta.get("avatar_filename")
            dest_avatar_path = DEFAULT_AVATAR
            if avatar_filename:
                src_avatar = temp_dir / avatar_filename
                if src_avatar.exists():
                    dest_avatar_path = avatar_dir / avatar_filename
                    shutil.copy(src_avatar, dest_avatar_path)
                    dest_avatar_path = str(dest_avatar_path)

            # TTS referanslarını kopyala
            dest_tts_refs = {}
            for emotion, filename in meta.get("tts_refs", {}).items():
                src_wav = temp_dir / filename
                if src_wav.exists():
                    dest_wav_path = tts_ref_dir / f"{char_name.lower()}_{filename}"
                    shutil.copy(src_wav, dest_wav_path)
                    dest_tts_refs[emotion] = str(dest_wav_path)

            # Karakteri DB'ye ekle
            self.add_character(
                name=char_name,
                rvc_model=str(dest_model_path) if dest_model_path else None,
                rvc_index=str(dest_index_path) if dest_index_path else None,
                color=meta.get("color", "#ffffff"),
                avatar=dest_avatar_path,
                regex_patterns=meta.get("regex_patterns"),
                tts_refs=dest_tts_refs
            )

            # Temizlik
            shutil.rmtree(temp_dir)
            logger.info(f"Karakter paketi başarıyla içe aktarıldı: {char_name}")
            return char_name

        except Exception as e:
            logger.error(f"Karakter içe aktarma hatası: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            return None
