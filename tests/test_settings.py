from homeshield.db import init_db
from homeshield.settings import DEFAULTS, SettingsStore


def test_defaults_and_persistence(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    s = SettingsStore(db)
    assert s.get("fire_confirm_frames") == DEFAULTS["fire_confirm_frames"]
    s.update({"yolo_confidence": 0.42, "fire_classes": "fire"})
    again = SettingsStore(db)
    assert again.get("yolo_confidence") == 0.42
    assert again.get("fire_classes") == "fire"


def test_legacy_process_fps_is_migrated(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    SettingsStore(db).update({"process_fps": 15})
    assert SettingsStore(db).get("process_fps") == 0
