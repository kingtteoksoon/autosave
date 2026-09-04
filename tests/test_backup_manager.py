"""backup_manager 핵심 로직(명명 규칙·저장번호 영속·보관 정책) 단위 테스트."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosave_program"))

from backup_manager import (  # noqa: E402
    BackupManager,
    Config,
    StateStore,
    sanitize_component,
    split_name,
)

BACKUP_NAME_RE = re.compile(r"^(?P<stem>.+)_(?P<num>\d{3,})_(?P<ts>\d{8}_\d{6})(?P<ext>\..+)$")


class BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.watched = self.root / "watched"
        self.backups = self.root / "backups"
        self.watched.mkdir()
        self.addCleanup(self._tmp.cleanup)

    def make_config(self, **overrides) -> Config:
        raw = {
            "watch_paths": [str(self.watched)],
            "backup_dir": str(self.backups),
            "interval_seconds": 900,
            "max_versions": 20,
            "verify_hash": True,
            "stabilize_delay_seconds": 0,  # 테스트에서는 안정화 대기 생략
            "stabilize_retries": 0,
        }
        raw.update(overrides)
        return Config.from_mapping(raw, self.root)

    def make_manager(self, config: Config = None, dry_run: bool = False):
        config = config or self.make_config()
        store = StateStore(self.root / "state.json")
        store.load()
        return BackupManager(config, store, dry_run=dry_run), store

    def write(self, relative: str, content: str, mtime: float = None) -> Path:
        path = self.watched / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def backup_names(self, folder: str):
        directory = self.backups / folder
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


class SplitNameTests(unittest.TestCase):
    def test_single_extension(self):
        self.assertEqual(split_name("poster.psd"), ("poster", ".psd"))

    def test_no_extension(self):
        self.assertEqual(split_name("Makefile"), ("Makefile", ""))

    def test_dotted_name_uses_last_extension(self):
        self.assertEqual(split_name("scene.v2.blend"), ("scene.v2", ".blend"))

    def test_compound_extension_exception_list(self):
        self.assertEqual(split_name("release.tar.gz", (".tar.gz",)), ("release", ".tar.gz"))

    def test_sanitize_component(self):
        self.assertEqual(sanitize_component('a/b:c*.psd'), "a_b_c_.psd")


class NamingAndCounterTests(BaseCase):
    def test_first_scan_creates_version_001(self):
        self.write("poster.psd", "v1")
        manager, _ = self.make_manager()
        stats = manager.scan_once()

        self.assertEqual(stats.backed_up, 1)
        names = self.backup_names("poster.psd")
        self.assertEqual(len(names), 1)
        match = BACKUP_NAME_RE.match(names[0])
        self.assertIsNotNone(match)
        self.assertEqual(match.group("stem"), "poster")
        self.assertEqual(match.group("num"), "001")
        self.assertEqual(match.group("ext"), ".psd")

    def test_unchanged_file_is_not_backed_up_again(self):
        self.write("main.py", "print(1)")
        manager, _ = self.make_manager()
        manager.scan_once()
        stats = manager.scan_once()

        self.assertEqual(stats.backed_up, 0)
        self.assertEqual(stats.skipped_unchanged, 1)
        self.assertEqual(len(self.backup_names("main.py")), 1)

    def test_counter_increments_on_each_save(self):
        path = self.write("main.py", "v1", mtime=1_700_000_000)
        manager, _ = self.make_manager()
        manager.scan_once()
        for index in range(2, 5):
            path.write_text(f"v{index}", encoding="utf-8")
            os.utime(path, (1_700_000_000 + index * 60, 1_700_000_000 + index * 60))
            manager.scan_once()

        numbers = [BACKUP_NAME_RE.match(name).group("num") for name in self.backup_names("main.py")]
        self.assertEqual(numbers, ["001", "002", "003", "004"])

    def test_touch_without_content_change_does_not_consume_number(self):
        path = self.write("character.blend", "mesh", mtime=1_700_000_000)
        manager, _ = self.make_manager()
        manager.scan_once()
        os.utime(path, (1_700_000_500, 1_700_000_500))
        stats = manager.scan_once()

        self.assertEqual(stats.backed_up, 0)
        self.assertEqual(len(self.backup_names("character.blend")), 1)

    def test_counter_survives_restart_via_state_file(self):
        path = self.write("poster.psd", "v1", mtime=1_700_000_000)
        manager, store = self.make_manager()
        manager.scan_once()
        store.flush(force=True)

        path.write_text("v2", encoding="utf-8")
        os.utime(path, (1_700_000_600, 1_700_000_600))

        # 새 프로세스를 모사: 동일한 state.json 을 다시 로드
        manager2, _ = self.make_manager()
        manager2.scan_once()

        numbers = [BACKUP_NAME_RE.match(name).group("num") for name in self.backup_names("poster.psd")]
        self.assertEqual(numbers, ["001", "002"])

    def test_counter_recovers_from_existing_backups_when_state_lost(self):
        path = self.write("poster.psd", "v1", mtime=1_700_000_000)
        manager, _ = self.make_manager()
        manager.scan_once()
        (self.root / "state.json").unlink(missing_ok=True)

        path.write_text("v2", encoding="utf-8")
        os.utime(path, (1_700_000_600, 1_700_000_600))
        manager2, _ = self.make_manager()  # state.json 유실 상태로 재시작
        manager2.scan_once()

        numbers = [BACKUP_NAME_RE.match(name).group("num") for name in self.backup_names("poster.psd")]
        self.assertEqual(numbers, ["001", "002"])

    def test_state_file_schema(self):
        self.write("poster.psd", "v1", mtime=1_700_000_000)
        manager, store = self.make_manager()
        manager.scan_once()
        store.flush(force=True)

        payload = json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        entry = payload["files"][str(self.watched / "poster.psd")]
        self.assertEqual(entry["save_count"], 1)
        self.assertEqual(entry["last_mtime"], 1_700_000_000)


class RetentionTests(BaseCase):
    def test_oldest_save_number_deleted_beyond_max_versions(self):
        path = self.write("poster.psd", "v1", mtime=1_700_000_000)
        manager, _ = self.make_manager(self.make_config(max_versions=3))
        manager.scan_once()
        for index in range(2, 7):
            path.write_text(f"v{index}", encoding="utf-8")
            os.utime(path, (1_700_000_000 + index * 60, 1_700_000_000 + index * 60))
            manager.scan_once()

        numbers = [BACKUP_NAME_RE.match(name).group("num") for name in self.backup_names("poster.psd")]
        self.assertEqual(numbers, ["004", "005", "006"])

    def test_retention_is_per_file(self):
        first = self.write("a.txt", "1", mtime=1_700_000_000)
        second = self.write("b.txt", "1", mtime=1_700_000_000)
        manager, _ = self.make_manager(self.make_config(max_versions=2))
        manager.scan_once()
        for index in range(2, 5):
            stamp = 1_700_000_000 + index * 60
            for target in (first, second):
                target.write_text(f"v{index}", encoding="utf-8")
                os.utime(target, (stamp, stamp))
            manager.scan_once()

        self.assertEqual(len(self.backup_names("a.txt")), 2)
        self.assertEqual(len(self.backup_names("b.txt")), 2)


class ScanBehaviourTests(BaseCase):
    def test_recursive_scan_and_backup_dir_is_excluded(self):
        self.write("sub/deep/character.blend", "mesh")
        manager, _ = self.make_manager()
        stats = manager.scan_once()
        self.assertEqual(stats.backed_up, 1)

        # 백업 디렉터리를 감시 경로 내부로 옮겨도 재귀 백업이 발생하지 않아야 한다.
        inner_backups = self.watched / "backups"
        config = self.make_config(backup_dir=str(inner_backups))
        manager2, _ = self.make_manager(config)
        manager2.scan_once()
        stats2 = manager2.scan_once()
        self.assertEqual(stats2.backed_up, 0)

    def test_excluded_patterns_are_skipped(self):
        self.write("draft.tmp", "temp")
        self.write(".hidden", "temp")
        self.write("real.txt", "keep")
        manager, _ = self.make_manager()
        stats = manager.scan_once()

        self.assertEqual(stats.backed_up, 1)
        self.assertTrue((self.backups / "real.txt").exists())
        self.assertFalse((self.backups / "draft.tmp").exists())

    def test_same_filename_in_different_folders_uses_separate_folders(self):
        self.write("projA/main.py", "a")
        self.write("projB/main.py", "b")
        manager, _ = self.make_manager()
        manager.scan_once()

        folders = sorted(p.name for p in self.backups.iterdir())
        self.assertEqual(len(folders), 2)
        self.assertIn("main.py", folders)
        for folder in folders:
            self.assertEqual(len(self.backup_names(folder)), 1)

    def test_dry_run_creates_no_files(self):
        self.write("poster.psd", "v1")
        manager, _ = self.make_manager(dry_run=True)
        stats = manager.scan_once()

        self.assertEqual(stats.backed_up, 1)
        self.assertFalse((self.backups / "poster.psd").exists())

    def test_compound_extension_naming(self):
        self.write("release.tar.gz", "archive")
        manager, _ = self.make_manager()
        manager.scan_once()

        names = self.backup_names("release.tar.gz")
        self.assertEqual(len(names), 1)
        self.assertTrue(names[0].startswith("release_001_"))
        self.assertTrue(names[0].endswith(".tar.gz"))


class ConfigTests(BaseCase):
    def test_invalid_json_falls_back_to_defaults(self):
        config_path = self.root / "config.json"
        config_path.write_text("{ this is not json", encoding="utf-8")
        config = Config.load(config_path)

        self.assertEqual(config.interval_seconds, 900)
        self.assertEqual(config.max_versions, 20)
        self.assertEqual(config.backup_dir, (self.root / "backups").resolve())

    def test_out_of_range_values_are_clamped(self):
        config = Config.from_mapping(
            {"interval_seconds": 0, "max_versions": -5, "watch_paths": ["./w"]}, self.root
        )
        self.assertEqual(config.interval_seconds, 1)
        self.assertEqual(config.max_versions, 1)

    def test_relative_paths_resolve_against_config_dir(self):
        config = Config.from_mapping({"watch_paths": ["./w"], "backup_dir": "./b"}, self.root)
        self.assertEqual(config.watch_paths, ((self.root / "w").resolve(),))
        self.assertEqual(config.backup_dir, (self.root / "b").resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
