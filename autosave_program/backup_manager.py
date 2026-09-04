"""파일 변경 감지 · 버전 백업 생성 · 보관 정책(retention)을 담당하는 핵심 모듈.

폴링(mtime/size 비교 + 내용 해시 이중 확인) 방식으로 저장 이벤트를 판정하고,
`{원본파일명}_{저장번호:03d}_{YYYYMMDD_HHMMSS}{확장자}` 규칙으로 중앙 백업
디렉터리에 사본을 만든 뒤, 파일별 최대 보관 개수를 초과하면 저장번호가 낮은
버전부터 삭제한다. 저장번호는 state.json 에 영속화되어 재시작 후에도 이어진다.
"""

from __future__ import annotations

import contextlib
import errno
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

#: config.json 이 없거나 파싱에 실패했을 때 사용하는 폴백 기본값.
DEFAULT_CONFIG: Dict[str, object] = {
    "watch_paths": ["./watched"],
    "backup_dir": "./backups",
    "interval_seconds": 900,
    "max_versions": 20,
    # --- 선택 항목(생략 시 아래 기본값 사용) ---
    "verify_hash": True,
    "stabilize_delay_seconds": 3,
    "stabilize_retries": 2,
    "exclude_patterns": [
        ".*",
        "~$*",
        "*.tmp",
        "*.temp",
        "*.partial",
        "*.crdownload",
        "*.swp",
        "*.lock",
        "Thumbs.db",
    ],
    "compound_extensions": [".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"],
}

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_WINDOWS_RESERVED_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# --------------------------------------------------------------------------- #
# 설정
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    """config.json 을 표현하는 불변 설정 객체."""

    watch_paths: Tuple[Path, ...]
    backup_dir: Path
    interval_seconds: int
    max_versions: int
    verify_hash: bool = True
    stabilize_delay_seconds: float = 3.0
    stabilize_retries: int = 2
    exclude_patterns: Tuple[str, ...] = ()
    compound_extensions: Tuple[str, ...] = ()

    @classmethod
    def load(cls, config_path: Path) -> "Config":
        """설정 파일을 읽어 Config 를 만든다. 실패 시 경고 로그 후 기본값으로 폴백."""
        raw: Dict[str, object] = dict(DEFAULT_CONFIG)
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, dict):
                raise ValueError("config.json 최상위는 객체여야 합니다.")
            raw.update(loaded)
        except FileNotFoundError:
            LOGGER.warning("설정 파일이 없어 기본값으로 실행합니다: %s", config_path)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            LOGGER.warning("설정 파일 파싱 실패(%s) — 기본값으로 폴백합니다: %s", exc, config_path)

        base_dir = config_path.parent.resolve()
        return cls.from_mapping(raw, base_dir)

    @classmethod
    def from_mapping(cls, raw: Dict[str, object], base_dir: Path) -> "Config":
        """딕셔너리를 검증·정규화한다. 상대 경로는 base_dir 기준으로 절대화한다."""

        def _resolve(value: object, fallback: str) -> Path:
            text = str(value) if isinstance(value, (str, os.PathLike)) else fallback
            path = Path(os.path.expandvars(os.path.expanduser(text)))
            return path if path.is_absolute() else (base_dir / path).resolve()

        watch_raw = raw.get("watch_paths") or DEFAULT_CONFIG["watch_paths"]
        if isinstance(watch_raw, (str, os.PathLike)):
            watch_raw = [watch_raw]
        if not isinstance(watch_raw, list) or not watch_raw:
            LOGGER.warning("watch_paths 값이 올바르지 않아 기본값을 사용합니다.")
            watch_raw = DEFAULT_CONFIG["watch_paths"]

        watch_paths = tuple(dict.fromkeys(_resolve(item, "./watched") for item in watch_raw))
        backup_dir = _resolve(raw.get("backup_dir"), str(DEFAULT_CONFIG["backup_dir"]))

        def _positive_int(key: str, minimum: int) -> int:
            fallback = int(DEFAULT_CONFIG[key])  # type: ignore[arg-type]
            try:
                value = int(raw.get(key, fallback))
            except (TypeError, ValueError):
                LOGGER.warning("%s 값이 정수가 아니어서 기본값(%s)을 사용합니다.", key, fallback)
                return fallback
            if value < minimum:
                LOGGER.warning("%s=%s 는 허용 범위 미만이라 %s 로 보정합니다.", key, value, minimum)
                return minimum
            return value

        def _non_negative_float(key: str) -> float:
            fallback = float(DEFAULT_CONFIG[key])  # type: ignore[arg-type]
            try:
                value = float(raw.get(key, fallback))
            except (TypeError, ValueError):
                LOGGER.warning("%s 값이 숫자가 아니어서 기본값(%s)을 사용합니다.", key, fallback)
                return fallback
            return max(0.0, value)

        def _str_tuple(key: str) -> Tuple[str, ...]:
            value = raw.get(key, DEFAULT_CONFIG[key])
            if not isinstance(value, list):
                LOGGER.warning("%s 값이 배열이 아니어서 기본값을 사용합니다.", key)
                value = DEFAULT_CONFIG[key]
            return tuple(str(item) for item in value)  # type: ignore[union-attr]

        compound = tuple(sorted({ext.lower() for ext in _str_tuple("compound_extensions")}, key=len, reverse=True))

        return cls(
            watch_paths=watch_paths,
            backup_dir=backup_dir,
            interval_seconds=_positive_int("interval_seconds", 1),
            max_versions=_positive_int("max_versions", 1),
            verify_hash=bool(raw.get("verify_hash", DEFAULT_CONFIG["verify_hash"])),
            stabilize_delay_seconds=_non_negative_float("stabilize_delay_seconds"),
            stabilize_retries=_positive_int("stabilize_retries", 0),
            exclude_patterns=_str_tuple("exclude_patterns"),
            compound_extensions=compound,
        )


# --------------------------------------------------------------------------- #
# 영속 상태(state.json)
# --------------------------------------------------------------------------- #
@dataclass
class FileState:
    """파일 하나에 대한 저장번호와 마지막으로 관측한 메타데이터."""

    save_count: int = 0
    last_mtime: float = 0.0
    last_size: int = -1
    last_hash: Optional[str] = None
    backup_folder: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "save_count": self.save_count,
            "last_mtime": self.last_mtime,
            "last_size": self.last_size,
        }
        if self.last_hash:
            data["last_hash"] = self.last_hash
        if self.backup_folder:
            data["backup_folder"] = self.backup_folder
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FileState":
        return cls(
            save_count=int(data.get("save_count", 0) or 0),
            last_mtime=float(data.get("last_mtime", 0.0) or 0.0),
            last_size=int(data.get("last_size", -1) if data.get("last_size") is not None else -1),
            last_hash=(str(data["last_hash"]) if data.get("last_hash") else None),
            backup_folder=(str(data["backup_folder"]) if data.get("backup_folder") else None),
        )


class StateStore:
    """state.json 로드/저장. 쓰기는 임시 파일 + os.replace 로 원자적으로 수행한다."""

    def __init__(self, state_path: Path) -> None:
        self._path = state_path
        self._files: Dict[str, FileState] = {}
        self._dirty = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            files = payload.get("files", {}) if isinstance(payload, dict) else {}
            if not isinstance(files, dict):
                raise ValueError("state.json 의 files 는 객체여야 합니다.")
            self._files = {
                str(key): FileState.from_dict(value)
                for key, value in files.items()
                if isinstance(value, dict)
            }
            LOGGER.info("상태 파일 로드 완료: %s (추적 파일 %d개)", self._path, len(self._files))
        except FileNotFoundError:
            LOGGER.info("상태 파일이 없어 새로 시작합니다: %s", self._path)
            self._files = {}
        except (json.JSONDecodeError, ValueError, OSError, TypeError) as exc:
            backup = self._path.with_suffix(self._path.suffix + ".corrupt")
            LOGGER.error("상태 파일이 손상되어 초기화합니다(%s). 원본 보존: %s", exc, backup)
            try:
                if self._path.exists():
                    os.replace(self._path, backup)
            except OSError:
                LOGGER.exception("손상된 상태 파일 보존에 실패했습니다.")
            self._files = {}

    def get(self, key: str) -> Optional[FileState]:
        return self._files.get(key)

    def set(self, key: str, state: FileState) -> None:
        self._files[key] = state
        self._dirty = True

    def used_backup_folders(self) -> Dict[str, str]:
        """백업 폴더명 → 소유 파일 경로 (중복 폴더 배정을 막기 위한 역인덱스)."""
        return {
            state.backup_folder: key
            for key, state in self._files.items()
            if state.backup_folder
        }

    def flush(self, force: bool = False) -> None:
        """변경분이 있을 때만 원자적으로 기록한다."""
        if not (self._dirty or force):
            return
        payload = {"files": {key: state.to_dict() for key, state in sorted(self._files.items())}}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._path)
            self._dirty = False
        except OSError:
            LOGGER.exception("상태 파일 저장 실패: %s", self._path)
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _is_same_or_inside(path: Path, root: Path) -> bool:
    """path 가 root 자신이거나 root 하위에 있으면 True (백업 폴더 재귀 방지)."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:  # pragma: no cover - 심볼릭 링크 루프 등
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


# --------------------------------------------------------------------------- #
# 유틸리티
# --------------------------------------------------------------------------- #
def split_name(filename: str, compound_extensions: Tuple[str, ...] = ()) -> Tuple[str, str]:
    """파일명을 (확장자 제외 이름, 확장자)로 분리한다.

    기본은 마지막 확장자만 분리하며, `.tar.gz` 처럼 예외 목록에 등록된
    복합 확장자는 통째로 확장자로 취급한다.
    """
    lowered = filename.lower()
    for ext in compound_extensions:
        if lowered.endswith(ext) and len(filename) > len(ext):
            return filename[: -len(ext)], filename[-len(ext) :]
    stem, ext = os.path.splitext(filename)
    return (stem, ext) if stem else (filename, "")


def sanitize_component(name: str) -> str:
    """경로 구성요소로 안전한 문자열로 정규화한다(윈도우 금지문자 포함)."""
    cleaned = _WINDOWS_RESERVED_CHARS.sub("_", name).strip().rstrip(". ")
    return cleaned or "_"


def file_digest(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """파일 내용의 MD5 다이제스트(무결성 비교 용도, 암호학적 용도 아님)."""
    try:
        digest = hashlib.md5(usedforsecurity=False)  # type: ignore[call-arg]
    except TypeError:  # Python < 3.9
        digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScanStats:
    """한 스캔 주기의 처리 결과 요약."""

    scanned: int = 0
    backed_up: int = 0
    skipped_unchanged: int = 0
    retried_later: int = 0
    failed: int = 0
    deleted_versions: int = 0
    retry_paths: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} backed_up={self.backed_up} "
            f"unchanged={self.skipped_unchanged} retry={self.retried_later} "
            f"failed={self.failed} pruned={self.deleted_versions}"
        )


# --------------------------------------------------------------------------- #
# 백업 매니저
# --------------------------------------------------------------------------- #
class BackupManager:
    """감시 대상 폴더를 스캔해 변경된 파일의 버전 백업을 생성한다."""

    #: 백업 파일명 파싱용: {stem}_{번호}_{YYYYMMDD_HHMMSS}{ext}
    _VERSION_RE_TEMPLATE = r"^{stem}_(\d{{3,}})_\d{{8}}_\d{{6}}{ext}$"

    def __init__(self, config: Config, state: StateStore, dry_run: bool = False) -> None:
        self.config = config
        self.state = state
        self.dry_run = dry_run

    # -- 공개 API ---------------------------------------------------------- #
    def scan_once(self) -> ScanStats:
        """감시 경로를 1회 순회하며 변경 파일을 백업한다."""
        stats = ScanStats()
        for path in self._iter_files():
            stats.scanned += 1
            try:
                self._process_file(path, stats)
            except Exception:  # 개별 파일 실패가 데몬 전체를 멈추지 않도록 격리
                stats.failed += 1
                LOGGER.exception("파일 처리 중 예기치 못한 예외: %s", path)
        self.state.flush()
        return stats

    # -- 파일 수집 --------------------------------------------------------- #
    def _iter_files(self) -> Iterator[Path]:
        backup_root = self.config.backup_dir
        for watch_path in self.config.watch_paths:
            if not watch_path.exists():
                LOGGER.warning("감시 경로가 존재하지 않습니다(스킵): %s", watch_path)
                continue
            if watch_path.is_file():
                if not self._is_excluded(watch_path):
                    yield watch_path
                continue
            for root, dirnames, filenames in os.walk(watch_path, followlinks=False):
                root_path = Path(root)
                # 백업 디렉터리가 감시 경로 안에 있어도 재귀 백업이 생기지 않도록 제외
                dirnames[:] = [
                    d
                    for d in dirnames
                    if not self._is_excluded(root_path / d, is_dir=True)
                    and not _is_same_or_inside(root_path / d, backup_root)
                ]
                for filename in filenames:
                    candidate = root_path / filename
                    if self._is_excluded(candidate):
                        continue
                    if _is_same_or_inside(candidate, backup_root):
                        continue
                    yield candidate

    def _is_excluded(self, path: Path, is_dir: bool = False) -> bool:
        name = path.name
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(str(path), pattern):
                LOGGER.debug("제외 패턴(%s)에 걸려 스킵: %s", pattern, path)
                return True
        if not is_dir and path.is_symlink():
            LOGGER.debug("심볼릭 링크는 감시 대상에서 제외: %s", path)
            return True
        return False

    # -- 파일 단위 처리 ---------------------------------------------------- #
    def _process_file(self, path: Path, stats: ScanStats) -> None:
        key = str(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return  # 스캔 도중 삭제됨
        except OSError as exc:
            stats.retried_later += 1
            stats.retry_paths.append(key)
            LOGGER.warning("메타데이터 조회 실패(다음 주기 재시도): %s (%s)", path, exc)
            return

        entry = self.state.get(key)
        if entry is None:
            entry = self._bootstrap_entry(path)
        elif stat.st_mtime == entry.last_mtime and stat.st_size == entry.last_size:
            stats.skipped_unchanged += 1
            return

        stable_stat = self._wait_until_stable(path, stat)
        if stable_stat is None:
            stats.retried_later += 1
            stats.retry_paths.append(key)
            LOGGER.info("쓰기 진행 중으로 판단되어 다음 주기에 재시도합니다: %s", path)
            return

        digest: Optional[str] = None
        if self.config.verify_hash:
            try:
                digest = file_digest(path)
            except OSError as exc:
                stats.retried_later += 1
                stats.retry_paths.append(key)
                LOGGER.warning("해시 계산 실패(다음 주기 재시도): %s (%s)", path, exc)
                return
            if entry.last_hash and digest == entry.last_hash:
                # mtime 만 바뀐 경우(touch 등) — 저장번호를 소모하지 않는다.
                entry.last_mtime = stable_stat.st_mtime
                entry.last_size = stable_stat.st_size
                self.state.set(key, entry)
                stats.skipped_unchanged += 1
                LOGGER.debug("내용 동일(메타데이터만 변경) — 백업 생략: %s", path)
                return

        self._backup_file(path, stable_stat, entry, digest, stats)

    def _bootstrap_entry(self, path: Path) -> FileState:
        """신규 추적 파일의 상태를 만든다.

        상태 파일이 유실되었더라도 기존 백업 폴더에 남아 있는 최대 저장번호를
        읽어 이어서 증가시킨다(덮어쓰기 방지).
        """
        entry = FileState()
        entry.backup_folder = self._assign_backup_folder(path)
        folder = self.config.backup_dir / entry.backup_folder
        stem, ext = split_name(path.name, self.config.compound_extensions)
        existing = self._existing_versions(folder, stem, ext)
        if existing:
            entry.save_count = max(number for number, _ in existing)
            LOGGER.info(
                "기존 백업(%d개)에서 저장번호를 복구했습니다: %s → %03d",
                len(existing),
                path,
                entry.save_count,
            )
        return entry

    def _assign_backup_folder(self, path: Path) -> str:
        """원본 파일명과 동일한 백업 하위 폴더명을 배정한다.

        서로 다른 경로에 같은 파일명이 있으면 경로 해시 접미사로 충돌을 피한다.
        배정 결과는 state.json 에 영속되어 이후에도 동일 폴더를 사용한다.
        """
        candidate = sanitize_component(path.name)
        owners = self.state.used_backup_folders()
        owner = owners.get(candidate)
        if owner is None or owner == str(path):
            return candidate
        suffix = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
        unique = f"{candidate}_{suffix}"
        LOGGER.info("백업 폴더명 충돌(%s) — 고유 폴더를 사용합니다: %s", candidate, unique)
        return unique

    def _wait_until_stable(self, path: Path, stat: os.stat_result):
        """저장이 끝났는지 수 초 간격으로 재확인한다(크기·mtime 동일 확정)."""
        delay = self.config.stabilize_delay_seconds
        if delay <= 0:
            return stat
        previous = stat
        for _ in range(self.config.stabilize_retries + 1):
            time.sleep(delay)
            try:
                current = path.stat()
            except OSError:
                return None
            if current.st_size == previous.st_size and current.st_mtime == previous.st_mtime:
                return current
            previous = current
        return None

    # -- 백업 생성 / 보관 정책 --------------------------------------------- #
    def _backup_file(
        self,
        path: Path,
        stat: os.stat_result,
        entry: FileState,
        digest: Optional[str],
        stats: ScanStats,
    ) -> None:
        key = str(path)
        stem, ext = split_name(path.name, self.config.compound_extensions)
        folder = self.config.backup_dir / (entry.backup_folder or self._assign_backup_folder(path))
        next_count = entry.save_count + 1
        timestamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y%m%d_%H%M%S")
        dest = folder / f"{stem}_{next_count:03d}_{timestamp}{ext}"

        if self.dry_run:
            LOGGER.info("[dry-run] 백업 생성 예정: %s → %s", path, dest)
            stats.backed_up += 1
            return

        tmp_path = folder / f".{dest.name}.partial"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, tmp_path)
            os.replace(tmp_path, dest)
        except OSError as exc:
            # 실패 시 저장번호를 증가시키지 않는다(카운터 롤백).
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            if exc.errno == errno.ENOSPC:
                stats.failed += 1
                LOGGER.error("디스크 공간 부족으로 백업 실패(이번 회차 스킵): %s", path)
            elif exc.errno in (errno.EACCES, errno.EBUSY, errno.EPERM, errno.ETXTBSY):
                stats.retried_later += 1
                stats.retry_paths.append(key)
                LOGGER.warning("파일 잠금/권한 문제로 다음 주기에 재시도합니다: %s (%s)", path, exc)
            else:
                stats.failed += 1
                LOGGER.error("백업 실패: %s → %s (%s)", path, dest, exc)
            return

        entry.save_count = next_count
        entry.last_mtime = stat.st_mtime
        entry.last_size = stat.st_size
        entry.last_hash = digest
        entry.backup_folder = folder.name
        self.state.set(key, entry)
        stats.backed_up += 1
        LOGGER.info("백업 생성: %s → %s (저장번호 %03d)", path, dest, next_count)

        stats.deleted_versions += self._apply_retention(folder, stem, ext)

    def _existing_versions(self, folder: Path, stem: str, ext: str) -> List[Tuple[int, Path]]:
        """백업 폴더에서 명명 규칙에 맞는 버전 목록을 (저장번호, 경로)로 반환."""
        pattern = re.compile(
            self._VERSION_RE_TEMPLATE.format(stem=re.escape(stem), ext=re.escape(ext))
        )
        versions: List[Tuple[int, Path]] = []
        try:
            for item in folder.iterdir():
                if not item.is_file():
                    continue
                match = pattern.match(item.name)
                if match:
                    versions.append((int(match.group(1)), item))
        except FileNotFoundError:
            return []
        except OSError as exc:
            LOGGER.warning("백업 폴더를 읽지 못했습니다: %s (%s)", folder, exc)
            return []
        versions.sort(key=lambda pair: (pair[0], pair[1].name))
        return versions

    def _apply_retention(self, folder: Path, stem: str, ext: str) -> int:
        """max_versions 초과분을 저장번호가 낮은 것부터 삭제하고 삭제 개수를 반환."""
        versions = self._existing_versions(folder, stem, ext)
        excess = len(versions) - self.config.max_versions
        if excess <= 0:
            return 0
        deleted = 0
        for number, victim in versions[:excess]:
            try:
                victim.unlink()
                deleted += 1
                LOGGER.info("보관 정책에 따라 오래된 버전 삭제: %s (저장번호 %03d)", victim, number)
            except OSError as exc:
                LOGGER.warning("오래된 버전 삭제 실패: %s (%s)", victim, exc)
        return deleted
