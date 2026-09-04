#!/usr/bin/env python3
"""자동저장(버전 백업) 데몬 진입점.

config.json 을 읽어 감시 경로를 `interval_seconds`(기본 900초) 주기로 폴링하고,
변경된 파일마다 버전 백업을 만든다. SIGINT/SIGTERM 수신 시 진행 중인 스캔을
마친 뒤 상태를 저장하고 정상 종료한다(graceful shutdown).

사용 예:
    python3 autosave_program.py                 # 데몬 실행(기본 설정 파일)
    python3 autosave_program.py --once          # 1회 스캔 후 종료(점검용)
    python3 autosave_program.py --dry-run       # 실제 복사 없이 동작만 확인
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Sequence

try:  # 패키지로 임포트될 때와 스크립트로 직접 실행될 때 모두 지원
    from .backup_manager import BackupManager, Config, StateStore
except ImportError:  # pragma: no cover - 직접 실행 경로
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backup_manager import BackupManager, Config, StateStore  # type: ignore[no-redef]

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_STATE_PATH = BASE_DIR / "state.json"
DEFAULT_LOG_PATH = BASE_DIR / "logs" / "autosave.log"

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

LOGGER = logging.getLogger("autosave")


def configure_logging(log_path: Path, level: str) -> None:
    """콘솔 + 회전 파일 핸들러를 구성한다. 파일 핸들러 실패 시 콘솔만 사용."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("로그 파일을 열 수 없어 콘솔 로그만 사용합니다: %s (%s)", log_path, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autosave_program",
        description="지정 폴더의 파일 저장을 감지해 버전 백업을 생성하는 헤드리스 데몬",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="설정 파일 경로 (기본: %(default)s)"
    )
    parser.add_argument(
        "-s", "--state", type=Path, default=DEFAULT_STATE_PATH, help="상태 파일 경로 (기본: %(default)s)"
    )
    parser.add_argument(
        "-l", "--log-file", type=Path, default=DEFAULT_LOG_PATH, help="로그 파일 경로 (기본: %(default)s)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="로그 레벨 (기본: %(default)s)",
    )
    parser.add_argument("--once", action="store_true", help="1회만 스캔하고 종료한다(점검용)")
    parser.add_argument(
        "--dry-run", action="store_true", help="실제 복사·삭제 없이 수행 계획만 로그로 남긴다"
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handle(signum: int, _frame) -> None:
        LOGGER.info("시그널 %s 수신 — 정상 종료를 시작합니다.", signal.Signals(signum).name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # pragma: no cover - 메인 스레드가 아닌 경우
            LOGGER.debug("시그널 핸들러 등록 실패: %s", sig)


def run(args: argparse.Namespace, stop_event: Optional[threading.Event] = None) -> int:
    """데몬 본체. 반환값은 프로세스 종료 코드."""
    stop_event = stop_event or threading.Event()
    config = Config.load(args.config)
    state = StateStore(args.state)
    state.load()
    manager = BackupManager(config, state, dry_run=args.dry_run)

    LOGGER.info("자동저장 데몬 v%s 시작 (pid=%d)", __version__, os.getpid())
    LOGGER.info("감시 경로: %s", ", ".join(str(p) for p in config.watch_paths))
    LOGGER.info(
        "백업 위치: %s | 주기: %d초 | 파일당 보관: %d개 | dry-run: %s",
        config.backup_dir,
        config.interval_seconds,
        config.max_versions,
        args.dry_run,
    )

    try:
        config.backup_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        LOGGER.exception("백업 디렉터리를 만들 수 없습니다: %s", config.backup_dir)
        return 1

    exit_code = 0
    try:
        while not stop_event.is_set():
            stats = manager.scan_once()
            LOGGER.info("스캔 완료: %s", stats.summary())
            if stats.retry_paths:
                LOGGER.info("다음 주기 재시도 대상 %d건: %s", len(stats.retry_paths), stats.retry_paths[:10])
            if args.once:
                break
            stop_event.wait(config.interval_seconds)
    except Exception:
        LOGGER.exception("데몬 루프에서 치명적 예외가 발생해 종료합니다.")
        exit_code = 1
    finally:
        state.flush(force=True)
        LOGGER.info("상태를 저장하고 종료합니다: %s", state.path)
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_file, args.log_level)
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    return run(args, stop_event)


if __name__ == "__main__":
    raise SystemExit(main())
