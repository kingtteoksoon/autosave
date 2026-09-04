# autosave — 범용 파일 자동 버전 백업 데몬

외부 애플리케이션(Photoshop, Blender, 3ds Max, Maya, ZBrush, Substance Painter, IDE 등)이
파일을 저장하면 이를 폴링으로 감지해 **버전별 백업본**을 중앙 백업 디렉터리에 생성하는
헤드리스 데몬. 자체 에디터 UI는 제공하지 않으며, 확장자와 무관하게 동작한다.

## 1. 요구 환경

- Python 3.9 이상 (표준 라이브러리만 사용, 외부 의존성 없음)
- Linux / macOS / Windows (경로·파일명 정규화 포함)

## 2. 디렉터리 구조

```
autosave_program/
├── config.json           # watch_paths, backup_dir, interval_seconds, max_versions 등
├── state.json            # 파일별 저장번호 카운터 (실행 시 자동 생성/갱신)
├── autosave_program.py   # 진입점: argparse, 로깅, 시그널 핸들링, 주기 루프
├── backup_manager.py     # 변경 감지 + 명명 규칙 + 보관 정책(retention)
└── logs/
    └── autosave.log      # 회전 로그 (5MB × 5개)
tests/
└── test_backup_manager.py
```

## 3. 실행

```bash
# 데몬 실행 (기본 설정 파일: autosave_program/config.json)
python3 autosave_program/autosave_program.py

# 설정을 지정해 실행
python3 autosave_program/autosave_program.py -c /etc/autosave/config.json -s /var/lib/autosave/state.json

# 1회 스캔 후 종료 (설정 점검용)
python3 autosave_program/autosave_program.py --once

# 실제 복사 없이 동작만 확인
python3 autosave_program/autosave_program.py --once --dry-run
```

| 옵션 | 설명 |
|---|---|
| `-c, --config` | 설정 파일 경로 (기본 `autosave_program/config.json`) |
| `-s, --state` | 상태 파일 경로 (기본 `autosave_program/state.json`) |
| `-l, --log-file` | 로그 파일 경로 (기본 `autosave_program/logs/autosave.log`) |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` (기본 `INFO`) |
| `--once` | 1회 스캔 후 종료 |
| `--dry-run` | 복사·삭제 없이 계획만 로그 출력 |

`SIGINT`(Ctrl+C) / `SIGTERM` 수신 시 진행 중인 스캔을 마치고 상태를 저장한 뒤 종료 코드 0으로 정상 종료한다.

## 4. 설정 (`config.json`)

```json
{
  "watch_paths": ["./watched"],
  "backup_dir": "./backups",
  "interval_seconds": 900,
  "max_versions": 20,
  "verify_hash": true,
  "stabilize_delay_seconds": 3,
  "stabilize_retries": 2,
  "exclude_patterns": [".*", "~$*", "*.tmp", "*.temp", "*.partial", "*.crdownload", "*.swp", "*.lock", "Thumbs.db"],
  "compound_extensions": [".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"]
}
```

| 키 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `watch_paths` | ● | `["./watched"]` | 재귀 감시 대상 폴더(또는 단일 파일) 목록 |
| `backup_dir` | ● | `"./backups"` | 원본과 분리된 중앙 백업 루트 |
| `interval_seconds` | ● | `900` | 스캔 주기(초). 사양상 15분 고정값을 기본으로 사용 |
| `max_versions` | ● | `20` | 파일당 최대 보관 버전 수 |
| `verify_hash` | | `true` | mtime/크기 변경 감지 후 MD5로 내용 변경 이중 확인 |
| `stabilize_delay_seconds` | | `3` | 저장 진행 중 오탐 방지를 위한 재확인 간격(초) |
| `stabilize_retries` | | `2` | 재확인 최대 횟수. 끝내 불안정하면 다음 주기로 이월 |
| `exclude_patterns` | | 위 목록 | 파일/폴더 제외 glob 패턴(파일명·전체 경로 모두 대조) |
| `compound_extensions` | | `.tar.*` | 복합 확장자 예외 목록(그 외에는 마지막 확장자만 분리) |

상대 경로는 **설정 파일이 위치한 디렉터리** 기준으로 절대화된다.
설정 파일이 없거나 JSON 파싱에 실패하면 경고 로그를 남기고 위 기본값으로 폴백하며,
범위를 벗어난 값(`interval_seconds <= 0` 등)은 최소값으로 보정한다.

## 5. 명명 규칙과 백업 배치

```
{backup_dir}/{원본파일명}/{원본파일명(확장자 제외)}_{저장번호:03d}_{YYYYMMDD_HHMMSS}{확장자}
```

| 원본 파일 | 백업 경로 |
|---|---|
| `/work/poster.psd` | `backups/poster.psd/poster_001_20260904_153000.psd` |
| `/work/character.blend` | `backups/character.blend/character_014_20260904_161200.blend` |
| `/src/main.py` | `backups/main.py/main_007_20260904_170500.py` |

- 타임스탬프는 **원본 파일의 수정 시각(mtime)** 을 로컬 시간으로 포맷한 값이다.
- 저장번호는 파일 경로 단위로 `state.json` 에 영속되어 재시작 후에도 이어서 증가한다.
- 서로 다른 폴더에 같은 파일명이 있으면(`projA/main.py`, `projB/main.py`) 두 번째 파일에는
  경로 해시 접미사가 붙은 폴더(`main.py_1f3c9ab2`)가 배정되고, 그 배정 결과도 상태에 영속된다.

## 6. 동작 흐름

1. `interval_seconds` 주기로 `watch_paths` 를 재귀 스캔한다(백업 디렉터리·제외 패턴·심볼릭 링크 제외).
2. `state.json` 의 `last_mtime`/`last_size` 와 비교해 변경을 감지한다.
3. 감지 즉시 `stabilize_delay_seconds` 간격으로 재확인해 쓰기가 끝났는지 확정한다
   (15분을 다시 기다리지 않는다). 끝내 크기가 계속 변하면 다음 주기로 이월한다.
4. `verify_hash` 가 켜져 있으면 MD5를 비교해, mtime만 바뀐 경우(touch 등)에는 저장번호를 소모하지 않는다.
5. 저장번호를 +1 하고, 임시 파일(`.<name>.partial`)로 복사한 뒤 `os.replace` 로 원자적으로 확정한다.
6. 파일별 보관 개수가 `max_versions` 를 초과하면 **저장번호가 가장 낮은 버전부터** 삭제한다.
7. `state.json` 은 임시 파일 + `fsync` + `os.replace` 로 원자적으로 갱신한다.

## 7. 예외 처리

| 상황 | 처리 |
|---|---|
| 저장 진행 중(쓰기 미완료) 파일 | 수 초 뒤 재확인으로 크기·mtime 동일 확정 후 백업, 불안정하면 다음 주기 재시도 |
| 디스크 공간 부족(`ENOSPC`) | 임시 파일 정리 → 오류 로그 → 이번 회차 스킵, **저장번호 롤백**(증가하지 않음) |
| 잠금/권한 오류(`EACCES`, `EBUSY`, `EPERM`) | 재시도 대상으로 로그 후 상태를 갱신하지 않아 다음 주기에 자동 재시도 |
| 설정 파일 파싱 실패 | 경고 로그 후 기본값 폴백 |
| `state.json` 손상 | `state.json.corrupt` 로 보존 후 초기화. 기존 백업 폴더의 최대 저장번호를 읽어 카운터 복구 |
| 개별 파일 처리 중 예외 | 해당 파일만 실패 처리하고 스캔은 계속 진행 |
| `SIGINT` / `SIGTERM` | 상태 저장 후 graceful shutdown |

## 8. 테스트

```bash
python3 -m unittest discover -s tests -v
```

명명 규칙, 저장번호 영속·복구, 보관 정책, 제외 패턴, 백업 디렉터리 재귀 방지,
설정 폴백 등 22개 케이스를 검증한다.

## 9. 서비스 등록 예시 (systemd)

```ini
# /etc/systemd/system/autosave.service
[Unit]
Description=Autosave version backup daemon
After=local-fs.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/autosave/autosave_program/autosave_program.py -c /etc/autosave/config.json
Restart=on-failure
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

## 10. 사양 대비 구현 노트

- **보관 개수**: 사양서 4.1 예시(`10`)와 5장(`20`)이 상이해, 최종 요구인 **20**을 기본값으로 채택했다.
- **최초 스캔**: 처음 발견한 파일은 기준 버전(`_001_`)으로 즉시 1회 백업한다.
- **이동/이름 변경**: 경로를 키로 사용하므로 새 항목으로 취급하며 이전 저장번호를 승계하지 않는다(사양 4.2 명시).
- **상태 스키마 확장**: 사양의 `save_count`, `last_mtime` 에 더해 `last_size`, `last_hash`,
  `backup_folder` 를 저장한다(크기 비교·MD5 이중 확인·폴더 충돌 방지에 필요).
