"""CLI 실행 유틸리티."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, cwd: Path | None = None) -> None:
    """CLI 명령을 실행하고 결과를 확인한다.

    Args:
        cmd: 실행할 명령 리스트 (예: ["python", "script.py", "--arg", "value"])
        description: 명령 설명 (로깅용)
        cwd: 작업 디렉터리 (기본: 현재 디렉터리)

    Raises:
        SystemExit: 명령 실패 시 프로그램 종료
    """
    print(f"[실행] {description}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[오류] {description} 실패:")
        print(result.stderr)
        sys.exit(1)
    print(f"[완료] {description}")
