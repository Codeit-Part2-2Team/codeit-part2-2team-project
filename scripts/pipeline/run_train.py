# isort: skip_file
# ruff: noqa: E402
"""Stage 1 학습 → Stage 2 학습 통합 실행 스크립트.

사용 예:
    python scripts/pipeline/run_train.py \
        --stage1-config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
        --stage2-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
        --data          data/processed/dataset.yaml \
        --crops         data/processed/crops/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            p
            for p in Path(__file__).resolve().parents
            if (p / "requirements.txt").exists()
        )
    ),
)

from src.utils.cli import run_command
from src.utils.path import PROJECT_ROOT

ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "scripts"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 학습 → Stage 2 학습 통합 실행"
    )
    parser.add_argument("--stage1-config", required=True, help="Stage 1 config 경로")
    parser.add_argument("--stage2-config", required=True, help="Stage 2 config 경로")
    parser.add_argument("--data", required=True, help="Stage 1 dataset.yaml 경로")
    parser.add_argument(
        "--crops",
        required=True,
        help="Stage 2 학습용 GT crop 루트 디렉터리 (<crops>/train, <crops>/val)",
    )
    parser.add_argument("--device", default=None, help="GPU 번호 또는 'cpu'")
    args = parser.parse_args()

    # Stage 1 학습
    s1_cmd = [
        sys.executable,
        str(SCRIPTS / "train.py"),
        "--config",
        args.stage1_config,
        "--data",
        args.data,
    ]
    if args.device:
        s1_cmd += ["--device", args.device]
    run_command(s1_cmd, "Stage 1 학습", cwd=ROOT)

    # Stage 2 학습
    run_command(
        [
            sys.executable,
            str(SCRIPTS / "pipeline" / "stage2_train.py"),
            "--config",
            args.stage2_config,
            "--data",
            args.crops,
        ],
        "Stage 2 학습",
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
