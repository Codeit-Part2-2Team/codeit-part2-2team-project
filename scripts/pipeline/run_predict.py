"""Stage 1 추론 → crop → Stage 2 추론 → submission.csv 통합 실행 스크립트.

사용 예:
    python scripts/pipeline/run_predict.py \
        --stage1-config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
        --stage2-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
        --source data/raw/test/ \
        --output submissions/submission.csv
"""

from __future__ import annotations

import argparse
import json
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
from src.utils.config import load_config
from src.utils.path import (
    PROJECT_ROOT,
    resolve_predictions_path,
    resolve_project_path,
    resolve_stage2_predictions_path,
    resolve_weights_path,
)
from src.utils.submission import save_submission

ROOT = PROJECT_ROOT
SCRIPTS = ROOT / "scripts"


def merge_predictions(
    manifest_path: str | Path, stage2_predictions_path: str | Path
) -> list[dict]:
    """Stage 2 예측 결과와 크롭 manifest를 crop_id 기준으로 병합한다.

    Returns:
        image_id별 detections 리스트 [{"image_id", "detections": [{"class_id", "class_name", "bbox", "score"}]}]
    """
    with Path(manifest_path).open(encoding="utf-8") as f:
        manifest = json.load(f)
    with Path(stage2_predictions_path).open(encoding="utf-8") as f:
        stage2_predictions = json.load(f)

    manifest_by_crop = {item["crop_id"]: item for item in manifest}
    merged_by_image: dict[str, list[dict]] = {}

    for record in stage2_predictions:
        crop_id = record["crop_id"]
        manifest_item = manifest_by_crop.get(crop_id)
        if manifest_item is None:
            raise KeyError(f"manifest에 없는 crop_id: {crop_id}")
        merged_by_image.setdefault(record["image_id"], []).append(
            {
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "bbox": manifest_item["bbox"],
                "score": record["score"],
            }
        )

    image_ids = {item["image_id"] for item in manifest}
    return [
        {"image_id": iid, "detections": merged_by_image.get(iid, [])}
        for iid in sorted(image_ids)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 추론 → crop → Stage 2 추론 → submission.csv"
    )
    parser.add_argument("--stage1-config", required=True, help="Stage 1 config 경로")
    parser.add_argument("--stage2-config", required=True, help="Stage 2 config 경로")
    parser.add_argument("--source", required=True, help="테스트 이미지 디렉터리")
    parser.add_argument("--output", required=True, help="submission.csv 저장 경로")
    parser.add_argument(
        "--stage1-weights",
        default=None,
        help="Stage 1 best.pt (미지정 시 config 자동 조합)",
    )
    parser.add_argument(
        "--stage2-weights",
        default=None,
        help="Stage 2 best.pt (미지정 시 config 자동 조합)",
    )
    parser.add_argument(
        "--crop-output",
        default="data/processed/crops/inference",
        help="crop 저장 디렉터리",
    )
    parser.add_argument(
        "--padding", type=float, default=0.05, help="bbox 여백 비율 (기본값 0.05)"
    )
    args = parser.parse_args()

    stage1_cfg = load_config(args.stage1_config)
    stage2_cfg = load_config(args.stage2_config)

    stage1_weights = resolve_weights_path(stage1_cfg, args.stage1_weights)
    stage1_predictions = resolve_predictions_path(stage1_cfg, None)
    stage2_weights = resolve_weights_path(stage2_cfg, args.stage2_weights)
    stage2_predictions = resolve_stage2_predictions_path(stage2_cfg, None)
    crop_output = resolve_project_path(args.crop_output)
    manifest_path = crop_output / "crops_manifest.json"

    run_command(
        [
            sys.executable,
            str(SCRIPTS / "predict.py"),
            "--config",
            args.stage1_config,
            "--weights",
            str(stage1_weights),
            "--source",
            args.source,
            "--output",
            str(stage1_predictions),
        ],
        f"Stage 1 추론: {args.source} → {stage1_predictions}",
        cwd=ROOT,
    )

    run_command(
        [
            sys.executable,
            str(SCRIPTS / "pipeline" / "crop.py"),
            "--predictions",
            str(stage1_predictions),
            "--source",
            args.source,
            "--output",
            str(crop_output),
            "--padding",
            str(args.padding),
        ],
        f"crop 생성: {crop_output}",
        cwd=ROOT,
    )

    run_command(
        [
            sys.executable,
            str(SCRIPTS / "pipeline" / "stage2_predict.py"),
            "--config",
            args.stage2_config,
            "--weights",
            str(stage2_weights),
            "--source",
            str(crop_output),
            "--output",
            str(stage2_predictions),
        ],
        f"Stage 2 추론: {crop_output} → {stage2_predictions}",
        cwd=ROOT,
    )

    print("[실행] 결과 병합 및 submission 생성 중...")
    merged = merge_predictions(manifest_path, stage2_predictions)
    output_path = resolve_project_path(args.output)
    save_submission(merged, output_path)
    print(f"[완료] submission.csv → {output_path}")


if __name__ == "__main__":
    main()
