"""Stage 1 → crop → Stage 2 → submission 통합 실행 스크립트.

사용 예:
    python scripts/pipeline/run.py \
        --stage1-config experiments/stage1_detection/config.yaml \
        --stage2-config experiments/stage2_classifier/exp_A/config.yaml \
        --source data/raw/test/ \
        --output submissions/submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from src.utils.cli import run_command
from src.utils.cli import run_command
from src.utils.config import load_config
from src.utils.path import (
    resolve_predictions_path,
    resolve_project_path,
    resolve_stage2_predictions_path,
    resolve_weights_path,
)
from src.utils.submission import save_submission


def merge_predictions(
    manifest_path: str | Path, stage2_predictions_path: str | Path
) -> list[dict]:
    """Stage 2 예측 결과와 크롭 manifest를 crop_id 기준으로 병합한다.

    Args:
        manifest_path: crops_manifest.json 경로
        stage2_predictions_path: stage2_predictions.json 경로

    Returns:
        image_id별 detections 리스트를 가진 predictions.json 스키마.
    """
    manifest_path = Path(manifest_path)
    stage2_predictions_path = Path(stage2_predictions_path)

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with stage2_predictions_path.open("r", encoding="utf-8") as f:
        stage2_predictions = json.load(f)

    manifest_by_crop = {item["crop_id"]: item for item in manifest}
    merged_by_image: dict[str, list[dict]] = {}

    for record in stage2_predictions:
        crop_id = record["crop_id"]
        manifest_item = manifest_by_crop.get(crop_id)
        if manifest_item is None:
            raise KeyError(f"stage2_predictions.json에 없는 crop_id가 있습니다: {crop_id}")

        image_id = record["image_id"]
        merged_by_image.setdefault(image_id, []).append(
            {
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "bbox": manifest_item["bbox"],
                "score": record["score"],
            }
        )

    # 빈 이미지도 유지하면 downstream이 더 예측 가능함.
    image_ids = {item["image_id"] for item in manifest}
    return [
        {"image_id": image_id, "detections": merged_by_image.get(image_id, [])}
        for image_id in sorted(image_ids)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 추론 → 크롭 생성 → Stage 2 추론 → submission.csv 생성"
    )
    parser.add_argument("--stage1-config", required=True, help="Stage 1 config.yaml 경로")
    parser.add_argument("--stage2-config", required=True, help="Stage 2 config.yaml 경로")
    parser.add_argument("--source", required=True, help="테스트 이미지 디렉터리")
    parser.add_argument("--output", required=True, help="submission.csv 저장 경로")
    parser.add_argument(
        "--stage1-weights",
        default=None,
        help="Stage 1 best.pt 경로 (미지정 시 config output에서 자동 조합)",
    )
    parser.add_argument(
        "--stage2-weights",
        default=None,
        help="Stage 2 best.pt 경로 (미지정 시 config output에서 자동 조합)",
    )
    parser.add_argument(
        "--crop-output",
        default="data/processed/crops/inference",
        help="Stage 1 bbox 기반 크롭 이미지 저장 디렉터리",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="bbox 여백 비율 (기본값 0.05)",
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

    # Stage 1 추론
    run_command([
        sys.executable, str(SCRIPTS / "predict.py"),
        "--config", args.stage1_config,
        "--weights", str(stage1_weights),
        "--source", args.source,
        "--output", str(stage1_predictions),
    ], f"Stage 1 추론: {args.source} → {stage1_predictions}", cwd=ROOT)

    # 크롭 생성
    run_command([
        sys.executable, str(SCRIPTS / "pipeline" / "crop.py"),
        "--predictions", str(stage1_predictions),
        "--source", args.source,
        "--output", str(crop_output),
        "--padding", str(args.padding),
    ], f"Stage 1 예측 bbox로 크롭 생성: {crop_output}", cwd=ROOT)

    # Stage 2 추론
    run_command([
        sys.executable, str(SCRIPTS / "pipeline" / "stage2_predict.py"),
        "--config", args.stage2_config,
        "--weights", str(stage2_weights),
        "--source", str(crop_output),
        "--output", str(stage2_predictions),
    ], f"Stage 2 분류기 추론: {crop_output} → {stage2_predictions}", cwd=ROOT)

    # 병합 및 제출 파일 생성
    print("[실행] Stage 2 결과 병합 및 제출 파일 생성 중...")
    merged_predictions = merge_predictions(manifest_path, stage2_predictions)
    output_path = resolve_project_path(args.output)
    save_submission(merged_predictions, output_path)
    print(f"[완료] submission.csv 생성 → {output_path}")


if __name__ == "__main__":
    main()
