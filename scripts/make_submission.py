# isort: skip_file
# ruff: noqa: E402
"""
Kaggle 제출 CSV 생성 CLI.

crops_manifest.json + stage2_predictions.json → submission.csv

submission.csv 포맷:
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    1, 1, 1, 156, 247, 211, 456, 0.91

score = det_score (Stage 1) × cls_score (Stage 2).

사용 예:
    python scripts/make_submission.py \\
        --manifest  experiments/.../stage1_crops/crops_manifest.json \\
        --s2-preds  experiments/.../stage2_predictions.json \\
        --output    submissions/submission.csv

    # Kaggle category_id 매핑 적용 (class_name → category_id)
    python scripts/make_submission.py ... \\
        --class-map data/processed/kaggle_class_map.json

    # Kaggle image_id 매핑 적용 (filename_stem → numeric id)
    python scripts/make_submission.py ... \\
        --image-id-map data/processed/kaggle_image_id_map.json
"""

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

from src.utils.submission import merge_predictions, save_submission


def _load_json_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Kaggle 제출 CSV 생성")
    parser.add_argument("--manifest", required=True, help="crops_manifest.json 경로")
    parser.add_argument(
        "--s2-preds", required=True, help="stage2_predictions.json 경로"
    )
    parser.add_argument("--output", required=True, help="submission.csv 저장 경로")
    parser.add_argument(
        "--class-map",
        default=None,
        help="Kaggle category_id 매핑 JSON {class_name: category_id} (미지정 시 내부 class_id 사용)",
    )
    parser.add_argument(
        "--image-id-map",
        default=None,
        help="Kaggle image_id 매핑 JSON {filename_stem: int} (미지정 시 파일명 그대로 사용)",
    )
    parser.add_argument(
        "--unknown-class-map",
        default=None,
        help="class_map 밖 예측 class_name을 Kaggle class_name/category_id로 치환하는 JSON",
    )
    parser.add_argument(
        "--strict-class-map",
        action="store_true",
        help="Raise an error instead of skipping class names missing from class_map.",
    )
    args = parser.parse_args()

    class_map = _load_json_map(args.class_map) if args.class_map else None
    image_id_map = _load_json_map(args.image_id_map) if args.image_id_map else None
    unknown_class_map = (
        _load_json_map(args.unknown_class_map) if args.unknown_class_map else None
    )

    predictions = merge_predictions(args.manifest, args.s2_preds)
    save_submission(
        predictions,
        args.output,
        class_map=class_map,
        image_id_map=image_id_map,
        unknown_class_map=unknown_class_map,
        strict_class_map=args.strict_class_map,
    )
    print(f"제출 파일 생성 완료: {args.output}")


if __name__ == "__main__":
    main()
