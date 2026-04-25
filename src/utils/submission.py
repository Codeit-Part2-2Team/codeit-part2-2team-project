"""제출 파일 변환 유틸. crops_manifest + stage2_preds → Kaggle submission CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def merge_predictions(
    manifest_path: str | Path,
    stage2_predictions_path: str | Path,
) -> list[dict]:
    """crops_manifest + stage2 predictions → image_id별 detections 병합.

    Returns:
        [{"image_id": str, "detections": [{"class_id", "class_name", "bbox", "score"}]}]
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


def predictions_to_df(predictions: list[dict]) -> pd.DataFrame:
    """predictions 리스트 → Kaggle submission DataFrame."""
    rows = []
    annotation_id = 1
    for item in predictions:
        image_id = item["image_id"]
        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": det["class_id"],
                    "bbox_x": round(x1),
                    "bbox_y": round(y1),
                    "bbox_w": round(x2 - x1),
                    "bbox_h": round(y2 - y1),
                    "score": det["score"],
                }
            )
            annotation_id += 1
    return pd.DataFrame(rows)


def save_submission(predictions: list[dict], output: str | Path) -> None:
    """predictions → CSV 저장."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions_to_df(predictions).to_csv(out, index=False)
