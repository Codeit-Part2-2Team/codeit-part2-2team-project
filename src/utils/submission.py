"""제출 파일 변환 유틸. predictions.json → Kaggle submission CSV."""

from pathlib import Path

import pandas as pd


def predictions_to_df(predictions: list[dict]) -> pd.DataFrame:
    """predictions.json 리스트 → submission DataFrame (xywh, annotation_id 포함)."""
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
    """predictions_to_df 호출 후 CSV로 저장한다."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions_to_df(predictions).to_csv(out, index=False)
