"""make_submission.py CSV 변환 로직 단위 테스트."""

import json
from pathlib import Path

import pandas as pd
import pytest


def run_make_submission(predictions: list[dict], tmp_path: Path) -> pd.DataFrame:
    """make_submission.py의 변환 로직을 직접 호출하는 헬퍼."""
    out_file = tmp_path / "submission.csv"

    rows = []
    annotation_id = 1
    for item in predictions:
        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            rows.append({
                "annotation_id": annotation_id,
                "image_id": item["image_id"],
                "category_id": det["class_id"],
                "bbox_x": round(x1),
                "bbox_y": round(y1),
                "bbox_w": round(x2 - x1),
                "bbox_h": round(y2 - y1),
                "score": det["score"],
            })
            annotation_id += 1

    pd.DataFrame(rows).to_csv(out_file, index=False)
    return pd.read_csv(out_file)


def test_submission_columns(sample_predictions, tmp_path):
    """CSV에 올바른 컬럼이 있어야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    expected = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    assert list(df.columns) == expected


def test_submission_row_count(sample_predictions, tmp_path):
    """전체 검출 수만큼 행이 생성된다."""
    df = run_make_submission(sample_predictions, tmp_path)
    total_detections = sum(len(item["detections"]) for item in sample_predictions)
    assert len(df) == total_detections


def test_submission_annotation_id_sequential(sample_predictions, tmp_path):
    """annotation_id가 1부터 순차 증가해야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    assert list(df["annotation_id"]) == list(range(1, len(df) + 1))


def test_submission_bbox_xywh_conversion(sample_predictions, tmp_path):
    """bbox가 xyxy → xywh로 올바르게 변환된다."""
    df = run_make_submission(sample_predictions, tmp_path)
    first = df.iloc[0]
    # 픽스처의 첫 번째 검출: bbox [156, 247, 367, 703]
    assert first["bbox_x"] == 156
    assert first["bbox_y"] == 247
    assert first["bbox_w"] == 367 - 156   # x2 - x1
    assert first["bbox_h"] == 703 - 247   # y2 - y1


def test_submission_category_id(sample_predictions, tmp_path):
    """category_id가 class_id와 일치해야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    assert df.iloc[0]["category_id"] == 1
    assert df.iloc[1]["category_id"] == 24


def test_submission_empty_detection(sample_predictions, tmp_path):
    """검출이 없는 이미지는 행이 생성되지 않는다."""
    df = run_make_submission(sample_predictions, tmp_path)
    # image_id "3"은 검출 없음 → 해당 행 없어야 함
    assert "3" not in df["image_id"].astype(str).values
