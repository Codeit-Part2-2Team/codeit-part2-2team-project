"""submission 유틸 단위 테스트."""

from pathlib import Path

import pandas as pd

from src.utils.submission import predictions_to_df, save_submission


def test_submission_columns(sample_predictions):
    """DataFrame에 올바른 컬럼이 있어야 한다."""
    df = predictions_to_df(sample_predictions)
    expected = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    assert list(df.columns) == expected


def test_submission_row_count(sample_predictions):
    """전체 검출 수만큼 행이 생성된다."""
    df = predictions_to_df(sample_predictions)
    total_detections = sum(len(item["detections"]) for item in sample_predictions)
    assert len(df) == total_detections


def test_submission_annotation_id_sequential(sample_predictions):
    """annotation_id가 1부터 순차 증가해야 한다."""
    df = predictions_to_df(sample_predictions)
    assert list(df["annotation_id"]) == list(range(1, len(df) + 1))


def test_submission_bbox_xywh_conversion(sample_predictions):
    """bbox가 xyxy → xywh로 올바르게 변환된다."""
    df = predictions_to_df(sample_predictions)
    first = df.iloc[0]
    # 픽스처의 첫 번째 검출: bbox [156, 247, 367, 703]
    assert first["bbox_x"] == 156
    assert first["bbox_y"] == 247
    assert first["bbox_w"] == 367 - 156
    assert first["bbox_h"] == 703 - 247


def test_submission_category_id(sample_predictions):
    """category_id가 class_id와 일치해야 한다."""
    df = predictions_to_df(sample_predictions)
    assert df.iloc[0]["category_id"] == 1
    assert df.iloc[1]["category_id"] == 24


def test_submission_empty_detection(sample_predictions):
    """검출이 없는 이미지는 행이 생성되지 않는다."""
    df = predictions_to_df(sample_predictions)
    assert "3" not in df["image_id"].astype(str).values


def test_save_submission_writes_csv(sample_predictions, tmp_path):
    """save_submission이 CSV 파일을 올바르게 저장한다."""
    out = tmp_path / "sub" / "submission.csv"
    save_submission(sample_predictions, out)

    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == sum(len(item["detections"]) for item in sample_predictions)
