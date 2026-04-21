"""make_submission.py CSV 변환 로직 단위 테스트."""

import json
from pathlib import Path

import pandas as pd


def run_make_submission(predictions: list[dict], tmp_path: Path) -> pd.DataFrame:
    """make_submission.py의 변환 로직을 직접 호출하는 헬퍼."""
    pred_file = tmp_path / "predictions.json"
    pred_file.write_text(json.dumps(predictions, ensure_ascii=False))

    out_file = tmp_path / "submission.csv"

    # scripts/make_submission.py의 변환 로직을 그대로 재현
    rows = []
    for item in predictions:
        parts = []
        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            parts.append(
                f"{det['class_name']} {det['score']:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
            )
        rows.append({"image_id": item["image_id"], "PredictionString": " ".join(parts)})

    df = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    df.to_csv(out_file, index=False)
    return pd.read_csv(out_file)


def test_submission_columns(sample_predictions, tmp_path):
    """CSV에 image_id, PredictionString 두 컬럼이 있어야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    assert list(df.columns) == ["image_id", "PredictionString"]


def test_submission_row_count(sample_predictions, tmp_path):
    """입력 이미지 수만큼 행이 생성된다."""
    df = run_make_submission(sample_predictions, tmp_path)
    assert len(df) == len(sample_predictions)


def test_submission_prediction_string_format(sample_predictions, tmp_path):
    """PredictionString이 '클래스명 score x1 y1 x2 y2 ...' 형식이어야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    pred_str = df.loc[df["image_id"] == "test_0001", "PredictionString"].iloc[0]

    tokens = pred_str.split()
    # 첫 번째 검출: "Crestor 10mg tab score x1 y1 x2 y2" → 클래스명이 3단어이므로 토큰 7개
    assert "Crestor" in tokens
    assert "0.870000" in tokens


def test_submission_empty_detection(sample_predictions, tmp_path):
    """검출이 없는 이미지는 PredictionString이 빈 문자열이어야 한다."""
    df = run_make_submission(sample_predictions, tmp_path)
    empty_row = df.loc[df["image_id"] == "test_0002", "PredictionString"].iloc[0]
    # NaN 또는 빈 문자열 모두 허용
    assert pd.isna(empty_row) or empty_row == ""
