"""submission 유틸 단위 테스트."""

import json

import pandas as pd

from src.utils.submission import merge_predictions, predictions_to_df, save_submission


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


def test_submission_skips_classes_missing_from_class_map(sample_predictions):
    """Kaggle class_map 밖의 detection은 내부 class_id로 제출하지 않는다."""
    df = predictions_to_df(
        sample_predictions,
        class_map={"Crestor 10mg tab": 16262},
    )

    assert len(df) == 1
    assert df.iloc[0]["category_id"] == 16262
    assert df.attrs["skipped_unknown_classes"] == {"Aspirin 100mg tab": 1}


def test_submission_strict_class_map_raises(sample_predictions):
    """strict 모드에서는 class_map 밖 class_name을 에러로 잡는다."""
    import pytest

    with pytest.raises(KeyError):
        predictions_to_df(
            sample_predictions,
            class_map={"Crestor 10mg tab": 16262},
            strict_class_map=True,
        )


def test_submission_remaps_classes_missing_from_class_map(sample_predictions):
    """unknown_class_map이 있으면 map 밖 class_name을 Kaggle class로 치환한다."""
    df = predictions_to_df(
        sample_predictions,
        class_map={
            "Crestor 10mg tab": 16262,
            "Mapped Aspirin": 1900,
        },
        unknown_class_map={"Aspirin 100mg tab": "Mapped Aspirin"},
    )

    assert len(df) == 2
    assert list(df["category_id"]) == [16262, 1900]
    assert df.attrs["skipped_unknown_classes"] == {}
    assert df.attrs["remapped_unknown_classes"] == {"Aspirin 100mg tab": 1}


def test_merge_predictions_uses_manifest_image_id_for_inference_crops(tmp_path):
    """Inference crop id(1_0)가 아니라 원본 test image id(1)로 제출해야 한다."""
    manifest = [
        {
            "image_id": "1",
            "crop_id": "1_0",
            "crop_path": "crops/1_0.jpg",
            "bbox": [10.0, 20.0, 50.0, 80.0],
            "score": 0.5,
        },
        {
            "image_id": "1",
            "crop_id": "1_1",
            "crop_path": "crops/1_1.jpg",
            "bbox": [60.0, 30.0, 100.0, 90.0],
            "score": 0.8,
        },
    ]
    stage2_preds = [
        {
            "image_id": "1_0",
            "crop_id": "1_0",
            "class_id": 3,
            "class_name": "pill_a",
            "score": 0.9,
        },
        {
            "image_id": "1_1",
            "crop_id": "1_1",
            "class_id": 4,
            "class_name": "pill_b",
            "score": 0.7,
        },
    ]
    manifest_path = tmp_path / "crops_manifest.json"
    preds_path = tmp_path / "stage2_predictions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preds_path.write_text(json.dumps(stage2_preds), encoding="utf-8")

    merged = merge_predictions(manifest_path, preds_path)
    df = predictions_to_df(merged)

    assert len(merged) == 1
    assert merged[0]["image_id"] == "1"
    assert len(merged[0]["detections"]) == 2
    assert list(df["image_id"].astype(str)) == ["1", "1"]
