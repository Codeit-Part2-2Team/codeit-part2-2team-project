"""pipeline/run.py 유틸 함수 단위 테스트."""

import json

import pytest

from scripts.pipeline.run_predict import (
    merge_predictions,
)
from src.utils.path import (
    resolve_predictions_path,
    resolve_stage2_predictions_path,
    resolve_weights_path,
)


def test_resolve_stage1_paths_from_config():
    cfg = {"output": {"project": "experiments", "name": "exp_stage1"}}
    from src.utils.path import PROJECT_ROOT

    assert (
        resolve_weights_path(cfg, None)
        == PROJECT_ROOT / "experiments/exp_stage1/weights/best.pt"
    )
    assert (
        resolve_predictions_path(cfg, None)
        == PROJECT_ROOT / "experiments/exp_stage1/results/predictions.json"
    )


def test_resolve_stage2_paths_from_config():
    cfg = {"output": {"project": "experiments/stage2", "name": "exp_stage2"}}
    from src.utils.path import PROJECT_ROOT

    assert (
        resolve_weights_path(cfg, None)
        == PROJECT_ROOT / "experiments/stage2/exp_stage2/weights/best.pt"
    )
    assert (
        resolve_stage2_predictions_path(cfg, None)
        == PROJECT_ROOT
        / "experiments/stage2/exp_stage2/results/stage2_predictions.json"
    )


def test_merge_predictions_with_manifest(tmp_path):
    manifest = [
        {
            "image_id": "test_0001",
            "crop_id": "test_0001_0",
            "crop_path": "data/processed/crops/inference/test_0001_0.jpg",
            "bbox": [120.0, 45.0, 380.0, 210.0],
            "score": 0.91,
        },
        {
            "image_id": "test_0002",
            "crop_id": "test_0002_0",
            "crop_path": "data/processed/crops/inference/test_0002_0.jpg",
            "bbox": [40.0, 20.0, 180.0, 200.0],
            "score": 0.82,
        },
    ]
    stage2_predictions = [
        {
            "image_id": "test_0001",
            "crop_id": "test_0001_0",
            "class_id": 42,
            "class_name": "Crestor 10mg tab",
            "score": 0.912,
        }
    ]

    manifest_path = tmp_path / "crops_manifest.json"
    stage2_path = tmp_path / "stage2_predictions.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    stage2_path.write_text(
        json.dumps(stage2_predictions, ensure_ascii=False), encoding="utf-8"
    )

    merged = merge_predictions(manifest_path, stage2_path)
    assert len(merged) == 2
    assert merged[1] == {"image_id": "test_0002", "detections": []}

    det = merged[0]
    assert det["image_id"] == "test_0001"
    assert len(det["detections"]) == 1
    d = det["detections"][0]
    assert d["class_id"] == 42
    assert d["class_name"] == "Crestor 10mg tab"
    assert d["bbox"] == [120.0, 45.0, 380.0, 210.0]
    assert d["score"] == pytest.approx(0.91 * 0.912)
