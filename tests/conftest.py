"""공통 픽스처."""

import pytest


@pytest.fixture
def sample_config() -> dict:
    """테스트용 최소 config 딕셔너리."""
    return {
        "seed": 42,
        "model": {"name": "yolo26n", "pretrained": False},
        "data": {"yaml": "data/processed/dataset.yaml", "imgsz": 640, "workers": 2},
        "train": {
            "epochs": 1,
            "batch": 2,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 0,
            "patience": 5,
        },
        "augment": {
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "val": {"conf": 0.25, "iou": 0.45, "max_det": 300},
        "output": {
            "project": "experiments",
            "name": "test_run",
            "save_period": -1,
        },
    }


@pytest.fixture
def sample_predictions() -> list[dict]:
    """테스트용 predictions.json 데이터."""
    return [
        {
            "image_id": "test_0001",
            "detections": [
                {"class_name": "Crestor 10mg tab", "bbox": [10.0, 20.0, 100.0, 200.0], "score": 0.87},
                {"class_name": "Aspirin 100mg tab", "bbox": [150.0, 50.0, 300.0, 180.0], "score": 0.72},
            ],
        },
        {
            "image_id": "test_0002",
            "detections": [],  # 검출 없음
        },
    ]
