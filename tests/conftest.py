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
def sample_stage2_config() -> dict:
    """테스트용 최소 Stage 2 config 딕셔너리."""
    return {
        "seed": 42,
        "stage": 2,
        "task": "classification",
        "classes": ["class_a", "class_b", "class_c"],
        "model": {
            "name": "efficientnetv2_s",
            "pretrained": False,
            "num_classes": 3,
        },
        "data": {
            "train": "data/processed/crops/train",
            "val": "data/processed/crops/val",
            "imgsz": 224,
            "workers": 2,
        },
        "train": {
            "epochs": 3,
            "batch": 8,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 1,
            "device": "cpu",
        },
        "val": {"top_k": [1, 3]},
        "output": {
            "project": "experiments/stage2_classifier",
            "name": "test_stage2",
            "save_period": -1,
        },
        "albumentations": {
            "brightness_contrast": {
                "p": 0.5,
                "brightness_limit": 0.15,
                "contrast_limit": 0.15,
            }
        },
    }


@pytest.fixture
def sample_predictions() -> list[dict]:
    """테스트용 predictions.json 데이터."""
    return [
        {
            "image_id": "1",
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "Crestor 10mg tab",
                    "bbox": [156.0, 247.0, 367.0, 703.0],  # xyxy
                    "score": 0.91,
                },
                {
                    "class_id": 24,
                    "class_name": "Aspirin 100mg tab",
                    "bbox": [498.0, 40.0, 958.0, 514.0],  # xyxy
                    "score": 0.78,
                },
            ],
        },
        {
            "image_id": "3",
            "detections": [],  # 검출 없음
        },
    ]
