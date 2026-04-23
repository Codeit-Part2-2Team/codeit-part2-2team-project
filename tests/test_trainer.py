"""Trainer 퍼사드 단위 테스트."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.trainer import Trainer


def _make_model(sample_config):
    model = MagicMock()
    model.cfg = sample_config
    return model


# ---------------------------------------------------------------------------
# _build_train_kwargs
# ---------------------------------------------------------------------------


def test_build_train_kwargs_contains_required_keys(sample_config):
    """_build_train_kwargs가 학습에 필요한 모든 키를 포함한다."""
    trainer = Trainer(_make_model(sample_config))
    kwargs = trainer._build_train_kwargs("data/processed/dataset.yaml", resume=False)

    for key in ("data", "imgsz", "workers", "epochs", "batch", "seed", "exist_ok", "resume"):
        assert key in kwargs, f"'{key}' 키가 없습니다"


def test_build_train_kwargs_no_duplicate_device(sample_config):
    """cfg['train']에 device가 있을 때 kwargs에 중복 없이 한 번만 포함된다."""
    sample_config["train"]["device"] = "cuda:0"
    trainer = Trainer(_make_model(sample_config))
    kwargs = trainer._build_train_kwargs("data/processed/dataset.yaml", resume=False)

    assert kwargs["device"] == "cuda:0"
    assert list(kwargs.keys()).count("device") == 1


def test_build_train_kwargs_seed_from_config(sample_config):
    """seed가 cfg에서 올바르게 주입된다."""
    sample_config["seed"] = 99
    trainer = Trainer(_make_model(sample_config))
    kwargs = trainer._build_train_kwargs("data/processed/dataset.yaml", resume=False)

    assert kwargs["seed"] == 99


def test_build_train_kwargs_resume_flag(sample_config):
    """resume 인자가 그대로 전달된다."""
    trainer = Trainer(_make_model(sample_config))
    assert trainer._build_train_kwargs("x.yaml", resume=True)["resume"] is True
    assert trainer._build_train_kwargs("x.yaml", resume=False)["resume"] is False


def test_build_train_kwargs_albumentations_override(sample_config):
    """albumentations 섹션이 있으면 YOLO 중복 augment 항목이 0.0으로 덮어씌워진다."""
    sample_config["albumentations"] = {"horizontal_flip": {"p": 0.5}}
    sample_config["augment"]["fliplr"] = 0.5
    sample_config["augment"]["hsv_h"] = 0.015
    trainer = Trainer(_make_model(sample_config))
    kwargs = trainer._build_train_kwargs("x.yaml", resume=False)

    for key in ("fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear"):
        assert kwargs[key] == 0.0, f"{key}가 0.0이어야 합니다"


def test_build_train_kwargs_no_albumentations_keeps_augment_values(sample_config):
    """albumentations 섹션이 없으면 augment 원래 값이 유지된다."""
    sample_config["augment"]["fliplr"] = 0.5
    sample_config["augment"]["hsv_h"] = 0.015
    trainer = Trainer(_make_model(sample_config))
    kwargs = trainer._build_train_kwargs("x.yaml", resume=False)

    assert kwargs["fliplr"] == 0.5
    assert kwargs["hsv_h"] == 0.015


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_returns_map_dict(sample_config):
    """train()이 mAP50 / mAP50_95 키를 가진 dict를 반환한다."""
    model = _make_model(sample_config)
    mock_results = MagicMock()
    mock_results.results_dict = {
        "metrics/mAP50(B)": 0.75,
        "metrics/mAP50-95(B)": 0.55,
    }
    model.raw_train.return_value = mock_results

    result = Trainer(model).train("data/processed/dataset.yaml")

    assert result == pytest.approx({"mAP50": 0.75, "mAP50_95": 0.55})


def test_train_missing_metric_defaults_to_zero(sample_config):
    """결과 dict에 지표가 없으면 0.0을 반환한다."""
    model = _make_model(sample_config)
    model.raw_train.return_value.results_dict = {}

    result = Trainer(model).train("data/processed/dataset.yaml")

    assert result == {"mAP50": 0.0, "mAP50_95": 0.0}


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_returns_map_dict(sample_config):
    """validate()가 mAP50 / mAP50_95 키를 가진 dict를 반환한다."""
    model = _make_model(sample_config)
    mock_results = MagicMock()
    mock_results.results_dict = {
        "metrics/mAP50(B)": 0.80,
        "metrics/mAP50-95(B)": 0.60,
    }
    model.raw_val.return_value = mock_results

    result = Trainer(model).validate("data/processed/dataset.yaml")

    assert result == pytest.approx({"mAP50": 0.80, "mAP50_95": 0.60})


def test_validate_passes_data_yaml_to_raw_val(sample_config):
    """validate()가 data_yaml을 raw_val에 data= 인자로 전달한다."""
    model = _make_model(sample_config)
    model.raw_val.return_value.results_dict = {}

    Trainer(model).validate("data/processed/dataset.yaml")

    call_kwargs = model.raw_val.call_args.kwargs
    assert Path(call_kwargs["data"]).name == "dataset.yaml"
