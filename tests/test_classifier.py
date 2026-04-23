"""Stage 2 Classifier 단위 테스트."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.models.classifier import Classifier


class _FakeModel:
    def __init__(self):
        self.device = None
        self.loaded_state_dict = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def eval(self):
        self.eval_called = True
        return self


@pytest.fixture
def fake_timm(monkeypatch):
    fake_model = _FakeModel()
    fake_module = SimpleNamespace(create_model=MagicMock(return_value=fake_model))
    monkeypatch.setattr("src.models.classifier.import_module", lambda _: fake_module)
    return fake_module, fake_model


def test_classifier_init_builds_timm_model(fake_timm, sample_stage2_config):
    fake_module, fake_model = fake_timm

    classifier = Classifier(sample_stage2_config)

    fake_module.create_model.assert_called_once_with(
        "efficientnetv2_rw_s",
        pretrained=False,
        num_classes=3,
    )
    assert classifier.model is fake_model
    assert classifier.device == "cpu"
    assert classifier.class_names == ["class_a", "class_b", "class_c"]


def test_classifier_load_weights_returns_self(fake_timm, sample_stage2_config, monkeypatch):
    _, fake_model = fake_timm
    checkpoint = {
        "model_state_dict": {"layer.weight": torch.tensor([1.0])},
        "class_names": ["drug_a", "drug_b"],
    }
    monkeypatch.setattr("src.models.classifier.torch.load", lambda *args, **kwargs: checkpoint)

    classifier = Classifier(sample_stage2_config)
    result = classifier.load_weights("weights/best.pt")

    assert result is classifier
    assert fake_model.loaded_state_dict == checkpoint["model_state_dict"]
    assert classifier.class_names == ["drug_a", "drug_b"]


def test_classifier_raises_on_unsupported_model_name(fake_timm, sample_stage2_config):
    sample_stage2_config["model"]["name"] = "convnext_tiny"

    with pytest.raises(ValueError, match="지원하지 않는 Stage 2 모델"):
        Classifier(sample_stage2_config)


def test_classifier_export_saves_onnx(fake_timm, sample_stage2_config, monkeypatch, tmp_path):
    _, fake_model = fake_timm
    export_calls = {}

    def _fake_export(model, dummy, output_path, opset_version):
        export_calls["model"] = model
        export_calls["dummy_shape"] = tuple(dummy.shape)
        export_calls["output_path"] = output_path
        export_calls["opset_version"] = opset_version

    sample_stage2_config["output"]["project"] = str(tmp_path)
    classifier = Classifier(sample_stage2_config)
    monkeypatch.setattr("src.models.classifier.torch.onnx.export", _fake_export)

    output_path = classifier.export()

    assert fake_model.eval_called is True
    assert export_calls["model"] is fake_model
    assert export_calls["dummy_shape"] == (1, 3, 224, 224)
    assert export_calls["opset_version"] == 17
    assert export_calls["output_path"] == output_path
    assert output_path == (
        Path(tmp_path)
        / "test_stage2"
        / "weights"
        / "efficientnetv2_s.onnx"
    )
