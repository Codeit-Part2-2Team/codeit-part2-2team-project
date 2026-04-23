"""Stage 2 Classifier 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from PIL import Image

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


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_top_k_metrics(fake_timm, sample_stage2_config):
    """evaluate()가 cfg top_k 키에 따른 정확도 dict를 반환한다."""
    classifier = Classifier(sample_stage2_config)

    # 3-class identity logits: row i → class i has highest score
    fixed_logits = torch.eye(3)
    mock_model = MagicMock(return_value=fixed_logits)
    mock_model.eval = MagicMock()
    classifier.model = mock_model

    images = torch.zeros(3, 3, 224, 224)
    labels = torch.tensor([0, 1, 2])
    loader = [(images, labels)]

    metrics = classifier.evaluate(loader)

    # top_k=[1,3], all 3 predictions correct
    assert metrics["top1_acc"] == pytest.approx(1.0)
    assert metrics["top3_acc"] == pytest.approx(1.0)
    assert set(metrics.keys()) == {"top1_acc", "top3_acc"}


def test_evaluate_partial_accuracy(fake_timm, sample_stage2_config):
    """evaluate()가 부분 정확도를 올바르게 계산한다."""
    classifier = Classifier(sample_stage2_config)

    # row 0 predicts class 0, row 1 predicts class 1, row 2 predicts class 2
    fixed_logits = torch.eye(3)
    mock_model = MagicMock(return_value=fixed_logits)
    mock_model.eval = MagicMock()
    classifier.model = mock_model

    images = torch.zeros(3, 3, 224, 224)
    labels = torch.tensor([0, 2, 1])  # 1/3 top-1 correct, 3/3 top-3 correct
    loader = [(images, labels)]

    metrics = classifier.evaluate(loader)

    assert metrics["top1_acc"] == pytest.approx(1 / 3)
    assert metrics["top3_acc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# predict_loader
# ---------------------------------------------------------------------------


def test_predict_loader_returns_correct_structure(fake_timm, sample_stage2_config):
    """predict_loader()가 class_id / class_name / score 필드를 올바르게 채운다."""
    classifier = Classifier(sample_stage2_config)
    classifier.class_names = ["class_a", "class_b", "class_c"]

    logits = torch.tensor([[0.0, 10.0, 0.0]])
    mock_model = MagicMock(return_value=logits)
    mock_model.eval = MagicMock()
    classifier.model = mock_model

    batch = {
        "image": torch.zeros(1, 3, 224, 224),
        "image_id": ["img_001"],
        "crop_id": ["img_001_0"],
    }
    loader = [batch]

    results = classifier.predict_loader(loader)

    assert len(results) == 1
    r = results[0]
    assert r["image_id"] == "img_001"
    assert r["crop_id"] == "img_001_0"
    assert r["class_id"] == 1
    assert r["class_name"] == "class_b"
    assert r["score"] == pytest.approx(
        torch.softmax(logits, dim=1)[0, 1].item(), abs=1e-5
    )


def test_predict_loader_no_class_names_uses_index(fake_timm, sample_stage2_config):
    """class_names가 비어 있으면 class_id의 문자열을 class_name으로 사용한다."""
    classifier = Classifier(sample_stage2_config)
    classifier.class_names = []

    logits = torch.tensor([[0.0, 0.0, 5.0]])
    mock_model = MagicMock(return_value=logits)
    mock_model.eval = MagicMock()
    classifier.model = mock_model

    batch = {
        "image": torch.zeros(1, 3, 224, 224),
        "image_id": ["img_002"],
        "crop_id": ["img_002_0"],
    }

    results = classifier.predict_loader([batch])

    assert results[0]["class_id"] == 2
    assert results[0]["class_name"] == "2"


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_saves_checkpoints_and_returns_metrics(fake_timm, sample_stage2_config, tmp_path):
    """fit()이 best.pt / last.pt를 저장하고 top-k 지표 dict를 반환한다."""
    sample_stage2_config["output"]["project"] = str(tmp_path)
    sample_stage2_config["train"]["epochs"] = 2
    sample_stage2_config["train"]["warmup_epochs"] = 1

    classifier = Classifier(sample_stage2_config)
    classifier.model = nn.Linear(4, 3)

    images = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    loader = [(images, labels)]

    metrics = classifier.fit(loader, loader)

    weights_dir = tmp_path / "test_stage2" / "weights"
    assert (weights_dir / "best.pt").exists()
    assert (weights_dir / "last.pt").exists()
    assert "top1_acc" in metrics
    assert "top3_acc" in metrics
    assert 0.0 <= metrics["top1_acc"] <= 1.0


# ---------------------------------------------------------------------------
# train (high-level)
# ---------------------------------------------------------------------------


def test_train_calls_fit_and_returns_metrics(fake_timm, sample_stage2_config, monkeypatch, tmp_path):
    """train()이 Stage2Dataset을 생성하고 fit()을 호출한 뒤 결과를 반환한다."""
    sample_stage2_config["output"]["project"] = str(tmp_path)

    fake_ds = MagicMock()
    fake_ds.classes = ["class_a", "class_b", "class_c"]
    fake_ds.__len__ = MagicMock(return_value=2)
    fake_ds.__getitem__ = MagicMock(return_value=(torch.zeros(3, 224, 224), 0))
    monkeypatch.setattr("src.data.stage2_dataset.Stage2Dataset", lambda *a, **kw: fake_ds)

    expected = {"top1_acc": 0.9, "top3_acc": 1.0}

    classifier = Classifier(sample_stage2_config)
    classifier.fit = MagicMock(return_value=expected)

    result = classifier.train(data_dir=str(tmp_path))

    classifier.fit.assert_called_once()
    assert result == expected
    assert classifier.class_names == ["class_a", "class_b", "class_c"]


# ---------------------------------------------------------------------------
# predict (high-level)
# ---------------------------------------------------------------------------


def test_predict_writes_json_and_returns_results(fake_timm, sample_stage2_config, tmp_path):
    """predict()가 추론 결과를 JSON으로 저장하고 같은 결과를 반환한다."""
    source = tmp_path / "crops"
    source.mkdir()
    Image.new("RGB", (64, 64)).save(source / "img_001_0.jpg")

    output = tmp_path / "results.json"

    sample_stage2_config["train"]["batch"] = 1
    sample_stage2_config["data"]["workers"] = 0

    classifier = Classifier(sample_stage2_config)
    classifier.class_names = ["class_a", "class_b", "class_c"]

    expected = [
        {
            "image_id": "img_001",
            "crop_id": "img_001_0",
            "class_id": 0,
            "class_name": "class_a",
            "score": 0.95,
        }
    ]
    classifier.predict_loader = MagicMock(return_value=expected)

    results = classifier.predict(source, output)

    assert results == expected
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == expected
