"""YOLOv26 래퍼 단위 테스트."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.models.yolo26 import YOLOv26, load_config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 42\nmodel:\n  name: yolo26n\n")

    cfg = load_config(config_file)

    assert cfg["seed"] == 42
    assert cfg["model"]["name"] == "yolo26n"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# YOLOv26.__init__
# ---------------------------------------------------------------------------

@patch("src.models.yolo26.YOLO")
def test_init_with_dict_config(mock_yolo, sample_config):
    """dict config을 넘기면 YOLO가 올바른 가중치명으로 초기화된다."""
    YOLOv26(sample_config)
    # pretrained=False → name 그대로 사용
    mock_yolo.assert_called_once_with("yolo26n")


@patch("src.models.yolo26.YOLO")
def test_init_pretrained_appends_pt(mock_yolo, sample_config):
    """pretrained=True 이면 모델명에 .pt를 붙여 허브 가중치를 요청한다."""
    sample_config["model"]["pretrained"] = True
    YOLOv26(sample_config)
    mock_yolo.assert_called_once_with("yolo26n.pt")


@patch("src.models.yolo26.YOLO")
def test_init_with_yaml_file(mock_yolo, sample_config, tmp_path):
    """yaml 파일 경로로 초기화해도 동일하게 동작한다."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(sample_config))

    YOLOv26(config_file)
    mock_yolo.assert_called_once_with("yolo26n")


# ---------------------------------------------------------------------------
# YOLOv26.predict
# ---------------------------------------------------------------------------

def _make_mock_box(x1, y1, x2, y2, score, cls_idx):
    """Ultralytics boxes 객체 하나를 흉내 낸 mock."""
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
    box.conf = [score]
    box.cls = [cls_idx]
    return box


@patch("src.models.yolo26.YOLO")
def test_predict_output_format(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 predictions.json 스키마를 올바르게 반환한다."""
    # YOLO 추론 결과 mock 구성
    mock_result = MagicMock()
    mock_result.path = "/data/test/test_0001.jpg"
    mock_result.names = {0: "Crestor 10mg tab", 1: "Aspirin 100mg tab"}
    mock_result.boxes = [
        _make_mock_box(10, 20, 100, 200, 0.87, 0),
        _make_mock_box(150, 50, 300, 180, 0.72, 1),
    ]

    mock_yolo_instance = mock_yolo_cls.return_value
    mock_yolo_instance.predict.return_value = [mock_result]

    output_path = tmp_path / "predictions.json"
    model = YOLOv26(sample_config)
    predictions = model.predict("data/raw/test/", output=output_path)

    # 반환값 검증
    assert len(predictions) == 1
    assert predictions[0]["image_id"] == "test_0001"
    assert len(predictions[0]["detections"]) == 2

    det = predictions[0]["detections"][0]
    assert det["class_name"] == "Crestor 10mg tab"
    assert det["bbox"] == [10, 20, 100, 200]
    assert det["score"] == pytest.approx(0.87)


@patch("src.models.yolo26.YOLO")
def test_predict_writes_json(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 predictions.json 파일을 실제로 저장한다."""
    mock_result = MagicMock()
    mock_result.path = "/data/test/test_0002.jpg"
    mock_result.names = {}
    mock_result.boxes = None  # 검출 없음

    mock_yolo_cls.return_value.predict.return_value = [mock_result]

    output_path = tmp_path / "out" / "predictions.json"
    YOLOv26(sample_config).predict("data/raw/test/", output=output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved[0]["image_id"] == "test_0002"
    assert saved[0]["detections"] == []


@patch("src.models.yolo26.YOLO")
def test_predict_val_params_passed(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 config의 val 파라미터를 YOLO.predict에 전달한다."""
    mock_yolo_cls.return_value.predict.return_value = []

    YOLOv26(sample_config).predict("data/raw/test/", output=tmp_path / "p.json")

    call_kwargs = mock_yolo_cls.return_value.predict.call_args.kwargs
    assert call_kwargs["conf"] == sample_config["val"]["conf"]
    assert call_kwargs["iou"] == sample_config["val"]["iou"]
