"""YOLOModel 래퍼 단위 테스트."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.yolo_model import YOLOModel

# ---------------------------------------------------------------------------
# YOLOModel.predict
# ---------------------------------------------------------------------------


def _make_mock_box(x1, y1, x2, y2, score, cls_idx):
    """Ultralytics boxes 객체 하나를 흉내 낸 mock."""
    box = MagicMock()
    box.xyxy = [MagicMock()]
    box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
    box.conf = [score]
    box.cls = [cls_idx]
    return box


@patch("src.models.yolo_model.YOLO")
def test_predict_output_format(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 predictions.json 스키마를 올바르게 반환한다."""
    mock_result = MagicMock()
    mock_result.path = "/data/test/test_0001.jpg"
    mock_result.names = {0: "Crestor 10mg tab", 1: "Aspirin 100mg tab"}
    mock_result.boxes = [
        _make_mock_box(10, 20, 100, 200, 0.87, 0),
        _make_mock_box(150, 50, 300, 180, 0.72, 1),
    ]

    mock_yolo_cls.return_value.predict.return_value = [mock_result]

    output_path = tmp_path / "predictions.json"
    model = YOLOModel(sample_config)
    predictions = model.predict("data/raw/test/", output=output_path)

    assert len(predictions) == 1
    assert predictions[0]["image_id"] == "test_0001"
    assert len(predictions[0]["detections"]) == 2

    det = predictions[0]["detections"][0]
    assert det["class_name"] == "Crestor 10mg tab"
    assert det["bbox"] == [10, 20, 100, 200]
    assert det["score"] == pytest.approx(0.87)


@patch("src.models.yolo_model.YOLO")
def test_predict_writes_json(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 predictions.json 파일을 실제로 저장한다."""
    mock_result = MagicMock()
    mock_result.path = "/data/test/test_0002.jpg"
    mock_result.names = {}
    mock_result.boxes = None  # 검출 없음

    mock_yolo_cls.return_value.predict.return_value = [mock_result]

    output_path = tmp_path / "out" / "predictions.json"
    YOLOModel(sample_config).predict("data/raw/test/", output=output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved[0]["image_id"] == "test_0002"
    assert saved[0]["detections"] == []


@patch("src.models.yolo_model.YOLO")
def test_predict_val_params_passed(mock_yolo_cls, sample_config, tmp_path):
    """predict()가 config의 val 파라미터를 YOLO.predict에 전달한다."""
    mock_yolo_cls.return_value.predict.return_value = []

    YOLOModel(sample_config).predict("data/raw/test/", output=tmp_path / "p.json")

    call_kwargs = mock_yolo_cls.return_value.predict.call_args.kwargs
    assert call_kwargs["conf"] == sample_config["val"]["conf"]
    assert call_kwargs["iou"] == sample_config["val"]["iou"]
