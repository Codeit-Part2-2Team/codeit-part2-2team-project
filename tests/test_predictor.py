"""Predictor 퍼사드 단위 테스트."""

import json
from unittest.mock import MagicMock

import pytest

from src.models.predictor import Predictor


def _make_model(sample_config):
    model = MagicMock()
    model.cfg = sample_config
    return model


def _make_mock_result(path, names, boxes_data):
    """Ultralytics Result 객체를 흉내 낸 mock.

    boxes_data: [(x1, y1, x2, y2, score, cls_idx), ...] 또는 None
    """
    r = MagicMock()
    r.path = path
    r.names = names
    if boxes_data is None:
        r.boxes = None
    else:
        r.boxes = []
        for x1, y1, x2, y2, score, cls_idx in boxes_data:
            box = MagicMock()
            box.xyxy = [MagicMock()]
            box.xyxy[0].tolist.return_value = [x1, y1, x2, y2]
            box.conf = [score]
            box.cls = [cls_idx]
            r.boxes.append(box)
    return r


# ---------------------------------------------------------------------------
# _parse_results
# ---------------------------------------------------------------------------


def test_parse_results_correct_schema(sample_config):
    """_parse_results가 predictions.json 스키마를 올바르게 반환한다."""
    predictor = Predictor(_make_model(sample_config))
    raw = [
        _make_mock_result(
            "/data/test/test_0001.jpg",
            {0: "Crestor 10mg tab", 1: "Aspirin 100mg tab"},
            [(10, 20, 100, 200, 0.87, 0), (150, 50, 300, 180, 0.72, 1)],
        )
    ]

    result = predictor._parse_results(raw)

    assert len(result) == 1
    assert result[0]["image_id"] == "test_0001"
    assert len(result[0]["detections"]) == 2

    det = result[0]["detections"][0]
    assert det["class_name"] == "Crestor 10mg tab"
    assert det["bbox"] == [10, 20, 100, 200]
    assert det["score"] == pytest.approx(0.87)


def test_parse_results_no_boxes(sample_config):
    """boxes가 None이면 detections가 빈 리스트다."""
    predictor = Predictor(_make_model(sample_config))
    raw = [_make_mock_result("/data/test/test_0002.jpg", {}, None)]

    result = predictor._parse_results(raw)

    assert result[0]["detections"] == []


def test_parse_results_image_id_from_stem(sample_config):
    """image_id가 파일 경로에서 확장자 없이 추출된다."""
    predictor = Predictor(_make_model(sample_config))
    raw = [_make_mock_result("/some/path/img_0042.jpg", {}, None)]

    result = predictor._parse_results(raw)

    assert result[0]["image_id"] == "img_0042"


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


def test_predict_writes_json(sample_config, tmp_path):
    """predict()가 predictions.json 파일을 저장한다."""
    model = _make_model(sample_config)
    model.raw_predict.return_value = [
        _make_mock_result("/data/test/test_0003.jpg", {}, None)
    ]

    output_path = tmp_path / "out" / "predictions.json"
    Predictor(model).predict("data/raw/test/", output=output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved[0]["image_id"] == "test_0003"


def test_predict_passes_tta_as_augment(sample_config):
    """tta=True이면 raw_predict의 augment 인자가 True로 전달된다."""
    model = _make_model(sample_config)
    model.raw_predict.return_value = []

    Predictor(model).predict("data/raw/test/", output="/tmp/p.json", tta=True)

    call_kwargs = model.raw_predict.call_args.kwargs
    assert call_kwargs["augment"] is True


def test_predict_val_params_passed(sample_config):
    """predict()가 config의 val 파라미터를 raw_predict에 전달한다."""
    model = _make_model(sample_config)
    model.raw_predict.return_value = []

    Predictor(model).predict("data/raw/test/", output="/tmp/p.json")

    call_kwargs = model.raw_predict.call_args.kwargs
    assert call_kwargs["conf"] == sample_config["val"]["conf"]
    assert call_kwargs["iou"] == sample_config["val"]["iou"]
    assert call_kwargs["max_det"] == sample_config["val"]["max_det"]
