"""YOLOModel (model_yolo) 단위 테스트."""

from unittest.mock import MagicMock, patch

import yaml

# ---------------------------------------------------------------------------
# YOLOModel.__init__
# ---------------------------------------------------------------------------


@patch("src.models.model_yolo.YOLO")
def test_init_with_dict_config(mock_yolo, sample_config):
    """dict config을 넘기면 YOLO가 올바른 가중치명으로 초기화된다."""
    from src.models.model_yolo import YOLOModel

    YOLOModel(sample_config)
    mock_yolo.assert_called_once_with("yolo26n")


@patch("src.models.model_yolo.YOLO")
def test_init_pretrained_appends_pt(mock_yolo, sample_config):
    """pretrained=True 이면 모델명에 .pt를 붙여 허브 가중치를 요청한다."""
    from src.models.model_yolo import YOLOModel

    sample_config["model"]["pretrained"] = True
    YOLOModel(sample_config)
    mock_yolo.assert_called_once_with("yolo26n.pt")


@patch("src.models.model_yolo.YOLO")
def test_init_with_yaml_file(mock_yolo, sample_config, tmp_path):
    """yaml 파일 경로로 초기화해도 동일하게 동작한다."""
    from src.models.model_yolo import YOLOModel

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(sample_config))

    YOLOModel(config_file)
    mock_yolo.assert_called_once_with("yolo26n")


# ---------------------------------------------------------------------------
# load_weights
# ---------------------------------------------------------------------------


@patch("src.models.model_yolo.YOLO")
def test_load_weights_returns_self(mock_yolo, sample_config):
    """load_weights()가 self를 반환해 체이닝이 가능하다."""
    from src.models.model_yolo import YOLOModel

    model = YOLOModel(sample_config)
    result = model.load_weights("experiments/best.pt")

    mock_yolo.return_value.load.assert_called_once_with("experiments/best.pt")
    assert result is model


# ---------------------------------------------------------------------------
# raw_predict
# ---------------------------------------------------------------------------


@patch("src.models.model_yolo.YOLO")
def test_raw_predict_passes_params(mock_yolo, sample_config):
    """raw_predict()가 인자를 그대로 YOLO.predict에 전달한다."""
    from src.models.model_yolo import YOLOModel

    mock_results = [MagicMock()]
    mock_yolo.return_value.predict.return_value = mock_results

    model = YOLOModel(sample_config)
    result = model.raw_predict(
        source="data/raw/test/", conf=0.25, iou=0.45, max_det=300, augment=False
    )

    mock_yolo.return_value.predict.assert_called_once_with(
        source="data/raw/test/",
        conf=0.25,
        iou=0.45,
        max_det=300,
        augment=False,
        save=False,
    )
    assert result is mock_results
