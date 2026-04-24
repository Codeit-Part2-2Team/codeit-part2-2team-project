"""Ultralytics YOLO 구체 구현. predictor/trainer가 경유하는 모델 레이어."""

from pathlib import Path

from ultralytics import YOLO

from src.utils.config import fix_seed, load_config, validate_config


class YOLOModel:
    """config 기반 YOLO 구체 구현체."""

    def __init__(self, config: str | Path | dict):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        validate_config(self.cfg)
        fix_seed(self.cfg.get("seed", 42))
        model_cfg = self.cfg["model"]
        weights = (
            f"{model_cfg['name']}.pt"
            if model_cfg.get("pretrained", True)
            else model_cfg["name"]
        )
        self.model = YOLO(weights)

    def load_weights(self, path: str | Path) -> "YOLOModel":
        self.model = YOLO(str(path))
        return self

    def raw_predict(
        self, source: str, conf: float, iou: float, max_det: int, augment: bool
    ):
        """Ultralytics Result 리스트를 그대로 반환. 포맷 변환은 Predictor가 담당."""
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            augment=augment,
            save=False,
        )

    def raw_train(self, **kwargs):
        """YOLO.train() 결과 객체를 그대로 반환. 지표 파싱은 Trainer가 담당."""
        return self.model.train(**kwargs)

    def raw_val(self, **kwargs):
        """YOLO.val() 결과 객체를 그대로 반환."""
        return self.model.val(**kwargs)

    def export(self, format: str = "onnx") -> None:
        self.model.export(format=format, imgsz=self.cfg["data"]["imgsz"])
