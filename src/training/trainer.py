"""학습·검증 퍼사드. Ultralytics를 직접 임포트하지 않는다."""

from pathlib import Path


class Trainer:
    """model_yolo (또는 미래 모델)을 경유해 학습·검증을 실행하는 퍼사드."""

    def __init__(self, model):
        self.model = model

    def train(self, data_yaml: str | Path | None = None, resume: bool = False) -> dict:
        """학습 실행 후 mAP 지표를 반환한다.

        Returns:
            {"mAP50": float, "mAP50_95": float}
        """
        kwargs = self._build_train_kwargs(
            str(Path(data_yaml or self.model.cfg["data"]["yaml"]).resolve()), resume
        )
        results = self.model.raw_train(**kwargs)
        return {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }

    def validate(self, data_yaml: str | Path | None = None) -> dict:
        """검증 실행 후 mAP 지표를 반환한다.

        Returns:
            {"mAP50": float, "mAP50_95": float}
        """
        data_yaml = str(Path(data_yaml or self.model.cfg["data"]["yaml"]).resolve())
        results = self.model.raw_val(data=data_yaml)
        return {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }

    def _build_train_kwargs(self, data_yaml: str, resume: bool) -> dict:
        """cfg 섹션을 병합해 raw_train()에 넘길 dict를 생성한다."""
        cfg = self.model.cfg
        # cfg["train"]에 device가 포함되므로 별도로 추가하지 않는다.
        return {
            **cfg["train"],
            **cfg["augment"],
            "data": data_yaml,
            "imgsz": cfg["data"]["imgsz"],
            "workers": cfg["data"]["workers"],
            "conf": cfg["val"]["conf"],
            "iou": cfg["val"]["iou"],
            "max_det": cfg["val"]["max_det"],
            "project": cfg["output"]["project"],
            "name": cfg["output"]["name"],
            "save_period": cfg["output"].get("save_period", -1),
            "seed": cfg.get("seed", 42),
            "exist_ok": True,
            "resume": resume,
        }
