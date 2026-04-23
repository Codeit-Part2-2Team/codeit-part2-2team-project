"""학습·검증 퍼사드. Ultralytics를 직접 임포트하지 않는다."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# albumentations가 담당하는 항목 — YOLO augment와 중복 적용 방지
_ALBUMENTATIONS_OVERRIDE = (
    "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear",
)


class Trainer:
    """model_yolo (또는 미래 모델)을 경유해 학습·검증을 실행하는 퍼사드."""

    def __init__(self, model):
        self.model = model

    def train(self, data_yaml: str | Path | None = None, resume: bool = False) -> dict:
        """학습 실행 후 mAP 지표를 반환한다.

        Returns:
            {"mAP50": float, "mAP50_95": float}
        """
        if self.model.cfg.get("albumentations"):
            self._register_albumentations_callback()
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

    def _register_albumentations_callback(self) -> None:
        """albumentations config 기반 커스텀 transform을 YOLO 학습 루프에 주입한다."""
        from src.data.augmentations import build_stage1_transforms

        compose = build_stage1_transforms(self.model.cfg)

        def on_train_start(ultralytics_trainer):
            _inject_into_dataset(ultralytics_trainer.train_loader.dataset, compose)

        self.model.model.add_callback("on_train_start", on_train_start)

    def _build_train_kwargs(self, data_yaml: str, resume: bool) -> dict:
        """cfg 섹션을 병합해 raw_train()에 넘길 dict를 생성한다."""
        cfg = self.model.cfg
        # cfg["train"]에 device가 포함되므로 별도로 추가하지 않는다.
        kwargs = {
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
        if cfg.get("albumentations"):
            for key in _ALBUMENTATIONS_OVERRIDE:
                kwargs[key] = 0.0
        return kwargs


def _inject_into_dataset(dataset, compose) -> None:
    """Ultralytics dataset의 Albumentations transform을 우리 compose로 교체한다.

    dataset.transforms.transforms 리스트에서 Albumentations 인스턴스를 찾아
    _AlbumentationsAdapter로 교체한다. 없으면 리스트 끝에 추가한다.
    Ultralytics 내부 구조에 의존하므로 버전이 달라지면 동작하지 않을 수 있다.
    """
    transforms_obj = getattr(dataset, "transforms", None)
    if transforms_obj is None:
        return
    transforms_list = getattr(transforms_obj, "transforms", None)
    if not isinstance(transforms_list, list):
        return
    adapter = _AlbumentationsAdapter(compose)
    for i, t in enumerate(transforms_list):
        if t.__class__.__name__ == "Albumentations":
            transforms_list[i] = adapter
            return
    transforms_list.append(adapter)


class _AlbumentationsAdapter:
    """Ultralytics labels dict ↔ albumentations Compose 어댑터.

    Ultralytics Albumentations.__call__과 동일한 labels dict 포맷을 따른다.
    bboxes는 normalized xywh(yolo) 포맷으로 전달된다.
    """

    def __init__(self, compose):
        self.transform = compose

    def __call__(self, labels: dict) -> dict:
        im = labels["img"]
        cls = labels["cls"]
        if not len(cls):
            return labels
        labels["instances"].convert_bbox("xywh")
        labels["instances"].normalize(*im.shape[:2][::-1])
        bboxes = labels["instances"].bboxes
        result = self.transform(image=im, bboxes=bboxes, class_labels=cls)
        if result["class_labels"]:
            labels["img"] = result["image"]
            labels["cls"] = np.array(result["class_labels"])
            labels["instances"].update(bboxes=np.array(result["bboxes"], dtype=np.float32))
        return labels
