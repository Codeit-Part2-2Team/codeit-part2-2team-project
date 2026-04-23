"""timm 기반 Stage 2 분류기 래퍼."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from src.utils.config import fix_seed, load_config, validate_config

_MODEL_NAME_MAP = {
    "resnet50": "resnet50",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnetv2_s": "efficientnetv2_rw_s",
}


class Classifier:
    """Stage 2 config를 받아 timm 분류기를 생성하는 래퍼."""

    def __init__(self, config: str | Path | dict):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        validate_config(self.cfg)
        fix_seed(self.cfg.get("seed", 42))

        self.device = _resolve_device(self.cfg["train"].get("device", ""))
        self.class_names = list(self.cfg.get("classes") or [])
        self.model = self._build_model().to(self.device)

    def load_weights(self, path: str | Path) -> "Classifier":
        """체크포인트 또는 state_dict를 로드한다."""
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.class_names = list(checkpoint.get("class_names", self.class_names))
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        return self

    def fit(self, train_loader, val_loader) -> dict:
        """학습 루프는 다음 단계에서 구현한다."""
        raise NotImplementedError("Classifier.fit() 학습 루프는 아직 구현되지 않았습니다.")

    def evaluate(self, val_loader) -> dict:
        """평가 루프는 다음 단계에서 구현한다."""
        raise NotImplementedError(
            "Classifier.evaluate() 평가 루프는 아직 구현되지 않았습니다."
        )

    def predict_loader(self, loader) -> list[dict]:
        """배치 추론 루프는 다음 단계에서 구현한다."""
        raise NotImplementedError(
            "Classifier.predict_loader() 추론 루프는 아직 구현되지 않았습니다."
        )

    def export(self, format: str = "onnx") -> Path:
        """현재 모델을 ONNX로 내보낸다."""
        if format != "onnx":
            raise ValueError(f"현재 지원하지 않는 export format입니다: {format}")

        output_path = self._build_export_path(format)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        imgsz = int(self.cfg["data"]["imgsz"])
        dummy = torch.randn(1, 3, imgsz, imgsz, device=self.device)
        self.model.eval()
        torch.onnx.export(self.model, dummy, output_path, opset_version=17)
        return output_path

    def _build_model(self) -> Any:
        timm = _load_timm()
        model_cfg = self.cfg["model"]
        model_name = _resolve_model_name(model_cfg["name"])
        return timm.create_model(
            model_name,
            pretrained=bool(model_cfg.get("pretrained", True)),
            num_classes=int(model_cfg["num_classes"]),
        )

    def _build_export_path(self, format: str) -> Path:
        output_cfg = self.cfg["output"]
        model_name = self.cfg["model"]["name"]
        return (
            Path(output_cfg["project"])
            / output_cfg["name"]
            / "weights"
            / f"{model_name}.{format}"
        )


def _load_timm():
    try:
        return import_module("timm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "timm 패키지가 필요합니다. Main/requirements.txt를 설치하세요."
        ) from exc


def _resolve_model_name(name: str) -> str:
    try:
        return _MODEL_NAME_MAP[name]
    except KeyError as exc:
        supported = sorted(_MODEL_NAME_MAP)
        raise ValueError(
            f"지원하지 않는 Stage 2 모델입니다: {name}. 지원 모델: {supported}"
        ) from exc


def _resolve_device(device: str | None) -> str:
    if device in {None, ""}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if "," in device:
        return f"cuda:{device.split(',')[0].strip()}"
    if device.isdigit():
        return f"cuda:{device}"
    return device
