from __future__ import annotations

__all__ = ["Classifier", "YOLOModel", "Predictor"]


def __getattr__(name: str):
    if name == "Classifier":
        from src.models.classifier import Classifier

        return Classifier
    if name == "YOLOModel":
        from src.models.model_yolo import YOLOModel

        return YOLOModel
    if name == "Predictor":
        from src.models.predictor import Predictor

        return Predictor
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
