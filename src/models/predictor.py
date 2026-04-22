"""추론 퍼사드. Ultralytics를 직접 임포트하지 않는다."""

import json
from pathlib import Path


class Predictor:
    """model_yolo (또는 미래 모델)을 경유해 추론을 실행하는 퍼사드."""

    def __init__(self, model):
        self.model = model

    def predict(
        self,
        source: str | Path,
        output: str | Path = "results/predictions.json",
        tta: bool = False,
    ) -> list[dict]:
        """추론 실행 → predictions.json 저장 + 동일 구조 리스트 반환.

        Returns:
            [{"image_id": str, "detections": [{"class_id": int, "class_name": str,
                                               "bbox": [x1,y1,x2,y2], "score": float}]}]
        """
        cfg = self.model.cfg["val"]
        raw = self.model.raw_predict(
            source=str(source),
            conf=cfg["conf"],
            iou=cfg["iou"],
            max_det=cfg["max_det"],
            augment=tta,
        )
        predictions = self._parse_results(raw)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        return predictions

    def _parse_results(self, raw_results) -> list[dict]:
        """Ultralytics Result 리스트 → predictions.json 스키마 변환."""
        predictions = []
        for r in raw_results:
            detections = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_idx = int(box.cls[0])
                    detections.append(
                        {
                            "class_id": cls_idx,
                            "class_name": r.names[cls_idx],
                            "bbox": [x1, y1, x2, y2],
                            "score": float(box.conf[0]),
                        }
                    )
            predictions.append(
                {"image_id": Path(r.path).stem, "detections": detections}
            )
        return predictions
