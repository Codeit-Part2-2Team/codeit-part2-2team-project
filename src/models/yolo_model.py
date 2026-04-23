"""
YOLO 모델 래퍼.

config.yaml 하나로 학습·추론·내보내기를 모두 제어한다.

사용 예:
    model = YOLOModel("experiments/exp_20260420_baseline_yolo26n/config.yaml")
    model.train()
    model.predict("data/raw/test/")
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


def load_config(path: str | Path) -> dict:
    """YAML 파일을 읽어 딕셔너리로 반환한다."""
    with open(path) as f:
        return yaml.safe_load(f)


class YOLOModel:
    """config 기반 YOLO 래퍼.

    config의 model.name으로 모델 종류를 지정하므로 yolo26n, yolo26s 등
    어떤 YOLO 모델이든 교체해서 사용할 수 있다.

    Args:
        config: config.yaml 경로 또는 이미 파싱된 딕셔너리.
    """

    def __init__(self, config: str | Path | dict):
        self.cfg = config if isinstance(config, dict) else load_config(config)
        self._fix_seed(self.cfg.get("seed", 42))

        model_cfg = self.cfg["model"]
        # pretrained=True 이면 Ultralytics 허브에서 사전학습 가중치를 받아온다.
        # pretrained=False 또는 resume 시에는 name에 .pt 경로를 직접 넣는다.
        weights = (
            f"{model_cfg['name']}.pt"
            if model_cfg.get("pretrained", True)
            else model_cfg["name"]
        )
        self.model = YOLO(weights)

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------

    def train(self, data_yaml: str | Path | None = None) -> dict:
        """모델을 학습하고 mAP 지표를 반환한다.

        Args:
            data_yaml: dataset.yaml 경로. None이면 config의 data.yaml을 사용한다.

        Returns:
            {"mAP50": float, "mAP50_95": float}
        """
        cfg = self.cfg
        # data_yaml은 상대경로로 넘기면 CWD에 따라 깨질 수 있으므로 절대경로로 변환한다.
        data_yaml = str(Path(data_yaml or cfg["data"]["yaml"]).resolve())

        out = cfg["output"]
        results = self.model.train(
            data=data_yaml,
            # 학습 하이퍼파라미터
            epochs=cfg["train"]["epochs"],
            batch=cfg["train"]["batch"],
            imgsz=cfg["data"]["imgsz"],
            workers=cfg["data"]["workers"],
            optimizer=cfg["train"]["optimizer"],
            lr0=cfg["train"]["lr0"],  # 초기 학습률
            lrf=cfg["train"]["lrf"],  # 최종 학습률 = lr0 * lrf
            momentum=cfg["train"]["momentum"],
            weight_decay=cfg["train"]["weight_decay"],
            warmup_epochs=cfg["train"]["warmup_epochs"],
            patience=cfg["train"]["patience"],  # EarlyStopping 기준 에폭 수
            # 증강
            mosaic=cfg["augment"]["mosaic"],
            mixup=cfg["augment"]["mixup"],
            copy_paste=cfg["augment"]["copy_paste"],
            flipud=cfg["augment"]["flipud"],
            fliplr=cfg["augment"]["fliplr"],
            hsv_h=cfg["augment"]["hsv_h"],
            hsv_s=cfg["augment"]["hsv_s"],
            hsv_v=cfg["augment"]["hsv_v"],
            # 검증 설정
            conf=cfg["val"]["conf"],
            iou=cfg["val"]["iou"],
            max_det=cfg["val"]["max_det"],
            device=cfg["train"]["device"],
            # 출력 경로: experiments/<name>/ 아래에 저장된다.
            project=out["project"],
            name=out["name"],
            save_period=out["save_period"],  # N 에폭마다 중간 가중치 저장
            seed=cfg.get("seed", 42),
            exist_ok=True,
        )

        return {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def predict(
        self, source: str | Path, output: str | Path = "results/predictions.json"
    ) -> list[dict]:
        """이미지 디렉터리에 대해 추론을 실행하고 predictions.json을 저장한다.

        Args:
            source: 이미지 디렉터리 경로.
            output: predictions.json 저장 경로.

        Returns:
            predictions.json과 동일한 구조의 리스트.
            [{"image_id": str, "detections": [{"class_name": str, "bbox": [x1,y1,x2,y2], "score": float}]}]
        """
        cfg = self.cfg["val"]
        results = self.model.predict(
            source=str(source),
            conf=cfg["conf"],  # 신뢰도 임계값 이하 박스 제거
            iou=cfg["iou"],  # NMS IoU 임계값
            max_det=cfg["max_det"],
            save=False,
        )

        predictions = []
        for r in results:
            detections = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 절대 픽셀 좌표
                    cls_idx = int(box.cls[0])
                    detections.append(
                        {
                            "class_id": cls_idx,
                            "class_name": r.names[cls_idx],
                            "bbox": [x1, y1, x2, y2],  # xyxy
                            "score": float(box.conf[0]),
                        }
                    )
            predictions.append(
                {"image_id": Path(r.path).stem, "detections": detections}
            )

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        return predictions

    # ------------------------------------------------------------------
    # 내보내기
    # ------------------------------------------------------------------

    def export(self, format: str = "onnx") -> None:
        """학습된 모델을 ONNX 또는 TFLite 포맷으로 내보낸다.

        Args:
            format: "onnx" 또는 "tflite".
        """
        self.model.export(format=format, imgsz=self.cfg["data"]["imgsz"])

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_seed(seed: int) -> None:
        """Python / NumPy / PyTorch 난수 시드를 고정해 재현성을 보장한다."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU 대비
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
