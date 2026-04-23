"""
추론 CLI.

사용 예:
    python scripts/predict.py \\
        --config experiments/exp_20260420_baseline_yolo26n/config.yaml \\
        --weights experiments/exp_20260420_baseline_yolo26n/weights/best.pt \\
        --source data/raw/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.yolo_model import YOLOModel


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 추론")
    parser.add_argument(
        "--config", required=True, help="config.yaml 경로 (val 파라미터 사용)"
    )
    parser.add_argument("--weights", required=True, help="best.pt 경로")
    parser.add_argument("--source", required=True, help="이미지 디렉터리 경로")
    parser.add_argument(
        "--output",
        default="results/predictions.json",
        help="predictions.json 저장 경로",
    )
    args = parser.parse_args()

    model = YOLOModel(args.config)
    model.model.load(
        args.weights
    )  # config에서 초기화한 모델에 학습된 가중치를 덮어쓴다.

    predictions = model.predict(args.source, output=args.output)
    print(f"완료: {len(predictions)}개 이미지 → {args.output}")


if __name__ == "__main__":
    main()
