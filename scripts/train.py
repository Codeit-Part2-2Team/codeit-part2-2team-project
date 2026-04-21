"""
학습 CLI.

사용 예:
    python scripts/train.py --config experiments/exp_20260420_baseline_yolo26n/config.yaml
    python scripts/train.py --config experiments/.../config.yaml --data data/processed/dataset.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.yolo26 import YOLOv26


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 학습")
    parser.add_argument("--config", required=True, help="config.yaml 경로")
    parser.add_argument("--data", default=None, help="dataset.yaml 경로 (config의 data.yaml을 덮어씀)")
    parser.add_argument("--device", default="0", help="GPU 번호 또는 'cpu'")
    args = parser.parse_args()

    model = YOLOv26(args.config)

    # CLI에서 device를 지정했을 때만 config를 덮어쓴다.
    if args.device:
        model.cfg["train"]["device"] = args.device

    metrics = model.train(data_yaml=args.data)
    print(f"mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
