"""
학습 CLI.

사용 예:
    python scripts/train.py --config experiments/exp_20260420_baseline_yolo26n/config.yaml
    python scripts/train.py --config experiments/.../config.yaml --data data/processed/dataset.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        next(
            p
            for p in Path(__file__).resolve().parents
            if (p / "requirements.txt").exists()
        )
    ),
)

from src.models.model_yolo import YOLOModel
from src.training.trainer import Trainer
from src.utils.timing import exp_dir_from_cfg, timed


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 학습")
    parser.add_argument("--config", required=True, help="config.yaml 경로")
    parser.add_argument(
        "--data", default=None, help="dataset.yaml 경로 (config의 data.yaml을 덮어씀)"
    )
    parser.add_argument(
        "--device", default=None, help="GPU 번호 또는 'cpu' (미지정 시 config 값 사용)"
    )
    args = parser.parse_args()

    model = YOLOModel(args.config)

    if args.device is not None:
        model.cfg["train"]["device"] = args.device

    with timed(exp_dir_from_cfg(model.cfg), "s1_train"):
        metrics = Trainer(model).train(data_yaml=args.data)
    print(f"mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
