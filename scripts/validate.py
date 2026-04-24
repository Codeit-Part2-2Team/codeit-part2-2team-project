"""
검증 CLI.

사용 예:
    python scripts/validate.py \\
        --config experiments/exp_20260420_baseline_yolo26n/config.yaml \\
        --weights experiments/exp_20260420_baseline_yolo26n/weights/best.pt \\
        --data data/processed/dataset.yaml
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
from src.utils.path import resolve_weights_path


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 검증")
    parser.add_argument("--config", required=True, help="config.yaml 경로")
    parser.add_argument(
        "--weights",
        default=None,
        help="best.pt 경로 (미지정 시 config output에서 자동 조합)",
    )
    parser.add_argument(
        "--data", default=None, help="dataset.yaml 경로 (config의 data.yaml을 덮어씀)"
    )
    args = parser.parse_args()

    model = YOLOModel(args.config)
    cfg = model.cfg

    # 우선순위: CLI --weights > 자동 조합 (output.project/output.name/weights/best.pt)
    weights = resolve_weights_path(cfg, args.weights)
    model.load_weights(str(weights))

    metrics = Trainer(model).validate(data_yaml=args.data)
    print(f"mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}")


if __name__ == "__main__":
    main()
