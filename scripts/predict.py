# isort: skip_file
# ruff: noqa: E402
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
from src.models.predictor import Predictor
from src.utils.path import resolve_weights_path, resolve_predictions_path
from src.utils.timing import exp_dir_from_cfg, timed


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 추론")
    parser.add_argument(
        "--config", required=True, help="config.yaml 경로 (val 파라미터 사용)"
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="best.pt 경로 (미지정 시 config output에서 자동 조합)",
    )
    parser.add_argument("--source", required=True, help="이미지 디렉터리 경로")
    parser.add_argument(
        "--output",
        default=None,
        help="predictions.json 저장 경로 (미지정 시 config 우선순위 적용)",
    )
    parser.add_argument(
        "--tta", action="store_true", help="TTA 활성화 (config val.tta보다 우선)"
    )
    args = parser.parse_args()

    model = YOLOModel(args.config)
    cfg = model.cfg

    # 우선순위: CLI --weights > 자동 조합 (output.project/output.name/weights/best.pt)
    weights = resolve_weights_path(cfg, args.weights)
    model.load_weights(str(weights))

    # 우선순위: CLI --output > 자동 조합 (output.project/output.name/results/predictions.json)
    output = resolve_predictions_path(cfg, args.output)

    # 우선순위: CLI --tta > config val.tta
    tta = args.tta or cfg["val"].get("tta", False)

    with timed(exp_dir_from_cfg(cfg), "s1_predict"):
        predictions = Predictor(model).predict(args.source, output=output, tta=tta)
    print(f"완료: {len(predictions)}개 이미지 → {output}")


if __name__ == "__main__":
    main()
