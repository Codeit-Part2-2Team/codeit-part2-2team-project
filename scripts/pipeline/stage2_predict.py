"""Stage 2 분류기 추론 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())))

from src.models.classifier import Classifier
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 분류기 추론")
    parser.add_argument("--config", required=True, help="Stage 2 config.yaml 경로")
    parser.add_argument("--source", required=True, help="크롭 이미지 디렉터리")
    parser.add_argument("--weights", help="가중치 경로 (미지정 시 config output에서 자동 조합)")
    parser.add_argument("--output", help="결과 JSON 저장 경로 (미지정 시 자동 조합)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    weights = args.weights or _resolve_weights(cfg)
    output = args.output or _resolve_output(cfg)

    classifier = Classifier(cfg).load_weights(weights)
    results = classifier.predict(source=args.source, output=output)
    print(f"추론 완료: {len(results)}개 → {output}")


def _resolve_weights(cfg: dict) -> Path:
    out = cfg["output"]
    return Path(out["project"]) / out["name"] / "weights" / "best.pt"


def _resolve_output(cfg: dict) -> Path:
    out = cfg["output"]
    return Path(out["project"]) / out["name"] / "results" / "stage2_predictions.json"


if __name__ == "__main__":
    main()
