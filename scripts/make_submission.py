"""
Kaggle 제출 CSV 생성 CLI.

predictions.json을 읽어 대회 제출 포맷(submission.csv)으로 변환한다.

predictions.json 포맷:
    [{"image_id": "test_0001", "detections": [{"class_name": str, "bbox": [x1,y1,x2,y2], "score": float}]}]

submission.csv 포맷:
    image_id,PredictionString
    test_0001,Crestor 10mg tab 0.870000 10.0 20.0 100.0 200.0 ...

사용 예:
    python scripts/make_submission.py \\
        --predictions results/predictions.json \\
        --output submissions/submission.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Kaggle 제출 CSV 생성")
    parser.add_argument("--predictions", required=True, help="predictions.json 경로")
    parser.add_argument("--output", required=True, help="submission.csv 저장 경로")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        parts = []
        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            # PredictionString 형식: "클래스명 score x1 y1 x2 y2" 를 검출 수만큼 이어붙인다.
            parts.append(
                f"{det['class_name']} {det['score']:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
            )
        rows.append({"image_id": item["image_id"], "PredictionString": " ".join(parts)})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image_id", "PredictionString"]).to_csv(
        out, index=False
    )
    print(f"제출 파일 생성 완료: {out}")


if __name__ == "__main__":
    main()
