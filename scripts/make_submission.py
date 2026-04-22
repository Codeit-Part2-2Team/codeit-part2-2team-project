"""
Kaggle 제출 CSV 생성 CLI.

predictions.json을 읽어 대회 제출 포맷(submission.csv)으로 변환한다.

predictions.json 포맷:
    [{"image_id": "1", "detections": [{"class_id": int, "class_name": str, "bbox": [x1,y1,x2,y2], "score": float}]}]

submission.csv 포맷:
    annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
    1, 1, 1, 156, 247, 211, 456, 0.91

사용 예:
    python scripts/make_submission.py \\
        --predictions results/predictions.json \\
        --output submissions/submission.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.submission import save_submission


def main():
    parser = argparse.ArgumentParser(description="Kaggle 제출 CSV 생성")
    parser.add_argument("--predictions", required=True, help="predictions.json 경로")
    parser.add_argument("--output", required=True, help="submission.csv 저장 경로")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)

    save_submission(predictions, args.output)
    print(f"제출 파일 생성 완료: {args.output}")


if __name__ == "__main__":
    main()
