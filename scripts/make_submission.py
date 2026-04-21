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
    annotation_id = 1
    for item in data:
        image_id = item["image_id"]
        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            # 제출 포맷은 xywh (COCO 스타일)
            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": det["class_id"],
                    "bbox_x": round(x1),
                    "bbox_y": round(y1),
                    "bbox_w": round(x2 - x1),
                    "bbox_h": round(y2 - y1),
                    "score": det["score"],
                }
            )
            annotation_id += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"제출 파일 생성 완료: {out} ({len(rows)}개 검출)")


if __name__ == "__main__":
    main()
