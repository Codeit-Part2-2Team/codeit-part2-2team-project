from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

EXCLUDE_IMAGE_STEMS = {
    "K-003351-016262-018357_0_2_0_2_75_000_200",
}


def should_exclude_image(image_file: str) -> bool:
    image_stem = Path(image_file).stem
    return image_stem in EXCLUDE_IMAGE_STEMS


def coco_bbox_to_yolo(
    bbox: List[float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    box_w = w / img_w
    box_h = h / img_h

    return x_center, y_center, box_w, box_h


def validate_yolo_bbox(
    x_center: float,
    y_center: float,
    w: float,
    h: float,
) -> bool:
    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < w <= 1.0
        and 0.0 < h <= 1.0
    )


def clamp_bbox_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def convert_raw_annotations(
    raw_annotation_dir: Path,
    raw_output_dir: Path,
    class_id: int = 0,
) -> None:
    """
    Stage 1용 함수.
    raw COCO-style JSON annotation을 YOLO txt 형식으로 변환한다.
    """

    raw_output_dir.mkdir(parents=True, exist_ok=True)

    raw_json_files = sorted(raw_annotation_dir.rglob("*.json"))
    print(f"[RAW] 발견된 json 파일 수: {len(raw_json_files)}")

    image_to_lines: Dict[str, List[str]] = defaultdict(list)

    error_count = 0
    skipped_excluded_count = 0
    skipped_invalid_bbox_count = 0

    for json_path in raw_json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_info = data["images"][0]
            annotations = data["annotations"]

            image_file = image_info["file_name"]
            img_w = image_info["width"]
            img_h = image_info["height"]

            if should_exclude_image(image_file):
                skipped_excluded_count += 1
                continue

            for ann in annotations:
                bbox = ann["bbox"]

                x_center, y_center, w, h = coco_bbox_to_yolo(
                    bbox=bbox,
                    img_w=img_w,
                    img_h=img_h,
                )

                if not validate_yolo_bbox(x_center, y_center, w, h):
                    skipped_invalid_bbox_count += 1
                    continue

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                image_to_lines[image_file].append(line)

        except Exception as e:
            error_count += 1
            print(f"[RAW][ERROR] {json_path}: {e}")

    saved_count = 0
    empty_skipped_count = 0

    for image_file, lines in image_to_lines.items():
        if len(lines) == 0:
            empty_skipped_count += 1
            continue

        save_path = raw_output_dir / f"{Path(image_file).stem}.txt"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        saved_count += 1

    print(f"[RAW] bbox가 모인 이미지 수: {len(image_to_lines)}")
    print(f"[RAW] 저장된 txt 수: {saved_count}")
    print(f"[RAW] 하드 제외된 샘플 수: {skipped_excluded_count}")
    print(f"[RAW] invalid bbox로 제외된 수: {skipped_invalid_bbox_count}")
    print(f"[RAW] 빈 라벨이라 저장 안 한 수: {empty_skipped_count}")
    print(f"[RAW] 에러 수: {error_count}")


def build_raw_crops(
    raw_ann_root: Path,
    raw_img_root: Path,
    raw_class_map: Dict[int, str],
    crop_root: Path,
) -> Dict[str, int]:
    """
    Stage 2용 함수.
    raw annotation JSON과 원본 이미지를 이용해 classification crop 이미지를 생성한다.
    """

    saved = 0
    skipped = 0
    invalid_bbox = 0
    errors = 0

    json_files = sorted(raw_ann_root.rglob("*.json"))

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            images = data.get("images", [])
            annotations = data.get("annotations", [])

            if not images or not annotations:
                skipped += 1
                continue

            image_info = images[0]
            annotation = annotations[0]

            file_name = image_info["file_name"]
            category_id = int(annotation["category_id"])
            bbox = annotation["bbox"]

            if category_id not in raw_class_map:
                skipped += 1
                continue

            img_path = raw_img_root / file_name

            if not img_path.exists():
                skipped += 1
                continue

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                x, y, w, h = bbox

                if w <= 1 or h <= 1:
                    invalid_bbox += 1
                    continue

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                x1, y1, x2, y2 = clamp_bbox_xyxy(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=width,
                    height=height,
                )

                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    invalid_bbox += 1
                    continue

                crop = img.crop((x1, y1, x2, y2))

                class_name = raw_class_map[category_id]
                save_dir = crop_root / class_name
                save_dir.mkdir(parents=True, exist_ok=True)

                save_name = f"raw_{Path(file_name).stem}_{category_id}.png"
                crop.save(save_dir / save_name)

                saved += 1

        except Exception:
            errors += 1

    return {
        "raw_json_files": len(json_files),
        "raw_saved": saved,
        "raw_skipped": skipped,
        "raw_invalid_bbox": invalid_bbox,
        "raw_errors": errors,
    }
