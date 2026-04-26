from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from src.data.parser_raw import clamp_bbox_xyxy


def yolo_box_area(parts: List[str]) -> float:
    """
    YOLO 라벨 한 줄을 split한 결과에서 bbox 면적을 계산한다.

    parts 예시:
    ["312", "0.5", "0.4", "0.2", "0.1"]
    """
    w = float(parts[3])
    h = float(parts[4])
    return w * h


def select_largest_yolo_bbox(
    lines: List[str],
) -> Tuple[int, float, float, float, float] | None:
    """
    YOLO txt 여러 줄 중 bbox 면적이 가장 큰 annotation 1개를 선택한다.

    Returns
    -------
    tuple | None
        (class_id, x_center, y_center, box_width, box_height)
    """

    candidates = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            area = bw * bh

            candidates.append((area, class_id, xc, yc, bw, bh))

        except ValueError:
            continue

    if not candidates:
        return None

    _, class_id, xc, yc, bw, bh = max(candidates, key=lambda x: x[0])

    return class_id, xc, yc, bw, bh


def convert_external_annotations(
    ext_label_roots: List[Path],
    ext_output_dir: Path,
    class_id: int = 0,
) -> None:
    """
    Stage 1용 함수.

    external YOLO txt 라벨을 처리한다.

    처리 내용
    1. 기존 class_id를 모두 단일 클래스 class_id로 통일
    2. bbox가 여러 개면 가장 큰 bbox 1개만 남김
    """

    ext_output_dir.mkdir(parents=True, exist_ok=True)

    txt_files: List[Path] = []

    for root in ext_label_roots:
        txt_files.extend(sorted(root.rglob("*.txt")))

    print(f"[EXT] 발견된 txt 파일 수: {len(txt_files)}")

    converted = 0
    multi_bbox_fixed = 0
    skipped = 0

    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            valid_parts: List[List[str]] = []

            for line in lines:
                parts = line.split()

                if len(parts) != 5:
                    continue

                valid_parts.append(parts)

            if len(valid_parts) >= 2:
                valid_parts = [max(valid_parts, key=yolo_box_area)]
                multi_bbox_fixed += 1

            new_lines: List[str] = []

            for parts in valid_parts:
                new_line = f"{class_id} {' '.join(parts[1:])}"
                new_lines.append(new_line)

            save_path = ext_output_dir / txt_path.name

            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            converted += 1

        except Exception as e:
            skipped += 1
            print(f"[EXT][ERROR] {txt_path}: {e}")

    print(f"[EXT] 변환 완료 수: {converted}")
    print(f"[EXT] 중복 bbox 정리 수: {multi_bbox_fixed}")
    print(f"[EXT] 스킵/에러 수: {skipped}")


def build_external_crops(
    external_root: Path,
    ext_class_map: Dict[int, str],
    crop_root: Path,
) -> Dict[str, int]:
    """
    Stage 2용 함수.

    external train/valid/test labels를 읽고,
    bbox 기준으로 classification crop 이미지를 생성한다.

    external txt에 bbox가 여러 개 있는 경우,
    normalized bbox area가 가장 큰 bbox 1개를 대표 bbox로 선택한다.
    """

    saved = 0
    skipped = 0
    errors = 0
    multi_bbox_files = 0
    invalid_label_files = 0
    split_counts = {}

    for split in ["train", "valid", "test"]:
        label_root = external_root / split / "labels"
        img_root = external_root / split / "images"

        txt_files = sorted(label_root.glob("*.txt"))
        split_counts[f"{split}_txt_files"] = len(txt_files)

        for txt_path in txt_files:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]

                if not lines:
                    skipped += 1
                    invalid_label_files += 1
                    continue

                if len(lines) >= 2:
                    multi_bbox_files += 1

                selected = select_largest_yolo_bbox(lines)

                if selected is None:
                    skipped += 1
                    invalid_label_files += 1
                    continue

                class_id, xc, yc, bw, bh = selected

                if class_id not in ext_class_map:
                    skipped += 1
                    continue

                img_path = img_root / f"{txt_path.stem}.jpg"

                if not img_path.exists():
                    img_path = img_root / f"{txt_path.stem}.png"

                if not img_path.exists():
                    skipped += 1
                    continue

                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    width, height = img.size

                    x_center = xc * width
                    y_center = yc * height
                    box_w = bw * width
                    box_h = bh * height

                    x1 = x_center - box_w / 2
                    y1 = y_center - box_h / 2
                    x2 = x_center + box_w / 2
                    y2 = y_center + box_h / 2

                    x1, y1, x2, y2 = clamp_bbox_xyxy(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        width=width,
                        height=height,
                    )

                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        skipped += 1
                        continue

                    crop = img.crop((x1, y1, x2, y2))

                    class_name = ext_class_map[class_id]
                    save_dir = crop_root / class_name
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_name = f"ext_{split}_{txt_path.stem}_{class_id}.png"
                    crop.save(save_dir / save_name)

                    saved += 1

            except Exception:
                errors += 1

    out = {
        "ext_saved": saved,
        "ext_skipped": skipped,
        "ext_errors": errors,
        "ext_multi_bbox_files": multi_bbox_files,
        "ext_invalid_label_files": invalid_label_files,
    }

    out.update(split_counts)

    return out
