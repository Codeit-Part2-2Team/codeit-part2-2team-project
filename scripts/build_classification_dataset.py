#!/usr/bin/env python3
"""
YOLO classification dataset build script

기능
1) raw train annotations(JSON) 기반 crop 생성
2) external train/valid/test labels(TXT) 기반 crop 생성
   - external txt에 bbox가 여러 개 있는 경우, normalized bbox area가 가장 큰 bbox 1개를 대표 bbox로 선택
3) 10장 이상 클래스 자동 유지 + under10 keep 목록 반영
4) 최종 keep 클래스만 모아 filtered crop dataset 생성
5) YOLO classification용 train/val/test 폴더 구조 생성
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO classification dataset from raw/external pill data.")
    parser.add_argument("--final-class-table", type=Path, required=True, help="CSV path for final_class_table.csv")
    parser.add_argument("--raw-ann-root", type=Path, required=True, help="Raw train annotation root (json files)")
    parser.add_argument("--raw-img-root", type=Path, required=True, help="Raw train image root")
    parser.add_argument("--external-root", type=Path, required=True, help="External extracted root containing train/valid/test")
    parser.add_argument("--crop-root", type=Path, required=True, help="Output root for all crops, e.g. crops_v1.0")
    parser.add_argument("--filtered-crop-root", type=Path, required=True, help="Output root for filtered crops, e.g. crops_v1.1")
    parser.add_argument("--cls-root", type=Path, required=True, help="Output root for YOLO classification dataset")
    parser.add_argument("--keep-under10-file", type=Path, default=None, help="Optional txt file, one class per line")
    parser.add_argument("--min-images", type=int, default=10, help="Minimum images to auto-keep")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-crop-root", action="store_true", help="Delete crop_root before rebuilding")
    parser.add_argument("--reset-filtered-crop-root", action="store_true", help="Delete filtered_crop_root before rebuilding")
    parser.add_argument("--reset-cls-root", action="store_true", help="Delete cls_root before rebuilding")
    return parser.parse_args()


def ensure_clean_dir(path: Path, reset: bool) -> None:
    if reset and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clamp_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def load_class_maps(final_class_table_path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    df = pd.read_csv(final_class_table_path)
    raw_map: Dict[int, str] = {}
    ext_map: Dict[int, str] = {}

    for _, row in df.iterrows():
        source = str(row["source"])
        class_id = int(row["source_class_id"])
        class_name = str(row["final_class_name"])

        if source == "raw":
            raw_map[class_id] = class_name
        elif source == "external":
            ext_map[class_id] = class_name

    return raw_map, ext_map


def build_raw_crops(raw_ann_root: Path, raw_img_root: Path, raw_class_map: Dict[int, str], crop_root: Path) -> Dict[str, int]:
    saved = skipped = invalid_bbox = errors = 0
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

                x1, y1 = x, y
                x2, y2 = x + w, y + h
                x1, y1, x2, y2 = clamp_bbox_xyxy(x1, y1, x2, y2, width, height)

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


def select_largest_yolo_bbox(lines: List[str]) -> Tuple[int, float, float, float, float] | None:
    """YOLO txt 여러 줄 중 bbox 면적이 가장 큰 annotation 1개를 선택합니다."""
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


def build_external_crops(external_root: Path, ext_class_map: Dict[int, str], crop_root: Path) -> Dict[str, int]:
    saved = skipped = errors = 0
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

                    x1, y1, x2, y2 = clamp_bbox_xyxy(x1, y1, x2, y2, width, height)

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


def count_crops(crop_root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in crop_root.iterdir():
        if class_dir.is_dir():
            rows.append((class_dir.name, len(list(class_dir.glob("*")))))

    return (
        pd.DataFrame(rows, columns=["final_class_name", "crop_count"])
        .sort_values("crop_count", ascending=False)
        .reset_index(drop=True)
    )


def load_keep_under10(path: Path | None) -> List[str]:
    if path is None or not path.exists():
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def build_filtered_crops(crop_root: Path, filtered_crop_root: Path, keep_classes: Iterable[str]) -> Dict[str, int]:
    copied_classes = copied_images = 0
    missing_classes = []

    for class_name in sorted(set(keep_classes)):
        src_dir = crop_root / class_name
        dst_dir = filtered_crop_root / class_name

        if not src_dir.exists():
            missing_classes.append(class_name)
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_dir.glob("*"):
            shutil.copy2(img_path, dst_dir / img_path.name)
            copied_images += 1

        copied_classes += 1

    return {
        "copied_classes": copied_classes,
        "copied_images": copied_images,
        "missing_classes": len(missing_classes),
    }


def split_files(file_list: List[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratio 합은 1이어야 합니다.")

    files = list(file_list)
    rnd = random.Random(seed)
    rnd.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def build_cls_dataset(
    filtered_crop_root: Path,
    cls_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    for split in ["train", "val", "test"]:
        (cls_root / split).mkdir(parents=True, exist_ok=True)

    records = []

    for class_dir in filtered_crop_root.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        img_files = list(class_dir.glob("*"))
        if not img_files:
            continue

        train_files, val_files, test_files = split_files(
            img_files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        split_map = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, files in split_map.items():
            dst_class_dir = cls_root / split_name / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy2(img_path, dst_class_dir / img_path.name)

        records.append(
            {
                "class_name": class_name,
                "total": len(img_files),
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
            }
        )

    return pd.DataFrame(records).sort_values("total", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    ensure_clean_dir(args.crop_root, args.reset_crop_root)
    ensure_clean_dir(args.filtered_crop_root, args.reset_filtered_crop_root)
    ensure_clean_dir(args.cls_root, args.reset_cls_root)

    print("[1/6] final_class_table 로드 및 class map 생성")
    raw_class_map, ext_class_map = load_class_maps(args.final_class_table)
    print(f"  raw classes: {len(raw_class_map)}")
    print(f"  external classes: {len(ext_class_map)}")

    print("[2/6] raw crop 생성")
    raw_stats = build_raw_crops(args.raw_ann_root, args.raw_img_root, raw_class_map, args.crop_root)
    for key, value in raw_stats.items():
        print(f"  {key}: {value}")

    print("[3/6] external crop 생성")
    ext_stats = build_external_crops(args.external_root, ext_class_map, args.crop_root)
    for key, value in ext_stats.items():
        print(f"  {key}: {value}")

    print("[4/6] crop count 집계")
    crop_df = count_crops(args.crop_root)
    count_path = args.crop_root.parent / f"{args.crop_root.name}_counts.csv"
    crop_df.to_csv(count_path, index=False, encoding="utf-8-sig")
    print(f"  crop classes: {len(crop_df)}")
    print(f"  total crops: {int(crop_df['crop_count'].sum())}")
    print(f"  crop count saved: {count_path}")

    keep_under10 = load_keep_under10(args.keep_under10_file)
    auto_keep = crop_df.loc[crop_df["crop_count"] >= args.min_images, "final_class_name"].tolist()
    final_keep_classes = sorted(set(auto_keep + keep_under10))

    print("[5/6] filtered crop dataset 생성")
    print(f"  auto keep (>= {args.min_images}): {len(auto_keep)}")
    print(f"  manual keep under10: {len(keep_under10)}")
    print(f"  final keep classes: {len(final_keep_classes)}")

    filtered_stats = build_filtered_crops(args.crop_root, args.filtered_crop_root, final_keep_classes)
    for key, value in filtered_stats.items():
        print(f"  {key}: {value}")

    print("[6/6] YOLO classification dataset split 생성")
    split_df = build_cls_dataset(
        args.filtered_crop_root,
        args.cls_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_summary_path = args.cls_root.parent / f"{args.cls_root.name}_split_summary.csv"
    split_df.to_csv(split_summary_path, index=False, encoding="utf-8-sig")

    print(f"  split classes: {len(split_df)}")
    print(f"  total images in filtered set: {int(split_df['total'].sum())}")
    print(f"  split summary saved: {split_summary_path}")

    for split in ["train", "val", "test"]:
        split_dir = args.cls_root / split
        class_count = len([d for d in split_dir.iterdir() if d.is_dir()])
        img_count = sum(len(list(d.glob("*"))) for d in split_dir.iterdir() if d.is_dir())
        print(f"  {split}: classes={class_count}, images={img_count}")

    keep_path = args.filtered_crop_root.parent / f"{args.filtered_crop_root.name}_keep_classes.txt"
    keep_path.write_text("\n".join(final_keep_classes), encoding="utf-8")
    print(f"  keep class list saved: {keep_path}")


if __name__ == "__main__":
    main()
