from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.data.class_map import load_class_maps
from src.data.parser_raw import build_raw_crops
from src.data.parser_external import build_external_crops
from src.data.split import build_cls_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO classification dataset from raw/external pill data."
    )

    parser.add_argument("--final-class-table", type=Path, required=True)
    parser.add_argument("--raw-ann-root", type=Path, required=True)
    parser.add_argument("--raw-img-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)

    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--filtered-crop-root", type=Path, required=True)
    parser.add_argument("--cls-root", type=Path, required=True)

    parser.add_argument("--keep-under10-file", type=Path, default=None)

    parser.add_argument("--min-images", type=int, default=10)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--reset-crop-root", action="store_true")
    parser.add_argument("--reset-filtered-crop-root", action="store_true")
    parser.add_argument("--reset-cls-root", action="store_true")

    return parser.parse_args()


def ensure_clean_dir(path: Path, reset: bool) -> None:
    if reset and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def count_crops(crop_root: Path) -> pd.DataFrame:
    rows = []

    for class_dir in crop_root.iterdir():
        if class_dir.is_dir():
            rows.append(
                (
                    class_dir.name,
                    len(list(class_dir.glob("*")))
                )
            )

    return (
        pd.DataFrame(rows, columns=["final_class_name", "crop_count"])
        .sort_values("crop_count", ascending=False)
        .reset_index(drop=True)
    )


def load_keep_under10(path: Path | None) -> List[str]:
    if path is None or not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()

    return [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def build_filtered_crops(
    crop_root: Path,
    filtered_crop_root: Path,
    keep_classes: Iterable[str],
) -> None:

    for class_name in sorted(set(keep_classes)):
        src_dir = crop_root / class_name
        dst_dir = filtered_crop_root / class_name

        if not src_dir.exists():
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_dir.glob("*"):
            shutil.copy2(img_path, dst_dir / img_path.name)


def main() -> None:
    args = parse_args()

    ensure_clean_dir(args.crop_root, args.reset_crop_root)
    ensure_clean_dir(
        args.filtered_crop_root,
        args.reset_filtered_crop_root,
    )
    ensure_clean_dir(args.cls_root, args.reset_cls_root)

    print("[1/6] class map 로드")
    raw_class_map, ext_class_map = load_class_maps(
        args.final_class_table
    )

    print(f"raw classes: {len(raw_class_map)}")
    print(f"external classes: {len(ext_class_map)}")

    print("[2/6] raw crop 생성")
    raw_stats = build_raw_crops(
        raw_ann_root=args.raw_ann_root,
        raw_img_root=args.raw_img_root,
        raw_class_map=raw_class_map,
        crop_root=args.crop_root,
    )

    for k, v in raw_stats.items():
        print(f"{k}: {v}")

    print("[3/6] external crop 생성")
    ext_stats = build_external_crops(
        external_root=args.external_root,
        ext_class_map=ext_class_map,
        crop_root=args.crop_root,
    )

    for k, v in ext_stats.items():
        print(f"{k}: {v}")

    print("[4/6] crop count 집계")
    crop_df = count_crops(args.crop_root)

    keep_under10 = load_keep_under10(args.keep_under10_file)

    auto_keep = crop_df.loc[
        crop_df["crop_count"] >= args.min_images,
        "final_class_name"
    ].tolist()

    final_keep_classes = sorted(
        set(auto_keep + keep_under10)
    )

    print(f"auto keep: {len(auto_keep)}")
    print(f"manual keep: {len(keep_under10)}")
    print(f"final keep: {len(final_keep_classes)}")

    print("[5/6] filtered crop 생성")
    build_filtered_crops(
        crop_root=args.crop_root,
        filtered_crop_root=args.filtered_crop_root,
        keep_classes=final_keep_classes,
    )

    print("[6/6] classification dataset split")
    split_df = build_cls_dataset(
        filtered_crop_root=args.filtered_crop_root,
        cls_root=args.cls_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(split_df.head())

    print("DONE")


if __name__ == "__main__":
    main()