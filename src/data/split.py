from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def split_files(
    file_list: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    이미지 파일 목록을 train/val/test로 나눈다.
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratio 합은 1이어야 합니다.")

    files = list(file_list)

    rnd = random.Random(seed)
    rnd.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def build_cls_dataset(
    filtered_crop_root: Path,
    cls_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """
    filtered crop dataset을 YOLO classification용 폴더 구조로 변환한다.

    생성 구조:
    cls_root/
    ├── train/
    │   ├── class_A/
    │   └── class_B/
    ├── val/
    │   ├── class_A/
    │   └── class_B/
    └── test/
        ├── class_A/
        └── class_B/
    """

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
            file_list=img_files,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

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

    return (
        pd.DataFrame(records)
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )