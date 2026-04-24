"""Stage 2 분류기 학습 CLI."""

from __future__ import annotations

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

from torch.utils.data import DataLoader

from src.data.stage2_dataset import Stage2Dataset
from src.models.classifier import Classifier
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 분류기 학습")
    parser.add_argument("--config", required=True, help="Stage 2 config.yaml 경로")
    parser.add_argument(
        "--data",
        help="크롭 루트 디렉터리 (미지정 시 config 값 사용). 지정 시 <data>/train, <data>/val 로 해석",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.data:
        base = Path(args.data)
        train_dir = base / "train"
        val_dir = base / "val"
    else:
        train_dir = Path(cfg["data"]["train"])
        val_dir = Path(cfg["data"]["val"])

    batch = cfg["train"]["batch"]
    workers = cfg["data"]["workers"]

    train_ds = Stage2Dataset(train_dir, cfg, split="train")
    val_ds = Stage2Dataset(val_dir, cfg, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers
    )

    # 실제 클래스 수로 모델 빌드 (config 값 덮어쓰기)
    cfg["model"]["num_classes"] = len(train_ds.classes)
    cfg["nc"] = len(train_ds.classes)

    classifier = Classifier(cfg)
    classifier.class_names = train_ds.classes
    metrics = classifier.fit(train_loader, val_loader)
    print(f"학습 완료: top1={metrics['top1_acc']:.4f}, top5={metrics['top5_acc']:.4f}")


if __name__ == "__main__":
    main()
