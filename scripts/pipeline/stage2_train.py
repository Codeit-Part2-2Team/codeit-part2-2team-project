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

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.stage2_dataset import Stage2Dataset
from src.models.classifier import Classifier
from src.utils.config import load_config
from src.utils.timing import exp_dir_from_cfg, timed


def _print_dataset_summary(train_ds: "Stage2Dataset", val_ds: "Stage2Dataset", cfg: dict) -> None:
    cfg_nc = cfg["model"]["num_classes"]
    train_nc = len(train_ds.classes)
    val_nc = len(val_ds.classes)

    test_dir = Path(cfg["data"]["train"]).resolve().parent / "test"
    test_part = ""
    if test_dir.exists():
        test_classes = [d for d in test_dir.iterdir() if d.is_dir()]
        test_samples = sum(len(list(d.iterdir())) for d in test_classes)
        test_part = f"  test={test_samples} ({len(test_classes)} cls)"

    nc_match = "OK" if cfg_nc == train_nc else "MISMATCH"
    print(
        f"[dataset] train={len(train_ds)} ({train_nc} cls)"
        f"  val={len(val_ds)} ({val_nc} cls)"
        f"{test_part}"
        f"  config nc={cfg_nc} [{nc_match}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 분류기 학습")
    parser.add_argument("--config", required=True, help="Stage 2 config.yaml 경로")
    parser.add_argument(
        "--data",
        help="크롭 루트 디렉터리 (미지정 시 config 값 사용). 지정 시 <data>/train, <data>/val 로 해석",
    )
    parser.add_argument(
        "--resume",
        help="이어서 학습할 체크포인트 경로 (예: experiments/.../weights/last.pt)",
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
    val_ds = Stage2Dataset(val_dir, cfg, split="val", classes=train_ds.classes)

    _print_dataset_summary(train_ds, val_ds, cfg)

    actual_nc = len(train_ds.classes)
    cfg_nc = cfg["model"]["num_classes"]
    if cfg_nc != actual_nc:
        raise ValueError(
            f"config model.num_classes={cfg_nc}이지만 "
            f"실제 데이터셋 클래스 수는 {actual_nc}개입니다. "
            f"config를 num_classes: {actual_nc}로 수정하세요."
        )

    sampler = WeightedRandomSampler(
        weights=train_ds.get_sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch, sampler=sampler, num_workers=workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers
    )

    classifier = Classifier(cfg)
    classifier.class_names = train_ds.classes
    with timed(exp_dir_from_cfg(cfg), "s2_train"):
        metrics = classifier.fit(train_loader, val_loader, resume_from=args.resume)
    print(f"학습 완료: top1={metrics['top1_acc']:.4f}, top5={metrics['top5_acc']:.4f}")


if __name__ == "__main__":
    main()
