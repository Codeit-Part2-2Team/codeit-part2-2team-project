"""노트북 시각화 유틸리티.

노트북에서 반복되는 plot 코드를 함수로 제공한다.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def _find_image(directory: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _group_by_image(items: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for item in items:
        result.setdefault(item["image_id"], []).append(item)
    return result


def plot_training_curves(
    results_csv: Path,
    save_dir: Path | None = None,
) -> None:
    """Loss / mAP 학습 곡선 시각화. save_dir 지정 시 training_curves.png 저장."""
    results_csv = Path(results_csv)
    if not results_csv.exists():
        print(f"⚠️  results.csv 없음: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    map_cols = [c for c in df.columns if "map" in c.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col in loss_cols:
        axes[0].plot(df["epoch"], df[col], label=col, linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for col in map_cols:
        axes[1].plot(df["epoch"], df[col], label=col, linewidth=2)
    axes[1].set_title("mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "training_curves.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ 저장: {out}")
    plt.show()


def plot_s1_gt_vs_pred(
    predictions: list[dict],
    val_img_dir: Path,
    val_label_dir: Path,
    n: int = 4,
) -> None:
    """Stage 1 GT(초록) vs 예측(빨강) bbox 비교 — 알약 2개 이상 케이스 위주."""
    val_img_dir = Path(val_img_dir)
    val_label_dir = Path(val_label_dir)

    def _gt_count(image_id: str) -> int:
        p = val_label_dir / f"{image_id}.txt"
        return len(p.read_text().strip().splitlines()) if p.exists() else 0

    multi = [p for p in predictions if _gt_count(p["image_id"]) >= 2]
    print(f"멀티 알약 이미지: {len(multi)}개 / 전체 {len(predictions)}개")

    samples = random.sample(multi, min(n, len(multi)))
    if not samples:
        print("⚠️  GT 기준 알약 2개 이상인 이미지가 없습니다")
        return

    fig, axes = plt.subplots(len(samples), 2, figsize=(14, 5 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for row, sample in enumerate(samples):
        img_path = _find_image(val_img_dir, sample["image_id"])
        if img_path is None:
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        label_path = val_label_dir / f"{sample['image_id']}.txt"
        gt_boxes = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                gt_boxes.append(
                    (
                        (cx - bw / 2) * W,
                        (cy - bh / 2) * H,
                        (cx + bw / 2) * W,
                        (cy + bh / 2) * H,
                    )
                )

        axes[row, 0].imshow(img)
        for x1, y1, x2, y2 in gt_boxes:
            axes[row, 0].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="green",
                    facecolor="none",
                )
            )
        axes[row, 0].set_title(
            f"GT  {sample['image_id']}  ({len(gt_boxes)}개)", fontsize=9
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img)
        for det in sample["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            axes[row, 1].add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )
            axes[row, 1].text(
                x1,
                y1 - 4,
                f"{det['score']:.2f}",
                color="red",
                fontsize=8,
                fontweight="bold",
            )
        axes[row, 1].set_title(
            f"예측  ({len(sample['detections'])}개 탐지)", fontsize=9
        )
        axes[row, 1].axis("off")

    plt.suptitle("Stage 1 멀티 알약 탐지 결과 — GT (초록) | 예측 (빨강)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_crop_showcase(
    manifest: list[dict],
    crop_dir: Path,
    img_dir: Path,
    save_dir: Path | None = None,
) -> None:
    """탐지 수 가장 많은 이미지 1장: 원본 + 각 crop 나란히. save_dir 지정 시 저장."""
    crop_dir = Path(crop_dir)
    img_dir = Path(img_dir)

    by_img = _group_by_image(manifest)
    showcase_id = max(by_img, key=lambda k: len(by_img[k]))
    crops = by_img[showcase_id]
    n = len(crops)

    img_path = _find_image(img_dir, showcase_id)
    if img_path is None:
        print(f"⚠️  이미지 없음: {showcase_id}")
        return
    orig_img = np.array(Image.open(img_path).convert("RGB"))

    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4))
    axes[0].imshow(orig_img)
    for i, item in enumerate(crops):
        color = plt.cm.tab10.colors[i % 10]
        x1, y1, x2, y2 = item["bbox"]
        axes[0].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        axes[0].text(x1, y1 - 5, str(i), color=color, fontsize=9, fontweight="bold")
    axes[0].set_title(f"원본  {showcase_id}", fontsize=8)
    axes[0].axis("off")

    for i, item in enumerate(crops):
        color = plt.cm.tab10.colors[i % 10]
        crop_path = crop_dir / f"{item['crop_id']}.jpg"
        if crop_path.exists():
            axes[i + 1].imshow(Image.open(crop_path).convert("RGB"))
        axes[i + 1].set_title(f"crop {i}\n{item['score']:.2f}", fontsize=7, color=color)
        axes[i + 1].axis("off")

    plt.suptitle("Stage 1 → crop 추출 중간 확인", fontsize=11)
    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "crop_samples.png"
        plt.savefig(out, dpi=150)
        print(f"✓ 저장: {out}")
    plt.show()


def plot_crop_grid(crop_dir: Path, n: int = 32) -> None:
    """랜덤 n개 crop 그리드 — 전반적인 crop 품질 확인."""
    crop_dir = Path(crop_dir)
    crop_files = sorted(crop_dir.glob("*.jpg"))
    sample_files = random.sample(crop_files, min(n, len(crop_files)))

    cols = 8
    rows = max(1, (len(sample_files) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
    ax_flat = axes.flatten() if rows > 1 else axes
    for ax, f in zip(ax_flat, sample_files):
        ax.imshow(Image.open(f).convert("RGB"))
        ax.set_title(f.stem[-8:], fontsize=6)
        ax.axis("off")
    for ax in ax_flat[len(sample_files) :]:
        ax.axis("off")

    plt.suptitle(
        f"crop 랜덤 샘플  (전체 {len(crop_files)}개 중 {len(sample_files)}개)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def plot_pipeline_overlay(
    pipeline_by_img: dict[str, list[dict]],
    img_dir: Path,
    n: int = 4,
    save_dir: Path | None = None,
) -> None:
    """2-Stage 통합 결과 오버레이 — 멀티 탐지 케이스 n장.

    pipeline_by_img 아이템: {"bbox", "det_score", "crop_id", "class_name", "class_score"}
    """
    img_dir = Path(img_dir)
    multi = {k: v for k, v in pipeline_by_img.items() if len(v) >= 2}
    print(f"멀티 탐지 이미지: {len(multi)}개 / 전체 {len(pipeline_by_img)}개")

    if not multi:
        print("⚠️  탐지 2개 이상인 이미지가 없습니다")
        return

    sample_ids = random.sample(list(multi.keys()), min(n, len(multi)))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, img_id in zip(axes.flatten(), sample_ids):
        img_path = _find_image(img_dir, img_id)
        if img_path is None:
            ax.axis("off")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        ax.imshow(img)

        for i, item in enumerate(multi[img_id]):
            x1, y1, x2, y2 = item["bbox"]
            color = plt.cm.tab10.colors[i % 10]
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2.5,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            label = (
                f"{item['class_name']}\n"
                f"S1:{item['det_score']:.2f} S2:{item['class_score']:.2f}"
            )
            ax.text(
                x1,
                y2 + 4,
                label,
                color="white",
                fontsize=7,
                fontweight="bold",
                bbox=dict(
                    facecolor=color, alpha=0.8, pad=2, edgecolor="white", linewidth=1
                ),
            )

        ax.set_title(f"{img_id}  ({len(multi[img_id])}개 탐지)", fontsize=9)
        ax.axis("off")

    for ax in axes.flatten()[len(sample_ids) :]:
        ax.axis("off")

    plt.suptitle(
        "2-Stage 통합 결과 — bbox + 약품명 + 신뢰도\n"
        "S1: Stage 1 탐지 신뢰도 | S2: Stage 2 분류 신뢰도",
        fontsize=13,
    )
    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "pipeline_overlay.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✓ 저장: {out}")
    plt.show()


def plot_pipeline_strip(
    pipeline_by_img: dict[str, list[dict]],
    crop_dir: Path,
    img_dir: Path,
    save_dir: Path | None = None,
) -> None:
    """탐지 수 가장 많은 이미지: 원본 + crop + 분류결과 스트립 뷰."""
    from matplotlib.gridspec import GridSpec

    crop_dir = Path(crop_dir)
    img_dir = Path(img_dir)

    strip_id = max(pipeline_by_img, key=lambda k: len(pipeline_by_img[k]))
    strip_items = pipeline_by_img[strip_id]
    n = len(strip_items)

    fig = plt.figure(figsize=(3 * (n + 1), 9))
    gs = GridSpec(3, n + 1, figure=fig)

    ax0 = fig.add_subplot(gs[:, 0])
    img_path = _find_image(img_dir, strip_id)
    if img_path:
        ax0.imshow(np.array(Image.open(img_path).convert("RGB")))
    for i, item in enumerate(strip_items):
        color = plt.cm.tab10.colors[i % 10]
        x1, y1, x2, y2 = item["bbox"]
        ax0.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax0.text(x1, y1 - 5, str(i), color=color, fontsize=9, fontweight="bold")
    ax0.set_title(f"원본  {strip_id}", fontsize=8)
    ax0.axis("off")

    for i, item in enumerate(strip_items):
        color = plt.cm.tab10.colors[i % 10]
        ax_num = fig.add_subplot(gs[0, i + 1])
        ax_crop = fig.add_subplot(gs[1, i + 1])
        ax_cls = fig.add_subplot(gs[2, i + 1])

        ax_num.text(
            0.5,
            0.5,
            str(i),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=color,
            transform=ax_num.transAxes,
        )
        ax_num.axis("off")

        crop_path = crop_dir / f"{item['crop_id']}.jpg"
        if crop_path.exists():
            ax_crop.imshow(Image.open(crop_path).convert("RGB"))
        ax_crop.set_title(f"det={item['det_score']:.2f}", fontsize=7)
        ax_crop.axis("off")

        ax_cls.text(
            0.5,
            0.5,
            f"{item['class_name']}\n{item['class_score']:.3f}",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            transform=ax_cls.transAxes,
            bbox=dict(facecolor=color, alpha=0.25, pad=3),
        )
        ax_cls.axis("off")

    plt.suptitle(f"2-Stage 스트립 — {strip_id}", fontsize=12)
    plt.tight_layout()
    if save_dir:
        out = Path(save_dir) / "pipeline_strip.png"
        plt.savefig(out, dpi=150)
        print(f"✓ 저장: {out}")
    plt.show()
