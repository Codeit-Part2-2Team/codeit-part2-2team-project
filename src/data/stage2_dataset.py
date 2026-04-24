"""Stage 2 분류기 학습/검증용 ImageFolder 방식 데이터셋."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentations import build_stage2_transforms

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Stage2Dataset(Dataset):
    """클래스별 서브디렉터리(ImageFolder) 구조를 읽는 Stage 2 분류 데이터셋.

    디렉터리 구조:
        root/
          <class_name>/
            img001.jpg
            ...
    """

    def __init__(self, root: str | Path, cfg: dict, split: str):
        self.root = Path(root)
        self.imgsz = int(cfg["data"]["imgsz"])
        self.transform = build_stage2_transforms(cfg, split)

        self.classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = _collect_samples(self.root, self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((self.imgsz, self.imgsz), Image.BILINEAR)
        )
        img_np = np.array(img, dtype=np.float32)

        result = self.transform(image=img_np)
        img_np = result["image"].astype(np.float32)

        img_np = (img_np / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        return tensor, label


def _collect_samples(root: Path, class_to_idx: dict) -> list[tuple[Path, int]]:
    samples = []
    for cls, idx in class_to_idx.items():
        for p in (root / cls).iterdir():
            if p.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append((p, idx))
    return samples
