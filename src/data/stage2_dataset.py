"""Stage 2 분류기 학습/검증용 manifest 기반 데이터셋."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentations import build_stage2_transforms

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Stage2Dataset(Dataset):
    """crops_manifest.json 기반 Stage 2 분류 데이터셋.

    root 디렉터리 또는 manifest JSON 파일 경로를 받는다.
    class_name이 없는 레코드(iOS 파일 등)는 자동으로 제외된다.
    """

    def __init__(
        self, root: str | Path, cfg: dict, split: str, classes: list[str] | None = None
    ):
        root = Path(root)
        manifest_path = root if root.suffix == ".json" else root / "crops_manifest.json"

        with open(manifest_path, encoding="utf-8") as f:
            records = [r for r in json.load(f) if r.get("class_name")]

        if classes is not None:
            self.classes = list(classes)
        else:
            self.classes = sorted({r["class_name"] for r in records})
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = [
            (Path(r["crop_path"]), self.class_to_idx[r["class_name"]])
            for r in records
            if r["class_name"] in self.class_to_idx
        ]
        self.imgsz = int(cfg["data"]["imgsz"])
        self.transform = build_stage2_transforms(cfg, split)

    def get_sample_weights(self) -> list[float]:
        """클래스 빈도 역수 기반 샘플 가중치 반환 (WeightedRandomSampler용)."""
        from collections import Counter

        label_counts = Counter(label for _, label in self.samples)
        return [1.0 / label_counts[label] for _, label in self.samples]

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
