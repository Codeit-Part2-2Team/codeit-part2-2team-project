"""timm 기반 Stage 2 분류기 래퍼."""

from __future__ import annotations

import copy
import json
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

from src.utils.config import fix_seed, load_config, validate_config

_MODEL_NAME_MAP = {
    "resnet50": "resnet50",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnetv2_s": "efficientnetv2_rw_s",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Classifier:
    """Stage 2 config를 받아 timm 분류기를 생성하는 래퍼."""

    def __init__(self, config: str | Path | dict):
        self.cfg = load_config(config) if not isinstance(config, dict) else config
        validate_config(self.cfg)
        fix_seed(self.cfg.get("seed", 42))

        self.device = _resolve_device(self.cfg["train"].get("device", ""))
        self.class_names: list[str] = list(self.cfg.get("classes") or [])
        self.model = self._build_model().to(self.device)

    def load_weights(self, path: str | Path) -> "Classifier":
        """체크포인트 또는 state_dict를 로드한다."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            self.class_names = list(checkpoint.get("class_names", self.class_names))
            # 저장된 클래스 수와 현재 모델이 다르면 모델을 재빌드
            actual_nc = len(self.class_names)
            if actual_nc and actual_nc != self.cfg["model"]["num_classes"]:
                self.cfg = copy.deepcopy(self.cfg)
                self.cfg["model"]["num_classes"] = actual_nc
                self.model = self._build_model().to(self.device)
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        return self

    # ------------------------------------------------------------------
    # High-level interface
    # ------------------------------------------------------------------

    def train(self, data_dir: str | Path | None = None) -> dict:
        """학습 실행. DataLoader 생성 후 fit()을 호출한다.

        Returns:
            {"top1_acc": float, "top5_acc": float}
        """
        from src.data.stage2_dataset import Stage2Dataset

        cfg = self.cfg
        if data_dir:
            base = Path(data_dir)
            train_dir, val_dir = base / "train", base / "val"
        else:
            train_dir = Path(cfg["data"]["train"])
            val_dir = Path(cfg["data"]["val"])

        batch = cfg["train"]["batch"]
        workers = cfg["data"]["workers"]

        train_ds = Stage2Dataset(train_dir, cfg, split="train")
        val_ds = Stage2Dataset(val_dir, cfg, split="val", classes=train_ds.classes)

        actual_nc = len(train_ds.classes)
        cfg_nc = self.cfg["model"]["num_classes"]
        if cfg_nc != actual_nc:
            raise ValueError(
                f"config model.num_classes={cfg_nc}이지만 "
                f"실제 데이터셋 클래스 수는 {actual_nc}개입니다. "
                f"config를 num_classes: {actual_nc}로 수정하세요."
            )
        self.class_names = train_ds.classes

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
        return self.fit(train_loader, val_loader)

    def predict(self, source: str | Path, output: str | Path) -> list[dict]:
        """크롭 이미지 디렉터리를 추론하고 JSON을 저장한다.

        Returns:
            [{"image_id", "crop_id", "class_id", "class_name", "score"}]
        """
        source = Path(source)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        manifest = _load_manifest(source)
        image_files = sorted(
            p for p in source.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        # ImageFolder 구조(class_name/ 하위) 대응: 최상위에 이미지가 없으면 재귀 탐색
        if not image_files:
            image_files = sorted(
                p for p in source.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
            )

        dataset = _InferenceDataset(image_files, manifest, self.cfg)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg["train"]["batch"],
            shuffle=False,
            num_workers=self.cfg["data"]["workers"],
        )

        results = self.predict_loader(loader)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    # ------------------------------------------------------------------
    # Low-level interface
    # ------------------------------------------------------------------

    def fit(
        self, train_loader, val_loader, resume_from: str | Path | None = None
    ) -> dict:
        """학습 루프 실행 후 best val 지표를 반환한다.

        Returns:
            {"top1_acc": float, "top5_acc": float}
        """
        train_cfg = self.cfg["train"]
        lr0 = train_cfg["lr0"]
        lrf = train_cfg["lrf"]
        warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
        epochs = int(train_cfg["epochs"])
        label_smoothing = float(train_cfg.get("label_smoothing", 0.1))
        weight_decay = float(train_cfg.get("weight_decay", 0.01))

        # Config 기반 optimizer 생성
        optimizer_name = train_cfg.get("optimizer", "AdamW")
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr0, weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            momentum = float(train_cfg.get("momentum", 0.9))
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {optimizer_name}. Supported: AdamW, Adam, SGD"
            )

        # Config 기반 scheduler 생성 (기본: cosine with warmup)
        scheduler_name = train_cfg.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=max(warmup_epochs, 1),
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr0 * lrf
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        else:
            raise ValueError(
                f"Unsupported scheduler: {scheduler_name}. Supported: cosine"
            )
        # Config 기반 criterion 생성
        criterion_name = train_cfg.get("criterion", "cross_entropy")
        if criterion_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif criterion_name == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif criterion_name == "focal":
            alpha = float(train_cfg.get("focal_alpha", 0.25))
            gamma = float(train_cfg.get("focal_gamma", 2.0))
            criterion = FocalLoss(
                alpha=alpha, gamma=gamma, label_smoothing=label_smoothing
            )
        else:
            raise ValueError(
                f"Unsupported criterion: {criterion_name}. Supported: cross_entropy, bce, focal"
            )

        out_cfg = self.cfg["output"]
        weights_dir = Path(out_cfg["project"]) / out_cfg["name"] / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        start_epoch = 0
        best_top1 = 0.0
        best_metrics: dict = {"top1_acc": 0.0, "top5_acc": 0.0}
        last_metrics: dict = best_metrics

        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
            ckpt_classes = ckpt.get("class_names", [])
            if ckpt_classes and len(ckpt_classes) != self.cfg["model"]["num_classes"]:
                raise ValueError(
                    f"resume checkpoint has {len(ckpt_classes)} classes, "
                    f"current model expects {self.cfg['model']['num_classes']}"
                )
            self.model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_top1 = ckpt["metrics"].get("top1_acc", 0.0)
            best_metrics = ckpt["metrics"]
            self.class_names = list(ckpt.get("class_names", self.class_names))
            print(
                f"resumed from epoch {start_epoch}/{epochs}  best_top1={best_top1:.4f}"
            )

        for epoch in range(start_epoch, epochs):
            self.model.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(images), labels).backward()
                optimizer.step()
            scheduler.step()

            last_metrics = self.evaluate(val_loader)
            top1 = last_metrics.get("top1_acc", 0.0)
            top5 = last_metrics.get("top5_acc", 0.0)
            lr = optimizer.param_groups[0]["lr"]
            marker = ""
            if top1 > best_top1:
                best_top1 = top1
                best_metrics = last_metrics
                marker = "  ✓ best"
                torch.save(
                    self._checkpoint(epoch, optimizer, scheduler, last_metrics),
                    weights_dir / "best.pt",
                )
            print(
                f"epoch {epoch + 1:>3}/{epochs}  "
                f"top1={top1:.4f}  top5={top5:.4f}  lr={lr:.2e}{marker}"
            )

        torch.save(
            self._checkpoint(epochs - 1, optimizer, scheduler, last_metrics),
            weights_dir / "last.pt",
        )
        return best_metrics

    def evaluate(self, val_loader) -> dict:
        """검증 루프 실행.

        Returns:
            {"top1_acc": float, "top5_acc": float}
        """
        top_k = self.cfg["val"].get("top_k", [1, 5])
        max_k = max(top_k)

        self.model.eval()
        correct = {k: 0 for k in top_k}
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                batch_size = labels.size(0)
                total += batch_size

                _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
                correct_mask = pred.t().eq(labels.view(1, -1).expand_as(pred.t()))
                for k in top_k:
                    correct[k] += int(correct_mask[:k].reshape(-1).float().sum().item())

        denom = total if total else 1
        return {f"top{k}_acc": correct[k] / denom for k in sorted(top_k)}

    def predict_loader(self, loader) -> list[dict]:
        """DataLoader 기반 배치 추론.

        loader의 각 배치는 {"image": Tensor, "image_id": list, "crop_id": list} 형식.
        Returns:
            [{"image_id", "crop_id", "class_id", "class_name", "score"}]
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                probs = torch.softmax(self.model(images), dim=1)
                scores, indices = probs.max(dim=1)

                for i, (class_idx, score) in enumerate(
                    zip(indices.cpu().tolist(), scores.cpu().tolist())
                ):
                    results.append(
                        {
                            "image_id": batch["image_id"][i],
                            "crop_id": batch["crop_id"][i],
                            "class_id": class_idx,
                            "class_name": (
                                self.class_names[class_idx]
                                if self.class_names
                                else str(class_idx)
                            ),
                            "score": float(score),
                        }
                    )

        return results

    def export(self, format: str = "onnx") -> Path:
        """현재 모델을 ONNX로 내보낸다."""
        if format != "onnx":
            raise ValueError(f"현재 지원하지 않는 export format입니다: {format}")

        output_path = self._build_export_path(format)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        imgsz = int(self.cfg["data"]["imgsz"])
        dummy = torch.randn(1, 3, imgsz, imgsz, device=self.device)
        self.model.eval()
        torch.onnx.export(self.model, dummy, output_path, opset_version=17)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        timm = _load_timm()
        model_cfg = self.cfg["model"]
        return timm.create_model(
            _resolve_model_name(model_cfg["name"]),
            pretrained=bool(model_cfg.get("pretrained", True)),
            num_classes=int(model_cfg["num_classes"]),
        )

    def _checkpoint(self, epoch: int, optimizer, scheduler, metrics: dict) -> dict:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "class_names": self.class_names,
            "cfg": self.cfg,
        }

    def _build_export_path(self, format: str) -> Path:
        out = self.cfg["output"]
        return (
            Path(out["project"])
            / out["name"]
            / "weights"
            / f"{self.cfg['model']['name']}.{format}"
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


class _InferenceDataset(Dataset):
    """추론용 flat 크롭 디렉터리 데이터셋."""

    def __init__(self, image_files: list[Path], manifest: dict | None, cfg: dict):
        self.image_files = image_files
        self.manifest = manifest
        self.imgsz = int(cfg["data"]["imgsz"])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_files[idx]
        stem = path.stem

        img = (
            Image.open(path)
            .convert("RGB")
            .resize((self.imgsz, self.imgsz), Image.BILINEAR)
        )
        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

        if self.manifest and stem in self.manifest:
            image_id = self.manifest[stem]["image_id"]
        else:
            parts = stem.rsplit("_", 1)
            image_id = parts[0] if len(parts) == 2 else stem

        return {"image": tensor, "image_id": image_id, "crop_id": stem}


def _load_manifest(source: Path) -> dict | None:
    manifest_path = source / "crops_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return {item["crop_id"]: item for item in json.load(f)}


def _load_timm():
    try:
        return import_module("timm")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "timm 패키지가 필요합니다. Main/requirements.txt를 설치하세요."
        ) from exc


def _resolve_model_name(name: str) -> str:
    try:
        return _MODEL_NAME_MAP[name]
    except KeyError as exc:
        raise ValueError(
            f"지원하지 않는 Stage 2 모델입니다: {name}. 지원 모델: {sorted(_MODEL_NAME_MAP)}"
        ) from exc


def _resolve_device(device: str | None) -> str:
    if device in {None, ""}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if "," in device:
        return f"cuda:{device.split(',')[0].strip()}"
    if str(device).isdigit():
        return f"cuda:{device}"
    return device


class FocalLoss(nn.Module):
    """Focal Loss for classification."""

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
