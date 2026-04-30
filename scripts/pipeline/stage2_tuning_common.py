# isort: skip_file
# ruff: noqa: E402
"""Stage 2 tuning runner 공통 유틸."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

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
from src.utils.timing import record_time


DEFAULT_GRID_SPACE = {
    "train.lr0": [3e-5, 1e-4, 3e-4],
    "train.weight_decay": [1e-3, 1e-2],
    "train.label_smoothing": [0.0, 0.05, 0.1],
}

DEFAULT_OPTUNA_SPACE = {
    "train.lr0": {"type": "float", "low": 1e-5, "high": 3e-4, "log": True},
    "train.lrf": {"type": "float", "low": 0.005, "high": 0.05, "log": True},
    "train.weight_decay": {
        "type": "float",
        "low": 1e-4,
        "high": 1e-2,
        "log": True,
    },
    "train.label_smoothing": {"type": "float", "low": 0.0, "high": 0.15},
}

RESULT_FIELDS = [
    "trial",
    "status",
    "score",
    "top1_acc",
    "top5_acc",
    "n_classes",
    "elapsed_sec",
    "config",
    "params",
    "error",
]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_search_space(path: str | Path | None, default: dict) -> dict:
    if path is None:
        return deepcopy(default)
    return flatten_mapping(load_yaml(path))


def flatten_mapping(mapping: dict, prefix: str = "") -> dict:
    """Nested YAML과 dotted-key YAML을 모두 dotted-key dict로 변환한다."""
    flat = {}
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and "type" not in value:
            flat.update(flatten_mapping(value, dotted))
        else:
            flat[dotted] = value
    return flat


def set_by_dotted_key(data: dict, dotted_key: str, value: Any) -> None:
    cur = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def build_trial_config(
    base_cfg: dict,
    params: dict[str, Any],
    output_dir: Path,
    trial_name: str,
    epochs: int | None = None,
    device: str | None = None,
) -> dict:
    cfg = deepcopy(base_cfg)
    for key, value in params.items():
        set_by_dotted_key(cfg, key, value)

    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if device is not None:
        cfg["train"]["device"] = device

    cfg["output"]["project"] = str(output_dir)
    cfg["output"]["name"] = trial_name
    return cfg


def iter_grid_params(search_space: dict) -> list[dict[str, Any]]:
    keys = list(search_space)
    values = [
        value if isinstance(value, list) else [value]
        for value in (search_space[key] for key in keys)
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def train_stage2(cfg: dict, data_root: str | Path | None = None) -> dict:
    """Stage 2 classifier를 학습하고 best validation metric을 반환한다."""
    if data_root:
        base = Path(data_root)
        train_dir = base / "train"
        val_dir = base / "val"
    else:
        train_dir = Path(cfg["data"]["train"])
        val_dir = Path(cfg["data"]["val"])

    train_ds = Stage2Dataset(train_dir, cfg, split="train")
    val_ds = Stage2Dataset(val_dir, cfg, split="val", classes=train_ds.classes)

    actual_nc = len(train_ds.classes)
    cfg_nc = int(cfg["model"]["num_classes"])
    if cfg_nc != actual_nc:
        raise ValueError(
            f"config model.num_classes={cfg_nc}이지만 "
            f"실제 데이터셋 클래스 수는 {actual_nc}개입니다."
        )

    batch = int(cfg["train"]["batch"])
    workers = int(cfg["data"]["workers"])
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
    metrics = classifier.fit(train_loader, val_loader)
    return {
        **metrics,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_classes": actual_nc,
    }


def record_trial_time(trial_dir: str | Path, elapsed_sec: float) -> float:
    """Record one HPO trial duration in trial_dir/timings.json."""
    elapsed_sec = round(elapsed_sec, 1)
    record_time(trial_dir, "hpo_trial", elapsed_sec)
    return elapsed_sec


def select_metric(metrics: dict, metric: str) -> float:
    try:
        return float(metrics[metric])
    except KeyError as exc:
        raise KeyError(
            f"metric={metric!r}이 metrics에 없습니다. 사용 가능: {sorted(metrics)}"
        ) from exc


def append_result(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = RESULT_FIELDS
    safe_row = {field: row.get(field, "") for field in fieldnames}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(safe_row)


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def result_row(
    trial_name: str,
    params: dict,
    metrics: dict,
    score: float,
    config_path: Path,
    elapsed_sec: float | None = None,
) -> dict:
    return {
        "trial": trial_name,
        "status": "ok",
        "score": score,
        "top1_acc": metrics.get("top1_acc"),
        "top5_acc": metrics.get("top5_acc"),
        "n_classes": metrics.get("n_classes"),
        "elapsed_sec": round(elapsed_sec, 1) if elapsed_sec is not None else "",
        "config": str(config_path),
        "params": json.dumps(params, sort_keys=True),
        "error": "",
    }


def print_best(results: list[dict], metric: str) -> None:
    if not results:
        return
    best = max(results, key=lambda row: float(row["score"]))
    print(
        f"[best] {best['trial']}  {metric}={float(best['score']):.4f}  "
        f"top1={float(best['top1_acc']):.4f}  top5={float(best['top5_acc']):.4f}"
    )


def load_base_config(path: str | Path) -> dict:
    return load_config(path)
