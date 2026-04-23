"""설정 유틸: YAML 로딩, 설정 검증, 시드 고정."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

_REQUIRED_KEYS = {"model", "data", "train", "augment", "val", "output"}


def load_config(path: str | Path) -> dict:
    """YAML 파일을 읽어 딕셔너리로 반환한다."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    """필수 키 존재 여부를 검증한다. 누락 시 ValueError."""
    missing = _REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"config에 필수 키가 없습니다: {sorted(missing)}")


def fix_seed(seed: int) -> None:
    """Python / NumPy / PyTorch 난수 시드를 고정해 재현성을 보장한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
