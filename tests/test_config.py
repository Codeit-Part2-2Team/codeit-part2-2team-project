"""src/utils/config.py 단위 테스트."""

import pytest
import torch

from src.utils.config import fix_seed, load_config, validate_config

# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("seed: 42\nmodel:\n  name: yolo26n\n")

    cfg = load_config(config_file)

    assert cfg["seed"] == 42
    assert cfg["model"]["name"] == "yolo26n"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


def test_validate_config_passes_with_all_keys(sample_config):
    validate_config(sample_config)  # 예외 없음


def test_validate_config_raises_on_missing_key(sample_config):
    del sample_config["model"]
    with pytest.raises(ValueError, match="model"):
        validate_config(sample_config)


def test_validate_config_raises_on_multiple_missing_keys(sample_config):
    del sample_config["train"]
    del sample_config["val"]
    with pytest.raises(ValueError):
        validate_config(sample_config)


# ---------------------------------------------------------------------------
# fix_seed
# ---------------------------------------------------------------------------


def test_fix_seed_sets_torch_seed():
    fix_seed(123)
    assert torch.initial_seed() == 123


def test_fix_seed_different_seeds_produce_different_results():
    fix_seed(0)
    a = torch.rand(1).item()
    fix_seed(1)
    b = torch.rand(1).item()
    assert a != b


def test_fix_seed_same_seed_is_reproducible():
    fix_seed(99)
    a = torch.rand(5).tolist()
    fix_seed(99)
    b = torch.rand(5).tolist()
    assert a == b
