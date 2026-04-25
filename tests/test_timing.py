"""src/utils/timing.py 단위 테스트."""

import json
import time

import pytest

from src.utils.timing import exp_dir_from_cfg, record_time, timed


# ── exp_dir_from_cfg ──────────────────────────────────────────────────────────


def test_exp_dir_from_cfg_s1(sample_config):
    """S1 config: project/name 형태를 반환한다."""
    # sample_config output: project=experiments, name=test_run
    result = exp_dir_from_cfg(sample_config)
    assert str(result) == "experiments/test_run"


def test_exp_dir_from_cfg_s2_via_s1_config(tmp_path, sample_stage2_config):
    """S2 config: project 디렉터리에 s1_config.yaml이 있으면 project 자체를 반환한다."""
    (tmp_path / "s1_config.yaml").write_text("dummy")
    sample_stage2_config["output"]["project"] = str(tmp_path)
    result = exp_dir_from_cfg(sample_stage2_config)
    assert result == tmp_path


def test_exp_dir_from_cfg_s2_via_weights_dir(tmp_path, sample_stage2_config):
    """S2 config: project 디렉터리에 weights/가 있으면 project 자체를 반환한다."""
    (tmp_path / "weights").mkdir()
    sample_stage2_config["output"]["project"] = str(tmp_path)
    result = exp_dir_from_cfg(sample_stage2_config)
    assert result == tmp_path


def test_exp_dir_from_cfg_empty_output():
    """output 섹션이 없으면 experiments 하위 경로를 반환한다."""
    result = exp_dir_from_cfg({})
    assert str(result) == "experiments"


# ── record_time ───────────────────────────────────────────────────────────────


def test_record_time_creates_file(tmp_path):
    record_time(tmp_path, "s1_train", 42.3)
    path = tmp_path / "timings.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["s1_train"] == pytest.approx(42.3, abs=0.05)


def test_record_time_accumulates_keys(tmp_path):
    record_time(tmp_path, "s1_train", 100.0)
    record_time(tmp_path, "crop", 5.0)
    data = json.loads((tmp_path / "timings.json").read_text())
    assert "s1_train" in data
    assert "crop" in data


def test_record_time_overwrites_same_key(tmp_path):
    record_time(tmp_path, "s1_train", 100.0)
    record_time(tmp_path, "s1_train", 50.0)
    data = json.loads((tmp_path / "timings.json").read_text())
    assert data["s1_train"] == pytest.approx(50.0, abs=0.05)


# ── timed ─────────────────────────────────────────────────────────────────────


def test_timed_records_elapsed(tmp_path):
    with timed(tmp_path, "test_key"):
        time.sleep(0.05)
    data = json.loads((tmp_path / "timings.json").read_text())
    assert "test_key" in data
    assert data["test_key"] >= 0.0


def test_timed_prints_elapsed(tmp_path, capsys):
    with timed(tmp_path, "crop"):
        pass
    out = capsys.readouterr().out
    assert "crop" in out
