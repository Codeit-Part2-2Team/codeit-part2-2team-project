"""src/utils/report.py 단위 테스트."""

import json

import pandas as pd
import pytest

from src.utils.report import load_s1_best_metrics, load_timings, print_timings


@pytest.fixture
def timings_dir(tmp_path):
    """timings.json이 있는 임시 실험 디렉터리."""
    data = {
        "s1_train": 120.0,
        "s1_predict": 10.0,
        "crop": 3.0,
        "s2_predict": 8.0,
        "pipeline_predict": 25.0,
    }
    (tmp_path / "timings.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def results_csv(tmp_path):
    """최소 results.csv (3 에폭, mAP50 최대값이 에폭 2)."""
    df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "metrics/mAP50(B)": [0.50, 0.80, 0.75],
            "metrics/mAP50-95(B)": [0.30, 0.55, 0.50],
            "metrics/precision(B)": [0.60, 0.85, 0.82],
            "metrics/recall(B)": [0.55, 0.83, 0.80],
        }
    )
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


# ── load_timings ──────────────────────────────────────────────────────────────


def test_load_timings_missing_returns_empty(tmp_path):
    assert load_timings(tmp_path) == {}


def test_load_timings_reads_values(timings_dir):
    t = load_timings(timings_dir)
    assert t["s1_train"] == 120.0
    assert t["crop"] == 3.0


# ── print_timings ─────────────────────────────────────────────────────────────


def test_print_timings_warns_when_no_file(tmp_path, capsys):
    print_timings(tmp_path)
    assert "없음" in capsys.readouterr().out


def test_print_timings_shows_recorded_keys(timings_dir, capsys):
    print_timings(timings_dir)
    out = capsys.readouterr().out
    assert "S1 학습" in out
    assert "120" in out


def test_print_timings_shows_overhead(timings_dir, capsys):
    """pipeline_predict - (s1_predict + crop + s2_predict) 오버헤드가 출력된다."""
    print_timings(timings_dir)
    out = capsys.readouterr().out
    # 오버헤드 = 25.0 - (10.0 + 3.0 + 8.0) = 4.0
    assert "4.0" in out


def test_print_timings_skips_missing_keys(tmp_path, capsys):
    """기록된 키만 출력하고 없는 키는 건너뛴다."""
    (tmp_path / "timings.json").write_text(json.dumps({"s1_train": 60.0}))
    print_timings(tmp_path)
    out = capsys.readouterr().out
    assert "S1 학습" in out
    assert "S2 학습" not in out


# ── load_s1_best_metrics ──────────────────────────────────────────────────────


def test_load_s1_best_metrics_no_file_returns_none(tmp_path):
    assert load_s1_best_metrics(tmp_path / "results.csv") is None


def test_load_s1_best_metrics_returns_best_epoch(results_csv):
    m = load_s1_best_metrics(results_csv)
    assert m is not None
    assert m["epoch"] == 1  # mAP50 최대값 에폭
    assert m["mAP50"] == pytest.approx(0.80)
    assert m["mAP50_95"] == pytest.approx(0.55)
    assert m["precision"] == pytest.approx(0.85)
    assert m["recall"] == pytest.approx(0.83)


def test_load_s1_best_metrics_keys(results_csv):
    m = load_s1_best_metrics(results_csv)
    assert set(m.keys()) == {"epoch", "mAP50", "mAP50_95", "precision", "recall"}
