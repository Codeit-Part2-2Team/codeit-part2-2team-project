"""실험 소요 시간 기록 유틸."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


def exp_dir_from_cfg(cfg: dict) -> Path:
    """config의 output 섹션에서 실험 디렉터리 경로를 반환한다.

    S1 config: project=experiments, name=exp_xxx  → experiments/exp_xxx
    S2 config: project=experiments/exp_xxx, name=stage2 → experiments/exp_xxx
    """
    out = cfg.get("output", {})
    project = Path(out.get("project", "experiments"))
    name = out.get("name", "")
    candidate = project / name
    # S2의 경우 project 자체가 실험 디렉터리 (name이 stage2 등 하위 폴더)
    if (project / "s1_config.yaml").exists() or (project / "weights").exists():
        return project
    return candidate


def record_time(exp_dir: Path | str, key: str, elapsed: float) -> None:
    path = Path(exp_dir) / "timings.json"
    timings = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    timings[key] = round(elapsed, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timings, ensure_ascii=False, indent=2))


@contextmanager
def timed(exp_dir: Path | str, key: str):
    """with timed(exp_dir, 's1_train'): ... 형태로 사용."""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    record_time(exp_dir, key, elapsed)
    print(f"⏱  {key}: {elapsed:.1f}초")
