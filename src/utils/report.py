"""실험 결과 요약 출력 유틸리티.

노트북에서 반복되는 지표 읽기·출력 코드를 함수로 제공한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_TIMING_LABELS: dict[str, str] = {
    "s1_train": "S1 학습",
    "s1_validate": "S1 검증",
    "s1_predict": "S1 추론",
    "crop": "Crop 생성",
    "s2_train": "S2 학습",
    "s2_predict": "S2 추론",
    "pipeline_predict": "파이프라인 추론",
}


def load_timings(exp_dir: Path) -> dict:
    """exp_dir/timings.json 로드. 없으면 빈 dict 반환."""
    path = Path(exp_dir) / "timings.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_timings(exp_dir: Path) -> None:
    """소요 시간 요약 출력. 파이프라인 오버헤드도 계산해서 표시."""
    timings = load_timings(exp_dir)
    if not timings:
        print("⚠️  timings.json 없음 — 스크립트 실행 후 자동 기록됩니다")
        return

    print("\n⏱  소요 시간 요약")
    print("=" * 44)
    for key, label in _TIMING_LABELS.items():
        if key in timings:
            sec = timings[key]
            print(f"  {label:<14s}: {sec:>7.1f}초  ({sec / 60:.1f}분)")

    pipeline = timings.get("pipeline_predict")
    sub_total = sum(timings.get(k, 0) for k in ["s1_predict", "crop", "s2_predict"])
    if pipeline and sub_total:
        print()
        print(f"  추론 합산 (S1+crop+S2): {sub_total:.1f}초")
        print(f"  전체 파이프라인:        {pipeline:.1f}초")
        print(f"  merge/save 오버헤드:    {pipeline - sub_total:.1f}초")


def load_s1_best_metrics(results_csv: Path) -> dict | None:
    """results.csv에서 mAP50 최고 에폭 지표 반환. 파일 없으면 None.

    Returns:
        {"epoch", "mAP50", "mAP50_95", "precision", "recall"} 또는 None
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    r = df.loc[df["metrics/mAP50(B)"].idxmax()]
    return {
        "epoch": int(r["epoch"]),
        "mAP50": float(r["metrics/mAP50(B)"]),
        "mAP50_95": float(r["metrics/mAP50-95(B)"]),
        "precision": float(r["metrics/precision(B)"]),
        "recall": float(r["metrics/recall(B)"]),
    }


def load_s2_best_metrics(best_pt_path: Path) -> dict | None:
    """Stage 2 best.pt 체크포인트에서 지표 반환. 파일 없으면 None.

    Returns:
        {"epoch", "top1_acc", "top5_acc", "n_classes"} 또는 None
    """
    # torch lazy-imported to keep report usable without GPU env
    import torch

    best_pt_path = Path(best_pt_path)
    if not best_pt_path.exists():
        return None
    ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {})
    return {
        "epoch": int(ckpt.get("epoch", -1)) + 1,
        "top1_acc": float(metrics.get("top1_acc", 0.0)),
        "top5_acc": float(metrics.get("top5_acc", 0.0)),
        "n_classes": len(ckpt.get("class_names", [])),
    }
