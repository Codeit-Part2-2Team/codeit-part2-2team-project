# isort: skip_file
# ruff: noqa: E402
"""Stage 2 분류기 자동 Grid Search runner.

목표는 Stage 2 classifier 자체의 validation metric(top1_acc 기본)을 빠르게
비교하는 것이다. 상위 trial은 별도로 E2E 평가를 수행한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

from scripts.pipeline.stage2_tuning_common import (
    DEFAULT_GRID_SPACE,
    append_result,
    build_trial_config,
    iter_grid_params,
    load_base_config,
    load_search_space,
    print_best,
    record_trial_time,
    result_row,
    save_yaml,
    select_metric,
    train_stage2,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Grid Search")
    parser.add_argument("--base-config", required=True, help="기준 s2_config.yaml")
    parser.add_argument(
        "--search-space",
        help="Grid search space YAML. 미지정 시 기본 탐색 공간 사용",
    )
    parser.add_argument(
        "--output",
        default="experiments/stage2_grid_search",
        help="trial 산출물 저장 루트",
    )
    parser.add_argument("--data", help="Stage 2 crop root (<data>/train, <data>/val)")
    parser.add_argument("--metric", default="top1_acc", help="최적화 metric")
    parser.add_argument("--max-trials", type=int, help="앞에서 N개 조합만 실행")
    parser.add_argument("--epochs", type=int, help="탐색용 epoch override")
    parser.add_argument("--device", help="탐색용 device override")
    args = parser.parse_args()

    base_cfg = load_base_config(args.base_config)
    output_dir = Path(args.output)
    search_space = load_search_space(args.search_space, DEFAULT_GRID_SPACE)
    all_params = iter_grid_params(search_space)
    if args.max_trials is not None:
        all_params = all_params[: args.max_trials]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(search_space, output_dir / "search_space.yaml")

    results: list[dict] = []
    for idx, params in enumerate(all_params):
        trial_name = f"trial_{idx:04d}"
        trial_dir = output_dir / trial_name
        cfg = build_trial_config(
            base_cfg,
            params,
            output_dir=output_dir,
            trial_name=trial_name,
            epochs=args.epochs,
            device=args.device,
        )
        config_path = trial_dir / "config.yaml"
        save_yaml(cfg, config_path)

        print(f"\n[grid] {trial_name}/{len(all_params):04d} params={params}")
        t0 = time.perf_counter()
        try:
            metrics = train_stage2(cfg, data_root=args.data)
            elapsed_sec = record_trial_time(trial_dir, time.perf_counter() - t0)
            score = select_metric(metrics, args.metric)
            row = result_row(
                trial_name, params, metrics, score, config_path, elapsed_sec
            )
        except Exception as exc:  # noqa: BLE001 - trial 실패를 기록하고 계속 진행
            elapsed_sec = record_trial_time(trial_dir, time.perf_counter() - t0)
            row = {
                "trial": trial_name,
                "status": "failed",
                "score": "",
                "top1_acc": "",
                "top5_acc": "",
                "n_classes": "",
                "elapsed_sec": round(elapsed_sec, 1),
                "config": str(config_path),
                "params": json.dumps(params, sort_keys=True),
                "error": repr(exc),
            }
            print(f"[grid] failed: {exc!r}")
        print(f"[grid] {trial_name} elapsed={elapsed_sec:.1f}s")

        append_result(output_dir / "results.csv", row)
        write_json(trial_dir / "result.json", row)
        if row.get("status") == "ok":
            results.append(row)

    write_json(output_dir / "best_trial.json", max(results, key=lambda r: r["score"]) if results else {})
    print_best(results, args.metric)


if __name__ == "__main__":
    main()
