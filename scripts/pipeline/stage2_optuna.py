# isort: skip_file
# ruff: noqa: E402
"""Stage 2 분류기 Optuna HPO runner.

Objective는 Stage 2 validation metric(top1_acc 기본)이다. Stage 1은 freeze하고,
상위 trial만 별도 E2E 평가로 검증하는 흐름을 권장한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    DEFAULT_OPTUNA_SPACE,
    append_result,
    build_trial_config,
    load_base_config,
    load_search_space,
    record_trial_time,
    result_row,
    save_yaml,
    select_metric,
    train_stage2,
    write_json,
)


def suggest_params(trial: Any, search_space: dict) -> dict[str, Any]:
    params = {}
    for key, spec in search_space.items():
        if not isinstance(spec, dict):
            params[key] = trial.suggest_categorical(
                key, spec if isinstance(spec, list) else [spec]
            )
            continue

        kind = spec.get("type", "categorical")
        if kind == "categorical":
            params[key] = trial.suggest_categorical(key, spec["choices"])
        elif kind == "float":
            params[key] = trial.suggest_float(
                key,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "int":
            params[key] = trial.suggest_int(
                key,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        else:
            raise ValueError(f"Unsupported Optuna search type: {kind}")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Optuna HPO")
    parser.add_argument("--base-config", required=True, help="기준 s2_config.yaml")
    parser.add_argument(
        "--search-space",
        help="Optuna search space YAML. 미지정 시 기본 탐색 공간 사용",
    )
    parser.add_argument(
        "--output",
        default="experiments/stage2_optuna",
        help="trial 산출물 저장 루트",
    )
    parser.add_argument("--data", help="Stage 2 crop root (<data>/train, <data>/val)")
    parser.add_argument("--metric", default="top1_acc", help="최적화 metric")
    parser.add_argument("--n-trials", type=int, default=20, help="trial 수")
    parser.add_argument("--epochs", type=int, help="탐색용 epoch override")
    parser.add_argument("--device", help="탐색용 device override")
    parser.add_argument("--study-name", default="stage2_optuna")
    args = parser.parse_args()

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna 패키지가 필요합니다. requirements.txt를 설치하세요."
        ) from exc

    base_cfg = load_base_config(args.base_config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = load_search_space(args.search_space, DEFAULT_OPTUNA_SPACE)
    save_yaml(search_space, output_dir / "search_space.yaml")

    storage = f"sqlite:///{output_dir / 'study.db'}"

    def objective(trial: Any) -> float:
        params = suggest_params(trial, search_space)
        trial_name = f"trial_{trial.number:04d}"
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

        print(f"\n[optuna] {trial_name} params={params}")
        t0 = time.perf_counter()
        try:
            metrics = train_stage2(cfg, data_root=args.data)
        except Exception as exc:  # noqa: BLE001 - 실패 trial도 CSV에 남긴다
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
            append_result(output_dir / "results.csv", row)
            write_json(trial_dir / "result.json", row)
            raise

        elapsed_sec = record_trial_time(trial_dir, time.perf_counter() - t0)
        score = select_metric(metrics, args.metric)

        row = result_row(trial_name, params, metrics, score, config_path, elapsed_sec)
        append_result(output_dir / "results.csv", row)
        write_json(trial_dir / "result.json", row)

        trial.set_user_attr("config", str(config_path))
        trial.set_user_attr("top1_acc", metrics.get("top1_acc"))
        trial.set_user_attr("top5_acc", metrics.get("top5_acc"))
        trial.set_user_attr("n_classes", metrics.get("n_classes"))
        trial.set_user_attr("elapsed_sec", round(elapsed_sec, 1))
        print(f"[optuna] {trial_name} elapsed={elapsed_sec:.1f}s")
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
    }
    write_json(output_dir / "best_trial.json", best)
    print(f"[best] trial_{study.best_trial.number:04d} {args.metric}={study.best_value:.4f}")


if __name__ == "__main__":
    main()
