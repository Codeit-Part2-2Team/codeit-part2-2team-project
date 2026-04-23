"""프로젝트 경로 관리 유틸리티."""

from __future__ import annotations

from pathlib import Path


# 프로젝트 루트 (Main/ 디렉터리)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """프로젝트 루트 기준으로 경로를 해석한다.

    Args:
        path: 상대 또는 절대 경로 문자열/객체.

    Returns:
        절대 경로. 입력이 절대 경로면 그대로, 상대 경로면 PROJECT_ROOT / path.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_weights_path(cfg: dict, weights_arg: str | None) -> Path:
    """config와 CLI 인자를 기반으로 weights 경로를 해석한다.

    Args:
        cfg: config 딕셔너리.
        weights_arg: CLI --weights 인자 (None이면 config 기반).

    Returns:
        절대 weights 경로.
    """
    if weights_arg:
        return resolve_project_path(weights_arg)
    output = cfg["output"]
    return PROJECT_ROOT / output["project"] / output["name"] / "weights" / "best.pt"


def resolve_predictions_path(cfg: dict, output_arg: str | None) -> Path:
    """config와 CLI 인자를 기반으로 predictions.json 경로를 해석한다.

    Args:
        cfg: config 딕셔너리.
        output_arg: CLI --output 인자 (None이면 config 기반).

    Returns:
        절대 predictions.json 경로.
    """
    if output_arg:
        return resolve_project_path(output_arg)
    output = cfg["output"]
    return PROJECT_ROOT / output["project"] / output["name"] / "results" / "predictions.json"


def resolve_stage2_predictions_path(cfg: dict, output_arg: str | None) -> Path:
    """config와 CLI 인자를 기반으로 stage2_predictions.json 경로를 해석한다.

    Args:
        cfg: config 딕셔너리.
        output_arg: CLI --output 인자 (None이면 config 기반).

    Returns:
        절대 stage2_predictions.json 경로.
    """
    if output_arg:
        return resolve_project_path(output_arg)
    output = cfg["output"]
    return PROJECT_ROOT / output["project"] / output["name"] / "results" / "stage2_predictions.json"