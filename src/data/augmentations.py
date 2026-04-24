"""Albumentations 기반 Stage 1/2 증강 빌더."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BBOX_KEY = "bbox"
_TRAIN_SPLIT = "train"
_EVAL_SPLITS = {"val", "valid", "validation", "test"}

_STAGE1_SUPPORTED = {
    "horizontal_flip",
    "vertical_flip",
    "random_rotate90",
    "shift_scale_rotate",
    "brightness_contrast",
    "random_gamma",
    "jpeg_compression",
    "downscale",
    "gaussian_blur",
    "motion_blur",
    "gauss_noise",
    "perspective",
}

_STAGE2_SUPPORTED = {
    "brightness_contrast",
    "random_gamma",
    "clahe",
    "jpeg_compression",
    "downscale",
    "gaussian_blur",
    "motion_blur",
    "bbox_jitter",
}


def build_stage1_transforms(cfg: dict) -> Any:
    """Stage 1 detection용 Albumentations Compose를 반환한다."""
    albumentations = _load_albumentations()
    transform_cfg = dict(cfg.get("albumentations") or {})
    _validate_transform_names(transform_cfg, _STAGE1_SUPPORTED, stage="stage1")

    bbox_params = _build_bbox_params(albumentations, transform_cfg.get(_BBOX_KEY))
    transforms = _build_transforms(
        albumentations,
        transform_cfg,
        allowed_names=_STAGE1_SUPPORTED,
    )
    return albumentations.Compose(transforms, bbox_params=bbox_params)


def build_stage2_transforms(cfg: dict, split: str) -> Any:
    """Stage 2 classification용 Albumentations Compose를 반환한다."""
    albumentations = _load_albumentations()
    split = split.lower()
    if split == _TRAIN_SPLIT:
        transform_cfg = dict(cfg.get("albumentations") or {})
        _validate_transform_names(transform_cfg, _STAGE2_SUPPORTED, stage="stage2")
        transforms = _build_transforms(
            albumentations,
            transform_cfg,
            allowed_names=_STAGE2_SUPPORTED,
        )
        return albumentations.Compose(transforms)

    if split in _EVAL_SPLITS:
        return albumentations.Compose([])

    raise ValueError(f"지원하지 않는 split입니다: {split}")


def _load_albumentations():
    try:
        return import_module("albumentations")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "albumentations 패키지가 필요합니다. Main/requirements.txt를 설치하세요."
        ) from exc


def _validate_transform_names(
    transform_cfg: dict,
    allowed_names: set[str],
    stage: str,
) -> None:
    unknown = sorted(
        name
        for name in transform_cfg
        if name != _BBOX_KEY and name not in allowed_names
    )
    if unknown:
        raise ValueError(f"{stage}에서 지원하지 않는 transform입니다: {unknown}")


def _build_bbox_params(albumentations, params: dict | None):
    params = dict(params or {})
    clip = bool(params.pop("clip", False))
    bbox_format = params.pop("format", "yolo")
    label_fields = params.pop("label_fields", ["class_labels"])
    min_visibility = float(params.pop("min_visibility", 0.0))
    if params:
        raise ValueError(
            f"bbox 설정에 지원하지 않는 파라미터가 있습니다: {sorted(params)}"
        )

    return albumentations.BboxParams(
        format=bbox_format,
        label_fields=label_fields,
        min_visibility=min_visibility,
        clip=clip,
    )


def _build_transforms(
    albumentations, transform_cfg: dict, allowed_names: set[str]
) -> list[Any]:
    transforms = []
    for name, params in transform_cfg.items():
        if name == _BBOX_KEY or name not in allowed_names:
            continue
        transforms.append(_build_transform(albumentations, name, params))
    return transforms


def _build_transform(albumentations, name: str, params: dict | None):
    params = dict(params or {})
    probability = float(params.pop("p", 0.5))

    if name == "horizontal_flip":
        _raise_on_unused(name, params)
        return albumentations.HorizontalFlip(p=probability)

    if name == "vertical_flip":
        _raise_on_unused(name, params)
        return albumentations.VerticalFlip(p=probability)

    if name == "random_rotate90":
        _raise_on_unused(name, params)
        return albumentations.RandomRotate90(p=probability)

    if name in {"shift_scale_rotate", "bbox_jitter"}:
        return albumentations.ShiftScaleRotate(
            shift_limit=params.pop("shift_limit", 0.0),
            scale_limit=params.pop("scale_limit", 0.0),
            rotate_limit=params.pop("rotate_limit", 0.0),
            p=probability,
        )

    if name == "brightness_contrast":
        return albumentations.RandomBrightnessContrast(
            brightness_limit=_as_range(params.pop("brightness_limit", 0.0)),
            contrast_limit=_as_range(params.pop("contrast_limit", 0.0)),
            p=probability,
        )

    if name == "random_gamma":
        return albumentations.RandomGamma(
            gamma_limit=_as_range(params.pop("gamma_limit", (80, 120)), cast=int),
            p=probability,
        )

    if name == "jpeg_compression":
        return albumentations.ImageCompression(
            quality_range=_as_range(
                (
                    params.pop("quality_lower", 99),
                    params.pop("quality_upper", 100),
                ),
                cast=int,
            ),
            p=probability,
        )

    if name == "downscale":
        return albumentations.Downscale(
            scale_range=_as_range(
                (params.pop("scale_min", 0.8), params.pop("scale_max", 0.95))
            ),
            p=probability,
        )

    if name == "gaussian_blur":
        return albumentations.GaussianBlur(
            blur_limit=_as_range(params.pop("blur_limit", (3, 5)), cast=int),
            p=probability,
        )

    if name == "motion_blur":
        return albumentations.MotionBlur(
            blur_limit=_as_range(params.pop("blur_limit", (3, 7)), cast=int),
            p=probability,
        )

    if name == "gauss_noise":
        return albumentations.GaussNoise(
            std_range=_as_range(params.pop("std_range", (0.01, 0.03))),
            p=probability,
        )

    if name == "perspective":
        return albumentations.Perspective(
            scale=_as_range(params.pop("scale", (0.02, 0.05))),
            p=probability,
        )

    if name == "clahe":
        clip_limit = params.pop("clip_limit", 2.0)
        tile_grid_size = params.pop("tile_grid_size", (8, 8))
        _raise_on_unused(name, params)
        return albumentations.CLAHE(
            clip_limit=clip_limit,
            tile_grid_size=tuple(tile_grid_size),
            p=probability,
        )

    raise ValueError(f"지원하지 않는 transform입니다: {name}")


def _as_range(value: Any, cast=float) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"범위 파라미터는 길이 2여야 합니다: {value}")
        return tuple(cast(v) for v in value)
    return cast(value)


def _raise_on_unused(name: str, params: dict) -> None:
    if params:
        raise ValueError(
            f"{name}에 지원하지 않는 파라미터가 있습니다: {sorted(params)}"
        )
