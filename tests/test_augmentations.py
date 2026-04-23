"""Albumentations 빌더 테스트."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.augmentations import build_stage1_transforms, build_stage2_transforms


class _FakeTransform:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCompose:
    def __init__(self, transforms, bbox_params=None):
        self.transforms = transforms
        self.bbox_params = bbox_params


class _FakeBboxParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAlbumentations(SimpleNamespace):
    def __init__(self):
        super().__init__(
            Compose=_FakeCompose,
            BboxParams=_FakeBboxParams,
            HorizontalFlip=type("HorizontalFlip", (_FakeTransform,), {}),
            VerticalFlip=type("VerticalFlip", (_FakeTransform,), {}),
            RandomRotate90=type("RandomRotate90", (_FakeTransform,), {}),
            ShiftScaleRotate=type("ShiftScaleRotate", (_FakeTransform,), {}),
            RandomBrightnessContrast=type(
                "RandomBrightnessContrast", (_FakeTransform,), {}
            ),
            RandomGamma=type("RandomGamma", (_FakeTransform,), {}),
            ImageCompression=type("ImageCompression", (_FakeTransform,), {}),
            Downscale=type("Downscale", (_FakeTransform,), {}),
            GaussianBlur=type("GaussianBlur", (_FakeTransform,), {}),
            MotionBlur=type("MotionBlur", (_FakeTransform,), {}),
            GaussNoise=type("GaussNoise", (_FakeTransform,), {}),
            Perspective=type("Perspective", (_FakeTransform,), {}),
            CLAHE=type("CLAHE", (_FakeTransform,), {}),
        )


@pytest.fixture
def fake_albumentations(monkeypatch):
    fake_module = _FakeAlbumentations()
    monkeypatch.setattr("src.data.augmentations.import_module", lambda _: fake_module)
    return fake_module


def test_build_stage1_transforms_includes_bbox_params(fake_albumentations):
    cfg = {
        "albumentations": {
            "bbox": {
                "format": "yolo",
                "label_fields": ["class_labels"],
                "min_visibility": 0.1,
                "clip": True,
            },
            "horizontal_flip": {"p": 0.5},
            "brightness_contrast": {
                "p": 0.3,
                "brightness_limit": 0.15,
                "contrast_limit": 0.15,
            },
        }
    }

    compose = build_stage1_transforms(cfg)

    assert isinstance(compose, _FakeCompose)
    assert len(compose.transforms) == 2
    assert compose.transforms[0].__class__.__name__ == "HorizontalFlip"
    assert compose.transforms[1].__class__.__name__ == "RandomBrightnessContrast"
    assert compose.bbox_params.kwargs["coord_format"] == "yolo"
    assert compose.bbox_params.kwargs["clip_bboxes_on_input"] is True
    assert compose.bbox_params.kwargs["clip_after_transform"] is True


def test_build_stage2_transforms_for_train_uses_flat_blocks(fake_albumentations):
    cfg = {
        "albumentations": {
            "jpeg_compression": {"p": 0.3, "quality_lower": 85, "quality_upper": 100},
            "bbox_jitter": {
                "p": 0.5,
                "shift_limit": 0.03,
                "scale_limit": 0.05,
                "rotate_limit": 5,
            },
        }
    }

    compose = build_stage2_transforms(cfg, split="train")

    assert isinstance(compose, _FakeCompose)
    assert compose.bbox_params is None
    assert [t.__class__.__name__ for t in compose.transforms] == [
        "ImageCompression",
        "ShiftScaleRotate",
    ]
    assert compose.transforms[0].kwargs["quality_range"] == (85, 100)


def test_build_stage2_transforms_for_val_returns_empty_compose(fake_albumentations):
    cfg = {"albumentations": {"brightness_contrast": {"p": 0.5}}}

    compose = build_stage2_transforms(cfg, split="val")

    assert isinstance(compose, _FakeCompose)
    assert compose.transforms == []


def test_build_stage2_transforms_raises_on_unknown_transform(fake_albumentations):
    cfg = {"albumentations": {"unknown_transform": {"p": 0.5}}}

    with pytest.raises(ValueError, match="지원하지 않는 transform"):
        build_stage2_transforms(cfg, split="train")


def test_build_stage2_transforms_raises_on_unknown_split(fake_albumentations):
    with pytest.raises(ValueError, match="지원하지 않는 split"):
        build_stage2_transforms({}, split="debug")
