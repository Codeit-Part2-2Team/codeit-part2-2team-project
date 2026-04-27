"""Stage2Dataset 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.stage2_dataset import Stage2Dataset


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MIN_CFG = {
    "data": {"imgsz": 32, "workers": 0},
    "albumentations": {},
}


def _write_manifest(root: Path, records: list[dict]) -> Path:
    manifest = root / "crops_manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    return manifest


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(path)


# ---------------------------------------------------------------------------
# class_to_idx 구성
# ---------------------------------------------------------------------------


def test_classes_sorted_from_manifest(tmp_path):
    """classes 미지정 시 manifest에서 정렬된 고유 클래스를 추출한다."""
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_c = tmp_path / "c.jpg"
    for p in [img_a, img_b, img_c]:
        _make_image(p)

    _write_manifest(tmp_path, [
        {"crop_path": str(img_b), "class_name": "beta"},
        {"crop_path": str(img_a), "class_name": "alpha"},
        {"crop_path": str(img_c), "class_name": "gamma"},
    ])

    ds = Stage2Dataset(tmp_path, _MIN_CFG, split="train")

    assert ds.classes == ["alpha", "beta", "gamma"]
    assert ds.class_to_idx == {"alpha": 0, "beta": 1, "gamma": 2}


def test_external_classes_overrides_manifest_order(tmp_path):
    """classes 파라미터 지정 시 해당 순서로 class_to_idx를 구성한다."""
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    for p in [img_a, img_b]:
        _make_image(p)

    _write_manifest(tmp_path, [
        {"crop_path": str(img_a), "class_name": "alpha"},
        {"crop_path": str(img_b), "class_name": "beta"},
    ])

    # train 기준 classes: beta=0, alpha=1 (역순)
    train_classes = ["beta", "alpha"]
    ds = Stage2Dataset(tmp_path, _MIN_CFG, split="val", classes=train_classes)

    assert ds.classes == ["beta", "alpha"]
    assert ds.class_to_idx["beta"] == 0
    assert ds.class_to_idx["alpha"] == 1


def test_train_val_index_alignment(tmp_path):
    """train_ds.classes를 val_ds에 전달하면 공유 클래스의 인덱스가 일치한다."""
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"

    # train: alpha, beta, gamma (3개)
    train_imgs = [train_dir / f"{c}.jpg" for c in ["alpha", "beta", "gamma"]]
    for p in train_imgs:
        _make_image(p)
    _write_manifest(train_dir, [
        {"crop_path": str(train_dir / "alpha.jpg"), "class_name": "alpha"},
        {"crop_path": str(train_dir / "beta.jpg"), "class_name": "beta"},
        {"crop_path": str(train_dir / "gamma.jpg"), "class_name": "gamma"},
    ])

    # val: beta, gamma만 있음 (alpha 없음 → 독립 구성 시 beta=0, gamma=1로 밀림)
    val_imgs = [val_dir / f"{c}.jpg" for c in ["beta", "gamma"]]
    for p in val_imgs:
        _make_image(p)
    _write_manifest(val_dir, [
        {"crop_path": str(val_dir / "beta.jpg"), "class_name": "beta"},
        {"crop_path": str(val_dir / "gamma.jpg"), "class_name": "gamma"},
    ])

    train_ds = Stage2Dataset(train_dir, _MIN_CFG, split="train")
    val_ds = Stage2Dataset(val_dir, _MIN_CFG, split="val", classes=train_ds.classes)

    assert val_ds.class_to_idx["beta"] == train_ds.class_to_idx["beta"]
    assert val_ds.class_to_idx["gamma"] == train_ds.class_to_idx["gamma"]


def test_val_only_classes_filtered_out(tmp_path):
    """val에만 존재하는 클래스(train에 없는)는 samples에서 제외된다."""
    img_known = tmp_path / "beta.jpg"
    img_unknown = tmp_path / "delta.jpg"
    _make_image(img_known)
    _make_image(img_unknown)

    _write_manifest(tmp_path, [
        {"crop_path": str(img_known), "class_name": "beta"},
        {"crop_path": str(img_unknown), "class_name": "delta"},  # train에 없음
    ])

    train_classes = ["alpha", "beta", "gamma"]
    ds = Stage2Dataset(tmp_path, _MIN_CFG, split="val", classes=train_classes)

    assert len(ds.samples) == 1
    assert ds.samples[0][1] == train_classes.index("beta")


def test_records_without_class_name_skipped(tmp_path):
    """class_name이 없는 레코드는 항상 제외된다."""
    img = tmp_path / "a.jpg"
    _make_image(img)

    _write_manifest(tmp_path, [
        {"crop_path": str(img), "class_name": "alpha"},
        {"crop_path": str(img)},  # class_name 없음
        {"crop_path": str(img), "class_name": ""},  # 빈 문자열
    ])

    ds = Stage2Dataset(tmp_path, _MIN_CFG, split="train")

    assert ds.classes == ["alpha"]
    assert len(ds.samples) == 1


# ---------------------------------------------------------------------------
# get_sample_weights
# ---------------------------------------------------------------------------


def test_get_sample_weights_inverse_frequency(tmp_path):
    """get_sample_weights()는 클래스 빈도 역수를 반환한다."""
    imgs = [tmp_path / f"{i}.jpg" for i in range(3)]
    for p in imgs:
        _make_image(p)

    # alpha 2개, beta 1개
    _write_manifest(tmp_path, [
        {"crop_path": str(imgs[0]), "class_name": "alpha"},
        {"crop_path": str(imgs[1]), "class_name": "alpha"},
        {"crop_path": str(imgs[2]), "class_name": "beta"},
    ])

    ds = Stage2Dataset(tmp_path, _MIN_CFG, split="train")
    weights = ds.get_sample_weights()

    assert len(weights) == 3
    # alpha 2개 → 1/2, beta 1개 → 1/1
    alpha_w = weights[0]
    beta_w = weights[2]
    assert alpha_w == pytest.approx(0.5)
    assert beta_w == pytest.approx(1.0)
