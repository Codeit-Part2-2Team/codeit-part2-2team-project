# isort: skip_file
# ruff: noqa: E402
"""2-Stage 파이프라인 end-to-end mAP50 평가 스크립트.

YOLO val label txt + 원본 이미지 → GT 구성
Stage 1 crops manifest + Stage 2 predictions → 예측 구성
→ end-to-end mAP50 계산

GT bbox는 YOLO label txt에서 읽고, GT 클래스명은 같은 split의
data/processed/crops/{split}/crops_manifest.json을 우선 사용한다.
manifest가 없거나 매칭되지 않는 경우에만 파일명 기반 추출로 fallback한다.

사용 예:
    python scripts/pipeline/evaluate_pipeline.py \\
        --gt-labels data/processed/labels/val \\
        --gt-images data/processed/images/val \\
        --s1-crops  experiments/exp_.../stage1_crops/crops_manifest.json \\
        --s2-preds  experiments/exp_.../stage2_predictions.json

    # 클래스별 AP 상위 20개 출력
    python scripts/pipeline/evaluate_pipeline.py ... --per-class
"""
from __future__ import annotations

import argparse
import json
import sys
import re as _re

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from PIL import Image

root = next(
    (p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists()),
    None,
)

if root is None:
    raise RuntimeError("project root (requirements.txt) not found")
sys.path.insert(0, str(root))

from scripts.pipeline.crop import extract_class_name

_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


# ---------------------------------------------------------------------------
# IoU / AP
# ---------------------------------------------------------------------------


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[1.0], precisions, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------------------
# mAP
# ---------------------------------------------------------------------------


def compute_map(
    gt: dict[str, list[dict]],
    preds: dict[str, list[dict]],
    iou_thrs: list[float] | None = None,
    known_classes: set[str] | None = None,
) -> dict:
    """여러 IoU threshold에 대해 mAP 계산.

    iou_thrs 기본값: [0.50, 0.55, ..., 0.95] (COCO 스타일)
    known_classes 지정 시 해당 클래스만 평가 (iOS 등 파일명 노이즈 제거).

    Returns:
        {
            "mAP@[0.50:0.95]": float,
            "mAP@[0.75:0.95]": float,
            "mAP@0.50": float,
            "mAP@0.75": float,
            "per_class": {cls: {"AP@0.50": ..., "AP@0.75": ..., "AP@[0.50:0.95]": ...}},
            "n_classes": int,
        }
    """
    if iou_thrs is None:
        iou_thrs = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

    gt_by_class: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    gt_count: dict[str, int] = defaultdict(int)
    for img_id, objs in gt.items():
        for obj in objs:
            cls = obj["class_name"]
            if known_classes and cls not in known_classes:
                continue
            gt_by_class[cls][img_id].append({"bbox": obj["bbox"]})
            gt_count[cls] += 1

    pred_by_class: dict[str, list[dict]] = defaultdict(list)
    for img_id, dets in preds.items():
        for det in dets:
            pred_by_class[det["class_name"]].append(
                {"image_id": img_id, "bbox": det["bbox"], "score": det["score"]}
            )

    # {cls: {iou_thr: ap}}
    ap_table: dict[str, dict[float, float]] = {}

    for cls in gt_count:
        class_preds = sorted(pred_by_class.get(cls, []), key=lambda x: -x["score"])
        n_gt = gt_count[cls]
        ap_table[cls] = {}

        for thr in iou_thrs:
            class_gt = {
                img_id: [{"bbox": g["bbox"], "matched": False} for g in gs]
                for img_id, gs in gt_by_class[cls].items()
            }
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            for i, pred in enumerate(class_preds):
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(class_gt.get(pred["image_id"], [])):
                    if g["matched"]:
                        continue
                    iou = _iou(pred["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thr:
                    class_gt[pred["image_id"]][best_j]["matched"] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / n_gt
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap_table[cls][thr] = _ap(recalls, precisions)

    def _mean_ap(thrs: list[float]) -> float:
        if not ap_table:
            return 0.0
        return float(
            np.mean(
                [np.mean([ap_table[cls].get(t, 0.0) for t in thrs]) for cls in ap_table]
            )
        )

    thrs_5095 = [t for t in iou_thrs if 0.50 <= t <= 0.95]
    thrs_7595 = [t for t in iou_thrs if 0.75 <= t <= 0.95]

    per_class = {
        cls: {
            "AP@0.50": ap_table[cls].get(0.50, 0.0),
            "AP@0.75": ap_table[cls].get(0.75, 0.0),
            "AP@[0.50:0.95]": float(
                np.mean([ap_table[cls].get(t, 0.0) for t in thrs_5095])
            ),
            "AP@[0.75:0.95]": float(
                np.mean([ap_table[cls].get(t, 0.0) for t in thrs_7595])
            ),
        }
        for cls in ap_table
    }

    return {
        "mAP@[0.50:0.95]": _mean_ap(thrs_5095),
        "mAP@[0.75:0.95]": _mean_ap(thrs_7595),
        "mAP@0.50": _mean_ap([0.50]),
        "mAP@0.75": _mean_ap([0.75]),
        "per_class": per_class,
        "n_classes": len(ap_table),
    }


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_")


def _strip_crop_suffix(image_id: str) -> str:
    """GT crop image_id 끝의 class/index suffix를 제거한다."""

    return _re.sub(r"_\d+$", "", image_id)


def _source_image_key(image_id: str) -> str:
    """Roboflow hash를 제외한 원본 이미지 기준 key를 만든다."""

    return _re.sub(r"\.rf\..+$", "", image_id)


def _extract_raw_k_ids(stem: str) -> list[str]:
    """raw_K 파일명에서 Kaggle category_id 목록을 추출한다."""

    match = _re.match(r"^raw_K-([^_]+)", stem)
    if not match:
        return []
    return [part.zfill(6) for part in match.group(1).split("-") if part]


def _resolve_raw_k_classes(
    stem: str,
    id_to_classes: dict[str, list[str]] | None,
    manifest_classes: deque[str],
) -> deque[str]:
    """raw_K category_id를 class_name queue로 변환한다.

    같은 category_id에 여러 alias가 있으면 GT crop manifest에 등장한 alias를
    우선 사용한다.
    """

    if id_to_classes is None:
        return deque()

    manifest_class_set = set(manifest_classes)
    classes: deque[str] = deque()
    for category_id in _extract_raw_k_ids(stem):
        candidates = id_to_classes.get(category_id, [])
        if not candidates:
            continue
        selected = next(
            (candidate for candidate in candidates if candidate in manifest_class_set),
            candidates[-1],
        )
        classes.append(selected)
    return classes


def _build_manifest_class_lookup(
    crops_root: Path, split: str
) -> dict[str, deque[str]]:
    """GT crop manifest에서 {원본 image_id → class_name 목록}을 빌드한다.

    ImageFolder 기반 GT crop manifest에는 bbox가 없으므로 bbox는 YOLO label에서
    읽고, 여기서는 class_name만 복원한다. 복수 알약 이미지는 manifest 순서를
    유지한 class queue로 반환한다.
    """

    lookup: dict[str, deque[str]] = defaultdict(deque)
    manifest_splits = [split] + [
        p.name for p in sorted(crops_root.iterdir()) if p.is_dir() and p.name != split
    ]

    for manifest_split in manifest_splits:
        manifest_path = crops_root / manifest_split / "crops_manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as f:
            for r in json.load(f):
                if not r.get("class_name"):
                    continue
                image_id = _strip_crop_suffix(r["image_id"])
                lookup[image_id].append(r["class_name"])

                source_key = _source_image_key(image_id)
                if source_key != image_id:
                    lookup[source_key].append(r["class_name"])
    return dict(lookup)


def _build_filename_class_lookup(crops_root: Path) -> dict[str, str]:
    """train/val crops_manifest.json에서 {추출된_파일명_키 → class_name} 빌드."""

    lookup: dict[str, str] = {}
    for split in ("train", "val"):
        manifest_path = crops_root / split / "crops_manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as f:
            for r in json.load(f):
                if not r.get("class_name"):
                    continue
                stem = _re.sub(r"_\d+$", "", r["image_id"])
                key = _normalize(extract_class_name(stem)).rstrip("_-")
                lookup[key] = r["class_name"]
    return lookup


def load_gt(
    label_dir: Path,
    image_dir: Path,
    filename_class_lookup: dict[str, str] | None = None,
    manifest_class_lookup: dict[str, deque[str]] | None = None,
    id_to_classes: dict[str, list[str]] | None = None,
) -> dict[str, list[dict]]:
    """YOLO label txt + 이미지 → {image_id: [{bbox(pixel), class_name}]}

    manifest_class_lookup이 있으면 GT crop manifest의 image_id 기준 class_name을
    우선 사용한다. 매칭되지 않으면 filename_class_lookup / 파일명 추출로 fallback한다.
    """
    gt: dict[str, list] = defaultdict(list)
    for label_file in sorted(label_dir.glob("*.txt")):
        stem = label_file.stem
        key = _normalize(extract_class_name(stem)).rstrip("_-")
        manifest_classes = deque()
        if manifest_class_lookup is not None:
            manifest_classes = deque(
                manifest_class_lookup.get(stem)
                or manifest_class_lookup.get(_source_image_key(stem))
                or []
            )
        raw_k_classes = _resolve_raw_k_classes(stem, id_to_classes, manifest_classes)
        fallback_class_name = (
            filename_class_lookup.get(key)
            if filename_class_lookup is not None
            else None
        )
        if fallback_class_name is None:
            fallback_class_name = key

        img_path = next(
            (
                image_dir / f"{stem}{e}"
                for e in _IMAGE_EXTS
                if (image_dir / f"{stem}{e}").exists()
            ),
            None,
        )
        if img_path is None:
            continue

        with Image.open(img_path) as img:
            W, H = img.size

        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            class_name = (
                raw_k_classes.popleft()
                if raw_k_classes
                else manifest_classes.popleft()
                if manifest_classes
                else fallback_class_name
            )
            gt[stem].append(
                {
                    "bbox": [
                        (cx - bw / 2) * W,
                        (cy - bh / 2) * H,
                        (cx + bw / 2) * W,
                        (cy + bh / 2) * H,
                    ],
                    "class_name": class_name,
                }
            )
    return dict(gt)


def load_preds(
    s1_crops_manifest: Path,
    s2_preds_path: Path,
    class_alias_map: dict[str, str] | None = None,
) -> tuple[dict[str, list[dict]], set[str]]:
    """S1 crops + S2 preds → ({image_id: [{bbox, class_name, score}]}, known_classes)

    score = det_score * cls_score
    """
    with open(s1_crops_manifest, encoding="utf-8") as f:
        s1_crops = json.load(f)
    with open(s2_preds_path, encoding="utf-8") as f:
        s2_preds = json.load(f)

    class_alias_map = class_alias_map or {}
    s2_by_crop = {r["crop_id"]: r for r in s2_preds}
    known_classes = {
        class_alias_map.get(r["class_name"], r["class_name"]) for r in s2_preds
    }

    preds: dict[str, list] = defaultdict(list)
    for crop in s1_crops:
        s2 = s2_by_crop.get(crop["crop_id"])
        if s2 is None:
            continue
        class_name = class_alias_map.get(s2["class_name"], s2["class_name"])
        preds[crop["image_id"]].append(
            {
                "bbox": crop["bbox"],
                "class_name": class_name,
                "score": crop["score"] * s2["score"],
            }
        )
    return dict(preds), known_classes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-Stage 파이프라인 end-to-end mAP 평가"
    )
    parser.add_argument("--gt-labels", required=True, help="YOLO label 디렉터리 (val)")
    parser.add_argument("--gt-images", required=True, help="원본 이미지 디렉터리 (val)")
    parser.add_argument("--s1-crops", required=True, help="Stage 1 crops manifest")
    parser.add_argument("--s2-preds", required=True, help="Stage 2 predictions JSON")
    parser.add_argument("--per-class", action="store_true", help="클래스별 AP 출력")
    parser.add_argument(
        "--kaggle-classes",
        default=None,
        help="Kaggle class map JSON ({class_name: category_id}). 지정 시 해당 클래스만 평가 (submission 기준과 일치)",
    )
    parser.add_argument(
        "--unknown-class-map",
        default=None,
        help="Stage 2 예측 alias class_name을 평가용 canonical class_name으로 치환하는 JSON",
    )
    args = parser.parse_args()

    # gt-labels 경로 기준으로 crops manifest 자동 탐색
    # (data/processed/labels/val → data/processed/crops/val/crops_manifest.json)
    gt_labels_path = Path(args.gt_labels)
    split = gt_labels_path.name
    crops_root = Path(args.gt_labels).parent.parent / "crops"
    manifest_class_lookup = (
        _build_manifest_class_lookup(crops_root, split) if crops_root.exists() else None
    )
    filename_class_lookup = (
        _build_filename_class_lookup(crops_root) if crops_root.exists() else None
    )
    id_to_classes = None
    if args.kaggle_classes:
        with open(args.kaggle_classes, encoding="utf-8") as f:
            kaggle_map = json.load(f)
        id_to_classes = defaultdict(list)
        for class_name, category_id in kaggle_map.items():
            if class_name == "_comment":
                continue
            id_to_classes[str(category_id).zfill(6)].append(class_name)

    class_alias_map = None
    if args.unknown_class_map:
        with open(args.unknown_class_map, encoding="utf-8") as f:
            class_alias_map = {
                k: v for k, v in json.load(f).items() if not k.startswith("_")
            }

    gt = load_gt(
        gt_labels_path,
        Path(args.gt_images),
        filename_class_lookup,
        manifest_class_lookup,
        id_to_classes,
    )
    preds, known_classes = load_preds(
        Path(args.s1_crops), Path(args.s2_preds), class_alias_map
    )

    if args.kaggle_classes:
        known_classes = {k for k in kaggle_map if k != "_comment"}

    n_gt_imgs = len(gt)
    n_gt_objs = sum(len(v) for v in gt.values())
    n_pred_objs = sum(len(v) for v in preds.values())

    result = compute_map(gt, preds, known_classes=known_classes)

    mode = "Kaggle 클래스 기준" if args.kaggle_classes else "Stage 2 학습 클래스 기준"
    print(f"GT     : {n_gt_imgs}장  {n_gt_objs}개 객체")
    print(f"Pred   : {n_pred_objs}개 / {result['n_classes']}클래스 평가  [{mode}]")
    print(f"mAP@[0.50:0.95] : {result['mAP@[0.50:0.95]']:.4f}  (COCO)")
    print(f"mAP@[0.75:0.95] : {result['mAP@[0.75:0.95]']:.4f}  (Kaggle)")
    print(f"mAP@0.50        : {result['mAP@0.50']:.4f}")
    print(f"mAP@0.75        : {result['mAP@0.75']:.4f}")

    if args.per_class:
        print("\n클래스별 AP@[0.75:0.95] (상위 20):")
        for cls, aps in sorted(
            result["per_class"].items(), key=lambda x: -x[1]["AP@[0.75:0.95]"]
        )[:20]:
            print(
                f"  {cls:<45s} "
                f"0.50={aps['AP@0.50']:.3f}  "
                f"0.75={aps['AP@0.75']:.3f}  "
                f"[0.75:0.95]={aps['AP@[0.75:0.95]']:.3f}"
            )


if __name__ == "__main__":
    main()
