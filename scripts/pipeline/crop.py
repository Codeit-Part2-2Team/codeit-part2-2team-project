"""알약 단위 크롭 이미지 생성 스크립트.

세 가지 모드를 지원한다:

  inference 모드 (--predictions 지정):
    Stage 1 predictions.json → flat 크롭 + crops_manifest.json
    추론 파이프라인(run_predict.py → stage2_predict.py)에서 사용.

  gt 모드 (--labels / --images 지정):
    GT YOLO label txt → flat 크롭 + crops_manifest.json (class_name 포함)
    Stage 2 학습/검증 데이터 준비에 사용.
    클래스명은 파일명에서 자동 탐지한다.

  convert 모드 (--imagefolder 지정):
    이미 크롭된 ImageFolder 구조 → crops_manifest.json 만 생성 (재크롭 없음)
    외부에서 제공받은 크롭 디렉터리에 crops_manifest.json 이 없을 때 사용.
    클래스명은 서브디렉터리명을 그대로 사용한다.

사용 예:
    # inference 모드
    python scripts/pipeline/crop.py \\
        --predictions experiments/.../val_predictions.json \\
        --source      data/processed/images/val/ \\
        --output      experiments/.../stage1_crops/

    # gt 모드
    python scripts/pipeline/crop.py \\
        --labels  data/processed/labels \\
        --images  data/processed/images \\
        --output  data/processed/crops \\
        --splits  train val

    # convert 모드 (이미 crop된 ImageFolder가 있을 때)
    python scripts/pipeline/crop.py \\
        --imagefolder data/processed/crops \\
        --splits      train val
"""
from __future__ import annotations
from PIL import Image
from src.utils.timing import timed

import argparse
import json
import re
import sys
from pathlib import Path

root = next(
    (p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists()),
    None,
)

if root is None:
    raise RuntimeError("project root (requirements.txt) not found")
sys.path.insert(0, str(root))


_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


# ---------------------------------------------------------------------------
# inference 모드
# ---------------------------------------------------------------------------


def crop_from_predictions(
    predictions_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    padding: float = 0.05,
) -> list[dict]:
    """Stage 1 predictions.json → flat 크롭 + crops_manifest.json."""
    predictions_path = Path(predictions_path)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(predictions_path) as f:
        predictions = json.load(f)

    manifest = []
    for item in predictions:
        image_id = item["image_id"]
        detections = item.get("detections", [])
        if not detections:
            continue

        image_path = _find_image(source_dir, image_id)
        if image_path is None:
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {image_id} (디렉터리: {source_dir})"
            )

        with Image.open(image_path) as img:
            img_w, img_h = img.size
            for idx, det in enumerate(detections):
                bbox = det["bbox"]
                crop_img = img.crop(_apply_padding(bbox, img_w, img_h, padding))
                crop_id = f"{image_id}_{idx}"
                crop_path = output_dir / f"{crop_id}.jpg"
                crop_img.save(crop_path, "JPEG")
                manifest.append(
                    {
                        "image_id": image_id,
                        "crop_id": crop_id,
                        "crop_path": str(crop_path),
                        "bbox": bbox,
                        "score": det.get("score", 0.0),
                    }
                )

    with open(output_dir / "crops_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


# ---------------------------------------------------------------------------
# gt 모드
# ---------------------------------------------------------------------------


def crop_from_gt(
    labels_dir: Path,
    images_dir: Path,
    output_dir: Path,
    splits: list[str],
    padding: float = 0.05,
    class_lookup: dict[str, dict[str, str]] | None = None,
) -> None:
    """GT YOLO label txt → flat 크롭 + crops_manifest.json (class_name 포함).

    class_lookup: {split: {image_stem: class_name}} 형태로 전달하면
    파일명 파싱 대신 해당 매핑에서 class_name을 가져온다.
    """
    for split in splits:
        label_dir = labels_dir / split
        image_dir = images_dir / split
        out_dir = output_dir / split

        if not label_dir.exists():
            print(f"[스킵] 레이블 디렉터리 없음: {label_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        manifest: list[dict] = []
        total, skipped = 0, 0
        class_counts: dict[str, int] = {}
        split_lookup = (class_lookup or {}).get(split, {})

        for label_file in sorted(label_dir.glob("*.txt")):
            stem = label_file.stem

            if split_lookup:
                class_name = split_lookup.get(stem)
                if class_name is None:
                    skipped += 1
                    continue
            else:
                class_name = extract_class_name(stem)

            img_path = _find_image(image_dir, stem)
            if img_path is None:
                skipped += 1
                continue

            lines = label_file.read_text().strip().splitlines()
            if not lines:
                continue

            with Image.open(img_path) as raw:
                img = raw.convert("RGB")
            W, H = img.size

            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                bbox = [
                    (cx - bw / 2) * W,
                    (cy - bh / 2) * H,
                    (cx + bw / 2) * W,
                    (cy + bh / 2) * H,
                ]
                crop_id = f"{stem}_{idx:04d}"
                crop_path = out_dir / f"{crop_id}.jpg"
                img.crop(_apply_padding(bbox, W, H, padding)).save(crop_path)
                manifest.append(
                    {
                        "image_id": stem,
                        "crop_id": crop_id,
                        "crop_path": str(crop_path),
                        "bbox": bbox,
                        "class_name": class_name,
                    }
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total += 1

        with open(out_dir / "crops_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(
            f"[{split}] 크롭 {total}개 / 클래스 {len(class_counts)}개"
            + (f" / 스킵 {skipped}개" if skipped else "")
        )


# ---------------------------------------------------------------------------
# convert 모드
# ---------------------------------------------------------------------------


def convert_from_imagefolder(
    root_dir: Path,
    splits: list[str],
) -> None:
    """기존 ImageFolder 구조 → crops_manifest.json 생성 (재크롭 없음).

    이미 크롭된 이미지가 class_name/img.jpg 형태로 있을 때 사용한다.
    crops_manifest.json 이 이미 존재하면 스킵한다.
    """
    for split in splits:
        split_dir = root_dir / split

        if not split_dir.exists():
            print(f"[스킵] 디렉터리 없음: {split_dir}")
            continue

        manifest_path = split_dir / "crops_manifest.json"
        if manifest_path.exists():
            print(f"[스킵] 이미 존재: {manifest_path}")
            continue

        manifest: list[dict] = []
        class_counts: dict[str, int] = {}

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                manifest.append(
                    {
                        "image_id": img_path.stem,
                        "crop_id": img_path.stem,
                        "crop_path": str(img_path),
                        "class_name": class_name,
                    }
                )
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(
            f"[{split}] manifest 생성: {len(manifest)}개 / 클래스 {len(class_counts)}개"
            f" → {manifest_path}"
        )


def build_class_lookup(
    imagefolder_root: Path, splits: list[str]
) -> dict[str, dict[str, str]]:
    """ImageFolder 구조에서 {split: {image_stem → class_name}} 매핑을 빌드한다.

    ImageFolder 내 크롭 파일명은 `{original_stem}_{crop_idx}` 형태를 가정한다.
    """
    result: dict[str, dict[str, str]] = {}
    for split in splits:
        split_dir = imagefolder_root / split
        if not split_dir.exists():
            continue
        mapping: dict[str, str] = {}
        conflicts: set[str] = set()
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                original_stem = re.sub(r"_\d+$", "", img_path.stem)
                if original_stem in mapping and mapping[original_stem] != class_name:
                    conflicts.add(original_stem)
                else:
                    mapping[original_stem] = class_name
        for stem in conflicts:
            del mapping[stem]
        if conflicts:
            print(
                f"[경고] {split}: 이미지 1장에 여러 클래스 충돌 {len(conflicts)}건 — 제외됨"
            )
        result[split] = mapping
    return result


def extract_class_name(stem: str) -> str:
    """파일명 stem에서 약품 클래스명을 추출한다."""
    if stem.startswith("ext_"):
        stem = stem[4:]
    stem = re.sub(r"\.rf\..+$", "", stem)
    stem = re.sub(r"_[Jj][Pp][Ee]?[Gg]$", "", stem)
    stem = re.sub(r"_[Pp][Nn][Gg]$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------


def _find_image(directory: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _apply_padding(
    bbox: list[float], img_w: int | float, img_h: int | float, padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        int(max(0, x1 - pad_x)),
        int(max(0, y1 - pad_y)),
        int(min(img_w, x2 + pad_x)),
        int(min(img_h, y2 + pad_y)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="알약 단위 크롭 이미지 생성")
    parser.add_argument(
        "--predictions", default=None, help="[inference] predictions.json 경로"
    )
    parser.add_argument(
        "--source", default=None, help="[inference] 원본 이미지 디렉터리"
    )
    parser.add_argument("--labels", default=None, help="[gt] 레이블 루트 디렉터리")
    parser.add_argument("--images", default=None, help="[gt] 이미지 루트 디렉터리")
    parser.add_argument(
        "--imagefolder",
        default=None,
        help="[convert] 이미 크롭된 ImageFolder 루트 디렉터리. "
        "crops_manifest.json 이 없을 때 manifest만 생성한다.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="[gt/convert] 처리할 split 목록",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="크롭 저장 루트 디렉터리 (inference/gt 모드 필수)",
    )
    parser.add_argument(
        "--padding", type=float, default=0.05, help="bbox 여백 비율 (기본값 0.05)"
    )
    args = parser.parse_args()

    if args.imagefolder and not args.labels:
        convert_from_imagefolder(Path(args.imagefolder), args.splits)
        return

    if not args.output:
        parser.error("inference/gt 모드: --output 이 필요합니다")

    output = Path(args.output)

    if args.predictions:
        if not args.source:
            parser.error("inference 모드: --source 가 필요합니다")
        with timed(output.parent, "crop"):
            manifest = crop_from_predictions(
                args.predictions, args.source, output, args.padding
            )
        print(f"크롭 완료: {len(manifest)}개 → {output}")
    elif args.labels:
        if not args.images:
            parser.error("gt 모드: --images 가 필요합니다")
        class_lookup = None
        if args.imagefolder:
            class_lookup = build_class_lookup(Path(args.imagefolder), args.splits)
            for split, mapping in class_lookup.items():
                print(f"[class_lookup] {split}: {len(mapping)}개 이미지 매핑")
        crop_from_gt(
            Path(args.labels),
            Path(args.images),
            output,
            args.splits,
            args.padding,
            class_lookup,
        )
    else:
        parser.error("--predictions, --labels, --imagefolder 중 하나를 지정하세요")


if __name__ == "__main__":
    main()
