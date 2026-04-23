"""Stage 1 예측 bbox 기준으로 알약 단위 크롭 이미지를 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


def crop_images(
    predictions_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    padding: float = 0.05,
) -> list[dict]:
    """predictions.json을 읽어 bbox 기준 크롭 이미지를 저장하고 manifest를 반환한다.

    Args:
        predictions_path: Stage 1 predictions.json 경로
        source_dir: 원본 이미지 디렉터리
        output_dir: 크롭 이미지 저장 디렉터리 (crops_manifest.json도 여기 저장)
        padding: bbox 상하좌우에 적용할 여백 비율 (bbox 크기 대비)

    Returns:
        crops_manifest 항목 리스트
    """
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
                bbox = det["bbox"]  # [x1, y1, x2, y2] xyxy
                padded = _apply_padding(bbox, img_w, img_h, padding)
                crop_img = img.crop(padded)

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

    manifest_path = output_dir / "crops_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def _find_image(source_dir: Path, image_id: str) -> Path | None:
    for ext in _IMAGE_EXTENSIONS:
        p = source_dir / f"{image_id}.{ext}"
        if p.exists():
            return p
    return None


def _apply_padding(
    bbox: list[float], img_w: int, img_h: int, padding: float
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 예측 bbox 기준 알약 크롭 이미지 생성"
    )
    parser.add_argument("--predictions", required=True, help="Stage 1 predictions.json 경로")
    parser.add_argument("--source", required=True, help="원본 이미지 디렉터리")
    parser.add_argument("--output", required=True, help="크롭 이미지 저장 디렉터리")
    parser.add_argument(
        "--padding", type=float, default=0.05, help="bbox 여백 비율 (기본값 0.05)"
    )
    args = parser.parse_args()

    manifest = crop_images(args.predictions, args.source, args.output, args.padding)
    print(f"크롭 완료: {len(manifest)}개 저장 → {args.output}")


if __name__ == "__main__":
    main()
