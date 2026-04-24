"""
convert_annotations.py

역할:
1) raw 데이터의 COCO-style json 어노테이션을 YOLO txt 형식으로 변환
2) external 데이터의 기존 YOLO txt 라벨을 단일 클래스(0: pill)로 통일
3) external 데이터에서 bbox가 여러 개인 경우 가장 큰 bbox 1개만 남김
4) raw 데이터의 이상치 샘플과 비정상 bbox를 자동 제외

주의:
- 이 스크립트는 "어노테이션 변환"만 담당한다.
- train/val/test split, merged dataset 생성, data.yaml 생성은 별도 스크립트에서 처리하는 것이 좋다.

예상 프로젝트 구조 예시:
pill_project/
├── raw/
│   └── extracted/
│       └── sprint_ai_project1_data/
│           ├── train_images/
│           ├── test_images/
│           └── train_annotations/
├── external/
│   └── extracted/
│       ├── train/
│       │   └── labels/
│       ├── valid/
│       │   └── labels/
│       └── test/
│           └── labels/
├── processed/
│   ├── raw_yolo_labels/
│   └── external_labels_unified/
└── scripts/
    └── convert_annotations.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# 하드 제외 대상: 이번에 실제로 발견한 raw 이상치 샘플
# stem 기준 (확장자 제외)
EXCLUDE_IMAGE_STEMS = {
    "K-003351-016262-018357_0_2_0_2_75_000_200",
}


def parse_args() -> argparse.Namespace:
    """
    커맨드라인 인자를 읽는다.
    기본값은 현재 프로젝트 구조 기준으로 설정했다.
    """
    parser = argparse.ArgumentParser(
        description="Convert raw/external annotations to YOLO format."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="프로젝트 루트 경로. 예: /content/drive/MyDrive/pill_project",
    )
    parser.add_argument(
        "--raw-annotation-dir",
        type=str,
        default="raw/extracted/sprint_ai_project1_data/train_annotations",
        help="raw annotation 폴더 경로 (project-root 기준 상대경로)",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=str,
        default="processed/raw_yolo_labels",
        help="raw YOLO txt 저장 폴더 (project-root 기준 상대경로)",
    )
    parser.add_argument(
        "--ext-label-roots",
        type=str,
        nargs="+",
        default=[
            "external/extracted/train/labels",
            "external/extracted/valid/labels",
            "external/extracted/test/labels",
        ],
        help="external label 폴더들 (project-root 기준 상대경로, 여러 개 가능)",
    )
    parser.add_argument(
        "--ext-output-dir",
        type=str,
        default="processed/external_labels_unified",
        help="external 단일클래스 txt 저장 폴더 (project-root 기준 상대경로)",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="단일 클래스 id. 기본값은 0 (pill)",
    )

    return parser.parse_args()


def coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    COCO 형식 bbox [x, y, w, h] 를 YOLO 형식 [x_center, y_center, w, h] (정규화)로 변환한다.

    Parameters
    ----------
    bbox : list
        [x_min, y_min, box_width, box_height]
    img_w : int
        이미지 가로 크기
    img_h : int
        이미지 세로 크기

    Returns
    -------
    tuple
        (x_center, y_center, width, height) 모두 0~1 범위
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w = w / img_w
    h = h / img_h

    return x_center, y_center, w, h


def validate_yolo_bbox(x_center: float, y_center: float, w: float, h: float) -> bool:
    """
    YOLO bbox 값이 정상 범위인지 검사한다.
    """
    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < w <= 1.0
        and 0.0 < h <= 1.0
    )


def should_exclude_image(image_file: str) -> bool:
    """
    파일 stem 기준으로 하드 제외 목록에 들어있는지 검사한다.
    """
    image_stem = Path(image_file).stem
    return image_stem in EXCLUDE_IMAGE_STEMS


def yolo_box_area(parts: List[str]) -> float:
    """
    YOLO 라벨 split 결과(parts)에서 bbox 면적을 계산한다.

    parts 예:
    ['312', '0.5', '0.4', '0.2', '0.1']
    """
    w = float(parts[3])
    h = float(parts[4])
    return w * h


def convert_raw_annotations(
    raw_annotation_dir: Path,
    raw_output_dir: Path,
    class_id: int = 0,
) -> None:
    """
    raw 어노테이션(json)을 YOLO txt로 변환한다.

    현재 raw 구조는 이미지 1장에 대해 여러 하위 폴더/파일이 있고,
    같은 image_file_name 으로 여러 bbox가 존재할 수 있다.
    따라서 같은 이미지 파일명 기준으로 bbox를 모두 모아서 txt 한 개에 저장한다.
    """
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    raw_json_files = sorted(raw_annotation_dir.rglob("*.json"))
    print(f"[RAW] 발견된 json 파일 수: {len(raw_json_files)}")

    # 같은 이미지 파일명에 해당하는 bbox 줄들을 모두 모음
    image_to_lines: Dict[str, List[str]] = defaultdict(list)

    error_count = 0
    skipped_excluded_count = 0
    skipped_invalid_bbox_count = 0

    for json_path in raw_json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 현재 확인된 raw json 구조 기준
            image_info = data["images"][0]
            annotations = data["annotations"]

            image_file = image_info["file_name"]
            img_w = image_info["width"]
            img_h = image_info["height"]

            # 이번에 발견한 하드 이상치 샘플 제외
            if should_exclude_image(image_file):
                skipped_excluded_count += 1
                continue

            for ann in annotations:
                bbox = ann["bbox"]  # COCO style: [x, y, w, h]
                x_center, y_center, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)

                # bbox 자동 이상치 필터
                if not validate_yolo_bbox(x_center, y_center, w, h):
                    skipped_invalid_bbox_count += 1
                    continue

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                image_to_lines[image_file].append(line)

        except Exception as e:
            error_count += 1
            print(f"[RAW][ERROR] {json_path}: {e}")

    saved_count = 0
    empty_skipped_count = 0

    for image_file, lines in image_to_lines.items():
        # 해당 이미지의 bbox가 모두 invalid여서 남은 줄이 없으면 저장 안 함
        if len(lines) == 0:
            empty_skipped_count += 1
            continue

        save_path = raw_output_dir / f"{Path(image_file).stem}.txt"

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        saved_count += 1

    print(f"[RAW] bbox가 모인 이미지 수: {len(image_to_lines)}")
    print(f"[RAW] 저장된 txt 수: {saved_count}")
    print(f"[RAW] 하드 제외된 샘플 수: {skipped_excluded_count}")
    print(f"[RAW] invalid bbox로 제외된 수: {skipped_invalid_bbox_count}")
    print(f"[RAW] 빈 라벨이라 저장 안 한 수: {empty_skipped_count}")
    print(f"[RAW] 에러 수: {error_count}")


def convert_external_annotations(
    ext_label_roots: List[Path],
    ext_output_dir: Path,
    class_id: int = 0,
) -> None:
    """
    external YOLO txt 라벨을 처리한다.

    처리 내용:
    1) 기존 class_id를 모두 단일 클래스(class_id=0)로 통일
    2) bbox가 여러 개면 가장 큰 박스 1개만 남김

    이유:
    external 데이터는 대부분 '한 이미지 = 알약 1개 중심' 구조이므로,
    여러 bbox는 중복/노이즈일 가능성이 높다.
    """
    ext_output_dir.mkdir(parents=True, exist_ok=True)

    txt_files: List[Path] = []
    for root in ext_label_roots:
        txt_files.extend(sorted(root.rglob("*.txt")))

    print(f"[EXT] 발견된 txt 파일 수: {len(txt_files)}")

    converted = 0
    multi_bbox_fixed = 0
    skipped = 0

    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            valid_parts: List[List[str]] = []

            for line in lines:
                parts = line.split()

                # 정상 YOLO 형식이 아니면 건너뜀
                if len(parts) != 5:
                    continue

                valid_parts.append(parts)

            # bbox가 여러 개면 가장 큰 박스 하나만 남김
            if len(valid_parts) >= 2:
                valid_parts = [max(valid_parts, key=yolo_box_area)]
                multi_bbox_fixed += 1

            new_lines: List[str] = []

            for parts in valid_parts:
                # class_id만 단일 클래스 0으로 통일
                new_line = f"{class_id} {' '.join(parts[1:])}"
                new_lines.append(new_line)

            save_path = ext_output_dir / txt_path.name
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))

            converted += 1

        except Exception as e:
            skipped += 1
            print(f"[EXT][ERROR] {txt_path}: {e}")

    print(f"[EXT] 변환 완료 수: {converted}")
    print(f"[EXT] 중복 bbox 정리 수: {multi_bbox_fixed}")
    print(f"[EXT] 스킵/에러 수: {skipped}")


def main() -> None:
    """
    스크립트 실행 진입점
    """
    args = parse_args()

    project_root = Path(args.project_root).resolve()

    raw_annotation_dir = project_root / args.raw_annotation_dir
    raw_output_dir = project_root / args.raw_output_dir

    ext_label_roots = [project_root / p for p in args.ext_label_roots]
    ext_output_dir = project_root / args.ext_output_dir

    print("=" * 70)
    print("convert_annotations.py")
    print("=" * 70)
    print(f"project_root       : {project_root}")
    print(f"raw_annotation_dir : {raw_annotation_dir}")
    print(f"raw_output_dir     : {raw_output_dir}")
    print(f"ext_label_roots    : {ext_label_roots}")
    print(f"ext_output_dir     : {ext_output_dir}")
    print(f"class_id           : {args.class_id}")
    print(f"exclude_stems      : {sorted(EXCLUDE_IMAGE_STEMS)}")
    print("=" * 70)

    if not raw_annotation_dir.exists():
        raise FileNotFoundError(
            f"raw annotation 폴더를 찾을 수 없습니다: {raw_annotation_dir}"
        )

    for root in ext_label_roots:
        if not root.exists():
            raise FileNotFoundError(f"external label 폴더를 찾을 수 없습니다: {root}")

    convert_raw_annotations(
        raw_annotation_dir=raw_annotation_dir,
        raw_output_dir=raw_output_dir,
        class_id=args.class_id,
    )

    convert_external_annotations(
        ext_label_roots=ext_label_roots,
        ext_output_dir=ext_output_dir,
        class_id=args.class_id,
    )

    print("[DONE] annotation 변환 완료")


if __name__ == "__main__":
    main()
