# isort: skip_file
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
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

from src.data.parser_raw import (
    EXCLUDE_IMAGE_STEMS,
    convert_raw_annotations,
)
from src.data.parser_external import convert_external_annotations


def parse_args() -> argparse.Namespace:
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
        help="raw annotation 폴더 경로",
    )

    parser.add_argument(
        "--raw-output-dir",
        type=str,
        default="processed/raw_yolo_labels",
        help="raw YOLO txt 저장 폴더",
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
        help="external label 폴더들",
    )

    parser.add_argument(
        "--ext-output-dir",
        type=str,
        default="processed/external_labels_unified",
        help="external 단일클래스 txt 저장 폴더",
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="단일 클래스 id. 기본값은 0",
    )

    return parser.parse_args()


def main() -> None:
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
