"""crops_manifest와 Stage 2 예측을 Kaggle 제출 CSV로 변환하는 유틸."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SUBMISSION_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def merge_predictions(
    manifest_path: str | Path,
    stage2_predictions_path: str | Path,
) -> list[dict]:
    """crop_id 기준으로 inference crop 메타와 Stage 2 분류 결과를 병합한다.

    제출용 원본 image_id와 bbox는 crops_manifest가 기준이다. Stage 2 결과는
    각 crop의 class_name, class_id, cls_score만 제공한다고 본다.
    """
    with Path(manifest_path).open(encoding="utf-8") as f:
        manifest = json.load(f)
    with Path(stage2_predictions_path).open(encoding="utf-8") as f:
        stage2_preds = json.load(f)

    manifest_by_crop = {item["crop_id"]: item for item in manifest}
    merged_by_image: dict[str, list[dict]] = {}

    for record in stage2_preds:
        crop_id = record["crop_id"]
        m = manifest_by_crop.get(crop_id)
        if m is None:
            raise KeyError(f"manifest에 없는 crop_id: {crop_id}")

        det_score = float(m.get("score", 1.0))
        cls_score = float(record["score"])
        image_id = m["image_id"]
        merged_by_image.setdefault(image_id, []).append(
            {
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "bbox": m["bbox"],
                "score": det_score * cls_score,
            }
        )

    image_ids = {item["image_id"] for item in manifest}
    return [
        {"image_id": iid, "detections": merged_by_image.get(iid, [])}
        for iid in sorted(image_ids)
    ]


def predictions_to_df(
    predictions: list[dict],
    class_map: dict[str, int] | None = None,
    image_id_map: dict[str, int] | None = None,
    unknown_class_map: dict[str, str | int] | None = None,
    strict_class_map: bool = False,
) -> pd.DataFrame:
    """병합된 예측을 Kaggle submission DataFrame으로 변환한다.

    class_map이 있으면 category_id는 반드시 해당 map을 통해 결정한다.
    class_map 밖 예측은 기본적으로 제외한다. 다만 unknown_class_map에
    `{예측 class_name: Kaggle class_name 또는 category_id}`를 지정하면
    해당 Kaggle 클래스로 치환해 제출한다.
    """
    rows = []
    annotation_id = 1
    skipped_unknown_classes: dict[str, int] = {}
    remapped_unknown_classes: dict[str, int] = {}

    for item in predictions:
        raw_image_id = item["image_id"]
        if image_id_map is not None:
            mapped = _lookup_id(image_id_map, raw_image_id)
            if mapped is None:
                raise KeyError(f"image_id_map에 없는 image_id: {raw_image_id!r}")
            image_id = mapped
        else:
            image_id = raw_image_id

        for det in item["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            if class_map is not None:
                class_name = det["class_name"]
                category_id = _resolve_category_id(
                    class_name, class_map, unknown_class_map
                )
                if category_id is None:
                    if strict_class_map:
                        raise KeyError(
                            f"class_map에 없는 class_name: {class_name!r}"
                        )
                    skipped_unknown_classes[class_name] = (
                        skipped_unknown_classes.get(class_name, 0) + 1
                    )
                    continue
                if class_name not in class_map:
                    remapped_unknown_classes[class_name] = (
                        remapped_unknown_classes.get(class_name, 0) + 1
                    )
            else:
                category_id = det["class_id"]

            rows.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox_x": round(x1),
                    "bbox_y": round(y1),
                    "bbox_w": round(x2 - x1),
                    "bbox_h": round(y2 - y1),
                    "score": det["score"],
                }
            )
            annotation_id += 1

    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    df.attrs["skipped_unknown_classes"] = skipped_unknown_classes
    df.attrs["remapped_unknown_classes"] = remapped_unknown_classes
    return df


def _resolve_category_id(
    class_name: str,
    class_map: dict[str, int],
    unknown_class_map: dict[str, str | int] | None,
) -> int | str | None:
    """class_name을 Kaggle category_id로 해석한다."""
    category_id = _lookup_id(class_map, class_name)
    if category_id is not None:
        return category_id

    if not unknown_class_map:
        return None

    fallback = _lookup_id(unknown_class_map, class_name)
    if fallback is None:
        return None

    mapped = _lookup_id(class_map, fallback)
    if mapped is not None:
        return mapped

    # unknown_class_map 값이 이미 category_id인 경우를 허용한다.
    if str(fallback).isdigit():
        direct_id = int(fallback)
        if direct_id in set(class_map.values()):
            return direct_id
    raise KeyError(
        f"unknown_class_map 값이 class_map에 없음: {class_name!r} -> {fallback!r}"
    )


def _lookup_id(mapping: dict, key: object) -> int | str | None:
    """JSON 문자열 키와 path-like image_id를 모두 허용해 값을 조회한다."""
    candidates = [key, str(key)]
    stem = Path(str(key)).stem
    if stem not in candidates:
        candidates.append(stem)

    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def save_submission(
    predictions: list[dict],
    output: str | Path,
    class_map: dict[str, int] | None = None,
    image_id_map: dict[str, int] | None = None,
    unknown_class_map: dict[str, str | int] | None = None,
    strict_class_map: bool = False,
) -> None:
    """병합된 예측을 submission CSV로 저장한다."""
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = predictions_to_df(
        predictions,
        class_map=class_map,
        image_id_map=image_id_map,
        unknown_class_map=unknown_class_map,
        strict_class_map=strict_class_map,
    )
    _print_class_map_summary(df)
    df.to_csv(out, index=False)


def _print_class_map_summary(df: pd.DataFrame) -> None:
    """unknown class 처리 요약을 출력한다."""
    remapped = df.attrs.get("remapped_unknown_classes", {})
    if remapped:
        total = sum(remapped.values())
        sample = _format_count_sample(remapped)
        print(
            f"[submission] class_map 밖 {len(remapped)}개 클래스 "
            f"{total}개 detection을 fallback으로 치환: {sample}"
        )

    skipped = df.attrs.get("skipped_unknown_classes", {})
    if skipped:
        total = sum(skipped.values())
        sample = _format_count_sample(skipped)
        print(
            f"[submission] class_map 밖 {len(skipped)}개 클래스 "
            f"{total}개 detection 제외: {sample}"
        )


def _format_count_sample(counts: dict[str, int], limit: int = 10) -> str:
    """클래스별 count 요약 문자열을 만든다."""
    sample = ", ".join(
        f"{name}={count}" for name, count in sorted(counts.items())[:limit]
    )
    return sample + (" ..." if len(counts) > limit else "")
