from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_class_maps(
    final_class_table_path: Path,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    final_class_table.csv를 읽어서 raw/external class_id → final_class_name 매핑을 만든다.

    Returns
    -------
    raw_map : Dict[int, str]
        raw 데이터의 source_class_id를 최종 클래스명으로 변환하는 딕셔너리

    ext_map : Dict[int, str]
        external 데이터의 source_class_id를 최종 클래스명으로 변환하는 딕셔너리
    """

    df = pd.read_csv(final_class_table_path)

    raw_map: Dict[int, str] = {}
    ext_map: Dict[int, str] = {}

    for _, row in df.iterrows():
        source = str(row["source"])
        class_id = int(row["source_class_id"])
        class_name = str(row["final_class_name"])

        if source == "raw":
            raw_map[class_id] = class_name

        elif source == "external":
            ext_map[class_id] = class_name

    return raw_map, ext_map