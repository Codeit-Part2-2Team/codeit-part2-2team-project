# src/data/

Stage 1 / Stage 2 데이터 파이프라인 공통 모듈입니다.
scripts/ 에서 호출하여 사용합니다.

## 파일 설명

| 파일 | 역할 |
|------|------|
| parser_raw.py | raw 데이터 파싱 / annotation 변환 / crop 생성 |
| parser_external.py | external 데이터 파싱 / bbox 정리 / crop 생성 |
| class_map.py | 클래스 매핑 CSV 로드 |
| split.py | train / val / test 분할 |

## 사용 위치

- scripts/convert_annotations.py
- scripts/build_classification_dataset.py