# tests/

단위 테스트 디렉터리입니다. GPU·실제 데이터 없이 실행 가능합니다.

## 실행

```bash
# 프로젝트 루트에서 실행
python -m pytest -v

# HTML 리포트 생성
python -m pytest -v --html=reports/test_report.html
```

## 테스트 파일 목록

| 파일 | 테스트 대상 | 주요 검증 내용 |
|------|------------|---------------|
| `test_config.py` | `src/utils/config.py` | YAML 로드, 필수 키 검증, 시드 고정 재현성 |
| `test_model_yolo.py` | `src/models/model_yolo.py` | 모델 초기화, 가중치 로드 체이닝, raw_predict 인자 전달 |
| `test_trainer.py` | `src/training/trainer.py` | kwargs 병합, device 중복 없음, mAP 반환 포맷 |
| `test_predictor.py` | `src/models/predictor.py` | _parse_results 스키마, JSON 저장, TTA 인자 전달 |
| `test_submission.py` | `src/utils/submission.py` | CSV 컬럼 구조, 행 수, bbox xywh 변환, 빈 검출 처리 |

## 구조

```
tests/
  ├── conftest.py           # 공통 픽스처 (sample_config, sample_predictions)
  ├── test_config.py
  ├── test_model_yolo.py
  ├── test_trainer.py
  ├── test_predictor.py
  └── test_submission.py
```

> Ultralytics YOLO 모델 로딩은 `unittest.mock`으로 대체하므로 가중치 파일 없이 동작합니다.
