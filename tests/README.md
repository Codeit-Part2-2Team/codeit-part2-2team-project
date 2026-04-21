# tests/

단위 테스트 디렉터리입니다. GPU·실제 데이터 없이 실행 가능합니다.

## 실행

```bash
# 프로젝트 루트에서 실행
pytest -v

# HTML 리포트 생성
pytest -v --html=reports/test_report.html
```

## 테스트 파일 목록

| 파일 | 테스트 대상 | 주요 검증 내용 |
|------|------------|---------------|
| `test_yolo_model.py` | `src/models/yolo_model.py` | config 로드, 모델 초기화, 추론 출력 포맷, JSON 저장 |
| `test_submission.py` | `scripts/make_submission.py` | CSV 컬럼 구조, 행 수, PredictionString 포맷, 빈 검출 처리 |

## 구조

```
tests/
  ├── conftest.py           # 공통 픽스처 (sample_config, sample_predictions)
  ├── test_yolo_model.py
  └── test_submission.py
```

> Ultralytics YOLO 모델 로딩은 `unittest.mock`으로 대체하므로 가중치 파일 없이 동작합니다.
