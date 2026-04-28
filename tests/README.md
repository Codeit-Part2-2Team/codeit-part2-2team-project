# tests/

단위 테스트 디렉터리입니다. GPU·실제 데이터·가중치 파일 없이 실행 가능합니다.

## 실행

```bash
# 전체 실행
python -m pytest -v

# 특정 파일만
python -m pytest -v tests/test_submission.py

# 특정 테스트만
python -m pytest -v tests/test_model_yolo.py -k "test_init"

# HTML 리포트 생성
python -m pytest -v --html=reports/test_report.html
```

## 테스트 파일 목록

| 파일 | 테스트 대상 | 주요 검증 내용 |
|------|------------|---------------|
| `test_config.py` | `src/utils/config.py` | YAML 로드, 필수 키 검증, 시드 고정 재현성 |
| `test_model_yolo.py` | `src/models/model_yolo.py` | 모델 초기화, 가중치 로드 체이닝, raw_predict 인자 전달 |
| `test_trainer.py` | `src/training/trainer.py` | kwargs 병합, device 중복 없음, mAP 반환 포맷 |
| `test_predictor.py` | `src/models/predictor.py` | `_parse_results` 스키마, JSON 저장, TTA 인자 전달 |
| `test_submission.py` | `src/utils/submission.py` | CSV 컬럼 구조, 행 수, bbox xyxy→xywh 변환, 빈 검출 처리 |
| `test_pipeline_run.py` | `scripts/pipeline/run.py` | 경로 해석, predictions 병합 로직 |
| `test_timing.py` | `src/utils/timing.py` | exp_dir 경로 해석(S1/S2), timings.json 누적·덮어쓰기, timed 컨텍스트 |
| `test_report.py` | `src/utils/report.py` | timings 로드·출력·오버헤드 계산, results.csv best epoch 파싱 |

## 구조

```
tests/
  ├── conftest.py           # 공통 픽스처 (sample_config, sample_predictions)
  ├── test_config.py
  ├── test_model_yolo.py
  ├── test_trainer.py
  ├── test_predictor.py
  ├── test_submission.py
  ├── test_pipeline_run.py  # pipeline/run.py 유틸 함수
  ├── test_timing.py        # timing.py: exp_dir_from_cfg, record_time, timed
  └── test_report.py        # report.py: load_timings, print_timings, load_s1_best_metrics
```

---

## conftest.py 픽스처

### `sample_config`

모든 테스트에서 공유하는 최소 config dict입니다.  
실제 학습이 일어나지 않도록 `epochs=1`, `batch=2`, `warmup_epochs=0`, `pretrained=False`로 설정돼 있습니다.

```python
{
    "seed": 42,
    "model": {"name": "yolo26n", "pretrained": False},
    "data": {"yaml": "data/processed/dataset.yaml", "imgsz": 640, "workers": 2},
    "train": {
        "epochs": 1, "batch": 2, "optimizer": "AdamW",
        "lr0": 0.001, "lrf": 0.01, "momentum": 0.937,
        "weight_decay": 0.0005, "warmup_epochs": 0, "patience": 5,
    },
    "augment": {
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    },
    "val": {"conf": 0.25, "iou": 0.45, "max_det": 300},
    "output": {"project": "experiments", "name": "test_run", "save_period": -1},
}
```

픽스처를 변형해서 쓸 경우 해당 테스트 함수 안에서만 수정하세요 (`del sample_config["model"]` 등). 픽스처 자체는 매 테스트마다 새로 생성됩니다.

### `sample_predictions`

predictions.json 형식의 리스트입니다. 이미지 2장: 첫 번째는 검출 2개, 두 번째는 검출 없음(빈 배열).

```python
[
    {
        "image_id": "1",
        "detections": [
            {"class_id": 1,  "class_name": "Crestor 10mg tab",  "bbox": [156.0, 247.0, 367.0, 703.0], "score": 0.91},
            {"class_id": 24, "class_name": "Aspirin 100mg tab", "bbox": [498.0, 40.0,  958.0, 514.0], "score": 0.78},
        ],
    },
    {
        "image_id": "3",
        "detections": [],  # 미검출 케이스
    },
]
```

---

## Mock 범위

Ultralytics 모델 로딩과 실제 학습·추론 호출은 모두 mock으로 대체합니다.

| 파일 | Mock 대상 | 방법 | 반환값 |
|------|-----------|------|--------|
| `test_model_yolo.py` | `YOLO` 클래스 (Ultralytics) | `@patch("src.models.model_yolo.YOLO")` | `MagicMock()` |
| `test_trainer.py` | `model.raw_train`, `model.raw_val` | `MagicMock()` 전체 교체 | `results_dict`를 가진 `MagicMock` |
| `test_predictor.py` | `model.raw_predict` | `MagicMock()` 전체 교체 | Ultralytics Result 구조를 흉내 낸 `MagicMock` |
| `test_config.py` | 없음 | — | 실제 YAML 파일 읽기 (`tmp_path` 사용) |
| `test_submission.py` | 없음 | — | 순수 데이터 변환, mock 불필요 |
| `test_timing.py` | 없음 | — | 순수 파일 I/O, `tmp_path` 사용 |
| `test_report.py` | 없음 | — | 순수 파일 I/O, `tmp_path` 사용 |

`test_predictor.py`의 `_make_mock_result()`는 Ultralytics `Results` 객체의 `.path`, `.names`, `.boxes[i].xyxy`, `.boxes[i].conf`, `.boxes[i].cls` 속성만 흉내 냅니다. 실제 객체와 동일한 인터페이스이므로 `_parse_results` 로직 검증에는 충분합니다.

---

## 의도적으로 제외된 케이스

| 제외 항목 | 이유 |
|-----------|------|
| 실제 Ultralytics 학습 루프 (`raw_train`) | GPU·데이터 필요, CI 환경에서 실행 불가 |
| 실제 추론 (`raw_predict`) with 가중치 | 가중치 파일 불필요 환경 유지 |
| 멀티 GPU 분기 | 환경 의존성 높음, 수동 검증으로 대체 |
| `src/data/` 관련 테스트 | 미구현 상태 |
| Stage 1 → Stage 2 통합 테스트 | `scripts/pipeline/run.py` 구현됨, 단위 테스트로 커버 |
| `src/utils/visualize.py` 시각화 함수 | matplotlib/PIL 렌더링은 단위 테스트에 부적합, 노트북 실행으로 수동 검증 |

---

## Submission Mapping Tests

`test_submission.py`는 제출 CSV의 기본 컬럼, bbox `xyxy -> xywh` 변환, `annotation_id` 증가뿐 아니라 inference crop ID 매핑을 검증한다.

중요 케이스:

- test 원본 이미지: `1.jpg`
- inference crop: `1_0.jpg`, `1_1.jpg`
- Stage 2 예측의 `crop_id`: `1_0`, `1_1`
- 최종 제출 `image_id`: `1`, `1`

즉, 제출 병합은 `stage2_predictions.image_id`가 아니라 `crop_id -> crops_manifest.json`을 통해 원본 `image_id`와 bbox를 가져와야 한다. 이 테스트는 GT crop manifest를 실수로 제출에 연결하는 회귀를 막기 위한 것이다.

`class_map`에 없는 예측 클래스는 기본 제출 모드에서 제외되는지도 함께 검증한다. `unknown_class_map`이 있으면 Kaggle class로 치환하고, `strict_class_map=True`를 쓰면 누락 class_name을 `KeyError`로 잡는다.
