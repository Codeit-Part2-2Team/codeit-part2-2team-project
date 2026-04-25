# src/

재사용 가능한 Python 모듈을 보관합니다.  
notebooks 및 scripts에서 `from src.models import ...` 형태로 import하세요.

```
src/
├── __init__.py
├── models/
│   ├── model_yolo.py   # YOLOModel: YOLO 구체 구현 (초기화·가중치·raw 호출) ✅
│   └── predictor.py    # Predictor: 추론 퍼사드 (파싱·JSON 저장) ✅
├── training/
│   └── trainer.py      # Trainer: 학습·검증 퍼사드 ✅
├── data/               # 데이터셋 클래스, 어노테이션 파서, 증강 파이프라인 🚧 미구현
└── utils/
    ├── config.py        # load_config, validate_config, fix_seed ✅
    ├── submission.py    # merge_predictions, predictions_to_df, save_submission ✅
    ├── timing.py        # exp_dir_from_cfg, record_time, timed (context mgr) ✅
    ├── report.py        # load_timings, print_timings, load_s1_best_metrics ✅
    ├── visualize.py     # 노트북 시각화 유틸 (학습곡선·GT비교·crop·오버레이) ✅
    └── classifier.py   # (src/models) Stage 2 분류기 래퍼 ✅
```

> `src/data/` 는 현재 미구현 상태입니다. `from src.data import ...` 사용 시 ImportError가 발생합니다.

---

## 모듈 구조 개요

```
scripts / notebooks
       │
       ▼
  YOLOModel          ← ultralytics.YOLO를 유일하게 import하는 레이어
  ├── Trainer        ← 학습·검증 퍼사드, Ultralytics 직접 import 없음
  └── Predictor      ← 추론 퍼사드, Ultralytics 직접 import 없음
       │
       ▼
  utils/config.py    ← 설정 로드·검증·시드 고정
  utils/submission.py← DataFrame 변환·CSV 저장
```

`Trainer`와 `Predictor`는 Ultralytics를 직접 import하지 않고 `YOLOModel.raw_*` 메서드를 경유합니다.  
덕분에 테스트에서 `YOLOModel`만 mock하면 Trainer/Predictor를 가중치 없이 검증할 수 있습니다.

---

## models/model_yolo.py — `YOLOModel`

Ultralytics YOLO를 감싸는 구체 구현체입니다.  
config 로딩, 검증, 시드 고정, 모델 초기화를 한 번에 처리합니다.

### 초기화 흐름

```
YOLOModel(config)
    │
    ├─ config가 파일 경로이면 → load_config()로 YAML 읽기
    ├─ validate_config() → 필수 섹션 6개 검사
    ├─ fix_seed()        → 전역 시드 고정
    └─ YOLO(weights)     → 가중치 로드
           ├─ pretrained=True  → "yolo26n.pt"  (Ultralytics Hub에서 다운로드)
           └─ pretrained=False → "yolo26n"     (랜덤 초기화, 아키텍처만 로드)
```

### 메서드

```python
# 추론 가중치 교체 (체이닝 가능)
model = YOLOModel(config).load_weights("experiments/.../weights/best.pt")

# Ultralytics Result 리스트 반환 — 포맷 변환은 Predictor가 담당
results = model.raw_predict(source, conf, iou, max_det, augment)

# Ultralytics train/val 결과 객체 반환 — 지표 파싱은 Trainer가 담당
results = model.raw_train(**kwargs)
results = model.raw_val(**kwargs)

# ONNX / TFLite 내보내기 (imgsz는 config에서 자동 참조)
model.export(format="onnx")     # → best.onnx
model.export(format="tflite")   # → best.tflite
```

---

## training/trainer.py — `Trainer`

학습과 검증을 실행하는 퍼사드입니다.  
config의 여러 섹션(`train`, `augment`, `data`, `val`, `output`)을 하나의 flat dict로 병합해 `raw_train()`에 전달합니다.

### `train()`

```python
Trainer(model).train(
    data_yaml: str | Path | None = None,  # None이면 config의 data.yaml 사용
    resume: bool = False,                 # 중단된 학습 재개
) -> {"mAP50": float, "mAP50_95": float}
```

`_build_train_kwargs()`가 아래 섹션을 병합해 Ultralytics에 전달합니다.

| 병합 섹션 | 대표 키 |
|-----------|---------|
| `train` | epochs, batch, optimizer, lr0, lrf, device, patience … |
| `augment` | mosaic, mixup, fliplr, hsv_h … |
| `data` | imgsz, workers |
| `val` | conf, iou, max_det |
| `output` | project, name, save_period |
| 최상위 | seed, exist_ok=True, resume |

### `validate()`

```python
Trainer(model).validate(
    data_yaml: str | Path | None = None,
) -> {"mAP50": float, "mAP50_95": float}
```

학습 없이 저장된 가중치로 val set을 평가합니다. `load_weights()` 후 호출하세요.

```python
model = YOLOModel(config).load_weights("experiments/.../weights/best.pt")
metrics = Trainer(model).validate()
```

---

## models/predictor.py — `Predictor`

추론을 실행하고 결과를 predictions.json으로 저장하는 퍼사드입니다.

### `predict()`

```python
Predictor(model).predict(
    source: str | Path,                               # 이미지 디렉터리
    output: str | Path = "results/predictions.json",  # 출력 경로 (상위 디렉터리 자동 생성)
    tta: bool = False,                                # Test Time Augmentation
) -> list[dict]  # 저장한 내용과 동일한 구조를 반환
```

`val.conf`, `val.iou`, `val.max_det`은 config에서 자동으로 읽습니다.  
`tta=True`이면 Ultralytics의 `augment=True`로 전달됩니다 (좌우 반전·스케일 앙상블).

### 출력 스키마

```json
[
  {
    "image_id": "test_0001",
    "detections": [
      {
        "class_id": 42,
        "class_name": "종근당_펠루비정",
        "bbox": [120.0, 45.0, 380.0, 210.0],
        "score": 0.91
      }
    ]
  }
]
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `image_id` | `str` | 파일명에서 확장자 제거 (`Path.stem`) |
| `bbox` | `[x1, y1, x2, y2]` | 절대 좌표 픽셀, xyxy 포맷 |
| `detections` | `list` | 미검출 시 빈 배열 `[]` |

---

## utils/config.py

### `load_config(path)`

YAML 파일을 읽어 dict를 반환합니다. 파일이 없으면 `FileNotFoundError`.

### `validate_config(cfg)`

필수 최상위 키 6개가 모두 있는지 검사합니다. 누락 시 `ValueError`.

```
필수 키: model, data, train, augment, val, output
```

### `fix_seed(seed)`

아래 6곳을 동시에 고정합니다. `YOLOModel.__init__`에서 자동 호출되므로 직접 호출할 일은 거의 없습니다.

```
random / numpy / torch.manual_seed / torch.cuda.manual_seed(_all)
+ cudnn.deterministic=True / cudnn.benchmark=False
```

---

## models/classifier.py — `Classifier`

Stage 2 timm 분류기 래퍼입니다.

### 체크포인트 포맷

`fit()` / `train()` 완료 시 저장되는 `.pt` 구조:

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "metrics": {          # val 지표 dict 전체 저장
        "top1_acc": float,
        "top5_acc": float,
    },
    "class_names": list[str],
    "cfg": dict,
}
```

> 구버전(`"top1_acc"` 단일 키)과 호환되지 않습니다. 기존 체크포인트는 재학습이 필요합니다.

### num_classes 불일치 동작

| 상황 | 동작 |
|------|------|
| `train()` / `stage2_train.py`: config ≠ 실제 클래스 수 | `ValueError` 발생 — config를 수동으로 수정하세요 |
| `load_weights()`: 체크포인트 ≠ config 클래스 수 | 체크포인트 기준으로 모델 자동 재빌드 (정상 동작) |

---

## utils/timing.py

### `exp_dir_from_cfg(cfg)`

config의 `output` 섹션에서 실험 디렉터리 경로를 반환합니다.

| config 종류 | output 예시 | 반환값 |
|-------------|-------------|--------|
| S1 | `project=experiments, name=exp_xxx` | `experiments/exp_xxx` |
| S2 | `project=experiments/exp_xxx, name=stage2` | `experiments/exp_xxx` |

S2 판별 기준: `project/` 안에 `s1_config.yaml` 또는 `weights/` 디렉터리가 있으면 `project` 자체를 반환합니다.

### `record_time(exp_dir, key, elapsed)`

`exp_dir/timings.json`에 `{key: elapsed_seconds}` 를 upsert합니다. 기존 파일이 있으면 병합합니다.

### `timed(exp_dir, key)` — context manager

```python
with timed(exp_dir, "s1_train"):
    trainer.train()
# → timings.json에 s1_train 소요 시간 자동 기록
```

모든 학습·추론 스크립트에서 사용합니다. 완료 시 `⏱  {key}: {elapsed:.1f}초` 를 출력합니다.

---

## utils/report.py

노트북에서 반복되는 지표 읽기·출력 코드를 담은 유틸입니다.

### `load_timings(exp_dir)` → `dict`

`exp_dir/timings.json` 로드. 파일 없으면 빈 dict 반환.

### `print_timings(exp_dir)`

소요 시간 요약 출력. 키가 없으면 경고 메시지만 표시합니다.  
`pipeline_predict - (s1_predict + crop + s2_predict)` 오버헤드도 함께 계산해 표시합니다.

### `load_s1_best_metrics(results_csv)` → `dict | None`

`results.csv`에서 `mAP50` 최고 에폭의 지표를 반환합니다. 파일 없으면 `None`.

```python
m = load_s1_best_metrics(Path("experiments/.../results.csv"))
# {"epoch": 42, "mAP50": 0.95, "mAP50_95": 0.81, "precision": 0.97, "recall": 0.94}
```

---

## utils/visualize.py

노트북 시각화 코드를 함수로 제공합니다. 반복되는 boilerplate를 제거하는 용도로만 사용합니다.

| 함수 | 역할 |
|------|------|
| `plot_training_curves(results_csv, save_dir)` | Loss / mAP 학습 곡선 |
| `plot_s1_gt_vs_pred(predictions, val_img_dir, val_label_dir, n)` | Stage 1 GT vs 예측 bbox 비교 |
| `plot_crop_showcase(manifest, crop_dir, img_dir, save_dir)` | 탐지 수 최다 이미지 + 각 crop |
| `plot_crop_grid(crop_dir, n)` | 랜덤 crop n개 그리드 |
| `plot_pipeline_overlay(pipeline_by_img, img_dir, n, save_dir)` | 2-Stage 오버레이 (멀티 탐지) |
| `plot_pipeline_strip(pipeline_by_img, crop_dir, img_dir, save_dir)` | 원본 + crop + 분류결과 스트립 |

`save_dir` 지정 시 해당 디렉터리에 PNG 저장, 미지정 시 화면에만 표시합니다.

---

## utils/submission.py

### `merge_predictions(manifest_path, stage2_predictions_path)`

`crops_manifest.json` + `stage2_predictions.json` → image_id별 detections 병합.

- `score = det_score (manifest) × cls_score (stage2)`
- `crop_id` 기준으로 두 파일을 join

### `predictions_to_df(predictions, class_map=None, image_id_map=None)`

merge_predictions() 결과를 Kaggle 제출 포맷 DataFrame으로 변환합니다.

- bbox **xyxy → xywh** 변환 (`w = x2 - x1`, `h = y2 - y1`, 반올림)
- `annotation_id` 1부터 순차 부여
- `class_map`: `{class_name: category_id}` — 미지정 시 내부 class_id 사용
- `image_id_map`: `{stem: int}` — 미지정 시 파일명 그대로 사용

반환 DataFrame 컬럼:

| 컬럼 | 설명 |
|------|------|
| `annotation_id` | 1부터 순차 증가 |
| `image_id` | image_id_map 적용 결과 (없으면 원본 문자열) |
| `category_id` | class_map 적용 결과 (없으면 내부 class_id) |
| `bbox_x`, `bbox_y` | bbox 좌상단 좌표 (정수) |
| `bbox_w`, `bbox_h` | bbox 너비·높이 (정수) |
| `score` | det_score × cls_score |

### `save_submission(predictions, output, class_map=None, image_id_map=None)`

`predictions_to_df()` 호출 후 CSV로 저장합니다. 상위 디렉터리가 없으면 자동 생성합니다.

---

## 전체 사용 예시

```python
from src.models.model_yolo import YOLOModel
from src.models.predictor import Predictor
from src.training.trainer import Trainer
from src.utils.submission import save_submission

EXP = "experiments/exp_20260420_baseline_yolo26n"

# 초기화 (config 로드 + 시드 고정 + 모델 로드 자동 처리)
model = YOLOModel(f"{EXP}/config.yaml")

# 학습
metrics = Trainer(model).train()
print(metrics)  # {"mAP50": 0.823, "mAP50_95": 0.541}

# 학습 재개
metrics = Trainer(model).train(resume=True)

# 검증 (저장된 가중치 사용)
model = YOLOModel(f"{EXP}/config.yaml").load_weights(f"{EXP}/weights/best.pt")
metrics = Trainer(model).validate()

# 추론 → predictions.json
predictions = Predictor(model).predict(
    "data/raw/test/",
    output=f"{EXP}/results/predictions.json",
)

# Kaggle 제출 CSV 생성
save_submission(predictions, "submissions/submission.csv")

# ONNX 내보내기
model.export(format="onnx")
```

---

## 코딩 규칙

- PEP 8 준수 (`ruff`, `black` 자동 포맷)
- 모든 public 함수·클래스에 docstring 작성
- 예외 처리 포함 필수
