# scripts/

CLI 실행 스크립트입니다. **프로젝트 루트**에서 실행하세요.

| 스크립트 | 역할 | 담당 |
|----------|------|------|
| `train.py` | config를 읽어 모델 학습 → 가중치 저장 | 승준 |
| `predict.py` | 저장된 가중치로 test 이미지 추론 → predictions.json 저장 | 승준 |
| `validate.py` | 저장된 가중치로 val set 평가 → mAP 출력 | 승준 |
| `make_submission.py` | predictions.json → Kaggle 제출용 submission.csv 변환 | 도혁 |
| `convert_annotations.py` | COCO/VOC 어노테이션 → YOLO txt 포맷 변환 🚧 미구현 | 소원 |

---

## 학습 파이프라인 흐름

```
data/raw/train/   experiments/.../config.yaml
        │                  │
        ▼                  ▼
  [ train.py ]  ──────────────────→  experiments/.../weights/best.pt
                                               │
  data/raw/test/ ◄──────────────────────────────
        │         experiments/.../config.yaml
        ▼                  │
  [ predict.py ] ──────────┘  →  experiments/.../results/predictions.json
                                               │
                                               ▼
                                   [ make_submission.py ]
                                               │
                                               ▼
                                   submissions/submission.csv
```

`validate.py`는 파이프라인과 별도로, 학습 중간이나 완료 후 val set 성능을 확인할 때 독립적으로 사용합니다.

---

## 각 스크립트 설명

### train.py

**입력**: config.yaml  
**출력**: `experiments/<name>/weights/best.pt`, `last.pt`, 학습 로그

config.yaml을 읽어 모델을 초기화하고 학습을 실행합니다.  
결과물은 config의 `output.project` / `output.name` 경로 아래에 저장됩니다.

```
experiments/
└── exp_20260420_baseline_yolo26n/   ← output.name
    ├── weights/
    │   ├── best.pt                  ← val mAP 기준 최고 체크포인트
    │   └── last.pt                  ← 마지막 에포크 체크포인트
    └── ...                          ← Ultralytics 학습 로그, 결과 이미지
```

`--device`를 지정하지 않으면 config의 `train.device` 값을 사용합니다 (`""` = GPU 자동 선택).

---

### predict.py

**입력**: config.yaml + 이미지 디렉터리 (+ best.pt, 선택)  
**출력**: predictions.json

저장된 가중치로 이미지를 추론해 알약 위치(bbox)와 클래스를 검출하고 결과를 JSON으로 저장합니다. 이 JSON이 시스템의 실질적인 추론 출력물이며, Kaggle 점수 산출이 필요한 경우 `make_submission.py`로 변환합니다.  
추론 파라미터(`conf`, `iou`, `max_det`)는 config의 `val` 섹션에서 읽습니다.

`--weights`를 생략하면 config의 `output.project`/`output.name`에서 `best.pt` 경로를 자동 조합합니다.  
`--output`을 생략하면 동일한 방식으로 `results/predictions.json` 경로를 자동 조합합니다.

---

### validate.py

**입력**: config.yaml (+ best.pt, 선택)  
**출력**: 터미널 mAP 출력

`predict.py`와 달리 test 이미지가 아닌 **val set**을 평가합니다.  
JSON 파일을 생성하지 않으며 mAP50 / mAP50-95 수치만 출력합니다.  
학습 완료 후 체크포인트 성능을 빠르게 확인하거나, 데이터 변경 후 재검증할 때 사용합니다.  
`--weights`를 생략하면 config의 `output.project`/`output.name`에서 `best.pt` 경로를 자동 조합합니다.

| | `validate.py` | `predict.py` |
|--|---------------|--------------|
| 대상 데이터 | val set (정답 있음) | 임의 이미지 |
| 출력 | mAP 수치 (터미널) | predictions.json |
| 용도 | 체크포인트 성능 확인 | 실제 추론 실행 (알약 탐지·분류) |

---

### make_submission.py

**입력**: predictions.json  
**출력**: submission.csv

`predict.py`가 생성한 predictions.json을 Kaggle 제출 포맷으로 변환합니다.  
bbox를 xyxy → xywh로 변환하고 `annotation_id`를 1부터 부여합니다.

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
1, test_0001, 42, 120, 45, 260, 165, 0.91
```

---

### convert_annotations.py 🚧 미구현

**예정 기능**: COCO / VOC 포맷 어노테이션 → YOLO txt 포맷 변환  
external 데이터셋 추가 시 사용할 예정입니다.

---

## 실행 예시

```bash
# 1. 학습
python scripts/train.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml

# 학습 (dataset.yaml 덮어쓰기, CPU 실행)
python scripts/train.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --data data/processed/dataset.yaml \
    --device cpu

# 2. 검증 (val set mAP 확인)
python scripts/validate.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --weights experiments/exp_20260420_baseline_yolo26n/weights/best.pt

# 3. 추론 (test 이미지 → predictions.json)
python scripts/predict.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --weights experiments/exp_20260420_baseline_yolo26n/weights/best.pt \
    --source data/raw/test/ \
    --output experiments/exp_20260420_baseline_yolo26n/results/predictions.json

# 4. 제출 파일 생성
python scripts/make_submission.py \
    --predictions experiments/exp_20260420_baseline_yolo26n/results/predictions.json \
    --output submissions/submission.csv
```

---

## CLI 인자 목록

### train.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | - | config.yaml 경로 |
| `--data` | | config | dataset.yaml 경로 (덮어쓰기) |
| `--device` | | config | GPU 번호 또는 `cpu` (예: `0`, `0,1`, `cpu`) |

### predict.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | - | config.yaml 경로 (val 파라미터 사용) |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--source` | ✅ | - | 이미지 디렉터리 경로 |
| `--output` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/results/predictions.json` |
| `--tta` | | `false` | 우선순위: CLI `--tta` → config `val.tta` |

### validate.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | - | config.yaml 경로 |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--data` | | config | dataset.yaml 경로 (덮어쓰기) |

### make_submission.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | - | predictions.json 경로 |
| `--output` | ✅ | - | submission.csv 저장 경로 |

---

## predictions.json 스키마

`predict.py` 출력 파일이자 `make_submission.py` 입력 파일입니다.

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
| `image_id` | `str` | 확장자 제외 파일명 |
| `detections` | `list` | 검출 결과 목록 (미검출 시 빈 배열 `[]`) |
| `bbox` | `[x1, y1, x2, y2]` | 절대 좌표 픽셀, xyxy 포맷 |
| `score` | `float` | confidence score |
