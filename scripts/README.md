# scripts/

CLI 실행 스크립트입니다. **프로젝트 루트**에서 실행하세요.

## 스크립트 목록

### Stage 1 — YOLO 탐지기

| 스크립트 | 역할 | 담당 |
|----------|------|------|
| `train.py` | config를 읽어 모델 학습 → 가중치 저장 | 승준 |
| `predict.py` | 저장된 가중치로 이미지 추론 → predictions.json 저장 | 승준 |
| `validate.py` | 저장된 가중치로 val set 평가 → mAP 출력 | 승준 |
| `make_submission.py` | predictions.json → Kaggle 제출용 submission.csv 변환 | 도혁 |
| `convert_annotations.py` | raw/external 어노테이션을 YOLO 라벨로 변환 | 소원 |

### Stage 2 — 분류기 파이프라인 🚧 구현 예정

| 스크립트 | 역할 |
|----------|------|
| `pipeline/crop.py` | Stage 1 predictions.json + 원본 이미지 → 알약 1개 단위 크롭 이미지 저장 |
| `pipeline/stage2_train.py` | 크롭 이미지로 분류기 학습 → 가중치 저장 |
| `pipeline/stage2_predict.py` | 크롭 이미지 분류 → stage2_predictions.json 저장 |
| `pipeline/run.py` | Stage 1 추론 → 크롭 → Stage 2 추론 → submission.csv 통합 실행 |

---

## 파이프라인 흐름

### Stage 1 단독 실행

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

### 2단계 파이프라인 전체 흐름

```
── 학습 ──────────────────────────────────────────────────────

  Stage 1 학습:  train.py  →  stage1/weights/best.pt

  Stage 2 학습:  (GT bbox 기준 크롭 준비)
                    └→  stage2_train.py  →  stage2/weights/best.pt

── 추론 ──────────────────────────────────────────────────────

  data/raw/test/
      │
      ▼
  [ predict.py ]            Stage 1 추론
      │  predictions.json
      ▼
  [ pipeline/crop.py ]      bbox 기준 알약 1개 단위 크롭
      │  crops/ + crops_manifest.json
      ▼
  [ pipeline/stage2_predict.py ]   브랜드명 분류
      │  stage2_predictions.json
      ▼
  [ make_submission.py ]    Stage 1 bbox + Stage 2 class 병합
      │
      ▼
  submissions/submission.csv

  ※ pipeline/run.py 로 위 추론 단계를 한 번에 실행 가능
```

`validate.py`는 파이프라인과 별도로, 학습 완료 후 val set 성능을 확인할 때 독립적으로 사용합니다.

---

## 실행 예시

### Stage 1

```bash
# 학습
python scripts/train.py \
    --config experiments/stage1_detection/config.yaml

# 검증 (val set mAP 확인)
python scripts/validate.py \
    --config  experiments/stage1_detection/config.yaml \
    --weights experiments/stage1_detection/weights/best.pt

# 추론 (test 이미지 → predictions.json)
python scripts/predict.py \
    --config  experiments/stage1_detection/config.yaml \
    --weights experiments/stage1_detection/weights/best.pt \
    --source  data/raw/test/

# 제출 파일 생성
python scripts/make_submission.py \
    --predictions experiments/stage1_detection/results/predictions.json \
    --output submissions/submission.csv

# 어노테이션 변환
python scripts/convert_annotations.py \
    --project-root .
```

### Stage 2 파이프라인

```bash
# 크롭 생성
python scripts/pipeline/crop.py \
    --predictions experiments/stage1_detection/results/predictions.json \
    --source      data/raw/test/ \
    --output      data/processed/crops/inference/

# Stage 2 학습
python scripts/pipeline/stage2_train.py \
    --config experiments/stage2_classifier/exp_A/config.yaml

# Stage 2 추론
python scripts/pipeline/stage2_predict.py \
    --config  experiments/stage2_classifier/exp_A/config.yaml \
    --source  data/processed/crops/inference/

# 전체 통합 실행 (Stage 1 추론 → 크롭 → Stage 2 추론 → submission.csv)
python scripts/pipeline/run.py \
    --stage1-config experiments/stage1_detection/config.yaml \
    --stage2-config experiments/stage2_classifier/exp_A/config.yaml \
    --source        data/raw/test/ \
    --output        submissions/submission.csv
```

---

## CLI 인자 목록

### train.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | config.yaml 경로 |
| `--data` | | config | dataset.yaml 경로 (덮어쓰기) |
| `--device` | | config | GPU 번호 또는 `cpu` (예: `0`, `0,1`, `cpu`) |

### predict.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | config.yaml 경로 (val 파라미터 사용) |
| `--source` | ✅ | — | 이미지 디렉터리 경로 |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--output` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/results/predictions.json` |
| `--tta` | | `false` | 우선순위: CLI `--tta` → config `val.tta` |

### validate.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | config.yaml 경로 |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--data` | | config | dataset.yaml 경로 (덮어쓰기) |

### make_submission.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | — | predictions.json 경로 |
| `--output` | ✅ | — | submission.csv 저장 경로 |

### pipeline/crop.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | — | Stage 1 predictions.json 경로 |
| `--source` | ✅ | — | 원본 이미지 디렉터리 |
| `--output` | ✅ | — | 크롭 이미지 저장 디렉터리 |
| `--padding` | | `0.05` | bbox 여백 비율 |

### pipeline/stage2_train.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | Stage 2 config.yaml 경로 |
| `--data` | | config | 크롭 루트 디렉터리 (덮어쓰기) |

### pipeline/stage2_predict.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | Stage 2 config.yaml 경로 |
| `--source` | ✅ | — | 크롭 이미지 디렉터리 |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--output` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/results/stage2_predictions.json` |

### pipeline/run.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--stage1-config` | ✅ | — | Stage 1 config.yaml |
| `--stage2-config` | ✅ | — | Stage 2 config.yaml |
| `--source` | ✅ | — | 테스트 이미지 디렉터리 |
| `--output` | ✅ | — | submission.csv 저장 경로 |
| `--stage1-weights` | | 자동 조합 | Stage 1 가중치 경로 |
| `--stage2-weights` | | 자동 조합 | Stage 2 가중치 경로 |
| `--padding` | | `0.05` | crop padding 비율 |

---

## 데이터 포맷

### predictions.json

`predict.py` 출력 / `crop.py` + `make_submission.py` 입력.

```json
[
  {
    "image_id": "test_0001",
    "detections": [
      {"bbox": [120.0, 45.0, 380.0, 210.0], "score": 0.91}
    ]
  }
]
```

Stage 1이 pill 단일 클래스로 동작하므로 `class_id` / `class_name` 없음.

### crops_manifest.json

`crop.py` 출력 / `stage2_predict.py` 입력. Stage 1 bbox 역추적에 사용.

```json
[
  {
    "image_id":  "test_0001",
    "crop_id":   "test_0001_0",
    "crop_path": "data/processed/crops/inference/test_0001_0.jpg",
    "bbox":      [120.0, 45.0, 380.0, 210.0],
    "score":     0.91
  }
]
```

### stage2_predictions.json

`stage2_predict.py` 출력 / `make_submission.py` 입력.

```json
[
  {
    "image_id":   "test_0001",
    "crop_id":    "test_0001_0",
    "class_id":   42,
    "class_name": "Crestor 10mg tab",
    "score":      0.912
  }
]
```

### submission.csv

`make_submission.py` 출력. Kaggle 제출 포맷.

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
1, test_0001, 42, 120, 45, 260, 165, 0.91
```

`crops_manifest.json`의 bbox + `stage2_predictions.json`의 class_id를 `crop_id` 기준으로 병합해 생성.
