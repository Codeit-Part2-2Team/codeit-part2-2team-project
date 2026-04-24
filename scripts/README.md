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
| `convert_annotations.py` | src/data 모듈을 이용해 raw/external 어노테이션을 YOLO 라벨로 변환 | 소원 |

### Stage 2 — 분류기 파이프라인

| 스크립트 | 역할 |
|----------|------|
| `build_classification_dataset.py` | src/data 모듈을 이용해 raw/external crop 생성 및 분류 데이터셋(train/val/test) 생성 |
| `pipeline/crop.py` | Stage 1 predictions.json + 원본 이미지 → 알약 1개 단위 크롭 이미지 저장 |
| `pipeline/stage2_train.py` | 크롭 이미지로 분류기 학습 → 가중치 저장 |
| `pipeline/stage2_predict.py` | 크롭 이미지 분류 → stage2_predictions.json 저장 |
| `pipeline/run_train.py` | Stage 1 학습 → Stage 2 학습 통합 실행 |
| `pipeline/run_predict.py` | Stage 1 추론 → 크롭(inference 모드 자동) → Stage 2 추론 → submission.csv 통합 실행 |

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

  1. Stage 1 학습
     data/processed/dataset.yaml
         │
         ▼
     [ train.py ]
         │
         ▼
     stage1/weights/best.pt

  2. GT 크롭 생성 (최초 1회, 이후 재사용)
     data/processed/labels/{train,val}/
     data/processed/images/{train,val}/
         │
         ▼
     [ pipeline/crop.py --labels ... --images ... ]  gt 모드
         │
         ▼
     data/processed/crops/{train,val}/
       *.jpg + crops_manifest.json  (class_name 포함)

  3. Stage 2 학습
     data/processed/crops/{train,val}/
         │
         ▼
     [ pipeline/stage2_train.py ]
         │
         ▼
     stage2/weights/best.pt

── 추론 (run_predict.py 로 한 번에) ─────────────────────────

  data/raw/test/
      │
      ▼
  [ predict.py ]                  Stage 1 추론
      │  predictions.json
      ▼
  [ pipeline/crop.py ]            inference 모드 (자동 호출)
      │  crops/inference/ + crops_manifest.json  (score 포함)
      ▼
  [ pipeline/stage2_predict.py ]  브랜드명 분류
      │  stage2_predictions.json
      ▼
  [ make_submission.py ]          bbox + class 병합
      │
      ▼
  submissions/submission.csv
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
python -m scripts.convert_annotations --project-root .
```

### Stage 2 파이프라인

```bash
# [최초 1회] GT 크롭 생성 (학습용) — 원본 이미지에서 직접 크롭
python scripts/pipeline/crop.py \
    --labels  data/processed/labels \
    --images  data/processed/images \
    --output  data/processed/crops \
    --splits  train val

# [대안] 외부 제공 ImageFolder 크롭이 이미 있을 때 manifest만 생성
python scripts/pipeline/crop.py \
    --imagefolder data/processed/crops \
    --splits      train val

# Stage 2 학습
python scripts/pipeline/stage2_train.py \
    --config experiments/stage2_classifier/config.yaml

# Stage 2 추론 (크롭이 이미 있는 경우 단독 실행)
python scripts/pipeline/stage2_predict.py \
    --config  experiments/stage2_classifier/config.yaml \
    --source  data/processed/crops/inference/

# 통합 추론 (Stage 1 추론 → 크롭 → Stage 2 추론 → submission.csv)
python scripts/pipeline/run_predict.py \
    --stage1-config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
    --stage2-config experiments/stage2_classifier/config.yaml \
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

> 참고: `predict.py`에서는 CLI 인자가 먼저 적용됩니다. `--weights`와 `--output`이 없을 경우에만 config의 `output.project/output.name` 기반 경로를 자동 조합합니다.

### validate.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | — | config.yaml 경로 |
| `--weights` | | 자동 조합 | 우선순위: CLI → `{project}/{name}/weights/best.pt` |
| `--data` | | config | dataset.yaml 경로 (덮어쓰기) |

> 참고: `validate.py`에서도 CLI 인자가 먼저 적용됩니다. `--weights`가 없으면 config의 output 설정 기준으로 weights 경로를 자동 조합하며, `--data`가 지정되면 config data 설정을 덮어씁니다.

### make_submission.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | — | predictions.json 경로 |
| `--output` | ✅ | — | submission.csv 저장 경로 |

### convert_annotations.py

| 인자                     | 필수 | 기본값                   | 설명                    |
| ---------------------- | -- | --------------------- | --------------------- |
| `--project-root`       |    | .                     | 프로젝트 루트 경로            |
| `--raw-annotation-dir` |    | raw/train_annotations | raw json 폴더 경로        |
| `--raw-output-dir`     |    | processed/raw_labels  | raw YOLO txt 저장 경로    |
| `--ext-label-roots`    |    | external/labels/*     | external 라벨 폴더들       |
| `--ext-output-dir`     |    | processed/ext_labels  | external 변환 txt 저장 경로 |
| `--class-id`           |    | 0                     | 단일 클래스 id (pill)      |

### build_classification_dataset.py

| 인자 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `--final-class-table` | ✅ | — | 클래스 매핑 CSV |
| `--raw-ann-root` | ✅ | — | raw annotation 폴더 |
| `--raw-img-root` | ✅ | — | raw 이미지 폴더 |
| `--external-root` | ✅ | — | external 데이터 루트 |
| `--crop-root` | ✅ | — | 전체 crop 저장 경로 |
| `--filtered-crop-root` | ✅ | — | 필터링 crop 저장 경로 |
| `--cls-root` | ✅ | — | 최종 분류 데이터셋 저장 경로 |
| `--keep-under10-file` |  | 없음 | 수동 유지 클래스 txt |
| `--min-images` |  | 10 | 최소 유지 이미지 수 |
| `--train-ratio` |  | 0.7 | train 비율 |
| `--val-ratio` |  | 0.15 | val 비율 |
| `--test-ratio` |  | 0.15 | test 비율 |
| `--seed` |  | 42 | 랜덤 시드 |

### pipeline/crop.py

세 모드 중 하나를 선택해 실행한다.

**inference 모드** (`--predictions` 지정 시)

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | — | Stage 1 predictions.json 경로 |
| `--source` | ✅ | — | 원본 이미지 디렉터리 |
| `--output` | ✅ | — | 크롭 저장 디렉터리 |
| `--padding` | | `0.05` | bbox 여백 비율 |

출력: `{output}/*.jpg` + `{output}/crops_manifest.json` (`score` 포함, `class_name` 없음)

**gt 모드** (`--labels` 지정 시)

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--labels` | ✅ | — | YOLO label 루트 (`labels/train/`, `labels/val/`) |
| `--images` | ✅ | — | 원본 이미지 루트 (`images/train/`, `images/val/`) |
| `--output` | ✅ | — | 크롭 저장 루트 디렉터리 |
| `--splits` | | `train val` | 처리할 split 목록 |
| `--padding` | | `0.05` | bbox 여백 비율 |

출력: `{output}/{split}/*.jpg` + `{output}/{split}/crops_manifest.json` (`class_name` 포함, `score` 없음)  
클래스명은 파일명에서 자동 탐지한다.

**convert 모드** (`--imagefolder` 지정 시)

외부에서 제공받은 크롭이 ImageFolder 구조(`class_name/img.jpg`)로 있고 `crops_manifest.json`이 없을 때 사용한다.  
이미지를 재크롭하지 않고 manifest만 생성한다. 클래스명은 서브디렉터리명을 그대로 사용한다.

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--imagefolder` | ✅ | — | ImageFolder 루트 디렉터리 |
| `--splits` | | `train val` | 처리할 split 목록 |

출력: `{imagefolder}/{split}/crops_manifest.json` (`class_name` 포함)  
`crops_manifest.json`이 이미 존재하는 split은 자동으로 스킵한다.

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

### pipeline/run_train.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--stage1-config` | ✅ | — | Stage 1 config.yaml |
| `--stage2-config` | ✅ | — | Stage 2 config.yaml |
| `--data` | ✅ | — | Stage 1 dataset.yaml 경로 |
| `--crops` | ✅ | — | Stage 2 학습용 GT crop 루트 디렉터리 |
| `--device` | | config | GPU 번호 또는 `cpu` |

### pipeline/run_predict.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--stage1-config` | ✅ | — | Stage 1 config.yaml |
| `--stage2-config` | ✅ | — | Stage 2 config.yaml |
| `--source` | ✅ | — | 테스트 이미지 디렉터리 |
| `--output` | ✅ | — | submission.csv 저장 경로 |
| `--stage1-weights` | | 자동 조합 | Stage 1 가중치 경로 |
| `--stage2-weights` | | 자동 조합 | Stage 2 가중치 경로 |
| `--crop-output` | | `data/processed/crops/inference` | 크롭 이미지 저장 디렉터리 |
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
      {
        "class_id": 0,
        "class_name": "pill",
        "bbox": [120.0, 45.0, 380.0, 210.0],
        "score": 0.91
      }
    ]
  }
]
```

Stage 1은 단일 클래스 탐지기이므로 `class_id`와 `class_name`은 기본적으로 하나의 pill 클래스를 의미합니다.

### crops_manifest.json

`crop.py` 출력. 모드에 따라 포함 필드가 다르다.

**inference 모드** — `stage2_predict.py` 입력 / Stage 1 bbox 역추적에 사용

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

**gt 모드** — `stage2_train.py` (Stage2Dataset) 입력

```json
[
  {
    "image_id":   "Acetaminophen_500mg_jpg.rf.abc123",
    "crop_id":    "Acetaminophen_500mg_jpg.rf.abc123_0000",
    "crop_path":  "data/processed/crops/train/Acetaminophen_500mg_jpg.rf.abc123_0000.jpg",
    "bbox":       [120.0, 45.0, 380.0, 210.0],
    "class_name": "Acetaminophen_500mg"
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
