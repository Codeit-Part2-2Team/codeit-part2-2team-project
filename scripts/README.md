# scripts/

CLI 실행 스크립트입니다. **프로젝트 루트**에서 실행하세요.

## 스크립트 목록

### Stage 1 — YOLO 탐지기

| 스크립트 | 역할 | 담당 |
|----------|------|------|
| `train.py` | config를 읽어 모델 학습 → 가중치 저장 | 승준 |
| `predict.py` | 저장된 가중치로 이미지 추론 → predictions.json 저장 | 승준 |
| `validate.py` | 저장된 가중치로 val set 평가 → mAP 출력 | 승준 |
| `make_submission.py` | crops_manifest + stage2_predictions → Kaggle 제출용 submission.csv 변환 (score = det×cls, category_id 매핑 적용) | 도혁 |
| `convert_annotations.py` | src/data 모듈을 이용해 raw/external 어노테이션을 YOLO 라벨로 변환 | 소원 |


### Stage 2 — 분류기 파이프라인

| 스크립트 | 역할 |
|----------|------|
| `build_classification_dataset.py` | src/data 모듈을 이용해 raw/external crop 생성 및 분류 데이터셋(train/val/test) 생성 |
| `pipeline/crop.py` | 알약 1개 단위 크롭 생성. **gt 모드**: GT label → flat 크롭 + manifest (학습용, 최초 1회). **inference 모드**: Stage 1 predictions → flat 크롭 + manifest (추론용). **convert 모드**: 외부 제공 ImageFolder → manifest만 생성 (재크롭 없음) |
| `pipeline/stage2_train.py` | GT 크롭으로 분류기 학습 → 가중치 저장 |
| `pipeline/stage2_predict.py` | 크롭 이미지 분류 → stage2_predictions.json 저장 |
| `pipeline/run_train.py` | Stage 1 학습 → Stage 2 학습 통합 실행 |
| `pipeline/run_predict.py` | Stage 1 추론 → 크롭(inference 모드 자동) → Stage 2 추론 → submission.csv 통합 실행 |
| `pipeline/evaluate_pipeline.py` | GT labels + S1 crops manifest + S2 predictions → end-to-end mAP@[0.50:0.95] / mAP@[0.75:0.95] 계산 (로컬 Kaggle 점수 추정) |

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
  [ make_submission.py ]          Kaggle category_id 매핑 + CSV 생성
      │  --manifest crops_manifest.json
      │  --s2-preds stage2_predictions.json
      │  --class-map kaggle_class_map.json
      ▼
  submissions/submission.csv
      score = det_score × cls_score
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

# 제출 파일 생성 (2-stage 파이프라인 결과)
python scripts/make_submission.py \
    --manifest  experiments/{EXP}/stage1_crops/crops_manifest.json \
    --s2-preds  experiments/{EXP}/stage2_predictions.json \
    --class-map data/processed/kaggle_class_map.json \
    --output    submissions/submission.csv

# 어노테이션 변환
python -m scripts.convert_annotations --project-root .
    --project-root .
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
| `--manifest` | ✅ | — | crops_manifest.json 경로 (Stage 1 crop 메타) |
| `--s2-preds` | ✅ | — | stage2_predictions.json 경로 |
| `--output` | ✅ | — | submission.csv 저장 경로 |
| `--class-map` | | — | Kaggle category_id 매핑 JSON `{class_name: id}` |
| `--image-id-map` | | — | Kaggle image_id 매핑 JSON `{stem: int}` |
| `--unknown-class-map` | | — | Stage 2 alias class_name을 Kaggle canonical class로 치환하는 JSON |
| `--strict-class-map` | | `false` | class_map 밖 class_name을 제외하지 않고 에러로 처리 |

score = `det_score` (manifest) × `cls_score` (stage2_predictions)

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
| `--class-map` | | — | Kaggle category_id 매핑 JSON `{class_name: id}` |
| `--image-id-map` | | — | Kaggle image_id 매핑 JSON `{stem: int}` |
| `--unknown-class-map` | | — | Stage 2 alias class_name을 Kaggle canonical class로 치환하는 JSON |
| `--strict-class-map` | | `false` | class_map 밖 class_name을 제외하지 않고 에러로 처리 |

### pipeline/evaluate_pipeline.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--gt-labels` | ✅ | — | YOLO label 디렉터리 |
| `--gt-images` | ✅ | — | 원본 이미지 디렉터리 |
| `--s1-crops` | ✅ | — | Stage 1 inference crops_manifest.json |
| `--s2-preds` | ✅ | — | Stage 2 predictions JSON |
| `--kaggle-classes` | | — | Kaggle class map JSON. 지정 시 해당 canonical class만 평가 |
| `--unknown-class-map` | | — | Stage 2 alias class_name을 평가용 canonical class로 치환 |
| `--per-class` | | `false` | 클래스별 AP 상위 20개 출력 |

GT bbox는 YOLO label에서 읽고, GT class는 GT crop manifest / Roboflow source key /
`raw_K-*` 파일명의 category_id를 사용해 복원한다. YOLO label은 Stage 1용
`pill=0` 단일 클래스이므로 E2E class 평가에는 별도 class 복원이 필요하다.

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
    "class_name": "crestor_tab_20mg",
    "score":      0.912
  }
]
```

### submission.csv

`make_submission.py` 출력. Kaggle 제출 포맷.

```
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
1, 1, 16262, 120, 45, 260, 165, 0.84
```

- `category_id`: `kaggle_class_map.json` 기준 Kaggle dl_idx (예: crestor_tab_20mg → 16262)
- `score`: `crops_manifest.json`의 det_score × `stage2_predictions.json`의 cls_score
- `crops_manifest.json`의 bbox + `stage2_predictions.json`의 class를 `crop_id` 기준으로 병합해 생성.

---

## Submission ID Mapping Rule

Kaggle 제출은 반드시 **test 원본 이미지 기준 ID**를 사용한다.

- `data/test/1.jpg`에서 Stage 1이 bbox 2개를 찾으면 inference crop은 `1_0.jpg`, `1_1.jpg`처럼 생성된다.
- 이때 `crop_id`는 `1_0`, `1_1`이지만 최종 `submission.csv`의 `image_id`는 둘 다 `1`이어야 한다.
- `make_submission.py`는 `stage2_predictions.json`의 `crop_id`로 inference `crops_manifest.json`을 join하고, 제출용 `image_id`와 bbox는 manifest의 `image_id`, `bbox`를 사용한다.
- 따라서 제출 생성 시 `--manifest`에는 GT crop manifest가 아니라 **test inference crop manifest**를 넣어야 한다.

Example:

```bash
python scripts/pipeline/crop.py \
    --predictions data/test/s1_predictions.json \
    --source      data/test \
    --output      data/test/s1_crops

python scripts/make_submission.py \
    --manifest  data/test/s1_crops/crops_manifest.json \
    --s2-preds  data/test/stage2_predictions.json \
    --class-map data/processed/kaggle_class_map.json \
    --unknown-class-map data/processed/kaggle_unknown_class_map.json \
    --output    submissions/submission.csv
```

`--class-map`을 지정한 경우 map에 없는 예측 `class_name`은 Kaggle 제출 대상 밖 클래스로 보고 기본 제외한다. 잘못된 내부 class index가 Kaggle `category_id`로 조용히 들어가는 상황을 막기 위함이다. 누락 클래스를 즉시 에러로 잡고 싶으면 `--strict-class-map`을 추가한다.

map 밖 클래스를 Kaggle 대상 클래스 중 하나로 치환해야 한다면 `--unknown-class-map`을 사용한다.

```json
{
  "40mg_isoptin_tab": "twynsta_tab_40_5mg",
  "some_external_class": 27733
}
```

값은 `kaggle_class_map.json`에 있는 class name 또는 category_id여야 한다. 임의의 `unknown` category는 Kaggle 평가 대상에 없으면 사용할 수 없다.

---

## Stage 2 자동 튜닝

Stage 2 자동 튜닝은 분류기 자체의 validation metric을 높이는 용도다. 기본 objective는 `top1_acc`이며, Stage 1은 고정한다. 최종 모델 후보는 별도로 E2E mAP를 확인한다.

> 현재 runner는 `top1_acc` / `top5_acc` objective를 지원한다. E2E mAP objective는 trial마다 Stage 2 prediction과 `evaluate_pipeline.py`까지 실행해야 하므로 후속 확장 대상으로 둔다.

### 공통 입력

| 인자 | 필수 | 설명 |
|------|------|------|
| `--base-config` | ✅ | 기준 Stage 2 config YAML. 보통 `experiments/exp_20260420_baseline_yolo26n/s2_config.yaml` |
| `--search-space` | 선택 | 탐색 공간 YAML. 생략 시 runner 내 기본 탐색 공간 사용 |
| `--output` | 선택 | trial 산출물 저장 경로 |
| `--data` | 선택 | crop root. 지정 시 `<data>/train`, `<data>/val`을 사용 |
| `--epochs` | 선택 | 탐색용 epoch override. 빠른 탐색은 15~30 권장 |
| `--device` | 선택 | `0`, `1`, `cpu` 등 device override |
| `--metric` | 선택 | objective metric. `top1_acc` 또는 `top5_acc` |

`--base-config`의 `model.num_classes`는 실제 crop train class 수와 같아야 한다. baseline freeze 기준은 305개 class set이다.

### Grid Search

정해진 후보 조합을 모두 실행한다. 작은 후보군을 명시적으로 비교할 때 사용한다.

```bash
python scripts/pipeline/stage2_grid_search.py \
    --base-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --search-space experiments/stage2_grid_search/grid_space.yaml \
    --output experiments/stage2_grid_search \
    --epochs 30 \
    --metric top1_acc
```

추가 옵션:

| 인자 | 설명 |
|------|------|
| `--max-trials` | 앞에서 N개 조합만 실행. smoke test나 비용 제한용 |

`--search-space`를 생략하면 기본 탐색 공간(`lr0`, `weight_decay`, `label_smoothing`)을 사용한다. 직접 지정하려면 아래처럼 YAML을 작성한다.

```yaml
# experiments/stage2_grid_search/grid_space.yaml
model:
  name: [resnet50, efficientnet_b2]

train:
  lr0: [0.00003, 0.0001, 0.0003]
  lrf: [0.005, 0.01]
  weight_decay: [0.001, 0.01]
  label_smoothing: [0.0, 0.05, 0.1]
```

YAML은 Stage 2 config 구조와 같은 nested 형식을 권장한다. 짧게 쓰고 싶으면 dotted key도 사용할 수 있다.

```yaml
# 위 설정과 동일
model.name: [resnet50, efficientnet_b2]
train.lr0: [0.00003, 0.0001, 0.0003]
train.lrf: [0.005, 0.01]
train.weight_decay: [0.001, 0.01]
train.label_smoothing: [0.0, 0.05, 0.1]
```

### search_space에 넣을 수 있는 주요 파라미터

Runner는 Stage 2 config의 값을 dotted key로 수정한다. 즉 `s2_config.yaml`에 존재하고 `Classifier` / `Stage2Dataset`이 실제로 읽는 값이면 search space에 넣을 수 있다. 우선 아래 파라미터를 권장한다.

아래 key 목록은 **Grid Search와 Optuna가 동일하게 사용**한다. 차이는 YAML 값 형식뿐이다.

- Grid Search: `lr0: [0.00003, 0.0001, 0.0003]`처럼 후보 리스트를 적는다.
- Optuna: `lr0: {type: float, low: ..., high: ..., log: true}`처럼 샘플링 규칙을 적는다.
- key는 nested 형식을 권장하며, dotted key도 옵션으로 지원한다.

| Nested 위치 | dotted alias | 타입 예시 | 설명 | 권장 범위/후보 |
|-------------|--------------|-----------|------|----------------|
| `model: name` | `model.name` | categorical | Stage 2 backbone | `resnet50`, `efficientnet_b2`, `efficientnetv2_s` |
| `train: lr0` | `train.lr0` | float log | 초기 learning rate | `1e-5` ~ `3e-4` |
| `train: lrf` | `train.lrf` | float log | cosine scheduler 최종 LR 비율 | `0.005` ~ `0.05` |
| `train: weight_decay` | `train.weight_decay` | float log | weight decay | `1e-4` ~ `1e-2` |
| `train: label_smoothing` | `train.label_smoothing` | float | CE/Focal label smoothing | `0.0` ~ `0.15` |
| `train: batch` | `train.batch` | categorical | batch size | `16`, `32` |
| `train: optimizer` | `train.optimizer` | categorical | optimizer | `AdamW`, `Adam`, `SGD` |
| `train: warmup_epochs` | `train.warmup_epochs` | int | warmup epoch 수 | `1` ~ `5` |
| `train: criterion` | `train.criterion` | categorical | loss 함수 | `cross_entropy`, `focal` |
| `train: focal_alpha` | `train.focal_alpha` | float | FocalLoss alpha | `0.1` ~ `0.75` |
| `train: focal_gamma` | `train.focal_gamma` | float | FocalLoss gamma | `1.0` ~ `3.0` |
| `albumentations: brightness_contrast: p` | `albumentations.brightness_contrast.p` | float | 밝기/대비 증강 확률 | `0.0` ~ `0.7` |
| `albumentations: brightness_contrast: brightness_limit` | `albumentations.brightness_contrast.brightness_limit` | float | 밝기 변화 폭 | `0.05` ~ `0.25` |
| `albumentations: brightness_contrast: contrast_limit` | `albumentations.brightness_contrast.contrast_limit` | float | 대비 변화 폭 | `0.05` ~ `0.25` |
| `albumentations: jpeg_compression: p` | `albumentations.jpeg_compression.p` | float | JPEG compression 증강 확률 | `0.0` ~ `0.5` |
| `albumentations: jpeg_compression: quality_lower` | `albumentations.jpeg_compression.quality_lower` | int | JPEG 품질 하한 | `70` ~ `95` |
| `albumentations: gaussian_blur: p` | `albumentations.gaussian_blur.p` | float | blur 증강 확률 | `0.0` ~ `0.4` |

주의:

- `model.num_classes`, `nc`는 search space에 넣지 않는다. 실제 crop class 수와 일치해야 하는 고정값이다.
- `data.train`, `data.val`, `output.project`, `output.name`은 runner가 관리하므로 search space에 넣지 않는다.
- `train.criterion: bce`는 현재 label 형식과 맞지 않을 수 있어 기본 탐색에서는 제외한다.
- `SGD`를 탐색할 때는 `train.momentum`도 함께 넣을 수 있다.

### Optuna

Optuna가 search space 안에서 후보를 샘플링한다. 연속형 하이퍼파라미터를 탐색할 때 사용한다.

```bash
python scripts/pipeline/stage2_optuna.py \
    --base-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --search-space experiments/stage2_optuna/search_space.yaml \
    --output experiments/stage2_optuna \
    --n-trials 30 \
    --epochs 30 \
    --metric top1_acc
```

추가 옵션:

| 인자 | 설명 |
|------|------|
| `--n-trials` | 실행할 trial 수 |
| `--study-name` | Optuna study 이름. 기본값 `stage2_optuna` |

Optuna search space YAML은 parameter별 타입을 명시한다.

```yaml
# experiments/stage2_optuna/search_space.yaml
model:
  name:
    type: categorical
    choices: [resnet50, efficientnet_b2, efficientnetv2_s]

train:
  lr0:
    type: float
    low: 0.00001
    high: 0.0003
    log: true
  lrf:
    type: float
    low: 0.005
    high: 0.05
    log: true
  weight_decay:
    type: float
    low: 0.0001
    high: 0.01
    log: true
  label_smoothing:
    type: float
    low: 0.0
    high: 0.15
  batch:
    type: categorical
    choices: [16, 32]
```

지원 타입:

| type | 필드 | 예시 |
|------|------|------|
| `categorical` | `choices` | 모델명, optimizer, batch 후보 |
| `float` | `low`, `high`, 선택 `log` | learning rate, weight decay |
| `int` | `low`, `high`, 선택 `step`, `log` | epoch, warmup 등 정수값 |

Optuna도 dotted key를 옵션으로 사용할 수 있다.

```yaml
train.lr0:
  type: float
  low: 0.00001
  high: 0.0003
  log: true
```

위 설정은 nested YAML의 `train: lr0:`과 동일하게 처리된다.

### 추천 탐색 순서

처음부터 큰 search space를 돌리지 말고 아래 순서로 좁힌다.

1. `--epochs 15~30`으로 빠른 탐색
2. `model.name`, `train.lr0`, `train.weight_decay`, `train.label_smoothing` 우선 탐색
3. 상위 3~5개 trial을 100 epoch로 재학습
4. 재학습된 `best.pt`로 Stage 2 prediction 생성
5. `evaluate_pipeline.py --kaggle-classes --unknown-class-map` 기준 E2E mAP 확인

추천 starting space:

```yaml
model:
  name:
    type: categorical
    choices: [resnet50, efficientnet_b2]
train:
  lr0:
    type: float
    low: 0.00003
    high: 0.0003
    log: true
  weight_decay:
    type: float
    low: 0.001
    high: 0.01
    log: true
  label_smoothing:
    type: float
    low: 0.0
    high: 0.1
```

공통 산출물:

```text
experiments/stage2_optuna/
├── search_space.yaml
├── study.db              # Optuna 사용 시
├── results.csv           # trial별 score/top1/top5/elapsed_sec/params/status
├── best_trial.json       # 최고 trial 요약
└── trial_0000/
    ├── config.yaml       # trial에 실제 사용한 config
    ├── result.json       # trial 결과 한 줄 요약
    ├── timings.json      # hpo_trial 소요 시간
    └── weights/
        ├── best.pt
        └── last.pt
```

`results.csv`의 `score`는 `--metric`으로 선택한 objective 값이다. trial 소요 시간은 각 `trial_*/timings.json`의 `hpo_trial`에 저장되며, 비교 편의를 위해 `results.csv`와 `result.json`에도 `elapsed_sec`로 함께 기록한다.

튜닝 결과를 채택하기 전에는 상위 trial의 `best.pt`로 Stage 2 prediction을 만들고, `evaluate_pipeline.py`로 Kaggle 기준 E2E mAP를 재확인한다.

```bash
python scripts/pipeline/stage2_predict.py \
    --config  experiments/stage2_optuna/trial_0000/config.yaml \
    --weights experiments/stage2_optuna/trial_0000/weights/best.pt \
    --source  experiments/exp_20260420_baseline_yolo26n/stage1_crops \
    --output  experiments/stage2_optuna/trial_0000/stage2_predictions.json

python scripts/pipeline/evaluate_pipeline.py \
    --gt-labels data/processed/labels/val \
    --gt-images data/processed/images/val \
    --s1-crops  experiments/exp_20260420_baseline_yolo26n/stage1_crops/crops_manifest.json \
    --s2-preds  experiments/stage2_optuna/trial_0000/stage2_predictions.json \
    --kaggle-classes data/processed/kaggle_class_map.json \
    --unknown-class-map data/processed/kaggle_unknown_class_map.json
```
