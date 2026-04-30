# scripts/pipeline/ - 2-Stage 파이프라인 상세 가이드

> 약물 탐지 2-Stage 파이프라인의 핵심 개념, 데이터 흐름, 각 스크립트의 역할을 다룹니다.

---

## 📌 핵심 개념: GT Crop vs Inference Crop

### 🎯 문제 정의

```
원본 이미지에 여러 약물이 섞여 있음
  ↓
두 가지 질문:
  1. 어디에 약물이 있는가?           → Stage 1 (탐지)
  2. 각 약물이 정확히 무엇인가?      → Stage 2 (분류)
```

---

## 🔄 데이터 흐름

### Phase 1: 모델 학습 (GT Crop 사용)

```
┌─ 학습 데이터 준비 ─────────────────────────────────────┐
│                                                        │
│ data/processed/                                        │
│ ├── images/train/*.jpg           (학습 이미지)        │
│ └── labels/train/*.txt           (정답 위치+약품명)   │
│                                                        │
└─ crop_from_gt() ────────────────────────────────────────┘
                  ↓
        GT Crop 생성 (정답 기반)

┌─ GT Crop (정답 있음) ──────────────────────────────────┐
│                                                        │
│ data/processed/crops/train/                           │
│ ├── 아스피린_0001.jpg  ← 정답 포함                    │
│ ├── 아스피린_0002.jpg     (폴더명 = 실제 약품)       │
│ ├── 감기약_0001.jpg                                   │
│ └── crops_manifest.json  (메타데이터)                │
│                                                        │
│ 각 이미지:                                            │
│ • 100% 정확한 위치 (GT bbox)                         │
│ • 약품명 정보 (폴더명)                               │
│ • 정답 제공 → 모델이 학습                            │
│                                                        │
└────────────────────────────────────────────────────────┘
                  ↓
       ┌─ Stage 2 분류 모델 학습 ─┐
       │                          │
       │ Input: crop 이미지        │
       │ Label: 폴더명 (약품명)    │
       │ Loss: |예측 - 정답|       │
       │       ↓                  │
       │ 가중치 업데이트           │
       │                          │
       └──────────────────────────┘
```

**특징:**
- ✅ 정답이 주어짐 (감독 학습)
- ✅ 100% 정확한 위치
- ✅ 모델: "이게 아스피린이 맞구나" → 학습
- 📊 목표: 높은 학습 정확도

---

### Phase 2: 모델 평가/추론 (Inference Crop 사용)

```
┌─ 검증 데이터 준비 ─────────────────────────────────────┐
│                                                        │
│ data/processed/                                        │
│ ├── images/val/*.jpg             (검증 이미지)       │
│ └── (정답 없음!)                                      │
│                                                        │
└─ Stage 1 모델 추론 ────────────────────────────────────┘
                  ↓
        Stage 1 예측 (오류 가능)

┌─ Inference Crop (정답 없음) ──────────────────────────┐
│                                                        │
│ experiments/{EXP_NAME}/stage1_crops/                  │
│ ├── IMG_001_0.jpg  ← 모델이 예측한 위치              │
│ ├── IMG_001_1.jpg     (실제와 다를 수 있음!)         │
│ ├── IMG_002_0.jpg     (정답 없음)                    │
│ └── crops_manifest.json  (메타데이터)                │
│                                                        │
│ 각 이미지:                                            │
│ • 모델이 예측한 위치 (추론 bbox)                     │
│ • 탐지 신뢰도 포함 (score: 0.95)                    │
│ • 약품명 정보 없음 ← 이제 분류기가 할 일!           │
│                                                        │
└────────────────────────────────────────────────────────┘
                  ↓
       ┌─ Stage 2 분류 모델 추론 ─┐
       │                          │
       │ Input: crop 이미지        │
       │ (정답 모름!)              │
       │       ↓                  │
       │ 모델: "이건 아스피린일     │
       │       거 같아 (0.92)"    │
       │                          │
       └──────────────────────────┘
                  ↓
       최종 결과: 위치 + 약품명 + 신뢰도
```

**특징:**
- ❌ 정답이 없음 (자율 학습 상황)
- ⚠️ 오류 가능 (Stage 1 오류 누적)
- ✅ 모델: "이게 아스피린인 것 같은데?" → 추론만
- 📊 목표: 현실적인 성능 평가

---

## 🎯 Stage 1: 약물 탐지 (Detection)

### 개요

```
원본 이미지
  ↓
[YOLO26n 모델]
  ↓
"여기 알약이 있어!" (bbox + 신뢰도)
  ↓
각 bbox를 crop으로 추출
  ↓
Stage 2 입력 준비 완료
```

**목표**: "약물이 어디에 있는가?" 를 정확하게 찾기

**메트릭**:
- Precision: 탐지한 것 중 정답 비율
- Recall: 정답 중 탐지한 비율  
- mAP50: 탐지 정확도 (관대함)
- mAP50-95: 탐지 정확도 (엄격함)

---

### 학습 데이터 형식 (YOLO)

**디렉토리 구조:**
```
data/processed/
├── images/
│   ├── train/
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   └── ...
│   └── val/
│       ├── IMG_100.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── IMG_001.txt  ← 정답 label
│   │   ├── IMG_002.txt
│   │   └── ...
│   └── val/
│       ├── IMG_100.txt
│       └── ...
└── dataset.yaml  ← 데이터셋 정의
```

**Label 형식 (YOLO txt):**
```
# IMG_001.txt 예시
# 한 줄에 하나의 약물
# 형식: class cx cy w h
#       (모두 정규화된 좌표, 0~1 범위)

0 0.50 0.50 0.20 0.30   ← 약물 1개 (이미지 중앙에 약간 오른쪽)
0 0.70 0.80 0.15 0.25   ← 약물 2개 (이미지 우측 하단)
```

**좌표 설명:**
```
bbox = [cx, cy, w, h]

cx (center_x) = 약물 중심 x좌표 (0~1)
cy (center_y) = 약물 중심 y좌표 (0~1)
w  (width)    = 약물 폭 (0~1)
h  (height)   = 약물 높이 (0~1)

예: [0.5, 0.5, 0.2, 0.3]
  → x1 = 0.5 - 0.2/2 = 0.4  (픽셀: 0.4 * 이미지_폭)
  → y1 = 0.5 - 0.3/2 = 0.35 (픽셀: 0.35 * 이미지_높이)
  → x2 = 0.5 + 0.2/2 = 0.6
  → y2 = 0.5 + 0.3/2 = 0.65
```

---

### 학습 단계

#### Step 1: Config 파일 준비

```yaml
# experiments/exp_20260420_baseline_yolo26n/config.yaml

model:
  name: yolo26n              # 사용 모델
  pretrained: true          # ImageNet 사전학습 가중치 사용

data:
  yaml: data/processed/dataset.yaml
  imgsz: 640                # 입력 이미지 크기 (640×640)
  workers: 4

train:
  epochs: 100               # 100번 반복 학습
  batch: 16                 # 한 번에 16장씩 처리
  lr0: 0.001                # 초기 학습률
  optimizer: auto           # 자동 선택 (SGD/Adam)
  patience: 20              # Early stopping (20 epoch 개선 없으면 중단)

augment:
  hsv_h: 0.015              # 색상 무작위 변환
  hsv_s: 0.7                # 채도 무작위 변환
  hsv_v: 0.4                # 밝기 무작위 변환
  fliplr: 0.5               # 50% 확률로 좌우 반전
  flipud: 0.0               # 상하 반전 없음
  mosaic: 1.0               # 모자이크 증강 (4개 이미지 합치기)
  mixup: 0.1                # 10% 확률로 이미지 믹스업
```

#### Step 2: 모델 학습

**코드:**
```python
from ultralytics import YOLO
import yaml

# Config 로드
with open("experiments/exp_20260420_baseline_yolo26n/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 모델 초기화 (ImageNet 사전학습 가중치 로드)
model = YOLO(f"{cfg['model']['name']}.pt")

# 학습 실행
results = model.train(
    data=cfg['data']['yaml'],
    epochs=cfg['train']['epochs'],
    imgsz=cfg['data']['imgsz'],
    batch=cfg['train']['batch'],
    device=0,  # GPU 0번 사용
    project="experiments",
    name="exp_20260420_baseline_yolo26n",
)
```

**내부 동작:**
```
1. 데이터 로드
   data/processed/dataset.yaml 읽기
   ↓
   학습 이미지 캐싱 (빠른 로드)

2. 에포크 반복 (100번)
   각 epoch마다:
     a) Batch 단위로 이미지 로드
     b) Data Augmentation 적용
     c) 모델 추론 → bbox 예측
     d) Loss 계산 (|예측 bbox - 정답 bbox|)
     e) 역전파 (gradients 계산)
     f) 가중치 업데이트

3. 검증 (매 epoch마다)
   val 데이터로 mAP 계산
   ↓
   Best mAP 갱신 시 best.pt 저장

4. 학습 완료
   experiments/exp_20260420_baseline_yolo26n/
   ├── weights/
   │   ├── best.pt    ← 최고 성능 모델
   │   └── last.pt    ← 마지막 epoch 모델
   ├── results.csv    ← Loss/mAP 곡선 데이터
   └── results/
       ├── confusion_matrix.png
       └── ...
```

---

### 검증 단계

**목표**: 모델 성능 평가

**코드:**
```bash
python scripts/pipeline/run_train.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --validate
```

**메트릭 해석:**
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        497        547      0.996      0.998      0.994      0.934

P (Precision) = 0.996
  → 탐지한 1000개 중 996개가 정답
  → "거짓 양성(오탐) 거의 없음" ✓

R (Recall) = 0.998
  → 정답 1000개 중 998개를 탐지
  → "놓친 약물 거의 없음" ✓

mAP50 = 0.994
  → IoU(겹침 비율) > 0.5일 때 평균 정밀도
  → IoU 해석:
      • 완벽 일치: 1.0
      • 반정도 겹침: 0.5
      • 약간만 겹침: 0.3
      • 안 겹침: 0.0

mAP50-95 = 0.934
  → IoU 0.5부터 0.95까지 모두 포함
  → 더 엄격한 평가 (위치가 정확해야 함)
  → 보통 mAP50보다 낮음
```

**목표값:**
- mAP50 > 0.95 (탐지 정확도)
- 수렴 확인: loss 곡선이 안정화

---

### 추론 단계 (Inference)

**목표**: 검증 이미지에 대해 Stage 1 결과 생성

**코드:**
```bash
python scripts/pipeline/run_predict.py \
    --config  experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --source  data/processed/images/val/ \
    --output  experiments/exp_20260420_baseline_yolo26n/val_predictions.json
```

**추론 과정:**
```
1. 모델 로드
   best.pt 읽기

2. 각 이미지마다:
   a) 이미지 로드 (jpg 읽기)
   b) 전처리 (640×640 리사이즈, 정규화)
   c) 모델 추론
      이미지 → [N, 6] 배열
      각 행: [x1, y1, x2, y2, confidence, class]
   d) NMS (Non-Maximum Suppression)
      · 겹치는 탐지 제거
      · 신뢰도 임계값 적용 (기본 0.1)
   e) 결과 정리

3. JSON 저장
```

**출력 형식:**
```json
[
  {
    "image_id": "IMG_001",
    "detections": [
      {
        "bbox": [100, 50, 180, 120],  // [x1, y1, x2, y2] 픽셀 좌표
        "score": 0.95                  // 탐지 신뢰도 (0~1)
      },
      {
        "bbox": [250, 200, 320, 280],
        "score": 0.87
      }
    ]
  },
  {
    "image_id": "IMG_002",
    "detections": [
      {
        "bbox": [150, 100, 220, 180],
        "score": 0.92
      }
    ]
  }
]
```

---

### Crop 추출 (Stage 1 → Stage 2 준비)

**목표**: Stage 2에 입력할 crop 이미지 생성

**코드:**
```bash
python scripts/pipeline/crop.py \
    --predictions experiments/exp_20260420_baseline_yolo26n/results/predictions.json \
    --source      data/processed/images/val/ \
    --output      experiments/exp_20260420_baseline_yolo26n/stage1_crops/ \
    --padding     0.05
```

**Crop 추출 과정:**
```
1. predictions.json 읽기
   
2. 각 이미지마다:
   a) 원본 이미지 로드
   b) 각 bbox마다:
      • bbox에서 패딩(5%) 추가
      • 해당 영역 crop
      • stage1_crops/{image_id}_{idx}.jpg 저장
   c) manifest에 메타데이터 기록

3. crops_manifest.json 생성
```

**패딩의 역할:**
```
패딩 없음:
  ┌─ bbox ─┐
  │ 알약   │
  └────────┘

패딩 5%:
  ┌── crop (bbox + 5%) ──┐
  │  배경 │ 알약 │ 배경   │
  └───────────────────────┘
  
효과: 알약 주변 컨텍스트 포함
      → Stage 2 분류 성능 향상
```

---

## 📊 전체 파이프라인 비교표

### Stage 1 vs Stage 2

| 항목 | Stage 1 (탐지) | Stage 2 (분류) |
|------|----------------|----------------|
| **입력** | 원본 이미지 | Crop 이미지 |
| **질문** | "어디에 약물이 있는가?" | "이 약물이 무엇인가?" |
| **출력** | bbox + 신뢰도 | 약품명 + 신뢰도 |
| **모델** | YOLO26n | ResNet50 |
| **학습 데이터** | 이미지 + YOLO label txt | Crop + 폴더명 (자동) |
| **메트릭** | mAP50, Precision, Recall | Top-1 Accuracy, Top-5 Accuracy |
| **목표값** | mAP50 > 0.95 | Top-1 > 0.80 |

### GT Crop vs Inference Crop

| 항목 | GT Crop | Inference Crop |
|------|---------|-----------------|
| **생성 기반** | 실제 정답 label | Stage 1 모델 예측 |
| **정확도** | 100% | Stage 1 오류 포함 |
| **위치 정보** | GT bbox (정확함) | 추론 bbox (근사값) |
| **약품명** | 있음 (폴더명) | 없음 ← Stage 2가 할 일 |
| **신뢰도** | 없음 | 있음 (detection score) |
| **용도** | Stage 2 학습용 | Stage 2 평가용 |
| **생성 함수** | `crop_from_gt()` | `crop_from_predictions()` |
| **저장 위치** | `data/processed/crops/` | `experiments/{EXP_NAME}/stage1_crops/` |
| **이미지 수** | ~5000개 (train+val) | 추론 결과에 따라 (보통 ~550개) |
| **재생성** | 최초 1회만 | 매번 (Stage 1 결과 반영) |

---

## 🔧 각 스크립트의 역할

### 1. `crop.py` - Crop 추출 엔진

**3가지 모드:**

#### Mode 1: `crop_from_gt()` - GT Crop 생성
```bash
python scripts/pipeline/crop.py \
    --labels  data/processed/labels \       # GT label txt 위치
    --images  data/processed/images \       # 원본 이미지 위치
    --output  data/processed/crops \        # 출력 위치
    --splits  train val                     # 처리할 split
```

**입력:**
```
data/processed/labels/train/
├── 아스피린_123.txt
│   └── "0 0.5 0.5 0.2 0.3"  (class cx cy w h)
└── 감기약_456.txt
    └── "0 0.3 0.7 0.15 0.25"
```

**출력:**
```
data/processed/crops/
├── train/
│   ├── 아스피린_123_0000.jpg  ← 첫 번째 bbox crop
│   ├── 감기약_456_0000.jpg
│   └── crops_manifest.json
│       [
│         {
│           "image_id": "아스피린_123",
│           "crop_id": "아스피린_123_0000",
│           "crop_path": "data/processed/crops/train/...",
│           "bbox": [100, 100, 200, 200],
│           "class_name": "아스피린",
│           "split": "train"
│         }
│       ]
└── crops_manifest.json  (통합 manifest)
```

---

#### Mode 2: `crop_from_predictions()` - Inference Crop 생성
```bash
python scripts/pipeline/crop.py \
    --predictions experiments/{EXP}/results/predictions.json \  # Stage 1 결과
    --source      data/processed/images/val/ \                  # 원본 이미지
    --output      experiments/{EXP}/stage1_crops/               # 출력
    --padding     0.05                                          # 여백
```

**입력:**
```
experiments/{EXP}/val_predictions.json
{
  "image_id": "IMG_001",
  "detections": [
    {
      "bbox": [100, 50, 180, 120],  # [x1, y1, x2, y2]
      "score": 0.95                  # 탐지 신뢰도
    },
    {
      "bbox": [250, 200, 320, 280],
      "score": 0.87
    }
  ]
}
```

**출력:**
```
experiments/{EXP}/stage1_crops/
├── IMG_001_0.jpg  ← 첫 번째 탐지 crop
├── IMG_001_1.jpg  ← 두 번째 탐지 crop
└── crops_manifest.json
    [
      {
        "image_id": "IMG_001",
        "crop_id": "IMG_001_0",
        "crop_path": "experiments/{EXP}/stage1_crops/IMG_001_0.jpg",
        "bbox": [100, 50, 180, 120],
        "score": 0.95        # ← 탐지 신뢰도
      }
    ]
```

---

#### Mode 3: `convert_from_imagefolder()` - ImageFolder → Manifest
```bash
python scripts/pipeline/crop.py \
    --imagefolder data/processed/crops \  # 이미 crop된 이미지들
    --splits      train val               # manifest만 생성
```

**용도:** 외부에서 제공받은 crop 이미지에 manifest만 추가

---

### 2. `run_train.py` - Stage 1 학습 래퍼

```bash
python scripts/pipeline/run_train.py \
    --config experiments/{EXP}/config.yaml \
    --data   data/processed/dataset.yaml
```

**내부 동작:**
```
config.yaml 읽기
  ↓
Ultralytics YOLO 초기화
  ↓
data/processed/dataset.yaml 에서 데이터 로드
  ↓
학습 (epochs만큼 반복)
  ↓
best.pt 저장
```

---

### 3. `run_predict.py` - Stage 1 추론

```bash
python scripts/pipeline/run_predict.py \
    --config  experiments/{EXP}/config.yaml \
    --source  data/processed/images/val/ \
    --output  experiments/{EXP}/val_predictions.json
```

**내부 동작:**
```
best.pt 로드
  ↓
각 이미지마다:
  • 모델 추론
  • bbox + score 추출
  ↓
val_predictions.json 저장
```

---

### 4. `stage2_train.py` - Stage 2 학습

```bash
python scripts/pipeline/stage2_train.py \
    --config experiments/stage2_classifier/config.yaml
```

**입력:**
- `data/processed/crops/train/` (GT crop)
- 폴더명 = class label (자동 인식)

**출력:**
- `experiments/stage2_classifier/.../weights/best.pt`

---

### 5. `stage2_predict.py` - Stage 2 추론

```bash
python scripts/pipeline/stage2_predict.py \
    --config  experiments/stage2_classifier/config.yaml \
    --weights experiments/stage2_classifier/.../best.pt \
    --source  experiments/{EXP}/stage1_crops/ \
    --output  experiments/{EXP}/stage2_predictions.json
```

**입력:** Inference crop (정답 없음)

**출력:** 
```json
{
  "crop_id": "IMG_001_0",
  "class_name": "아스피린",
  "score": 0.92
}
```
---

### 6. `evaluate_pipeline.py` - End-to-end mAP 평가

```bash
python scripts/pipeline/evaluate_pipeline.py \
    --gt-labels data/processed/labels/val \
    --gt-images data/processed/images/val \
    --s1-crops  experiments/{EXP}/stage1_crops/crops_manifest.json \
    --s2-preds  experiments/{EXP}/stage2_predictions.json

# Kaggle canonical class 기준 평가 + Stage 2 alias 정규화
python scripts/pipeline/evaluate_pipeline.py \
    --gt-labels data/processed/labels/val \
    --gt-images data/processed/images/val \
    --s1-crops  experiments/{EXP}/stage1_crops/crops_manifest.json \
    --s2-preds  experiments/{EXP}/stage2_predictions.json \
    --kaggle-classes data/processed/kaggle_class_map.json \
    --unknown-class-map data/processed/kaggle_unknown_class_map.json

# 클래스별 AP 상위 20개 추가 출력
python scripts/pipeline/evaluate_pipeline.py ... --per-class
```

**출력:**
```
GT     : 497장  547개 객체
Pred   : 547개 / 44클래스 평가
mAP@[0.50:0.95] : 0.xxxx  (COCO)
mAP@[0.75:0.95] : 0.xxxx  (Kaggle 평가지표)
mAP@0.50        : 0.xxxx
mAP@0.75        : 0.xxxx
```

**주의:**
- GT bbox는 YOLO label에서 읽고, GT class는 GT crop manifest / Roboflow source key / `raw_K-*` 파일명의 category_id로 복원한다.
- YOLO label은 Stage 1용 `pill=0` 단일 클래스이므로 E2E class 평가에는 별도 class 복원이 필요하다.
- `--unknown-class-map`은 Stage 2가 출력한 alias class_name을 Kaggle canonical class_name으로 치환해 평가한다.
- score = `det_score × cls_score` (crops_manifest × stage2_predictions)

---

### 7. Stage 2 자동 튜닝

Stage 2 튜닝 runner는 **분류기 자체의 validation metric**을 최적화한다. 기본 objective는 `top1_acc`이며, Stage 1은 baseline freeze 상태로 고정한다. 상위 trial만 별도로 E2E 평가를 수행해 Kaggle mAP 개선 여부를 확인한다.

> 현재 runner는 `top1_acc` / `top5_acc` objective를 지원한다. E2E mAP objective는 trial마다 Stage 2 prediction과 `evaluate_pipeline.py`까지 실행해야 하므로 후속 확장 대상으로 둔다.

#### 공통 옵션

| 인자 | 필수 | 설명 |
|------|------|------|
| `--base-config` | ✅ | 기준 Stage 2 config YAML. baseline freeze 기준은 `experiments/exp_20260420_baseline_yolo26n/s2_config.yaml` |
| `--search-space` | 선택 | 탐색 공간 YAML. 생략 시 runner 내 기본 탐색 공간 사용 |
| `--output` | 선택 | trial 산출물 저장 루트 |
| `--data` | 선택 | Stage 2 crop root. 지정 시 `<data>/train`, `<data>/val`을 사용 |
| `--epochs` | 선택 | 탐색용 epoch override. 15~30 epoch 권장 |
| `--device` | 선택 | `0`, `1`, `cpu` 등 device override |
| `--metric` | 선택 | objective metric. `top1_acc` 또는 `top5_acc` |

주의:
- `--base-config`의 `model.num_classes`는 실제 crop train class 수와 같아야 한다.
- baseline freeze 기준 Stage 2 class set은 305개다.
- 탐색 runner는 baseline config 파일을 직접 수정하지 않고 trial별 `config.yaml`을 새로 저장한다.

#### Grid Search

Grid Search는 정해진 후보 조합을 모두 실행한다. 작은 후보군을 명시적으로 비교할 때 사용한다.

```bash
python scripts/pipeline/stage2_grid_search.py \
    --base-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --output experiments/stage2_grid_search \
    --epochs 30 \
    --max-trials 12
```

추가 옵션:

| 인자 | 설명 |
|------|------|
| `--max-trials` | 앞에서 N개 조합만 실행. smoke test나 비용 제한용 |

검색 공간을 파일로 지정할 수도 있다. YAML은 Stage 2 config 구조와 같은 nested 형식을 권장한다.

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

#### search_space에 넣을 수 있는 주요 파라미터

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

```bash
python scripts/pipeline/stage2_grid_search.py \
    --base-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --search-space experiments/stage2_grid_search/grid_space.yaml \
    --output experiments/stage2_grid_search \
    --epochs 30
```

#### Optuna

Optuna는 search space 안에서 후보를 샘플링한다. 연속형 하이퍼파라미터를 탐색할 때 사용한다.

```bash
python scripts/pipeline/stage2_optuna.py \
    --base-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
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

Optuna 검색 공간은 다음 형식을 사용한다.

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
| `int` | `low`, `high`, 선택 `step`, `log` | warmup epoch 등 정수값 |

Optuna도 dotted key를 옵션으로 사용할 수 있다.

```yaml
train.lr0:
  type: float
  low: 0.00001
  high: 0.0003
  log: true
```

위 설정은 nested YAML의 `train: lr0:`과 동일하게 처리된다.

#### 산출물

각 runner는 trial별 산출물을 분리해 저장한다.

```
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

#### 운영 원칙

- 튜닝 objective는 기본적으로 `top1_acc`를 사용한다.
- 탐색 단계에서는 `--epochs 15~30` 정도로 짧게 돌린다.
- best trial 상위 3~5개만 100 epoch로 재학습하고 E2E mAP를 확인한다.
- Stage 2 튜닝 실험은 baseline config를 직접 덮어쓰지 않고 별도 `experiments/stage2_*` 경로에 저장한다.
- HPO 결과가 좋아도 최종 채택 전에는 `evaluate_pipeline.py --kaggle-classes --unknown-class-map` 기준으로 E2E 검증을 수행한다.

상위 trial E2E 검증 예시:

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

---

## 🔄 전체 파이프라인 통합

```
1️⃣ GT Crop 생성
   crop_from_gt()
   → data/processed/crops/{train,val}/

2️⃣ Stage 2 분류기 학습
   stage2_train.py
   → experiments/stage2_classifier/.../best.pt

3️⃣ Stage 1 추론
   predict.py
   → experiments/{EXP}/results/predictions.json

4️⃣ Inference Crop 생성
   crop.py (inference 모드)
   → experiments/{EXP}/stage1_crops/ + crops_manifest.json

5️⃣ Stage 2 추론
   stage2_predict.py
   → experiments/{EXP}/stage2_predictions.json

6️⃣ [선택] End-to-end mAP 평가
   evaluate_pipeline.py --kaggle-classes ... --unknown-class-map ...
   → mAP@[0.50:0.95], mAP@[0.75:0.95] 출력

7️⃣ Kaggle 제출 CSV 생성
   make_submission.py --class-map kaggle_class_map.json --unknown-class-map ...
   → submissions/submission.csv
      score = det_score × cls_score
```

---

## 💡 핵심 인사이트

### Stage 2 모델의 관점

```python
# 학습 중 (GT Crop 사용)
crop_image = cv2.imread("data/processed/crops/train/아스피린_001.jpg")
label = "아스피린"  # 폴더명에서 가져옴

output = model(crop_image)  # 예측: [0.2, 0.8, 0.0, ...]
loss = cross_entropy(output, label)  # |예측 - 정답| 계산
# 손실이 줄어들도록 가중치 업데이트

# 추론 중 (Inference Crop 사용)
crop_image = cv2.imread("experiments/.../stage1_crops/IMG_001_0.jpg")
# 정답이 없음!

output = model(crop_image)  # 예측: [0.1, 0.75, 0.15]
pred_class = argmax(output)  # 가장 높은 값 선택 → "감기약"
confidence = max(output)  # 0.75 (신뢰도)

# 결과: "이 약물은 감기약일 거 같아 (75% 확신)"
```

---

## 🎯 주의사항

### ⚠️ GT Crop 생성은 **최초 1회만**

```python
# GT crop이 이미 존재하면 재생성하지 않음
if (GT_CROP_DIR / "train" / "crops_manifest.json").exists():
    print("✓ GT crop 이미 존재")
else:
    crop_from_gt(...)  # 최초 1회만 실행
```

**이유:**
- 시간이 오래 걸림 (~5분)
- 정답은 변하지 않음
- Stage 2 학습 시작 전 반드시 생성 필요

### ⚠️ Inference Crop은 **매번 재생성**

```python
# Stage 1 추론 결과가 달라질 수 있으므로
crop_from_predictions()  # 매번 새로 생성

# 왜?
# • Stage 1 모델 업데이트 시 추론 결과 변함
# • 다른 config로 추론하면 bbox 위치 달라짐
# • 최신 결과로 Stage 2 평가해야 함
```

---
