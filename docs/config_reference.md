# Config 필드 참조

`config.yaml` 작성 시 참조하는 목표 스펙 문서입니다.  
Stage 1 Detection과 Stage 2 Classification은 공통 섹션 이름을 최대한 맞추되, 데이터 입력 방식과 증강 파이프라인은 stage별로 일부 다릅니다.

## 공통 구조 요약

- 공통 필수 섹션: `model`, `data`, `train`, `val`, `output`
- Stage 1 Detection 전용 섹션: `augment`
- Stage 1 / Stage 2 공통 선택 섹션: `albumentations` (transform별 optional block)
- 공통 메타데이터: `seed`, `stage`, `task`, `classes`, `nc`

---

## 최상위 필드

| 필드 | 필수 | 기본 동작 | 설명 |
|------|------|-----------|------|
| `seed` | 권장 | 명시 권장 | 전역 랜덤 시드 (`random`, `numpy`, `torch` 동시 고정). 재현성을 위해 모든 실험에서 명시 권장 |
| `stage` | | 메타데이터 | 파이프라인 스테이지 번호 (`1` 또는 `2`) |
| `task` | | 메타데이터 | `detection` / `classification` |
| `classes` | | 메타데이터 | 클래스 이름 목록 |
| `nc` | | 메타데이터 | 클래스 수 |

---

## Stage 1 — Detection config

Stage 1은 **Ultralytics 기본 증강(`augment`)** 과 **Albumentations 확장 증강(`albumentations`)** 을 함께 사용할 수 있습니다.  
`augment`는 YOLO 학습 루프에 직접 들어가는 기본 증강이고, `albumentations`는 필요한 transform만 골라 붙이는 확장 증강입니다.

### 전체 예시

```yaml
seed: 42
stage: 1
task: detection
classes: [pill]
nc: 1

model:
  name: yolo26n
  pretrained: true

data:
  yaml: data/processed/dataset.yaml
  imgsz: 640
  workers: 4
  version: "v1.0"
  description: "baseline detection dataset"

train:
  epochs: 100
  batch: 16
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  patience: 20
  device: ""

augment:
  mosaic: 1.0
  mixup: 0.1
  copy_paste: 0.0
  flipud: 0.0
  fliplr: 0.5
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4

albumentations:
  bbox:
    format: yolo
    label_fields: [class_labels]
    min_visibility: 0.10
    clip: true
  horizontal_flip:
    p: 0.5
  vertical_flip:
    p: 0.2
  random_rotate90:
    p: 0.2
  shift_scale_rotate:
    p: 0.6
    shift_limit: 0.03
    scale_limit: 0.08
    rotate_limit: 8
  brightness_contrast:
    p: 0.5
    brightness_limit: 0.15
    contrast_limit: 0.15
  jpeg_compression:
    p: 0.3
    quality_lower: 85
    quality_upper: 100
  gaussian_blur:
    p: 0.2
    blur_limit: [3, 5]
  gauss_noise:
    p: 0.15
    std_range: [0.01, 0.03]

val:
  conf: 0.25
  iou: 0.45
  max_det: 300
  tta: false

output:
  project: experiments
  name: exp_20260420_baseline_yolo26n
  save_period: 10
```

### model

```yaml
model:
  name: yolo26n
  pretrained: true
```

| 필드 | 필수 | 허용 값 | 설명 |
|------|------|---------|------|
| `name` | ✅ | `yolo26n` `yolo26s` `yolo26m` `yolov8s` `yolo11s` | Ultralytics 모델 식별자 |
| `pretrained` | ✅ | `true` / `false` | COCO 사전학습 가중치 사용 여부 |

### data

```yaml
data:
  yaml: data/processed/dataset.yaml
  imgsz: 640
  workers: 4
  version: "v1.0"        # 선택
  description: "..."     # 선택
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `yaml` | ✅ | YOLO 포맷 데이터셋 정의 파일 경로 (프로젝트 루트 기준 상대 경로) |
| `imgsz` | ✅ | 입력 해상도. 기본 베이스라인은 `640` |
| `workers` | ✅ | DataLoader 워커 수 |
| `version` | | 데이터셋 버전 태그. 재현성 추적용 메타데이터 |
| `description` | | 데이터셋 설명 |

### train

```yaml
train:
  epochs: 100
  batch: 16
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  patience: 20
  device: ""
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `epochs` | ✅ | 최대 학습 에포크 수 |
| `batch` | ✅ | 배치 크기. `-1`이면 AutoBatch |
| `optimizer` | ✅ | `AdamW` `Adam` `SGD` `NAdam` `RAdam` `RMSProp` `auto` |
| `lr0` | ✅ | 초기 학습률 |
| `lrf` | ✅ | 최종 학습률 비율. `lr_final = lr0 × lrf` |
| `momentum` | ✅ | SGD momentum / Adam beta1 |
| `weight_decay` | ✅ | L2 정규화 계수 |
| `warmup_epochs` | ✅ | lr을 0에서 `lr0`까지 선형 증가시키는 워밍업 에포크 수 |
| `patience` | ✅ | mAP 개선이 없을 때 조기 종료를 판단하는 에포크 수 |
| `device` | ✅ | `""` = auto, `"0"` = GPU 0, `"0,1"` = 멀티GPU, `"cpu"` = CPU |

### augment

`augment`는 Ultralytics가 직접 해석하는 기본 증강 섹션입니다.  
Stage 1에서는 이 섹션을 유지하고, 보다 복잡한 촬영 환경 시뮬레이션은 `albumentations`로 확장하는 것을 권장합니다.

```yaml
augment:
  mosaic: 1.0
  mixup: 0.1
  copy_paste: 0.0
  flipud: 0.0
  fliplr: 0.5
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `mosaic` | prob | 4장 이미지를 합쳐 1장으로 만드는 Mosaic 적용 확률 |
| `mixup` | prob | 두 이미지를 alpha-blend하는 MixUp 적용 확률 |
| `copy_paste` | prob | 객체를 다른 이미지에 붙여넣는 Copy-Paste 적용 확률 |
| `flipud` | prob | 상하 반전 확률 |
| `fliplr` | prob | 좌우 반전 확률 |
| `hsv_h` | intensity | Hue 변화 폭 |
| `hsv_s` | intensity | Saturation 변화 폭 |
| `hsv_v` | intensity | Value 변화 폭 |

### albumentations

`albumentations`는 Stage 1 / Stage 2 공통 확장 증강 섹션입니다.  
인터페이스는 최대한 단순하게 유지하며, **transform 이름을 키로 두고 필요한 block만 추가**하는 방식으로 사용합니다.

#### 해석 규칙

- `albumentations` 섹션 자체가 없으면 전체 미적용으로 간주한다
- transform block이 없으면 해당 transform은 비활성화로 간주한다
- transform block이 있으면 해당 transform은 활성화로 간주한다
- 생략한 세부 파라미터는 구현 기본값을 사용한다
- 범위형 파라미터는 `[min, max]` 형식으로 작성한다
- 알 수 없는 transform 이름은 무시하지 말고 에러로 처리하는 것을 권장한다

#### 공통 제어 스키마

```yaml
albumentations:
  bbox:                  # detection에서만 사용
    format: yolo
    label_fields: [class_labels]
    min_visibility: 0.10
    clip: true
  <transform_name>:
    p: 0.5
    ...
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `bbox` | Detection만 권장 | bbox-aware transform을 위한 설정. Classification에서는 생략 |
| `bbox.format` | | bbox 포맷. Detection 기준 `yolo` |
| `bbox.label_fields` | | bbox와 함께 동기화할 라벨 필드명 목록 |
| `bbox.min_visibility` | | 변환 후 유지할 최소 bbox visibility |
| `bbox.clip` | | 이미지 밖으로 나간 bbox를 경계 내로 clip |
| `albumentations.<transform_name>` | | 해당 transform 설정 block |
| `albumentations.<transform_name>.p` | | transform 적용 확률 |
| `albumentations.<transform_name>.<param>` | | transform 세부 파라미터 |

#### Stage 1 지원 transform

| transform | 주요 파라미터 | 설명 |
|-----------|---------------|------|
| `horizontal_flip` | `p` | 좌우 반전 |
| `vertical_flip` | `p` | 상하 반전 |
| `random_rotate90` | `p` | 90/180/270도 회전 |
| `shift_scale_rotate` | `p`, `shift_limit`, `scale_limit`, `rotate_limit` | 위치 변화, 확대/축소, 각도 변화 |
| `brightness_contrast` | `p`, `brightness_limit`, `contrast_limit` | 조명 변화 대응 |
| `random_gamma` | `p`, `gamma_limit` | 밝기 변화와 감마 왜곡 대응 |
| `jpeg_compression` | `p`, `quality_lower`, `quality_upper` | 압축 아티팩트 시뮬레이션 |
| `downscale` | `p`, `scale_min`, `scale_max` | 저해상도 촬영 / 메신저 전송본 품질 저하 시뮬레이션 |
| `gaussian_blur` | `p`, `blur_limit` | 초점 불량 / 흔들림 대응 |
| `motion_blur` | `p`, `blur_limit` | 손떨림 / 이동 중 촬영 blur 대응 |
| `gauss_noise` | `p`, `std_range` | 센서 노이즈 / 저화질 촬영 대응 |
| `perspective` | `p`, `scale` | 비스듬한 촬영 각도 시뮬레이션. 저강도만 권장 |

#### Stage 1 운영 규칙

- 동일한 의미의 증강은 `augment`와 `albumentations`에 중복 정의하지 않는다
- `mosaic`, `mixup`, `copy_paste`는 YOLO 기본 `augment`에서 관리하는 것을 권장한다
- flip, 색상/밝기, blur, noise, compression, perspective 계열은 `albumentations`에서 관리하는 것을 권장한다
- `albumentations`가 flip, photometric, geometry 계열을 담당하는 경우 대응되는 `augment` 키는 생략하거나 `0.0`으로 설정한다
- 예시: `horizontal_flip` / `vertical_flip`을 사용하면 `augment.fliplr`, `augment.flipud`는 생략하거나 `0.0`으로 둔다
- 예시: `brightness_contrast`, `random_gamma` 등 photometric 변환을 적극 사용하면 `augment.hsv_*`는 생략하거나 최소값으로 둔다
- `shift_scale_rotate`를 강하게 사용할 경우, geometry 계열 변환은 한 번에 하나씩만 올려 과증강을 방지한다

#### Stage 1 추가 실험 후보

| transform | 권장 시작값 | 추천 상황 | 주의점 |
|-----------|-------------|-----------|--------|
| `downscale` | `p=0.15`, `scale_min=0.7`, `scale_max=0.95` | 테스트 이미지 해상도 저하나 전송본 품질 열화가 걱정될 때 | `jpeg_compression`과 동시에 너무 강하게 쓰지 않기 |
| `motion_blur` | `p=0.1`, `blur_limit=[3, 7]` | 손떨림, 빠른 촬영, 초점 불량을 반영하고 싶을 때 | 소형 알약 경계가 무너질 수 있으므로 저확률 권장 |
| `random_gamma` | `p=0.2`, `gamma_limit=[80, 120]` | 실내외 조도 차이나 노출 변화가 큰 데이터일 때 | `brightness_contrast`와 중복 강화를 피하기 |
| `perspective` | `p=0.1`, `scale=[0.02, 0.05]` | 비스듬한 각도의 촬영을 일부 반영하고 싶을 때 | 강하게 주면 bbox 분포가 비현실적으로 왜곡될 수 있음 |

#### 설계 원칙

- Stage 1에서는 `augment`와 `albumentations`를 **중복이 아니라 역할 분담**으로 본다
- `augment`는 YOLO 기본 증강의 빠른 탐색용
- `albumentations`는 실험자가 촬영 품질, 회전, 압축, 노이즈를 transform 단위로 세밀 조정하는 용도
- transform이 누락되면 비활성화로 간주한다

### val

```yaml
val:
  conf: 0.25
  iou: 0.45
  max_det: 300
  tta: false
```

| 필드 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `conf` | ✅ | `0.25` | confidence threshold. Kaggle 제출 전 `0.001`로 낮춰 재검토 권장 |
| `iou` | ✅ | `0.45` | NMS IoU threshold |
| `max_det` | ✅ | `300` | 이미지 1장당 최대 검출 수 |
| `tta` | | `false` | `true`이면 predict 시 TTA 사용. CLI 플래그가 있으면 CLI 우선 |

> `val.conf`를 높이면 precision은 오르고 recall은 내려갑니다. Kaggle 평가는 다양한 conf threshold를 평균하므로, 검증 단계에서는 `conf: 0.001` 기준도 함께 확인하는 것을 권장합니다.

### output

```yaml
output:
  project: experiments
  name: exp_20260420_baseline_yolo26n
  save_period: 10
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `project` | ✅ | 결과 저장 루트 디렉터리 |
| `name` | ✅ | 실험 디렉터리 이름. `project/name/` 아래 가중치와 로그 저장 |
| `save_period` | ✅ | 중간 체크포인트 저장 주기. `-1`이면 비활성화 |

---

## Stage 2 — Classification config

Stage 2는 `augment` 대신 `albumentations`만 사용합니다.  
Stage 1과 같은 flat optional transform block 구조를 쓰되, bbox-aware 설정 없이 이미지 변환 위주로 구성합니다.

### 전체 예시

```yaml
seed: 42
stage: 2
task: classification

model:
  name: efficientnetv2_s   # resnet50 / efficientnet_b2 / efficientnetv2_s
  pretrained: true
  num_classes: 371

data:
  train: data/processed/crops/train/
  val: data/processed/crops/val/
  imgsz: 224
  workers: 8

train:
  epochs: 30
  batch: 32
  optimizer: AdamW
  lr0: 0.001
  lrf: 0.01
  weight_decay: 0.01
  warmup_epochs: 3
  scheduler: cosine
  criterion: cross_entropy
  label_smoothing: 0.1
  device: ""

albumentations:
  brightness_contrast:
    p: 0.5
    brightness_limit: 0.15
    contrast_limit: 0.15
  jpeg_compression:
    p: 0.3
    quality_lower: 85
    quality_upper: 100
  gaussian_blur:
    p: 0.2
    blur_limit: [3, 5]
  bbox_jitter:
    p: 0.5
    shift_limit: 0.03
    scale_limit: 0.05
    rotate_limit: 5

val:
  top_k: [1, 5]

output:
  project: experiments/stage2_classifier
  name: exp_YYYYMMDD_efficientnetv2s_baseline
  save_period: -1
```

### Stage 2에서 달라지는 필드

#### model

| 필드 | 필수 | 설명 |
|------|------|------|
| `name` | ✅ | 사용할 분류기 모델 |
| `pretrained` | ✅ | ImageNet 사전학습 가중치 사용 여부 |
| `num_classes` | ✅ | 분류 클래스 수 |

#### data

| 필드 | 필수 | 설명 |
|------|------|------|
| `train` | ✅ | 학습용 크롭 이미지 루트. `<클래스명>/이미지.jpg` 구조 |
| `val` | ✅ | 검증용 크롭 이미지 루트 |
| `imgsz` | ✅ | 모델 입력 해상도 |
| `workers` | ✅ | 데이터 로딩 워커 수 |

#### train

| 필드 | 필수 | 설명 |
|------|------|------|
| `epochs` | ✅ | 최대 학습 에포크 수 |
| `batch` | ✅ | 미니배치 크기 |
| `optimizer` | ✅ | 옵티마이저 종류 (`AdamW`, `Adam`, `SGD`) |
| `lr0` | ✅ | 초기 학습률 |
| `lrf` | ✅ | 최종 학습률 비율 |
| `weight_decay` | ✅ | L2 정규화 강도 |
| `warmup_epochs` | ✅ | 워밍업 에포크 수 |
| `scheduler` | | LR 스케줄러 종류 (`cosine` 기본) |
| `criterion` | | Loss 함수 종류 (`cross_entropy` 기본, `bce`, `focal` 지원) |
| `focal_alpha` | | Focal Loss alpha (기본 0.25) |
| `focal_gamma` | | Focal Loss gamma (기본 2.0) |
| `label_smoothing` | ✅ | label smoothing 강도 |
| `device` | ✅ | 실행 장치 |

#### albumentations

Stage 2도 Stage 1과 동일한 flat optional transform block 구조를 사용합니다.  
차이는 `bbox` 섹션이 없고, bbox-aware transform 대신 분류용 crop robustness 중심 변환을 사용한다는 점입니다.

| transform | 주요 파라미터 | 설명 |
|-----------|---------------|------|
| `brightness_contrast` | `p`, `brightness_limit`, `contrast_limit` | 조명 변화 대응 |
| `random_gamma` | `p`, `gamma_limit` | 조도 및 노출 변화 대응 |
| `clahe` | `p`, `clip_limit`, `tile_grid_size` | 저대비 표면과 각인 대비 개선 |
| `jpeg_compression` | `p`, `quality_lower`, `quality_upper` | 압축 아티팩트 대응 |
| `downscale` | `p`, `scale_min`, `scale_max` | 저해상도 crop / 전송본 품질 저하 대응 |
| `gaussian_blur` | `p`, `blur_limit` | 초점 불량 대응 |
| `motion_blur` | `p`, `blur_limit` | 아주 약한 손떨림 blur 대응 |
| `bbox_jitter` | `p`, `shift_limit`, `scale_limit`, `rotate_limit` | Stage 1 bbox 오차 시뮬레이션 |

#### Stage 2 추가 실험 후보

| transform | 권장 시작값 | 추천 상황 | 주의점 |
|-----------|-------------|-----------|--------|
| `clahe` | `p=0.2`, `clip_limit=[1, 4]`, `tile_grid_size=[8, 8]` | 각인이나 표면 대비가 낮은 crop이 많을 때 | 과하면 색감과 질감이 비현실적으로 바뀔 수 있음 |
| `downscale` | `p=0.15`, `scale_min=0.8`, `scale_max=0.95` | crop 해상도 저하, 작은 알약 표면 디테일 손실에 대비할 때 | Stage 2는 텍스처 손실에 민감하므로 약하게 시작 |
| `motion_blur` | `p=0.1`, `blur_limit=[3, 5]` | 촬영 흔들림이 실제 서비스 환경에 많을 때 | 문자/각인이 무너지지 않게 매우 약하게 사용 |
| `random_gamma` | `p=0.2`, `gamma_limit=[90, 110]` | 조명이나 노출 차이가 심한 crop이 많을 때 | `brightness_contrast`와 동시에 크게 올리지 않기 |

#### val

| 필드 | 필수 | 설명 |
|------|------|------|
| `top_k` | ✅ | 출력할 Top-K 목록. 예: `[1, 5]` |

#### output

| 필드 | 필수 | 설명 |
|------|------|------|
| `project` | ✅ | 실험 결과 저장 루트 |
| `name` | ✅ | 실험 이름. `exp_YYYYMMDD_<모델>_<가설>` 형식 권장 |
| `save_period` | ✅ | 중간 체크포인트 저장 주기. `-1`이면 비활성화 |

---

## 추천 운영 방식

- 빠른 베이스라인: `augment`만 사용
- 촬영 환경 강건성 실험: `augment + albumentations`
- Stage 2 crop robustness 실험: `albumentations`만 사용
- 실험 로그에는 사용한 transform 이름과 각 transform의 세부 파라미터를 함께 기록
