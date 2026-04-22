# Config 필드 참조

`config.yaml` 작성 시 참조하는 문서입니다.  
`validate_config`가 검사하는 **필수 섹션**: `model`, `data`, `train`, `augment`, `val`, `output`

---

## 최상위 필드

| 필드 | 필수 | 기본 동작 | 설명 |
|------|------|-----------|------|
| `seed` | | 없으면 랜덤 | 전역 랜덤 시드 (`random`, `numpy`, `torch` 동시 고정) |
| `stage` | | 미사용 | 파이프라인 스테이지 번호 (1 또는 2). 실험 추적용 메타데이터이며 학습에는 영향 없음 |
| `task` | | 미사용 | `detection` / `classification`. 위와 동일하게 메타데이터 용도 |
| `classes` | | 미사용 | 클래스 이름 목록 (예: `[pill]`). 메타데이터 용도 |
| `nc` | | 미사용 | 클래스 수. 메타데이터 용도 (`stage`, `task`와 함께 사용) |

---

## model

```yaml
model:
  name: yolo26n
  pretrained: true
```

| 필드 | 필수 | 허용 값 | 설명 |
|------|------|---------|------|
| `name` | ✅ | `yolo26n` `yolo26s` `yolo26m` `yolov8s` `yolo11s` | Ultralytics 모델 식별자. 값이 Ultralytics에서 지원되지 않으면 로드 오류 |
| `pretrained` | ✅ | `true` / `false` | COCO 사전학습 가중치 사용 여부. `false`이면 random init |

---

## data

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
| `imgsz` | ✅ | 입력 해상도. 베이스라인은 `640`. 크게 할수록 소형 알약 탐지에 유리하나 VRAM 증가 |
| `workers` | ✅ | DataLoader 워커 수. 코어 수를 초과하면 오히려 느려짐 |
| `version` | | 데이터셋 버전 태그 (예: `v1.0`). 재현성 추적용 메타데이터 |
| `description` | | 데이터셋 설명. 메타데이터 용도 |

---

## train

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
| `epochs` | ✅ | 최대 학습 에포크 수. `patience`에 의한 조기 종료 가능 |
| `batch` | ✅ | 배치 크기. `-1`이면 Ultralytics AutoBatch |
| `optimizer` | ✅ | `AdamW` `Adam` `SGD` `NAdam` `RAdam` `RMSProp` `auto` |
| `lr0` | ✅ | 초기 학습률 |
| `lrf` | ✅ | 최종 학습률 비율. `lr_final = lr0 × lrf` (코사인 스케줄 끝 값) |
| `momentum` | ✅ | SGD momentum / Adam beta1 |
| `weight_decay` | ✅ | L2 정규화 계수 |
| `warmup_epochs` | ✅ | lr을 0 → lr0으로 선형 증가하는 워밍업 에포크 수 |
| `patience` | ✅ | 이 에포크 동안 mAP 개선 없으면 조기 종료. `0`이면 비활성화 |
| `device` | ✅ | `""` = auto (GPU 있으면 GPU 0 사용), `"0"` = GPU 0, `"0,1"` = 멀티GPU, `"cpu"` = CPU |

---

## augment

모든 값은 `0.0 ~ 1.0` 범위입니다.  
확률(`prob`) 타입은 해당 변환이 적용될 확률, 강도(`intensity`) 타입은 변환 세기입니다.

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
| `mosaic` | prob | 4장 이미지를 합쳐 1장으로 만드는 Mosaic 증강 적용 확률. 소형 객체 탐지에 효과적 |
| `mixup` | prob | 두 이미지를 alpha-blend하는 MixUp 적용 확률 |
| `copy_paste` | prob | 객체를 다른 이미지에 붙여넣는 Copy-Paste 적용 확률. 현재 비활성화 |
| `flipud` | prob | 상하 반전 확률. 알약은 방향 무관하나 라벨 의존 시 주의 |
| `fliplr` | prob | 좌우 반전 확률 |
| `hsv_h` | intensity | 색조(Hue) 변화 폭 (±값). 알약 색상 변동에 대한 강건성 |
| `hsv_s` | intensity | 채도(Saturation) 변화 폭 |
| `hsv_v` | intensity | 명도(Value) 변화 폭 |

---

## val

```yaml
val:
  conf: 0.25
  iou: 0.45
  max_det: 300
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `conf` | ✅ | confidence threshold. 이 값 미만 검출 결과는 무시. **Kaggle 제출 전 0.001로 낮춰 재검토 권장** |
| `iou` | ✅ | NMS IoU threshold. 겹치는 박스 제거 기준. Kaggle 평가 기준(IoU@0.5)과 별개 |
| `max_det` | ✅ | 이미지 1장당 최대 검출 수 |
| `tta` | | `false` | `true`이면 predict.py 실행 시 TTA(좌우 반전·스케일 앙상블) 자동 활성화. CLI `--tta`로 override 가능 |

> `val.conf`를 높이면 precision은 오르고 recall은 내려갑니다. Kaggle 평가는 mAP(다양한 conf threshold의 평균)이므로 검증 시 `conf: 0.001`로 두면 실제 mAP에 가까운 수치를 확인할 수 있습니다.

---

## output

```yaml
output:
  project: experiments
  name: exp_20260420_baseline_yolo26n
  save_period: 10
```

| 필드 | 필수 | 설명 |
|------|------|------|
| `project` | ✅ | 결과 저장 루트 디렉터리 |
| `name` | ✅ | 실험 디렉터리 이름. `project/name/` 아래 가중치·로그 저장 |
| `save_period` | ✅ | N 에포크마다 중간 체크포인트 저장 (`last.pt`와 별도). `0`이면 비활성화 |
