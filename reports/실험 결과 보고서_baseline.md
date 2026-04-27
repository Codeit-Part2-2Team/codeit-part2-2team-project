# 실험 결과 보고서_Baseline

## 목적

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 프로젝트 | 경구약제 알약 탐지 |
| 목표 | 이미지에서 알약 위치(bbox) + 57개 약품명(class) 동시 예측 |
| 팀 | PM 호정, ME 승준, EL 도혁, DE 소원, AL 찬우 |
| 기간 | 2026-04-20 ~ 05-11 (이 보고서는 04-27 기준) |
| 최종 아키텍처 | 2-Stage 파이프라인 (YOLOv26n 탐지 + 분류기) |

---

## 실험 흐름 (시계열)

### Phase 1 — 프로젝트 셋업 (04-13 ~ 04-19)

저장소 초기 구성과 팀 협업 인프라를 갖추는 단계였다. `main` 브랜치 직접 커밋 방지 훅을 설정하고, conda 환경 및 CUDA 12.x 설치 경로를 문서화하는 PR(#4, #5)을 통해 모든 팀원이 동일한 환경에서 작업할 수 있는 기반을 마련했다.

### Phase 2 — 모델 파이프라인 초안 (04-20 ~ 04-21)

`승준_PipeLine_초안` 브랜치에서 YOLOModel 래퍼 클래스와 `train.py` / `predict.py` CLI 진입점을 작성하며 학습·추론 파이프라인의 뼈대를 완성했다. ruff/black 포맷을 처음 적용한 시점이기도 하다. **이 단계에서 치명적인 버그 3개가 발견되어 수정했다** → [P1], [P2], [P3] 참조.

### Phase 3 — Stage 1 설계 확정 (04-21 ~ 04-22)

단일 YOLO로 279개 클래스를 직접 탐지하는 방식을 검토했으나, 클래스당 학습 데이터가 평균 10장 수준으로 너무 부족하다는 결론에 도달했다. 이를 계기로 **Stage 1은 'pill' 단일 클래스 탐지에만 집중하고, 분류는 Stage 2에 위임하는 2-Stage 설계를 팀 전략으로 확정했다.** PR #39에서 `s1_config.yaml`의 클래스 수를 319 → 1로 수정했으며, YOLOv8s·YOLO11s 비교 베이스라인 config도 함께 추가해 모델 선택지를 관리했다.

### Phase 4 — 2-Stage 파이프라인 구현 (04-22 ~ 04-23)

PR #84(`feat/nighthom-2stage-pipeline`)에서 전체 파이프라인의 핵심을 구현했다.

- `scripts/pipeline/crop.py` — YOLO bbox → 알약 크롭 이미지 생성
- `src/models/classifier.py` — ResNet50 기반 분류기 (AdamW, CosineScheduler, FocalLoss 지원)
- `scripts/pipeline/stage2_train.py` / `stage2_predict.py` — Stage 2 학습·추론
- `scripts/pipeline/run_train.py` / `run_predict.py` — 전체 파이프라인 통합 실행
- 데이터 모듈 — `parser_raw`, `parser_external`, `class_map`, `split` 유틸

구현 완료 후 코드를 검토해 설계-구현 불일치를 발견했다. 핵심 파이프라인 자체는 설계 의도대로 동작했으나, Stage 2 `class_names` 미저장 문제([P7]) 등 즉시 수정이 필요한 항목을 발견하였다.

### Phase 5 — 베이스라인 학습 및 버그 수정 (04-23 ~ 04-24)

실제 학습을 돌리는 과정에서 런타임 버그가 연달아 터졌다. `s2_config.yaml`의 `num_classes` 값이 실제 데이터와 맞지 않아 Stage 2 모델이 지속적으로 에러를 뱉었고, crop.py에서 GT 크롭과 inference 크롭 로직이 섞여 있어 학습 데이터가 오염될 위험이 있었다([P5]). 두 문제를 수정한 뒤 Stage 2를 재학습했다.

### Phase 6 — E2E 평가 및 제출 준비 (04-24 ~ 04-26)

`evaluate_pipeline.py`를 새로 추가해 Stage 1 bbox + Stage 2 분류를 COCO 방식으로 통합 평가할 수 있게 됐다. confidence 점수를 `det_score × cls_score`로 통합하는 방식으로 제출 품질을 개선했고, Kaggle category_id 매핑 로직도 추가했다. 이 시점에 Kaggle mAP가 내부 mAP 대비 낮게 나오는 구조적 원인([P6])을 분석하였다.

---

## 발견된 문제 및 수정 이력

### [P1] device 설정 미반영 (2026-04-21) (✅ 해결됨)

`yolo_model.py`의 `train()` 호출 시 config의 `device` 값을 Ultralytics에 전달하지 않아, GPU를 명시해도 무시되는 버그였다. config에서 device를 읽어 kwargs로 전달하는 방식으로 수정했다. `commit 6de628c`

### [P2] CPU에만 시드 고정 (2026-04-21) (✅ 해결됨)

`_fix_seed()`가 `torch.manual_seed()`만 호출하고 `torch.cuda.manual_seed_all()`을 누락해, GPU 연산에서 실험 재현성이 깨지는 문제였다. 같은 config로 두 번 실행해도 결과가 달라지기 때문에 실험 비교 자체를 신뢰하기 어렵게 만드는 치명적인 버그다. `torch.cuda.manual_seed_all(seed)`와 `torch.backends.cudnn.deterministic = True`를 추가해 해결했다. `commit 24d6349`

### [P3] submission.csv Kaggle 포맷 불일치 (2026-04-21) (✅ 해결됨)

`make_submission.py`가 생성하는 CSV의 컬럼명과 값 포맷이 Kaggle 요구 양식과 달라 제출 즉시 오류가 났다. 실제 점수를 확인조차 할 수 없게 만드는 **가장 치명적인 버그**였다. 포맷을 재설계하고 `test_submission.py`도 함께 동기화했다. `commit 315073a`

### [P4] s2_config num_classes 오기입 (2026-04-24) (✅ 해결됨)

`s2_config.yaml`의 `nc` 값이 실제 DE 제공 클래스 수와 달라 ResNet50 분류기의 head 크기가 틀리게 생성됐다. 처음에는 config 값을 직접 수정했고(`2e4a5cd`), 이후 코드에서 실제 폴더 클래스 수로 자동 보정하는 로직을 추가했으며(`3575ac2`), 불일치 시 명확한 예외 메시지도 붙였다(`dc4aeb7`). 총 3개 커밋에 걸쳐 단계적으로 보강됐다.

### [P5] crop.py 모드 미분리 (2026-04-24) (✅ 해결됨)

학습용 GT 크롭, 추론용 inference 크롭, manifest 변환의 세 로직이 하나의 함수에 섞여 있었다. Stage 2 학습 시 ground truth 레이블이 누락되거나 오염될 수 있는 구조적 위험이었다. `gt` / `inference` / `convert` 세 모드로 명시 분리하고 `stage2_dataset.py`에도 반영했다. `commit d869006`

### [P6] Stage 2 class_names 미저장 (✅ 해결됨)

`stage2_train.py` 경로로 학습할 때 `classifier.fit()` 호출 전 `class_names` 갱신이 누락돼, checkpoint에 `class_names=[]`로 저장되는 문제였다. 모델 로드 후 추론 매핑이 실패할 수 있는 버그로, `fit()` 호출 전 `classifier.class_names = train_ds.classes`를 추가해 수정했다. `commit 36ecf1c`

이 버그로 인해서 Stage 2에서 어떤 파라미터를 수정해도 학습이 제대로 이루어지지 않았었다. 아래와 같은 정확도가 나왔는데, 거의 잘못된 학습을 수행한 셈이다.
| 지표 | 점수 |
|-----------|----------|
| Top-1 Acc | 0.0692 | 
| Top-5 Acc  | 0.0881 | 

이 버그를 해결하고 난 이후 정확도는 다음과 같다.
| 지표 | 점수 |
|-----------|----------|
| Top-1 Acc | 0.9031 | 
| Top-5 Acc  | 0.9774 | 

### [P7] Kaggle mAP 낮음 — 클래스 커버리지 문제 - TODO

Kaggle 평가 기준 57개 클래스 중 22개가 DE 제공 데이터에 포함되지 않아, 내부 mAP ≈ 0.77임에도 Kaggle mAP ≈ 0.40으로 보이는 현상으로, 이를 어떻게 해결할지 검토가 필요하다. 

필자가 생각하는 해결방안은 다음 두 가지다.
1. Kaggle Class 기준으로 계산
2. 우리 모델이 받는 클래스를 기반으로 내부계산

### [P8] submission 제출 포맷 에러 - TODO 

현행 submission 제출 파일인 make_submission.py에서의 산출물은 다음과 같다.
| annotation_id	| image_id	| category_id	| bbox_x	| bbox_y	| bbox_w | 	bbox_h	| score | 
|-|-|-|-|-|-|-|-|
| 1	| ext_0-25mg-Mirapex-tab_3_jpg.rf.0b4a7ceabeba62... | 	0	| 257	| 291| 	99	| 101	| 0.495044 | 

2번 산출물인 image_id는 텍스트 값이 아닌, 숫자로된 image id를 내놓아야 하는데 이것이 제대로 지켜지지 않고 있다. 

score도 현재 det score를 기준으로 계산되고 있는데, 이를 mAP@[75-95]로 수정하는 작업이 필요하다.

---

## 최종 성능 지표 (베이스라인 기준, 2026-04-26)

| 단계 | 지표 | val | test |
|------|------|-----|------|
| Stage 1 (YOLOv26n, 1클래스) | mAP@0.50 | 0.9947 | 0.9950 |
| Stage 1 | mAP@[0.50:0.95] | 0.8718 | 0.9410 |
| Stage 1 | Precision / Recall | 0.9963 / 0.9826 | ~1.000 / 0.9950 |
| Stage 2 (ResNet50, 279클래스) | Top-1 Acc | 0.9031 | 0.9042 |
| Stage 2 | Top-5 Acc | 0.9774 | — |
| E2E (내부 279클래스) | mAP@[0.75:0.95] | 0.7709 | 0.7763 |
| E2E (Kaggle 57클래스, 참고용) | mAP@[0.75:0.95] | 0.3979 | 0.3754 |

Stage 1은 mAP@0.50 기준 0.99를 넘어 탐지 자체는 충분히 강력하다. Stage 2 Top-1 Acc 0.90과 E2E mAP 0.77 사이의 격차(≈0.13)는 Stage 1의 미탐지·오탐·과탐이 그대로 E2E 손실로 이어지기 때문이다. 현 시점에서 가장 큰 개선 레버리지는 Stage 1 recall을 끌어올리거나, Stage 2 분류 정확도를 높이는 두 방향이다.

---

## 다음 Step에서 진행해야 할 것

| 순위 | 내용 | 기대 효과 |
|------|------|---------|
| 1 | submission 제출 포맷 개선 | 제출 시 오기입 문제 해결 |
| 2 | 클래스 커버리지 문제 | mAP 개선 | 
| 3 | Stage 2 분류 정확도 높이기 | 분류 성능 향상 |
| 4 | Stage 1 Crop 미탐지/과탐지 줄이기 | mAP 향상 | 


## 다음 실험 우선순위

| 순위 | 내용 | 기대 효과 |
|------|------|---------|
| ★★ | Stage 2 모델 업그레이드 (ResNet50 → EfficientNet / ConvNeXt) | Top-1 Acc 개선 |
| ★ | Stage 1 conf threshold 최적화 | mAP vs recall tradeoff 조정 |
| ★ | crop-level Augmentation 강화 | 소수 클래스 robustness |

---