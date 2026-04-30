# 실험 결과 보고서_Baseline

## 목적

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 프로젝트 | 경구약제 알약 탐지 |
| 목표 | 이미지에서 알약 위치(bbox) + Kaggle 기준 약품명(class) 동시 예측 |
| 팀 | PM 호정, ME 승준, EL 도혁, DE 소원, AL 찬우 |
| 기간 | 2026-04-20 ~ 05-11 (이 보고서는 04-29 기준) |
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

`evaluate_pipeline.py`를 새로 추가해 Stage 1 bbox + Stage 2 분류를 COCO 방식으로 통합 평가할 수 있게 됐다. confidence 점수를 `det_score × cls_score`로 통합하는 방식으로 제출 품질을 개선했고, Kaggle category_id 매핑 로직도 추가했다. 이 시점에 Kaggle mAP가 내부 mAP 대비 낮게 나오는 구조적 원인([P7])을 분석하였다.

### Phase 7 — Kaggle 기준 E2E 평가 재검증 및 클래스 매핑 보정 (04-29)

Baseline 실험 노트북에서 Kaggle 기준 E2E 지표가 내부 평가 대비 낮게 나타나 원인을 재검증했다. 이 과정에서 문제는 두 층으로 나뉘어 있었다. 첫째, Stage 2 학습/예측 대상 클래스 커버리지가 부족했다. 이는 DE와 논의한 결과 해결하였다. 둘째는, 실제로 캐글 클래스를 매핑해서 가져오는 과정에서 버그가 있었던 것이다. 

다만 Kaggle 원본 기준 클래스 수 자체가 잘못 매핑된 것은 아니었다. `sprint_ai_project1_data/train_annotations`를 기준으로 확인한 결과 raw annotation의 unique `dl_idx`는 56개였고, `kaggle_class_map.json`도 동일한 56개 category_id를 모두 커버하고 있었다. 따라서 이후 작업은 Stage 2 클래스 커버리지 확장과 Kaggle 기준 평가/제출 매핑 보정을 함께 진행하는 방향으로 정리했다.

이에 따라 `evaluate_pipeline.py`에서 YOLO label의 단일 class id에 의존하지 않고, GT crop manifest와 `raw_K-*` 파일명에 포함된 category id를 이용해 GT class를 복원하도록 수정했다. 또한 Stage 2가 출력하는 alias class_name 중 `kaggle_class_map.json`에 없는 항목을 `kaggle_unknown_class_map.json`으로 canonical Kaggle class_name에 매핑하도록 보정했다. 동일한 보정 흐름을 실험 노트북과 튜토리얼 노트북, README에도 반영했다.

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

### [P7] Kaggle mAP 낮음 — 클래스 커버리지 및 Kaggle 매핑 문제 (2026-04-29) (✅ 해결됨)

초기에는 Kaggle 평가 기준 클래스 중 일부가 DE 제공 데이터에 포함되지 않아 내부 mAP 대비 Kaggle mAP가 낮게 보이는 것으로 판단했다. 실제로 Stage 2 기준에서는 클래스 커버리지 개선이 필요했으며, baseline config의 `nc`와 `model.num_classes`를 279에서 305로 갱신해 확장된 crop class set을 반영했다.

동시에 raw annotation을 재검증한 결과, Kaggle 원본 category 기준에서는 `train_annotations`의 unique `dl_idx`가 56개였고 `kaggle_class_map.json`도 동일한 56개 category_id를 모두 포함하고 있었다. 즉, Kaggle category_id 자체가 누락된 것이 아니라, Stage 2 class set 확장과 E2E 평가/제출 매핑 보정이 함께 필요했던 문제였다.

실제 문제는 E2E 평가에서 GT class를 YOLO label 또는 제한적인 manifest 정보로 복원하면서 Kaggle category와 정확히 연결하지 못한 점이었다. 이를 `GT crop manifest → source image key → raw_K category id → filename fallback` 순서로 복원하도록 수정했다. 이 변경 이후 Kaggle 기준 E2E 평가에서 클래스 커버리지 손실 없이 GT와 prediction을 비교할 수 있게 됐다.

### [P8] submission/evaluate class_name 매핑 누락 (2026-04-29) (✅ 해결됨)

Stage 2 예측 결과 중 일부 class_name이 `kaggle_class_map.json`에 존재하지 않아 제출 생성 또는 Kaggle 기준 평가에서 누락되는 문제가 있었다. 대표적으로 `gabapentin_tab_800mg_dong_a`, `januvia_tab_50mg`, `trajenta_tab`, `kanarb_tab_60mg` 등이 Kaggle 제출용 canonical class_name과 달랐다.

이를 별도의 `kaggle_unknown_class_map.json`으로 관리하고, `evaluate_pipeline.py`와 `make_submission.py`에서 Stage 2 alias를 Kaggle canonical class_name으로 정규화한 뒤 `kaggle_class_map.json`을 통해 category_id로 변환하도록 보정했다. 이 작업은 모델 성능 자체를 바꾼 것이 아니라, 기존 예측 결과가 Kaggle 기준 class id로 올바르게 집계되도록 만든 평가/제출 경로 수정이다.

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
| E2E (Kaggle 기준, 초기 참고용) | mAP@[0.75:0.95] | 0.3979 | 0.3754 |

Stage 1은 mAP@0.50 기준 0.99를 넘어 탐지 자체는 충분히 강력하다. Stage 2 Top-1 Acc 0.90과 E2E mAP 0.77 사이의 격차(≈0.13)는 Stage 1의 미탐지·오탐·과탐이 그대로 E2E 손실로 이어지기 때문이다. 현 시점에서 가장 큰 개선 레버리지는 Stage 1 recall을 끌어올리거나, Stage 2 분류 정확도를 높이는 두 방향이다.

---

## 갱신된 E2E 및 제출 성능 (2026-04-29)

Kaggle 기준 E2E 평가를 재검증하면서 Stage 2 클래스 커버리지를 279개에서 305개로 확장하고, GT class 복원과 Stage 2 alias 매핑을 보정한 결과 내부 평가와 실제 Kaggle 제출 점수가 모두 개선됐다. 특히 실제 제출 점수인 `mAP@0.75:0.95`가 0.68914에서 0.90845로 상승해, 낮은 점수의 주요 원인이 모델 자체의 탐지 실패만이 아니라 클래스 커버리지 부족과 평가/제출 클래스 매핑 누락이 함께 작용한 결과였음을 확인했다.

| 기준 | 이전 | 보정 후 | 변화 |
|------|------|---------|------|
| Kaggle 제출 mAP@0.75:0.95 | 0.68914 | 0.90845 | +0.21931 |

이 개선은 `kaggle_class_map.json` 자체의 category_id 커버리지 문제가 아니라, Stage 2 class set 확장과 Stage 2 출력 class_name/Kaggle 제출 class_name 사이의 alias 차이 보정이 함께 반영된 효과다. 따라서 이후 실험에서는 Stage 2 모델 성능을 비교할 때 반드시 동일한 class set, alias 정규화, GT class 복원 기준을 적용해야 한다.

---

## Baseline Freeze 선언 (2026-04-29)

본 보고서의 baseline은 2026-04-29 기준으로 freeze한다. 이후 실험은 아래 조건을 고정된 비교 기준으로 사용한다.

| 항목 | Freeze 기준 |
|------|-------------|
| Stage 1 | YOLOv26n, pill 단일 클래스 탐지 |
| Stage 2 | ResNet50 분류기, 305개 crop class set |
| E2E 평가 | GT crop manifest / raw_K category id 기반 GT class 복원 |
| Kaggle 매핑 | `kaggle_class_map.json` + `kaggle_unknown_class_map.json` 적용 |
| 제출 score | `det_score × cls_score` |
| 대표 제출 성능 | Kaggle mAP@0.75:0.95 = 0.90845 |

이 시점 이후 baseline 파일과 결과는 임의로 수정하지 않고, 새로운 모델·augmentation·threshold·HPO 실험은 별도 실험명과 별도 config로 분리해 비교한다. 단, 평가/제출 재현성을 높이기 위한 문서화나 config 파일 위치 정리처럼 baseline 성능 자체를 바꾸지 않는 관리 작업은 후속 PR에서 별도로 다룬다.

---

## 다음 Step에서 진행해야 할 것

| 순위 | 내용 | 기대 효과 |
|------|------|---------|
| 1 | submission 제출 포맷 개선 | 제출 시 오기입 문제 해결 |
| 2 | Stage 2 분류 정확도 높이기 | 분류 성능 향상 |
| 3 | Stage 1 Crop 미탐지/과탐지 줄이기 | mAP 향상 |
| 4 | class map / alias map 설정 파일 위치 정리 | 평가·제출 재현성 개선 |


## 다음 실험 우선순위

| 순위 | 내용 | 기대 효과 |
|------|------|---------|
| ★★ | Stage 2 모델 업그레이드 (ResNet50 → EfficientNet / ConvNeXt) | Top-1 Acc 개선 |
| ★ | Stage 1 conf threshold 최적화 | mAP vs recall tradeoff 조정 |
| ★ | crop-level Augmentation 강화 | 소수 클래스 robustness |

---
