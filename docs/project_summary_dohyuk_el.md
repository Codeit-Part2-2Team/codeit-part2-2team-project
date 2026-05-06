# 알약 탐지 프로젝트 — EL (도혁) 관점 정리

작성일: 2026-05-06

---

## 1. EL 역할 개요

EL (Experiment Lead)은 실험 설계, HPO 실행, 결과 취합, Kaggle 제출 조율을 담당했습니다.

- 실험 설계 및 config 관리
- Grid Search / Optuna HPO 직접 실행
- 실험 결과 취합 및 experiment_summary.md 관리
- Kaggle 제출 승인 및 이력 관리
- 팀원 간 실험 방향 조율 및 의사결정 근거 문서화

---

## 2. 실험 설계 원칙

- 한 번에 하나의 변수만 변경
- GT bbox vs Pred bbox 성능 분리 기록
- CV vs LB 괴리 추적
- Kaggle 10회/일 한도 내 EL이 제출 조율
- Verdict 기준: keep / observe / drop
- Best model은 validation 기준으로만 선정

---

## 3. HPO 실험 흐름

### 3-1. Grid Search (12조합)

| 변수 | 탐색 범위 |
|------|----------|
| lr0 | 3e-05, 1e-04, 3e-04 |
| label_smoothing | 0.0, 0.05, 0.1, 0.2 |
| weight_decay | 0.001, 0.01 |

- Best: trial_0009, lr0=0.0003, label_smoothing=0.05, weight_decay=0.001
- Top-1: 0.8786
- 주요 발견: lr0=0.0003 구간이 일관되게 최적

### 3-2. Optuna HPO (15 trials / 20 epochs)

- 탐색 공간: lr0 0.0001~0.001 세밀 탐색
- Best: trial_0008
  - lr0: 0.000645
  - label_smoothing: 0.0695
  - lrf: 0.09057
  - weight_decay: 0.01013
  - Top-1: **0.9041**
- ResNet50 Baseline(0.8750) 대비 +2.91%p → 모델 교체 기준(+2%p) 충족

---

## 4. 모델 교체 실험 결과

| 모델 | GT Top-1 | Kaggle LB | 비고 |
|------|----------|-----------|------|
| ResNet50 (Baseline) | 0.8750 | 0.90845 | 기준선 (Public LB) / 내부 E2E val 0.9365 |
| ResNet50 (Optuna) | 0.9041 | - | HPO 최적값 |
| EfficientNet-B2 | 0.8932 | 0.94731 | Optuna 파라미터 적용 |
| EfficientNetV2-S (lr0 미조정) | 0.8726 | 0.95028 | lr0 과다, epoch 33 정체 |
| **EfficientNetV2-S (lr0=0.0003)** | **0.8883** | **0.96044** | **Final Model** |

---

## 5. 핵심 인사이트

### 5-1. 모델 교체 시 HPO 파라미터 이식 주의
B2(9M) 최적 lr0=0.000645를 V2-S(24M)에 그대로 적용하면 모델 크기 차이로 인해 lr0가 상대적으로 과도하게 작용합니다. 초반 과수렴 후 지역 최솟값 고착이 발생했으며, lr0를 0.0003으로 재조정 후 성능이 개선됐습니다.

### 5-2. 평가 기준은 Kaggle LB 우선
내부 GT Top-1이 낮아도 Kaggle LB는 높을 수 있습니다. 실제 평가 기준인 Kaggle LB 기준으로 판단하는 것이 맞습니다.

### 5-3. 305클래스 학습 모델의 범용성
Kaggle 평가 기준 57클래스에만 맞춰 학습하지 않고 305클래스로 학습했음에도 LB 0.96044 달성. 경구약제 전반적 형태에 대한 범용 분류 능력 확보.

### 5-4. HPO 시간 효율
15 trials / 20 epochs로 30 trials 대비 절반 시간에 충분한 탐색 가능. 마감 기한 고려 시 적절한 트레이드오프.

---

## 6. GT vs Pred 분리 분석 결과

Pred bbox 기준 Stage 2 Top-1 직접 산출은 inference crop 라벨 연결 구조 부재로 불가능함을 확인 (ME 승준 확인).

GT bbox Top-1과 E2E mAP는 평가 방식이 달라 직접 비교하지 않으며, E2E mAP val 0.9361 / test 0.9215 및 Kaggle LB 0.96044를 Pred bbox가 반영된 최종 파이프라인 성능 proxy로 해석한다.

---

## 7. Final Model

| 항목 | 값 |
|------|-----|
| Stage 1 | YOLOv26n |
| Stage 2 | EfficientNetV2-S |
| Stage 2 lr0 | 0.0003 |
| Kaggle LB | 0.96044 |
| E2E mAP val | 0.9361 |
| E2E mAP test | 0.9215 |
| GT bbox Top-1 | 0.8883 |
| GT bbox Top-5 | 0.9745 |

---

## 8. D-5 남은 과제

- [x] Stage 1 raw multi-object 신뢰성 간접 확인 완료 (ME 승준 보고서 기준 재현율 0.9826)
- Final submission 재현성 확보 (Weight Path, Inference Command)
- Main / Backup submission 정리
- 보고서 연계 (AL 찬우)
- 협업 일지 작성

---

## 9. 한계 및 향후 과제

- 손바닥 위 알약, 복잡한 배경, 다양한 조명 조건 등 실제 서비스 환경 데이터 부재
- 모바일 앱 배포 미완성
- GT vs Pred 직접 측정 파이프라인 미구현
- Stage 1 raw multi-object 신뢰성 직접 검증 미완료 (간접 확인으로 대체)