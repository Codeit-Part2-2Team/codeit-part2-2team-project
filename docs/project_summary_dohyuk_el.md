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

---

## 10. EL Decision Story (최종 의사결정 논리)

### 10-1. 왜 2-Stage 파이프라인인가?
305클래스, 클래스당 평균 10장 수준의 극소량 데이터 환경에서 단일 YOLO로 직접 분류를 시도하면 클래스당 학습 신호가 지나치게 희박해진다. Stage 1을 단일 알약(pill) 클래스 detector로 고정하여 탐지 문제를 단순화하고, 잘린(Crop) 이미지를 Stage 2 분류기에 넘기는 구조가 데이터 부족 문제를 우회하는 공학적 최선이었다. 이 구조는 detection과 classification 병목을 분리해 분석할 수 있고, 실험 통제와 성능 개선 방향을 명확하게 만든다.

### 10-2. 왜 Stage 1을 일찍 Freeze했는가?
YOLOv26n 모델이 mAP@50 0.9947, 다중 객체 재현율 0.9826을 기록하며 탐지 성능이 사실상 포화 상태에 도달했다. 여기에 컴퓨팅 자원을 추가 투입하기보다 Stage 2 분류기 아키텍처 개선에 집중하는 것이 E2E 성능 향상에 더 직접적이라고 판단했다.

### 10-3. 왜 ResNet50 → EfficientNet-B2 → EfficientNetV2-S 순서로 갔는가?
Stage 2는 ResNet50을 baseline으로 설정했다. 모델 교체 기준을 baseline 대비 Top-1 +2%p 이상으로 사전 설정했고, ResNet50 Optuna HPO에서 +2.91%p 개선을 확인한 뒤 더 높은 표현력을 가진 EfficientNet 계열로 확장했다. EfficientNet-B2는 비교적 효율적인 파라미터 규모로 안정적인 성능을 보였고, EfficientNetV2-S는 더 큰 표현력을 바탕으로 최종 Kaggle LB 기준 최고 성능을 기록했다.

### 10-4. 왜 EfficientNetV2-S에서 lr0=0.0003을 선택했는가?
이전 HPO에서 확인된 lr0=0.000645를 EfficientNetV2-S에 그대로 적용했을 때 best epoch가 33에서 형성된 뒤 성능 정체가 발생했다. 모델 Capacity가 커질수록 Loss Landscape가 복잡해지며, 높은 lr0는 최적점을 지나쳐 Overshooting 및 진동 현상을 유발할 수 있다. lr0를 0.0003으로 낮춰 재실험했고, best epoch가 33 → 71로 이동하며 더 안정적인 수렴 흐름을 보였다. 결과적으로 Kaggle LB는 0.95028 → 0.96044로 개선되었다. 따라서 lr0=0.0003은 단순 경험값이 아니라 학습률 조정 전후의 수렴 시점과 LB 개선을 근거로 선택한 최종 값이다.

### 10-5. 왜 D-5 이후 대규모 실험을 중단했는가?
남은 기간 대비 새로운 변수 도입의 리스크가 압도적으로 크다. 새 변수를 추가하면 기존 결과 해석이 흔들리고 복구 시간을 확보하기 어렵다. 0.96 이상의 고정밀 영역에서는 성능 탐색보다 확정된 Final Model의 재현성과 제출 안정화가 최우선이라고 판단했다.

### 10-6. 최종 모델 선택의 리스크와 한계
- Pred bbox 기준 Stage 2 Top-1을 직접 산출하지 못했고, E2E mAP와 Kaggle LB를 최종 파이프라인 성능 proxy로 활용했다.
- GT bbox Top-1과 E2E mAP는 서로 다른 지표이므로 직접 비교하지 않는다.
- val/test E2E mAP gap은 약 1.46%p로, split별 데이터 분포 차이에 따른 변동 가능성이 있다.
- 손바닥 위 알약, 복잡한 배경, 다양한 조명 등 실제 사용자 환경 데이터는 충분히 반영되지 않았다.

---

## 11. 잔여 성능 손실 원인 추정 및 향후 고도화 전략

### 11-1. 잔여 성능 손실 원인 (정성적 분석)
Kaggle LB 0.96044는 매우 높은 최종 성능이지만, mAP 기준으로 일부 클래스·bbox·confidence 영역에서 아직 개선 여지가 남아 있다. 단, mAP는 단순 accuracy가 아니므로 이를 오답률로 해석하지 않는다.

1. **각인 유사 클래스 혼동**: 외형이 유사하고 각인만 다른 클래스의 경우, 클래스당 평균 10장의 데이터로는 세밀한 특징 추출에 한계가 존재한다.
2. **2-Stage 구조의 Trade-off**: Stage 1 YOLO의 bbox가 알약 각인 부위를 미세하게 잘라낼 경우, 손실된 정보가 Stage 2 분류기로 전파되어 구조적 오분류를 유발할 수 있다. 이는 단일 모델의 한계를 극복하기 위해 2-Stage를 선택한 데 따른 필연적 트레이드오프다.
3. **소수 클래스 학습 한계**: 소수 클래스는 증강으로 보완했지만 원본 샘플 다양성 자체가 부족하여 다양한 조명·배경 변화에 대한 일반화 성능이 제한될 수 있다.

### 11-2. 향후 고도화 전략 (Domain Generalization)
본 프로젝트는 통제된 환경에서 최적화된 모델이다. 실제 헬스케어 서비스로의 상용화를 위해 아래 전략이 향후 과제로 요구된다.

1. **Domain Shift 대응**: 손바닥 위 알약, 복잡한 배경, 다양한 조명 조건 등 실전 데이터(In-the-wild) 수집 및 증강 적용.
2. **On-Device AI 배포**: 현재 분리된 2-Stage 모델을 ONNX 형식으로 경량화 및 통합하여 모바일 기기에 이식하는 엔지니어링 파이프라인 구축.
3. **Pred bbox 기준 평가 파이프라인 구현**: Stage 1 bbox 기준 crop 이미지에 대한 직접 정확도 측정 구조 마련.