# 실험 기록 — Stage 2 EfficientNet-B2

## 실험 정보
- 실험명: exp_20260503_efficientnetb2_baseline
- 실험자: ME (승준)
- 날짜: 2026-05-03
- 브랜치명: exp/Nighthom-baseline

## 실험 목적
ResNet50 Baseline 대비 EfficientNet-B2 모델 교체 후 성능 비교

## 변경 변수
- 변경 항목: Stage 2 모델 교체 (ResNet50 → EfficientNet-B2)
- 변경 전: ResNet50, Top-1 0.8750
- 변경 후: EfficientNet-B2, Optuna 최적 파라미터 적용

## 고정 변수
- Image size: 224
- Epoch: 100
- Batch size: 32
- Optimizer: AdamW
- LR: 0.000645 (lrf: 0.09057, CosineAnnealingLR)
- label_smoothing: 0.0695
- weight_decay: 0.01013
- Seed: 42
- Dataset: v2.0 (305클래스)

## 실험 설정
- 모델: EfficientNet-B2 (pretrained)
- 데이터 조건: GT bbox 기준 크롭 이미지, 305클래스, Albumentations 증강 적용
- 특이사항: Optuna HPO (trial_0008) 최적 파라미터 그대로 적용

## 실험 결과

### Stage 2 (Classifier) - GT bbox 기준
- Top-1 Accuracy (val): 0.8932
- Top-5 Accuracy (val): 0.9697
- Best epoch: 93
- 학습 시간: 157분

### E2E 평가
| 기준 | val | test |
|------|-----|------|
| Kaggle mAP@[0.75:0.95] | 0.9065 | 0.9045 |
| Kaggle Public LB | 0.94731 | - |

## 결과 해석
- ResNet50 Baseline(0.8750) 대비 Top-1 +1.82%p 개선
- Kaggle LB 0.94731 달성
- EfficientNetV2-S 대비 LB 낮아 V2-S lr0 재조정 실험으로 이어짐

## 실험 중단 여부
- [x] 계속 진행 (V2-S 비교 후 최종 결정)

## EL (도혁) 코멘트
Optuna 파라미터 적용 시 B2에서 안정적인 학습 확인. ResNet50 대비 개선됐으나 V2-S가 Kaggle LB 기준으로 더 높아 Final Model에서 제외됨.

## 가중치/결과 위치
- 가중치: Google Drive (PR #165 참고 — ME 승준 문의)
- 결과 로그: experiments/exp_20260503_efficientnet-b2/