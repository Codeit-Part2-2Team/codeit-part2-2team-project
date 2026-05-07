# 실험 기록 — Stage 2 EfficientNetV2-S (lr0 미조정)

## 실험 정보
- 실험명: exp_20260503_efficientnetv2s_baseline
- 실험자: ME (승준)
- 날짜: 2026-05-03
- 브랜치명: exp/Nighthom-baseline

## 실험 목적
EfficientNet-B2 대비 EfficientNetV2-S 모델 교체 후 성능 비교

## 변경 변수
- 변경 항목: Stage 2 모델 교체 (EfficientNet-B2 → EfficientNetV2-S)
- 변경 전: EfficientNet-B2, Top-1 0.8932
- 변경 후: EfficientNetV2-S, B2용 Optuna 파라미터 그대로 이식

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
- 모델: EfficientNetV2-S (pretrained)
- 데이터 조건: GT bbox 기준 크롭 이미지, 305클래스, Albumentations 증강 적용
- 특이사항: B2용 Optuna 파라미터(lr0=0.000645)를 V2-S에 그대로 이식 → lr0 과다 문제 발생

## 실험 결과

### Stage 2 (Classifier) - GT bbox 기준
- Top-1 Accuracy (val): 0.8726
- Top-5 Accuracy (val): 0.9563
- Best epoch: 33 (이후 정체)
- 학습 시간: 221분

### E2E 평가
| 기준 | val | test |
|------|-----|------|
| Kaggle mAP@[0.75:0.95] | 0.8923 | 0.9316 |
| Kaggle Public LB | 0.95028 | - |

## 결과 해석
- GT bbox Top-1은 B2(0.8932) 대비 낮음 (-0.021)
- 그러나 Kaggle LB는 B2(0.94731) 대비 높음 (+0.003)
- epoch 33 이후 성능 갱신 없음 → lr0 과다로 인한 지역 최솟값 고착으로 판단
- B2(9M)보다 큰 V2-S(24M)에 동일 lr0 적용 시 과도한 수렴 발생

## 실험 중단 여부
- [x] lr0 재조정 후 재실험 (exp_20260503_efficientnetv2-s_lr3e-4 참고)

## EL (도혁) 코멘트
B2 Optuna 파라미터 V2-S에 그대로 이식 시 lr0 과다 문제 확인. 모델 크기(24M vs 9M) 차이로 인해 동일 lr0 적용이 부적절함을 파악. lr0=0.0003으로 재조정 후 재실험 진행.

## 가중치/결과 위치
- 가중치: Google Drive (PR #173 코멘트 참고 — ME 승준 제공)
- 결과 로그: experiments/exp_20260503_efficientnetv2-s/