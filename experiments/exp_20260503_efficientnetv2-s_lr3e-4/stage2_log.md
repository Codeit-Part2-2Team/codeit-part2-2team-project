# 실험 기록 — Stage 2 EfficientNetV2-S (lr0=0.0003, Final Model)

## 실험 정보
- 실험명: exp_20260503_efficientnetv2s_lr3e-4
- 실험자: ME (승준)
- 날짜: 2026-05-03 ~ 2026-05-04
- 브랜치명: exp/efficientnetv2_s_lr3e4

## 실험 목적
EfficientNetV2-S lr0 재조정 후 성능 개선 확인 및 Final Model 확정

## 변경 변수
- 변경 항목: lr0 재조정 (0.000645 → 0.0003)
- 변경 전: EfficientNetV2-S, lr0=0.000645, Kaggle LB 0.95028
- 변경 후: EfficientNetV2-S, lr0=0.0003, Kaggle LB 0.96044

## 고정 변수
- Image size: 224
- Epoch: 100
- Batch size: 32
- Optimizer: AdamW
- LR: 0.0003 (lrf: 0.09057, CosineAnnealingLR)
- label_smoothing: 0.0695
- weight_decay: 0.01013
- Seed: 42
- Dataset: v2.0 (305클래스)

## 실험 설정
- 모델: EfficientNetV2-S (pretrained)
- 데이터 조건: GT bbox 기준 크롭 이미지, 305클래스, Albumentations 증강 적용
- 특이사항: lr0를 B2 최적값(0.000645)의 절반 수준인 0.0003으로 낮춰 재조정

## 실험 결과

### Stage 2 (Classifier) - GT bbox 기준
- Top-1 Accuracy (val): 0.8883
- Top-5 Accuracy (val): 0.9745
- Top-1 Accuracy (test): 0.8844
- Best epoch: 71
- 학습 시간: 225분

### E2E 평가
| 기준 | val | test |
|------|-----|------|
| Kaggle mAP@[0.75:0.95] | 0.9361 | 0.9215 |
| Kaggle Public LB | 0.96044 | - |

## 결과 해석
- lr0 재조정으로 best epoch 33 → 71로 이동, 수렴 안정성 크게 향상
- GT bbox Top-1 0.8883으로 이전 V2-S(0.8726) 대비 개선
- Kaggle LB 0.96044로 전체 실험군 중 최고 성능 달성
- val/test E2E mAP gap 약 1.46%p — 데이터 분포 차이로 기록

## 실험 중단 여부
- [x] Final Model 확정 (팀 합의 완료, 2026-05-04)

## EL (도혁) 코멘트
lr0=0.0003 재조정으로 V2-S 지역 최솟값 고착 해소 확인. Kaggle LB 0.96044로 전체 실험군 최고 성능 기록. 모델 크기에 맞는 lr0 조정이 핵심 인사이트. Final Model로 확정.

## 가중치/결과 위치
- 가중치: Google Drive (PR #166 참고 — ME 승준 문의)
- 결과 로그: experiments/exp_20260503_efficientnetv2-s_lr3e-4/
