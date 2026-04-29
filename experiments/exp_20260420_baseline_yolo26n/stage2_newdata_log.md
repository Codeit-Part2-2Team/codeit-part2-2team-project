# 실험 기록 — Stage 2 신규 데이터셋 Baseline (Hold)

## 실험 정보
- 실험명: exp_20260429_resnet50_newdata
- 실험자: ME (승준)
- 날짜: 2026-04-29
- 브랜치명: exp/Nighthom-baseline
- 상태: **Hold** (279클래스 커버리지 불충분, 새 데이터셋 재실험 예정)

## 실험 목적
신규 데이터셋 기반 Stage 2 Baseline 산출 및 Kaggle 커버리지 확인

## 변경 변수
- 변경 항목: 데이터셋 (소원 신규 데이터셋 적용)
- 변경 전: v1.0 (279클래스)
- 변경 후: 신규 데이터셋 (279클래스, 10장 미만 제거 후)

## 고정 변수
- Image size: 224
- Epoch: 100
- Batch size: 32
- Optimizer: AdamW (weight_decay: 0.01)
- LR: 0.0001 (lrf: 0.01, CosineAnnealingLR)
- Seed: 42
- Dataset: 신규 데이터셋 (279클래스)

## 실험 결과

### Stage 2 (Classifier) - GT bbox 기준
- Top-1 Accuracy (val): 0.8579
- Top-5 Accuracy (val): 0.9572
- Top-1 Accuracy (test): 0.9042
- best epoch: 73
- 학습 시간: 7181.2초 (119.7분)

### E2E 평가
| 기준 | val | test |
|------|-----|------|
| 내부 mAP@[0.75:0.95] | 0.7863 | 0.7763 |
| Kaggle mAP@[0.75:0.95] | 0.3451 | 0.3754 |
| Kaggle 커버 클래스 | 15/57 | 20/57 |

## 결과 해석
- GT bbox 기준 Top-1 0.8579로 v1.0 Baseline(0.9031) 대비 낮음
- Kaggle 커버 클래스 15~20/57 — 279클래스 중 Kaggle 평가 56클래스의 절반만 커버
- 근본 원인: 10장 미만 제거 과정에서 Kaggle 평가 대상 클래스 25~26개 누락
- HPO 진입 전 데이터셋 재구성 필요

## 실험 중단 여부
- [x] Hold (데이터셋 재구성 후 재실험 예정)

## Hold 사유
- Kaggle 평가 56클래스 중 25~26개 미보유
- DE (소원) missing class 수집 + 증강 + merge 작업 진행 중
- 새 데이터셋 배포 완료 후 재실험 예정

## 다음 액션
- DE (소원) 새 데이터셋 배포 완료 대기
- 새 데이터셋 기반 Baseline 재실험 (ME (승준))
- 재실험 완료 후 EL (도혁) 결과 분석 및 HPO 진입

## EL (도혁) 코멘트
279클래스 데이터셋으로 실험했으나 Kaggle 평가 클래스 커버리지 불충분으로 Hold 처리. 데이터셋 재구성 후 재실험 필요. Submission 결과는 0.68914로 포맷 수정 효과는 확인됨.

## 가중치/결과 위치
- 가중치: Google Drive 공유 (PM (호정) 정책 기준, 위치 확인 필요)
- 결과 로그: experiments/exp_20260420_baseline_yolo26n/