# 실험 기록 — Stage 1 Baseline

## 실험 정보
- 실험명: exp_20260420_baseline_yolo26n
- 실험자: ME (승준)
- 날짜: 2026-04-20 ~ 2026-04-27
- 브랜치명: exp/Nighthom-baseline

## 실험 목적
2-Stage 파이프라인의 Stage 1 Detection 성능 기준선 확보

## 변경 변수
- 변경 항목: 없음 (Baseline)
- 변경 전: -
- 변경 후: -

## 고정 변수
- Image size: 640
- Epoch: 100
- Batch size: 16
- Optimizer: auto
- LR: 0.001 (lrf: 0.01)
- Seed: 42
- Dataset: v1.0 (raw + external 통합, 단일 클래스 pill, nc=1)

## 실험 설정
- 모델: YOLOv26n (pretrained)
- 데이터 조건: raw + external 통합, 단일 클래스 pill (nc=1), Albumentations 증강 적용

## 실험 결과

### Stage 1 (Detection)
- mAP@50 (val): 0.9947
- mAP@50 (test): 0.9950
- mAP@50-95 (val): 0.8718
- Inference time (ms): 미측정

### Stage 2 (Classifier) - GT bbox 기준
- 해당 없음 (Stage 1 전용 실험)

### Stage 2 (Classifier) - Pred bbox 기준
- 해당 없음 (Stage 1 전용 실험)

### GT vs Pred 괴리 분석
- 해당 없음 (Stage 1 단독 실험)

## 결과 해석
- mAP@50 0.9947로 Detection 성능 기준선 확보
- external 데이터 단일 알약 편향으로 과대평가 가능성 있음
- raw only mAP / multi-object recall 별도 검증 필요

## 실험 중단 여부
- [x] 계속 진행 (Baseline Freeze, 이후 파생 실험으로 분리)

## CV-LB 괴리 분석
- Local CV (val mAP@50): 0.9947
- Public LB: E2E 기준 분석 필요 (Stage 1 단독 LB 미측정)
- 괴리 해석: Kaggle mAP 저하 원인은 submission 포맷 문제 가능성 우선 (P0)

## Kaggle mAP 저하 원인 체크
- [x] submission image_id 포맷 확인 → 수정 완료 (ME (승준))
- [ ] category_id 매핑 정합성 확인 → 재제출 후 점검 예정
- [ ] Kaggle 평가 클래스 커버리지 확인 → 재제출 후 점검 예정
- [ ] class index / label mapping mismatch 확인 → 재제출 후 점검 예정

## 다음 액션
submission 재제출 결과 확인 후 Kaggle mAP 회복 여부 기준으로 Stage 1 신뢰성 검증 진행

## EL (도혁) 코멘트
Baseline 기준선 확보 완료. mAP@50 0.9947은 external 단일 알약 편향 가능성 있어 raw only / multi-object 기준 별도 검증 필요. Kaggle mAP 저하는 Stage 1보다 submission 포맷 문제가 1순위 판단.

## 가중치/결과 위치
- 가중치: Google Drive 공유 (EL (도혁) 문의)
- 결과 로그: experiments/exp_20260420_baseline_yolo26n/