# 실험 기록 — Stage 2 Baseline

## 실험 정보
- 실험명: exp_20260426_resnet50_baseline
- 실험자: ME (승준)
- 날짜: 2026-04-26 ~ 2026-04-27
- 브랜치명: exp/Nighthom-baseline

## 실험 목적
2-Stage 파이프라인의 Stage 2 Classification 성능 기준선 확보

## 변경 변수
- 변경 항목: 없음 (Baseline)
- 변경 전: -
- 변경 후: -

## 고정 변수
- Image size: 224
- Epoch: 100
- Batch size: 32
- Optimizer: AdamW (weight_decay: 0.01)
- LR: 0.0001 (lrf: 0.01, CosineAnnealingLR)
- Seed: 42
- Dataset: v1.0 (raw + external 통합, 279 클래스 / 371 → 279, 10장 미만 제거, 2026-04-29 확정)

## 실험 설정
- 모델: ResNet50 (pretrained)
- 데이터 조건: GT bbox 기준 크롭 이미지, 279 클래스 (371 → 279, 10장 미만 제거, 2026-04-29 확정), Albumentations 증강 적용
- 특이사항: 버그 수정 후 정상화 (수정 전 결과 무효)

## 실험 결과

### Stage 1 (Detection)
- 해당 없음 (Stage 2 전용 실험)

### Stage 2 (Classifier) - GT bbox 기준
- Top-1 Accuracy (val): 0.9031
- Top-5 Accuracy (val): 0.9774
- Top-1 Accuracy (test): 0.9042

### Stage 2 (Classifier) - Pred bbox 기준
- Top-1 Accuracy: 직접 측정 불가 (inference crop 라벨 연결 구조 부재)
- Top-5 Accuracy: 직접 측정 불가 (inference crop 라벨 연결 구조 부재)

### GT vs Pred 괴리 분석
- GT Top-1: 0.9031
- Pred Top-1: 직접 산출 불가 — E2E mAP를 최종 파이프라인 성능 proxy로 해석
- Gap: 직접 계산 불가
- 괴리 원인 추정: E2E mAP 기준 간접 확인 완료

## 결과 해석
- GT bbox 기준 Top-1 0.9031로 분류기 기준선 확보
- Pred bbox 기준 성능은 GT vs Pred 분리 분석 완료 후 병목 위치 파악 예정
- 버그 수정 전 결과는 무효 처리

## 실험 중단 여부
- [x] 계속 진행 (Baseline Freeze, 이후 파생 실험으로 분리)

## CV-LB 괴리 분석
- Local CV (val Top-1): 0.9031
- Public LB (E2E mAP@[0.75:0.95]): 0.9458 (305cls, mapping 수정 완료)
- Gap: +0.0427 (E2E 기준)
- 괴리 해석: 포맷 수정 + 305cls 보강으로 해소 완료 (2026-04-30)

## Kaggle mAP 저하 원인 체크
- [x] submission image_id 포맷 확인 → 수정 완료 (ME (승준))
- [x] category_id 매핑 정합성 확인 → 완료 (PR #144, #145)
- [x] Kaggle 평가 클래스 커버리지 확인 → 완료 (305cls 보강, DE 소원)
- [x] class index / label mapping mismatch 확인 → 완료 (PR #144, #145)

## 다음 액션
- [x] submission 재제출 후 Kaggle mAP 회복 확인 완료 (0.9458, 2026-04-30)
- [x] GT vs Pred 분리 분석 완료 — Pred bbox 직접 측정 불가 확인, E2E mAP proxy로 간접 확인
- [ ] Final Model (EfficientNetV2-S lr0=0.0003) 기준 결과는 stage2_log 별도 문서 참고

## EL (도혁) 코멘트
GT bbox 기준 Top-1 0.9031로 기준선 확보. Kaggle mAP 0.3754와의 괴리는 submission 포맷 수정 후 재평가 예정. Pred bbox 기준 성능 미측정 상태로 GT vs Pred 분리 분석이 다음 우선 과제.

## 가중치/결과 위치
- 가중치: Google Drive 공유 (EL (도혁) 문의)
- 결과 로그: experiments/exp_20260420_baseline_yolo26n/