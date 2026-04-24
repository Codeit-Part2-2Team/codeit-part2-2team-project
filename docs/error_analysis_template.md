# Error Analysis Template

## 개요
Baseline 학습 완료 후 오답 패턴을 분석하기 위한 템플릿이다.
단순 점수 기록이 아닌 "왜 틀렸는가"를 추적하여 다음 실험 방향을 결정한다.

## 사용 시점
- Stage 1 또는 Stage 2 Baseline 결과 확인 후
- Confusion Matrix에서 특정 클래스 오답이 반복될 때
- GT vs Pred 괴리가 CV-LB 기준 -2%p 이상 발생할 때

## Error 기록 양식

| Sample ID | Source | GT | Pred | Error Type | Root Cause | Next Action |
|-----------|--------|-----|------|------------|------------|-------------|
| | raw/external | | | | | |

## Error Type 분류 기준

### Stage 1 (Detection)
- `missed_detection`: GT bbox 있으나 Pred 없음
- `false_positive`: GT 없으나 Pred 있음
- `partial_crop`: bbox가 알약 일부만 포함
- `multi_miss`: multi-object 이미지에서 일부 알약 누락

### Stage 2 (Classification)
- `visually_similar`: 외형 유사 클래스 혼동 (색상/모양)
- `imprint_miss`: 각인 인식 실패
- `background_noise`: 배경 노이즈로 인한 오분류
- `low_freq_class`: 학습 데이터 부족 클래스 오답
- `bbox_error_propagation`: Stage 1 bbox 오차로 인한 크롭 품질 저하

## Root Cause 분류

- Detection 문제 → Stage 1 개선 필요
- Classifier 문제 → Stage 2 모델/증강 개선 필요
- 데이터 문제 → DE (소원)과 협의
- bbox 오차 전파 → Bbox Jittering 실험 검토

## 분석 우선순위

Baseline 결과 나오면 아래 순서로 확인한다.

1. raw vs external 분리 성능 비교
2. multi-object 이미지 recall 확인
3. Confusion Matrix 상위 오답 클래스 추출
4. Val set에 없는 클래스 Blind Spot 확인 (DE 소원 제공 자료 기준)

## 업데이트 규칙
- 실험 완료 후 EL (도혁)이 주요 오답 샘플 최소 5개 이상 기록
- Root Cause 미확인 시 "조사 중"으로 표시 후 다음 실험 전 반드시 확인
- AL (찬우) 보고서 작성 시 이 문서 참고