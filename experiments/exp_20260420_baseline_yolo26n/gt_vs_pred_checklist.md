# GT vs Pred 분리 분석 체크리스트

> 작성: EL (도혁) | 연계 이슈: #80
> 승준 신규 데이터셋 Baseline 결과 나오는 즉시 실행

---

## Step 1. 기본 지표 확인

- [ ] GT bbox 기준 Top-1 Accuracy
- [ ] GT bbox 기준 Top-5 Accuracy
- [ ] Pred bbox 기준 Top-1 Accuracy
- [ ] Pred bbox 기준 Top-5 Accuracy
- [ ] GT vs Pred Gap 계산 (GT Top-1 - Pred Top-1)

---

## Step 2. 병목 판단

| GT Top-1 | Pred Top-1 | Gap | 판단 | 액션 |
|----------|------------|-----|------|------|
| 높음 | 낮음 | 5%p 이상 | Stage 1 병목 | bbox Jittering 실험 검토 |
| 낮음 | 낮음 | 근소 | Stage 2 병목 | 모델 교체 / 증강 전략 변경 |
| 높음 | 높음 | 근소 | 전체 구조 안정 | 다음 실험 단계 진행 |

> GT 낮음 / Pred 높음: 파이프라인 구조상 발생 가능성 매우 낮음
> 예외 발생 시 평가 로직 이상 여부 먼저 점검

---

## Step 3. 클래스 구간별 성능 분리

- [ ] 0~10개 구간 클래스 GT / Pred 성능
- [ ] 11~20개 구간 클래스 GT / Pred 성능
- [ ] 21~30개 구간 클래스 GT / Pred 성능
- [ ] 31~40개 구간 클래스 GT / Pred 성능
- [ ] 41개 이상 구간 클래스 GT / Pred 성능

> 전체 평균만 보면 long-tail 영향 착시 발생 가능
> 구간별 성능 차이로 증강 전략 재조정 여부 판단

---

## Step 4. Crop 품질 확인

- [ ] crop 실패율 (bbox가 너무 작거나 잘못 잘린 케이스)
- [ ] 빈 crop 발생 여부
- [ ] 알약 일부만 포함된 partial crop 비율
- [ ] 잘못된 객체 포함 케이스

---

## Step 5. 오답 분석

- [ ] Top-10 오답 클래스 추출
- [ ] 오답 유형 분류

| 오답 유형 | 설명 |
|-----------|------|
| bbox_miss | GT bbox 있으나 Pred 없음 |
| partial_crop | bbox가 알약 일부만 포함 |
| visually_similar | 외형 유사 클래스 혼동 |
| imprint_miss | 각인 인식 실패 |
| background_noise | 배경 노이즈로 인한 오분류 |
| low_freq_class | 학습 데이터 부족 클래스 오답 |
| bbox_error_propagation | Stage 1 bbox 오차로 인한 crop 품질 저하 |

---

## Step 6. Source 분리 분석

- [ ] raw 이미지 기준 GT / Pred 성능
- [ ] external 이미지 기준 GT / Pred 성능
- [ ] raw vs external 성능 차이 (컷오프: 10%p 이상 차이 시 Stage 1 증강 재설계)

---

## Step 7. DE (소원) 연계 전달

분석 완료 후 아래 형식으로 DE (소원)에게 전달

---

## Step 8. Kaggle mAP 미회복 시 추가 점검

- [ ] class index / category_id mapping mismatch 확인
- [ ] Kaggle 57-class 커버리지 확인
- [ ] submission 샘플 10개 수동 검증
- [ ] 내부 metric vs Kaggle metric 정의 차이 확인

---

## 분석 결과 기록

| 항목 | GT 기준 | Pred 기준 | Gap | 판단 |
|------|---------|-----------|-----|------|
| Top-1 Accuracy | - | - | - | - |
| Top-5 Accuracy | - | - | - | - |
| raw Top-1 | - | - | - | - |
| external Top-1 | - | - | - | - |

**병목 위치**: (Stage 1 / Stage 2 / 전체 안정)

**EL (도혁) 코멘트**:

**PM (호정) 보고 필요 여부**: (Y / N)