# Stage 1 신뢰성 검증 체크리스트

> 작성: EL (도혁)
> mAP@50 0.9947은 external 단일 알약 편향으로 과대평가 가능성 있음
> 아래 항목으로 실전 성능 별도 검증 필요
> 현재 상태: 미실행 — EfficientNet-B2 학습 완료 후 진행 예정 (2026-05-03 기준)

---

## Step 1. Raw vs External 분리 성능 확인

- [ ] raw only mAP@50 측정 (다중 알약 이미지 기준)
- [ ] external only mAP@50 측정 (단일 알약 이미지 기준)
- [ ] raw vs external mAP 차이 계산

판단 기준:
- raw mAP가 전체 mAP보다 **10%p 이상 낮으면** Stage 1 증강 재설계 검토

---

## Step 2. Multi-Object Recall 확인

- [ ] raw 이미지에서 알약 여러 개 중 몇 개 탐지되는지 확인
- [ ] multi-object 이미지 기준 recall 측정
- [ ] miss detection 발생 케이스 수집

판단 기준:
- multi-object recall이 single-object recall보다 **5%p 이상 낮으면** Stage 1 증강 재설계 검토

---

## Step 3. False Positive 분석

- [ ] GT 없으나 Pred 있는 케이스 수 확인
- [ ] false positive 비율 측정
- [ ] 주요 발생 조건 정리 (배경 노이즈 / 유사 객체 등)

---

## Step 4. Confidence 분포 확인

- [ ] Stage 1 예측 confidence 분포 시각화
- [ ] 낮은 confidence (0.1~0.3) bbox 비율 확인
- [ ] confidence threshold 조정 필요 여부 판단

현재 설정: conf=0.10

---

## Step 5. Bbox 품질 확인

- [ ] bbox가 알약 전체를 포함하는지 확인
- [ ] partial crop 발생 비율 측정
- [ ] bbox 크기 분포 이상 여부 확인

---

## 검증 결과 기록

| 항목 | 결과 | 판단 |
|------|------|------|
| raw only mAP@50 | - | - |
| external only mAP@50 | - | - |
| raw vs external 차이 | - | - |
| multi-object recall | - | - |
| false positive 비율 | - | - |

**EL (도혁) 코멘트**:

**Stage 1 신뢰성 판단**: (신뢰 가능 / 재검증 필요 / 증강 재설계 필요)

**PM (호정) 보고 필요 여부**: (Y / N)