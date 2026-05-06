# 신규 데이터셋 Baseline 결과 해석 체크리스트

> 작성: EL (도혁)
> 승준 새 baseline 결과 나오는 즉시 실행
> PM 부재 중 단정 표현 금지 — 확인 후 보고 원칙

---

## Step 1. 기본 지표 확인

| 항목 | 기존 baseline | 새 baseline | 비고 |
|------|-------------|-------------|------|
| 전체 Top-1 Acc (val) | 0.9031 | 0.8750 | 하락 but Kaggle LB 상승 (정상) |
| 전체 Top-5 Acc (val) | 0.9774 | 0.9612 | - |
| Kaggle LB mAP | 0.68914 | 0.90845 (Public LB) / 내부 E2E test 0.9458 | 핵심 지표 — 회복 완료 |
| 학습 시간 | 119.7분 | - | - |

> 전체 Top-1 하락이 있어도 Kaggle LB 상승이면 긍정적으로 판단

---

## Step 2. 병목 판단 기준표

| 상황 | 판단 | 다음 액션 |
|------|------|---------|
| 전체 Top-1 하락 + LB 상승 | 성공 가능성 높음 | HPO 진입 검토 |
| 전체 Top-1 하락 + LB 하락 | 원인 분석 필요 | mapping/split/coverage 점검 |
| 전체 Top-1 상승 + LB 하락 | target mismatch 의심 | Kaggle 56/59 기준 재확인 |
| 전체 Top-1 상승 + LB 상승 | 이상적 케이스 | HPO 진입 |
| loss 불안정 / val 발산 | baseline 불안정 | 데이터셋 재검토 |

---

## Step 3. Kaggle Target-Class 성능 확인

- [x] Kaggle 평가 기준 클래스 수 확정 완료 (57클래스, category_id 기준, 2026-04-30)
- [ ] Kaggle target class 중 새 데이터셋에서 커버되는 클래스 수 확인
- [ ] 기존 279클래스 중 Kaggle target에 포함된 클래스 성능 별도 확인

> Kaggle 평가 기준 클래스 수 57클래스로 확정 완료 (2026-04-30)

---

## Step 4. Missing Class 성능 확인 (Trigger 방식)

아래 이상 신호가 보이면 DE (소원)에게 split 방식 확인 요청

**이상 신호 체크리스트:**
- [ ] rare class (원본 1~3장)에서 val Acc가 비정상적으로 높은가 (1.0 가까움)
- [ ] rare class의 LB 기여가 거의 없는가
- [ ] 전체 Top-1은 높은데 Kaggle LB가 오히려 낮은가
- [ ] rare class와 일반 class 간 성능 격차가 비정상적으로 큰가

> PM (호정) 부재 중이므로 이상 신호 확인 시 PM 복귀 후 보고 원칙 유지

---

## Step 5. GT vs Pred Gap 확인

- [ ] GT bbox 기준 Top-1 확인
- [ ] Pred bbox 기준 Top-1 확인
- [ ] Gap 계산 (GT - Pred)

| 케이스 | 판단 |
|--------|------|
| Gap 5%p 이상 | Stage 1 병목 (bbox 오차 전파) |
| Gap 근소 | Stage 2 병목 또는 전체 안정 |

---

## Step 6. HPO 진입 판단

**HPO 진입 전 필수 확인:**
- [x] Kaggle 57클래스 기준 확정 완료 (2026-04-30)
- [ ] new baseline loss curve 정상 수렴 확인
- [ ] mapping mismatch 없음 확인
- [ ] GT vs Pred gap 확인
- [ ] rare class split 이상 신호 없음 확인

**모두 통과 시 Pilot HPO 진입 가능**

---

## 결과 기록

| 항목 | 결과 | 판단 |
|------|------|------|
| 전체 Top-1 (val) | 0.8750 | 기존 대비 하락, Kaggle LB 상승으로 정상 판단 |
| 전체 Top-5 (val) | 0.9612 | - |
| Kaggle LB | 0.90845 (Public LB) / 내부 E2E test 0.9458 | 회복 완료 (305cls, mapping 수정) |
| GT vs Pred gap | - | 측정 예정 |
| rare class 이상 신호 | 없음 | 정상 |
| HPO 진입 가능 여부 | 완료 | Grid Search + Optuna 진행 완료 |

**EL (도혁) 코멘트**: 305cls 기반 새 baseline 확정 완료. Kaggle Public LB 0.90845 / 내부 E2E test mAP 0.9458 회복. Grid Search 12조합 → Optuna 15 trials 완료 (Best Top-1 0.9041). GT vs Pred gap은 EfficientNet-B2 학습 완료 후 측정 예정.

**PM (호정) 보고 필요 여부**: Y (Optuna 결과 보고 완료 — PR #160)