# Split 무결성 검증 체크리스트

> 작성: EL (도혁)
> 실행 조건: new_baseline_checklist.md Step 4 이상 신호 발견 시
> PM (호정) 부재 중 — 이상 신호 확인 후 PM 복귀 시 보고 원칙
> 현재 상태: 미실행 — 305cls baseline 결과 이상 신호 없음 확인 (2026-05-03 기준)
> 이상 신호 없음으로 split 무결성 검증 불필요 판단. HPO 진행 완료.

---

## Step 1. DE (소원)에게 확인 요청

**요청 타이밍**: 이상 신호 확인 후, PM 복귀 전 사전 확인 목적으로만 요청

**요청 목적**: baseline 결과 해석 시 leakage 또는 val 성능 착시 가능성 확인
**역할 범위**: 증강 정책 수정 요청이 아닌 split 방식 확인 요청

---

## Step 2. 확인 항목

### 2-1. Split 방식 확인

- [ ] train/val/test split이 원본 이미지 기준으로 먼저 나뉘었는가
- [ ] augmentation은 train에만 적용됐는가
- [ ] val/test에 random augmentation이 포함되어 있지 않은가
- [ ] 동일 원본에서 파생된 이미지가 train/val/test에 동시에 들어가지 않았는가

### 2-2. Rare Class 분포 확인

| class_name | 원본 수 | train | val | test | 비고 |
|------------|---------|-------|-----|------|------|
| - | - | - | - | - | - |

### 2-3. Augmentation Leakage 의심 확인

- [ ] 파일명 규칙 확인 (원본 ID가 파일명에 포함되어 있는가)
- [ ] 동일 원본 기반 이미지가 train/val에 동시에 존재하는가
- [ ] val/test 샘플이 train 증강 이미지와 시각적으로 유사한가

---

## Step 3. 판단 기준

| 확인 결과 | 판단 | 다음 액션 |
|-----------|------|---------|
| split 정상, leakage 없음 | val 성능 신뢰 가능 | HPO 진입 |
| val에 증강 이미지 포함 | val 성능 착시 가능성 | PM 보고 후 데이터셋 재구성 검토 |
| 동일 원본 train/val 혼재 | leakage 확정 | PM 보고 후 재split 필요 |
| rare class val 샘플 1장뿐 | metric 불안정 (정상) | 샘플 수 고려한 해석 필요 |

---

## Step 4. PM (호정) 보고 내용

**보고 시점**: PM 복귀 후 즉시

**보고 내용 템플릿:**

발견된 이상 신호:
- (내용 기입)

DE (소원) 확인 결과:
- (내용 기입)

EL 판단:
- (split 정상 / leakage 의심 / 재확인 필요)

다음 액션 제안:
- (HPO 진입 / 데이터셋 재검토 / 추가 확인 필요)

---

## 검증 결과 기록

| 항목 | 결과 | 판단 |
|------|------|------|
| split 방식 | - | - |
| val 증강 포함 여부 | - | - |
| leakage 여부 | - | - |
| rare class 분포 | - | - |
| 최종 판단 | - | - |

**EL (도혁) 코멘트**: 305cls baseline 결과 이상 신호 없음 확인. Split 무결성 검증 실행 조건 미충족으로 미실행. HPO 진행 완료 (Grid Search → Optuna → 모델 교체 → Final Model 확정).

**PM (호정) 보고 완료 여부**: Y (HPO 결과 보고 완료)