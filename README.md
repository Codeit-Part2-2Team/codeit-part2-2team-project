# 💊 경구약제 객체 인식 (pill-detection-team2)

K-DT AI 엔지니어 과정 초급 팀 프로젝트 — 경구약제 이미지에서 알약의 **클래스(약품명)** 와 **바운딩 박스**를 검출하는 Object Detection 태스크입니다.

> **기간**: 2026-04-20 ~ 05-11 (3주)  
> **대회**: Private Kaggle Competition + GitHub repo 제출

---

## 팀 구성

| 이름 | 역할 |
|------|------|
| 호정 | |
| 승준 | |
| 도혁 | |
| 소원 | |
| 찬우 | |

---

## 프로젝트 구조

```
pill-detection-team2/
├── docs/          # 기획·의사결정 문서
├── data/          # 데이터 (실제 파일은 .gitignore 처리)
│   ├── raw/       # 원본 데이터
│   ├── processed/ # 전처리 완료 데이터
│   └── external/  # 외부 데이터
├── notebooks/     # Jupyter Notebooks (EDA, baseline, experiments, inference)
├── src/           # 공유 Python 모듈
│   ├── data/      # 데이터셋·증강 관련
│   ├── models/    # 모델 정의
│   ├── training/  # 학습 루프·콜백
│   └── utils/     # 공통 유틸리티
├── scripts/       # 실행 스크립트
├── experiments/   # 실험별 config, log, results, weights
├── reports/       # 보고서·발표자료 PDF
└── .github/       # PR 템플릿, Issue 템플릿
```

---

## 환경 설정

### 사전 요구사항

- Python 3.10+
- CUDA 12.x (선택, GPU 학습 시)

### 설치

```bash
git clone https://github.com/Yopkigom/codeit-part2-2team-project.git
cd codeit-part2-2team-project

# 가상환경 생성 (conda 권장)
conda create -n codeit-part2-2team-project python=3.10 -y
conda activate codeit-part2-2team-project

pip install -r requirements.txt
```

### 데이터 배치

```
data/
└── raw/
    ├── train/        # 학습 이미지
    ├── test/         # 테스트 이미지
    └── annotations/  # 어노테이션 파일
```

> ⚠️ 금지 데이터: AI Hub `TL_2_조합.zip`, `TS_2_조합.zip` 사용 불가

---

## 재현 방법

### Baseline 학습

```bash
python scripts/train.py \
    --config experiments/baseline/config.yaml \
    --data data/processed/dataset.yaml
```

### 추론 및 제출 파일 생성

```bash
python scripts/predict.py \
    --weights experiments/baseline/weights/best.pt \
    --source data/raw/test/

python scripts/make_submission.py \
    --predictions experiments/baseline/results/predictions.json \
    --output submissions/submission.csv
```

---

## 브랜치 전략

| 브랜치 | 용도 |
|--------|------|
| `main` | 항상 동작 가능한 상태 유지. PR + 1인 리뷰 필수 |
| `feat/<이름>-<기능>` | 기능 개발 |
| `exp/<이름>-<실험>` | 실험 전용 |

---

## 제출 규칙

- 팀당 **10회/일** 제한, **개인 제출 금지**
- 제출 전 `#제출이슈` 채널에 공지 후 진행

---

## 라이선스

본 프로젝트는 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)을 사용합니다.  
교육 목적 오픈소스 활용이므로 프로젝트 내 제약 없음.
