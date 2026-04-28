# 💊 경구약제 객체 인식 (pill-detection-team2)

K-DT AI 엔지니어 과정 초급 팀 프로젝트 — 경구약제 이미지에서 알약의 **클래스(약품명)** 와 **바운딩 박스**를 검출하는 Object Detection 태스크입니다.

> **기간**: 2026-04-20 ~ 05-11 (3주)  
> **대회**: Private Kaggle Competition + GitHub repo 제출

---

## 팀 구성

| 이름 | 역할 | 담당 |
| --- | --- | --- |
| 호정 | PM (Project Manager) | 일정 관리, 이슈 조율, PR 리뷰 |
| 승준 | ME (Model Engineer) | 모델 설계 및 파이프라인 구현 |
| 도혁 | EL (Experiment Lead) | 실험 설계, config 관리, 결과 비교 |
| 소원 | DE (Data Engineer) | 데이터 수집, 전처리, EDA |
| 찬우 | AL (Analysis Lead) | 결과 분석, 시각화, 보고서 |

---

## 프로젝트 파이프라인

```
2-Stage Detection & Classification Pipeline

Stage 1 - Detection (YOLO)
  입력: 원본 이미지 (raw + external 통합)
  모델: 모델: YOLOv26n (Baseline) → YOLOv26s (HPO P2 후보)
  출력: 알약 위치 bbox 좌표 (단일 클래스 pill)

        ↓ bbox 좌표 전달

Stage 2 - Classification
  입력: Stage 1 bbox 기반 크롭 이미지
  모델: ResNet50 (Baseline) → EfficientNet-B2 → EfficientNetV2-S
  출력: 브랜드명 분류 (371 클래스)
```

> 학습 시 GT bbox 기준 크롭 / 추론 시 Stage 1 예측 bbox 기준 크롭

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
git clone https://github.com/Codeit-Part2-2Team/codeit-part2-2team-project.git
cd codeit-part2-2team-project

# 가상환경 생성 (conda 권장)
conda create -n codeit-part2-2team-project python=3.10 -y
conda activate codeit-part2-2team-project

pip install -r requirements.txt
pip install -e .
```

`pip install -e .`는 개발 중인 현재 프로젝트를 Python 환경에 등록합니다.
노트북 또는 다른 작업 디렉터리에서 실행해도 `from src...` import를 안정적으로 사용할 수 있습니다.

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

### 통합 실행

```bash
# 학습: Stage 1 → Stage 2 순차 실행
python scripts/pipeline/run_train.py \
    --stage1-config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
    --stage2-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --data          data/processed/dataset.yaml \
    --crops         data/processed/crops/

# 추론: Stage 1 추론 → 크롭 → Stage 2 추론 → submission.csv
python scripts/pipeline/run_predict.py \
    --stage1-config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
    --stage2-config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --source        data/raw/test/ \
    --output        submissions/submission.csv
```

### Stage 1 단독 실행

```bash
# 학습
python scripts/train.py \
    --config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
    --data data/processed/dataset.yaml

# 검증
python scripts/validate.py \
    --config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml

# 추론
python scripts/predict.py \
    --config experiments/exp_20260420_baseline_yolo26n/s1_config.yaml \
    --source data/raw/test/

python scripts/make_submission.py \
    --manifest  experiments/exp_20260420_baseline_yolo26n/stage1_crops/crops_manifest.json \
    --s2-preds  experiments/exp_20260420_baseline_yolo26n/stage2_predictions.json \
    --class-map data/processed/kaggle_class_map.json \
    --output    submissions/submission.csv
```

### Stage 2 단독 실행

```bash
# 학습 (GT 크롭 이미지 필요)
python scripts/pipeline/stage2_train.py \
    --config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --data data/processed/crops/

# 추론
python scripts/pipeline/stage2_predict.py \
    --config experiments/exp_20260420_baseline_yolo26n/s2_config.yaml \
    --source data/processed/crops/inference/
```

> `--weights`, `--output` 미지정 시 config의 `output.project/output.name` 기반으로 경로를 자동 조합합니다.

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
