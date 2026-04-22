# scripts/

CLI 실행 스크립트입니다. **프로젝트 루트**에서 실행하세요.

| 스크립트 | 설명 | 담당 |
|----------|------|------|
| `train.py` | 모델 학습 | 승준 |
| `predict.py` | 추론 실행 | 승준 |
| `make_submission.py` | Kaggle 제출 파일 생성 | 도혁 |
| `convert_annotations.py` | raw/external 어노테이션을 YOLO 라벨로 변환 | 소원 |

## 실행 예시

```bash
# 학습
python scripts/train.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml

# 학습 (dataset.yaml 덮어쓰기, CPU 실행)
python scripts/train.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --data data/processed/dataset.yaml \
    --device cpu

# 추론
python scripts/predict.py \
    --config experiments/exp_20260420_baseline_yolo26n/config.yaml \
    --weights experiments/exp_20260420_baseline_yolo26n/weights/best.pt \
    --source data/raw/test/

# 제출 파일 생성
python scripts/make_submission.py \
    --predictions results/predictions.json \
    --output submissions/submission.csv

# 어노테이션 변환
python scripts/convert_annotations.py \
    --project-root .
```

## CLI 인자 목록

### train.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | - | config.yaml 경로 |
| `--data` | | config 값 사용 | dataset.yaml 경로 (덮어쓰기) |
| `--device` | | `0` | GPU 번호 또는 `cpu` |

### predict.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--config` | ✅ | - | config.yaml 경로 (val 파라미터 사용) |
| `--weights` | ✅ | - | best.pt 경로 |
| `--source` | ✅ | - | 이미지 디렉터리 경로 |
| `--output` | | `results/predictions.json` | 출력 경로 |

### make_submission.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--predictions` | ✅ | - | predictions.json 경로 |
| `--output` | ✅ | - | submission.csv 저장 경로 |

### convert_annotations.py

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `--project-root` |  | . | 프로젝트 루트 경로 |
| `--raw-annotation-dir` |  | raw/extracted/sprint_ai_project1_data/train_annotations | raw json 어노테이션 폴더 |
| `--raw-output-dir` |  | processed/raw_yolo_labels | raw 변환 txt 저장 경로 |
| `--ext-label-roots` |  | external/extracted/train/labels external/extracted/valid/labels external/extracted/test/labels | external 라벨 폴더 목록 |
| `--ext-output-dir` |  | processed/external_labels_unified | external 변환 txt 저장 경로 |
| `--class-id` |  | 0 | 단일 클래스 id (pill) |