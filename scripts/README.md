# scripts/

CLI 실행 스크립트입니다. **프로젝트 루트**에서 실행하세요.

| 스크립트 | 설명 | 담당 |
|----------|------|------|
| `train.py` | 모델 학습 | 승준 |
| `predict.py` | 추론 실행 | 승준 |
| `make_submission.py` | Kaggle 제출 파일 생성 | 도혁 |
| `convert_annotations.py` | 어노테이션 포맷 변환 | 소원 |

## 실행 예시

```bash
# 학습
python scripts/train.py --config experiments/baseline/config.yaml

# 추론
python scripts/predict.py \
    --weights experiments/baseline/weights/best.pt \
    --source data/raw/test/

# 제출 파일 생성
python scripts/make_submission.py \
    --predictions experiments/baseline/results/predictions.json \
    --output submissions/submission.csv
```
