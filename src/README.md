# src/

재사용 가능한 Python 모듈을 보관합니다.  
notebooks 및 scripts에서 `from src.models import ...` 형태로 import하세요.

```
src/
├── __init__.py
├── models/
│   ├── model_yolo.py   # YOLOModel: YOLO 구체 구현 (초기화·가중치·raw 호출) ✅
│   └── predictor.py    # Predictor: 추론 퍼사드 (파싱·JSON 저장) ✅
├── training/
│   └── trainer.py      # Trainer: 학습·검증 퍼사드 ✅
├── data/               # 데이터셋 클래스, 어노테이션 파서, 증강 파이프라인
└── utils/
    ├── config.py        # load_config, validate_config, fix_seed ✅
    └── submission.py    # predictions_to_df, save_submission ✅
```

## 모듈 사용 예시

```python
from src.models.model_yolo import YOLOModel
from src.models.predictor import Predictor
from src.training.trainer import Trainer

# config.yaml로 초기화 — 모델 로드, 시드 고정 자동 처리
model = YOLOModel("experiments/exp_20260420_baseline_yolo26n/config.yaml")

# 학습
metrics = Trainer(model).train()
# → {"mAP50": float, "mAP50_95": float}

# 추론 → results/predictions.json
predictions = Predictor(model).predict("data/raw/test/")

# ONNX 내보내기
model.export(format="onnx")
```

## 코딩 규칙

- PEP 8 준수 (`ruff`, `black` 자동 포맷)
- 모든 public 함수·클래스에 docstring 작성
- 예외 처리 포함 필수
