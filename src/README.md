# src/

재사용 가능한 Python 모듈을 보관합니다.  
notebooks 및 scripts에서 `from src.data import ...` 형태로 import하세요.

```
src/
├── __init__.py
├── models/
│   └── yolo_model.py   # YOLOModel 래퍼 (학습·추론·내보내기) ✅ 구현 완료
├── data/               # 데이터셋 클래스, 어노테이션 파서, 증강 파이프라인
├── training/           # 학습 설정, 콜백, 평가 함수
└── utils/              # 공통 유틸리티 (로거, 시각화, 파일 I/O)
```

## 모듈 사용 예시

```python
from src.models.yolo_model import YOLOModel

# config.yaml로 초기화 — 모델 로드, 시드 고정 자동 처리
model = YOLOModel("experiments/exp_20260420_baseline_yolo26n/config.yaml")

model.train()                          # 학습
model.predict("data/raw/test/")        # 추론 → results/predictions.json
model.export(format="onnx")            # ONNX 내보내기
```

## 코딩 규칙

- PEP 8 준수 (`ruff`, `black` 자동 포맷)
- 모든 public 함수·클래스에 docstring 작성
- 예외 처리 포함 필수
