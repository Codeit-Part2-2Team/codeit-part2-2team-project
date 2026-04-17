# src/

재사용 가능한 Python 모듈을 보관합니다.  
notebooks 및 scripts에서 `from src.data import ...` 형태로 import하세요.

```
src/
├── __init__.py
├── data/       # 데이터셋 클래스, 어노테이션 파서, 증강 파이프라인
├── models/     # 모델 래퍼 (Ultralytics 확장)
├── training/   # 학습 설정, 콜백, 평가 함수
└── utils/      # 공통 유틸리티 (로거, 시각화, 파일 I/O)
```

## 코딩 규칙

- PEP 8 준수 (`ruff`, `black` 자동 포맷)
- 모든 public 함수·클래스에 docstring 작성
- 예외 처리 포함 필수
