# experiments/

실험별로 독립 디렉토리를 생성합니다.

## 네이밍 규칙

```
experiments/
└── YYYYMMDD_<담당자>_<실험명>/      # 예: 20260422_승준_baseline_yolov8n
    ├── config.yaml                  # 실험 설정 ✅ Git 관리
    ├── log/                         # 학습 로그  ❌ .gitignore
    ├── results/                     # mAP 곡선, 혼동행렬 ✅ 이미지만
    └── weights/                     # best.pt, last.pt ❌ .gitignore
```

## 결과 기록 방법

1. 실험 완료 후 `config.yaml`에 최종 mAP 메모 추가
2. 결과 이미지(`results/`) 스냅샷 커밋
3. 가중치는 Google Drive / Kaggle Dataset에 별도 보관 후 링크 공유

## 현재 실험 목록

| 실험 | 날짜 | 담당 | mAP@0.5 | 비고 |
|------|------|------|---------|------|
| baseline_yolov8n | - |  | - | 진행 예정 |
