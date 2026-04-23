# experiments/

실험별로 독립 디렉토리를 생성합니다.

## 네이밍 규칙

```
experiments/
└── exp_YYYYMMDD_<model>_<purpose>/   # 예: exp_20260422_yolov8s_baseline
    ├── config.yaml                    # 실험 설정 ✅ Git 관리
    ├── log/                           # 학습 로그  ❌ .gitignore
    ├── results/                       # mAP 곡선, 혼동행렬 ✅ 이미지만
    └── weights/                       # best.pt, last.pt ❌ .gitignore
```

### 네이밍 예시
- `exp_20260422_yolov8s_baseline`
- `exp_20260422_yolo11s_baseline`
- `exp_20260422_yolo26s_baseline`
- `exp_20260423_yolo26s_augmentation`
- `exp_20260424_yolo26s_musgd`

### 주의사항
- 한 번에 하나의 변수만 변경
- 실험명에 변경한 핵심 변수 반영
- 날짜는 실험 시작일 기준

## 결과 기록 방법

1. 실험 완료 후 `config.yaml`에 최종 mAP 메모 추가
2. 결과 이미지(`results/`) 스냅샷 커밋
3. 가중치는 Google Drive / Kaggle Dataset에 별도 보관 후 링크 공유
4. 실험 로그는 `experiment_log_template.md` 양식에 맞춰 기록

## 현재 실험 목록

| 실험 | 날짜 | 담당 | mAP@0.5 | 비고 |
|------|------|------|---------|------|
| exp_20260420_yolo26n_baseline | 2026-04-20 | 승준 | - | 진행 예정 |
| exp_20260422_yolov8s_baseline | 2026-04-22 | 도혁 | - | 진행 예정 |
| exp_20260422_yolo11s_baseline | 2026-04-22 | 도혁 | - | 진행 예정 |

## 실험 네이밍 규칙 적용 정책

- 신규 실험부터는 예외 없이 `exp_YYYYMMDD_<model>_<purpose>` 형식을 적용한다
- 기존 legacy 폴더명(exp_baseline_yolov8s 등)은 순차적으로 정리한다
- 정리 전 legacy 실험은 예외로 두되 신규 실험에서는 반드시 규칙을 준수한다
