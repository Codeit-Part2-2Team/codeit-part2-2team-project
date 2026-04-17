# notebooks/

| 노트북 | 목적 | 담당 |
|--------|------|------|
| `01_eda.ipynb` | 데이터 탐색·클래스 분포 분석 |  |
| `02_baseline.ipynb` | YOLOv8n Baseline 학습 |  |
| `03_experiments.ipynb` | 실험 로그·비교 |  |
| `04_inference.ipynb` | 추론·결과 시각화 | 공용 |

## 규칙

- 파일명 번호 접두사 필수 (`01_`, `02_`, …)
- 제출 전 **Kernel → Restart & Run All** 확인 필수
- 출력 포함 상태로 커밋 (재현 증거 보존)
- 무거운 학습 코드는 `scripts/train.py`로 분리하고 노트북은 호출만
