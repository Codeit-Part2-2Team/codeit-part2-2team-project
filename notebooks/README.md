# notebooks/

| 노트북 | 목적 | 담당 |
|--------|------|------|
| `01_eda.ipynb` | 데이터 탐색·클래스 분포 분석 | 소원 |
| `02_yolo_detection_dataset_build` | stage 1 yolo 모델 Dataset 구축 | 소원 |
| `03_yolo_classification_dataset_build` | stage 2 yolo 모델 Dataset 구축 | 소원 |
| `04_baseline.ipynb` | YOLOv8n Baseline 학습 |  |
| `05_experiments.ipynb` | 실험 로그·비교 |  |
| `06_inference.ipynb` | 추론·결과 시각화 | 공용 |
| `pipeline_tutorial.ipynb` | 2-Stage 파이프라인 튜토리얼 | 승준 |

## 규칙

- 파일명 번호 접두사 필수 (`01_`, `02_`, …)
- 제출 전 **Kernel → Restart & Run All** 확인 필수
- 출력 포함 상태로 커밋 (재현 증거 보존)
- 무거운 학습 코드는 `scripts/train.py`로 분리하고 노트북은 호출만
