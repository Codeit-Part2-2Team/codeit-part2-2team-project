# 2-Stage 파이프라인 튜토리얼 가이드

> 약물 탐지 프로젝트의 핵심 파이프라인을 이해하고 직접 실행해보는 가이드입니다.  
> **대상**: 팀 신입, ML 초보자 ~ 데이터 사이언티스트  
> **소요 시간**: 약 30분 (GPU 환경 기준)

---

## 🚀 빠른 시작 (5분)

### 1단계: 노트북 열기
```bash
cd /home/n132/Projects/Project1/Main
jupyter notebook notebooks/04_pipeline_tutorial.ipynb
```

### 2단계: 셀 순서대로 실행
`Shift + Enter` 로 위에서 아래로 순서대로 실행합니다.

### 3단계: 결과 시각화 확인
최종 결과에서 약물 위치(bbox) + 약품명 + 신뢰도가 시각화됩니다.

---

## 📌 핵심 개념: 왜 2-Stage인가?

### 문제 상황
```
원본 이미지
  ↓
  내용: 여러 종류의 알약이 섞여 있다
  
질문:
  1. 어디에 알약이 있는가?  (위치 탐지)
  2. 각 알약이 무엇인가?    (약품명 분류)
```

### 해결책: 2-Stage 파이프라인

```
┌─ Stage 1: 탐지 (Detection) ────────┐
│                                     │
│  모델: YOLO26n                      │
│  입력: 원본 이미지                   │
│  출력: bbox 리스트                   │
│  {image_id, detections[bbox, score]}│
│                                     │
└────────┬──────────────────────────┘
         │ crop 추출 (bbox 기준)
         ↓
┌─ Stage 2: 분류 (Classification) ───┐
│                                     │
│  모델: ResNet50                     │
│  입력: crop 이미지들                │
│  출력: 약품명 + 신뢰도              │
│  {crop_id, class_name, score}      │
│                                     │
└────────┬──────────────────────────┘
         │
         ↓
    최종 결과
  [약물위치, 약품명, 신뢰도]
```

---

## 🎯 파이프라인 상세 흐름

### Stage 1: 약물 탐지

#### 1-1. 모델 학습
```
데이터: data/processed/images/train/
레이블: YOLO format (class cx cy w h)
모델: yolo26n (pretrained)
설정: s1_config.yaml

↓ 학습 완료

출력: experiments/{EXP_NAME}/weights/best.pt
```

**주요 설정값**:
- `epochs: 100` — 100번 반복 학습
- `batch: 16` — 한 번에 16장씩 처리
- `imgsz: 640` — 모든 이미지 640×640으로 리사이즈

#### 1-2. 모델 검증
```
데이터: data/processed/images/val/

↓ 평가

메트릭:
  · Precision (P): 탐지한 것 중 정답 비율
  · Recall (R): 정답 중 탐지한 비율
  · mAP50: IoU 0.5 기준 평균 정밀도
  · mAP50-95: IoU 0.5~0.95 기준 평균 정밀도 (더 엄격)

목표: mAP50 > 0.95
```

#### 1-3. 추론 (Inference)
```
입력: data/processed/images/val/*.jpg (검증 이미지)

↓ 모델 실행

출력: val_predictions.json
{
  "image_id": "IMG_001",
  "detections": [
    {
      "bbox": [100, 50, 180, 120],  // [x1, y1, x2, y2] (픽셀 좌표)
      "score": 0.95                  // 신뢰도 (0~1)
    },
    { ... }
  ]
}
```

#### 1-4. Crop 추출
```
입력: val_predictions.json + 원본 이미지

↓ bbox를 기준으로 이미지 자르기

출력:
  · stage1_crops/*.jpg (개별 crop 이미지)
  · crops_manifest.json (메타데이터: image_id, crop_id, bbox 등)
```

---

### Stage 2: 약물 분류

#### 2-0. GT Crop 생성 (최초 1회만)
```
입력: data/processed/labels/ (GT label txt)
      data/processed/images/

↓ GT bbox를 기준으로 crop 생성

출력:
  · data/processed/crops/train/*.jpg
  · data/processed/crops/val/*.jpg
  · 폴더명 = 약품명 (class label)

예시:
  data/processed/crops/train/
    ├── 아스피린_001.jpg
    ├── 아스피린_002.jpg
    ├── 감기약_001.jpg
    └── ...
```

#### 2-1. 분류 모델 학습
```
입력: data/processed/crops/{train,val}/
      (crop 이미지들 + 폴더명으로 자동 레이블링)

↓ ResNet50 학습

출력: experiments/stage2_classifier/weights/best.pt

학습 시간: ~10분 (GPU)
```

#### 2-2. 분류 추론
```
입력: stage1_crops/*.jpg (Stage 1에서 추출한 crop들)

↓ 분류 모델 실행

출력: stage2_predictions.json
{
  "crop_id": "IMG_001_0",
  "class_name": "아스피린",
  "score": 0.92
}
```

---

## 📊 입출력 형식 정리

| 단계 | 입력 | 출력 | 형식 |
|------|------|------|------|
| **S1 학습** | `data/processed/images/train/` | `best.pt` | `.pt` (PyTorch) |
| **S1 검증** | `data/processed/images/val/` | 메트릭 | Console |
| **S1 추론** | `data/processed/images/val/` | `val_predictions.json` | JSON |
| **Crop 추출** | `val_predictions.json` | `stage1_crops/` | JPG |
| **S2 학습** | `data/processed/crops/train/` | `best.pt` | `.pt` |
| **S2 추론** | `stage1_crops/` | `stage2_predictions.json` | JSON |

---

## 💡 성능 메트릭 해석

### Stage 1 (탐지)

```
Precision = TP / (TP + FP)
예: 탐지한 100개 중 95개가 정답 → Precision = 0.95

Recall = TP / (TP + FN)
예: 정답 100개 중 95개를 탐지 → Recall = 0.95

mAP50 = 평균 정밀도 (IoU threshold = 0.5)
  · IoU (Intersection over Union): 예측 bbox와 GT bbox의 겹침 비율
  · IoU > 0.5 이면 "정답"으로 간주
  
mAP50-95 = 더 엄격한 평가 (IoU 0.5~0.95 모두 포함)
```

**목표값**:
- Stage 1: `mAP50 > 0.95` (위 약물 하나씩 탐지)

### Stage 2 (분류)

```
Top-1 Accuracy = 1순위 예측이 정답인 비율
Top-5 Accuracy = 상위 5순위에 정답이 포함된 비율
```

**목표값**:
- Stage 2: `Top-1 > 0.80`, `Top-5 > 0.95`

---

## ⚙️ 실행 환경 체크리스트

실행 전 반드시 확인하세요:

- [ ] GPU 할당됨: `nvidia-smi` 로 확인
- [ ] Conda 환경 활성화: `conda activate codeit-part2-2team-project`
- [ ] 필수 라이브러리 설치: `pip install -r requirements.txt`
- [ ] 데이터셋 존재:
  - `data/processed/images/{train,val,test}/` ✓
  - `data/processed/labels/{train,val}/` ✓
  - `data/processed/dataset.yaml` ✓

---

## 🔧 트러블슈팅

### 문제 1: GPU 메모리 부족
```
CUDA out of memory
```

**해결**:
```yaml
# s1_config.yaml 수정
train:
  batch: 8  # 16 → 8 (배치 크기 감소)
```

### 문제 2: 데이터셋 경로 오류
```
FileNotFoundError: data/processed/images/train
```

**해결**:
```bash
# 프로젝트 루트에서 실행하는지 확인
pwd  # /home/n132/Projects/Project1/Main 이어야 함
```

### 문제 3: 라이브러리 import 오류
```
ModuleNotFoundError: No module named 'koreanize_matplotlib'
```

**해결**:
```bash
pip install koreanize-matplotlib
```

---

## 📈 다음 단계

### 1단계: 결과 분석
- ✅ 파이프라인 이해 완료
- 📊 성능 메트릭 확인 (mAP, Accuracy)
- 🔍 실패 사례 분석 (어디서 틀렸는가?)

### 2단계: 개선 실험
1. **데이터 증강**: `s1_config.yaml` 의 `albumentations` 설정 조정
2. **모델 업그레이드**: `yolo26n` → `yolo26m` (더 큰 모델)
3. **Hyperparameter 조정**: `epochs`, `lr0` (학습률), `batch` 등

### 3단계: 새 실험 시작
```bash
# 새 config 작성
experiments/exp_20260425_yolo26m_augmented/config.yaml

# 학습 실행
python scripts/train.py --config experiments/exp_20260425_yolo26m_augmented/config.yaml
```

---

## 📚 참고 자료

| 자료 | 위치 | 설명 |
|------|------|------|
| **Config 참조** | `docs/config_reference.md` | 각 설정값의 의미와 범위 |
| **데이터셋 규칙** | `docs/dataset_rules.md` | 데이터셋 버전 관리 규칙 |
| **실험 관리** | `experiments/README.md` | 실험 네이밍 및 기록 방법 |
| **스크립트 가이드** | `scripts/README.md` | 학습/추론/제출 스크립트 사용법 |
| **노트북 가이드** | `notebooks/README.md` | 각 노트북별 목적 |

---

## ❓ FAQ

**Q: 다시 처음부터 학습하고 싶어요**
```bash
# weights 폴더 삭제
rm -rf experiments/{EXP_NAME}/weights

# 노트북에서 Step 1-1 셀부터 다시 실행
```

**Q: 중간에 멈췄는데 이어서 할 수 있나요?**
```python
# 아니요. 매번 Step 1-1부터 새로 학습해야 합니다.
# (기존 가중치가 있으면 덮어씌워집니다)
```

**Q: 내 모델이 성능이 안 좋아요**
```
1. Stage 1 mAP50 < 0.90이면:
   → 데이터 증강 강화 (augmentation 설정)
   → 모델 크기 늘리기 (yolo26m/l)
   → Epoch 더 학습

2. Stage 2 Top-1 < 0.70이면:
   → GT crop 품질 확인 (시각화)
   → 클래스 불균형 확인 (train/val crop 수 비교)
   → 학습률/배치 크기 조정
```

---

## 📞 추가 질문

문제 발생 시:
1. 위 FAQ 확인
2. `docs/config_reference.md` 에서 설정값 확인
3. 팀 Slack `#기술-문제` 채널에 질문
4. 이슈 등록: https://github.com/Codeit-Part2-2Team/codeit-part2-2team-project/issues

---

**작성**: 2026-04-24  
**최종 수정**: 2026-04-24
