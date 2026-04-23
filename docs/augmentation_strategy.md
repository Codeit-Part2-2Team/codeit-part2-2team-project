# Albumentations 증강 전략

이 문서는 Stage 1 Detection과 Stage 2 Classification에서 사용할 **Albumentations 기반 증강 설계 원칙**을 정리합니다.  
핵심 목표는 실험자가 **필요한 transform만 config에 추가**하고, 각 transform의 확률과 세부 파라미터를 직접 조절할 수 있게 만드는 것입니다.

---

## 설계 원칙

- Stage 1은 `augment`와 `albumentations`를 함께 사용할 수 있다
- `augment`는 YOLO 기본 증강, `albumentations`는 촬영 환경 시뮬레이션과 세밀 제어 담당
- 각 transform은 독립적인 optional block으로 정의한다
- Stage 1은 bbox를 보존해야 하므로 bbox-aware transform만 허용한다
- Stage 2는 bbox 대신 crop robustness를 높이는 변환을 사용한다

---

## 공통 config 구조

```yaml
albumentations:
  bbox:                  # detection에서만 사용
    format: yolo
    label_fields: [class_labels]
    min_visibility: 0.10
    clip: true
  <transform_name>:
    p: 0.5
    ...
```

### 제어 레벨

| 레벨 | 예시 | 역할 |
|------|------|------|
| 섹션 | `albumentations` | Albumentations 사용 여부 |
| 변환 | `gaussian_blur.p` | 개별 transform 실행 확률 |
| 파라미터 | `blur_limit`, `rotate_limit` | transform 강도와 범위 제어 |

### 해석 규칙

- `albumentations` 섹션이 없으면 전체 미적용으로 간주한다
- transform block이 없으면 해당 transform은 비활성화로 간주한다
- transform block이 있으면 해당 transform은 활성화로 간주한다
- 생략한 세부 파라미터는 구현 기본값을 사용한다
- 모르는 transform 이름은 무시하지 말고 에러로 처리하는 것을 권장한다

---

## Stage 1 권장 정책

Stage 1의 목표는 **알약 존재 여부와 위치 탐지**이므로, 형태를 크게 훼손하지 않으면서도 촬영 환경 변화와 검출 위치 흔들림에 강해지는 방향으로 잡습니다.

### 추천 파이프라인

```yaml
albumentations:
  bbox:
    format: yolo
    label_fields: [class_labels]
    min_visibility: 0.10
    clip: true
  horizontal_flip:
    p: 0.5
  vertical_flip:
    p: 0.2
  random_rotate90:
    p: 0.2
  shift_scale_rotate:
    p: 0.6
    shift_limit: 0.03
    scale_limit: 0.08
    rotate_limit: 8
  brightness_contrast:
    p: 0.5
    brightness_limit: 0.15
    contrast_limit: 0.15
  jpeg_compression:
    p: 0.3
    quality_lower: 85
    quality_upper: 100
  gaussian_blur:
    p: 0.2
    blur_limit: [3, 5]
  gauss_noise:
    p: 0.15
    std_range: [0.01, 0.03]
```

### 변환별 의도

| 범주 | 변환 | 권장값 | 의도 |
|------|------|--------|------|
| `geometry` | `horizontal_flip` | `p=0.5` | 좌우 배치 편향 완화 |
| `geometry` | `vertical_flip` | `p=0.2` | 촬영 방향 다양성 반영 |
| `geometry` | `random_rotate90` | `p=0.2` | 90/180/270도 방향 변화 반영 |
| `geometry` | `shift_scale_rotate` | `p=0.6`, `shift=0.03`, `scale=0.08`, `rotate=8` | 프레임 내 위치 변화, 거리 변화, 약한 각도 변화 반영 |
| `photometric` | `brightness_contrast` | `p=0.5`, `limit=0.15` | 밝기와 대비 변화 대응 |
| `photometric` | `jpeg_compression` | `p=0.3`, `quality=85~100` | 메신저 전송, 캡처본, 압축 이미지 대응 |
| `degradation` | `gaussian_blur` | `p=0.2`, `blur=3~5` | 초점 불량, 흔들림 촬영 대응 |
| `degradation` | `gauss_noise` | `p=0.15`, `std=0.01~0.03` | 센서 노이즈, 저화질 촬영 대응 |

### 운영 메모

- config는 flat하게 유지하지만, 비슷한 transform을 동시에 크게 올리면 과증강이 되기 쉽다
- `brightness_contrast`, `random_gamma`, `jpeg_compression`은 동시에 강하게 올리지 않는 것을 권장한다
- `gaussian_blur`, `motion_blur`, `downscale`도 동시에 크게 올리지 않는 것을 권장한다
- Stage 1에서는 YOLO `augment`와 의미가 겹치는 항목을 중복 정의하지 않는 것이 중요하다

### Stage 1 추가 후보 변환

| transform | 권장 범위 | 추천 이유 | 주의점 |
|-----------|-----------|-----------|--------|
| `downscale` | `scale_min=0.7~0.85`, `scale_max=0.9~0.95` | 메신저 전송본, 리사이즈된 이미지, 저해상도 촬영 대응 | compression과 동시에 강하게 쓰면 정보 손실이 과해질 수 있음 |
| `motion_blur` | `blur_limit=3~7` | 손떨림, 이동 중 촬영, 초점 흔들림 대응 | 소형 객체 경계가 흐려지므로 낮은 확률 권장 |
| `random_gamma` | `gamma_limit=80~120` | 실내외 조명 차이, 노출 편차 대응 | brightness_contrast와 함께 과도하게 키우지 않기 |
| `perspective` | `scale=0.02~0.05` | 살짝 비스듬한 촬영 각도 반영 | 강하게 주면 bbox 분포가 부자연스러워짐 |

---

## Stage 2 권장 정책

Stage 2는 **알약 crop 분류**가 목적이므로, Stage 1보다 geometry 증강을 보수적으로 가져갑니다.  
좌우반전이나 과한 회전은 각인, 로고, 문자 방향 정보를 망가뜨릴 수 있으므로 기본 정책에서는 제외합니다.

### 추천 파이프라인

```yaml
albumentations:
  brightness_contrast:
    p: 0.5
    brightness_limit: 0.15
    contrast_limit: 0.15
  jpeg_compression:
    p: 0.3
    quality_lower: 85
    quality_upper: 100
  gaussian_blur:
    p: 0.2
    blur_limit: [3, 5]
  bbox_jitter:
    p: 0.5
    shift_limit: 0.03
    scale_limit: 0.05
    rotate_limit: 5
```

### 변환별 의도

| 범주 | 변환 | 권장값 | 의도 |
|------|------|--------|------|
| `photometric` | `brightness_contrast` | `p=0.5` | 조명 변화 대응 |
| `photometric` | `random_gamma` | `p=0.2` | 노출 및 감마 변화 대응 |
| `photometric` | `clahe` | `p=0.2` | 저대비 crop에서 표면 대비와 각인 가독성 보완 |
| `photometric` | `jpeg_compression` | `p=0.3` | 압축 아티팩트 대응 |
| `photometric` | `downscale` | `p=0.15` | 저해상도 crop 품질 저하 대응 |
| `photometric` | `gaussian_blur` | `p=0.2` | 초점 불량 대응 |
| `photometric` | `motion_blur` | `p=0.1` | 아주 약한 손떨림 blur 대응 |
| `crop_robustness` | `bbox_jitter` | `p=0.5`, `shift=0.03`, `scale=0.05`, `rotate=5` | Stage 1 bbox 예측 오차 시뮬레이션 |

### Stage 2 추가 후보 변환

| transform | 권장 범위 | 추천 이유 | 주의점 |
|-----------|-----------|-----------|--------|
| `clahe` | `clip_limit=1~4`, `tile_grid_size=[8,8]` | 어두운 표면이나 대비 낮은 각인 정보 보완 | 과하면 표면 질감이 비현실적으로 강조될 수 있음 |
| `downscale` | `scale_min=0.8~0.9`, `scale_max=0.9~0.95` | 작은 crop, 재압축된 이미지, 서비스 입력 품질 저하 대응 | Stage 2는 디테일 민감하므로 약하게만 사용 |
| `motion_blur` | `blur_limit=3~5` | 실사용 촬영 흔들림 반영 | 문자/각인이 손상되지 않도록 저확률 권장 |
| `random_gamma` | `gamma_limit=90~110` | 조명 편차 큰 crop 대응 | brightness_contrast와 동시 강화 주의 |

---

## 실험자 튜닝 가이드

### 강도를 올리고 싶을 때

- 하나의 transform `p`를 크게 올리기 전에 비슷한 종류의 transform이 이미 켜져 있는지 먼저 본다
- geometry보다 photometric / degradation 계열을 먼저 늘리는 편이 안전하다
- `rotate_limit`, `scale_limit`, `std_range`는 작은 폭으로 점진 조정한다

### 강도를 낮추고 싶을 때

- blur, noise, compression 같이 화질 열화를 만드는 transform부터 먼저 낮춘다
- Stage 2에서는 글자 방향 정보가 손상될 수 있는 transform을 우선 제거한다
- 효과가 겹치는 transform 두 개를 동시에 쓰고 있다면 하나를 끄는 것이 가장 단순한 완화 방법이다

### 로그에 반드시 남길 항목

- 사용한 transform 목록
- 각 transform의 `p`
- 범위형 파라미터 (`brightness_limit`, `quality_lower`, `rotate_limit` 등)

---

## 권장 실험 순서

1. Stage 1 베이스라인은 `augment`만 고정하고 기준 score를 먼저 확보한다
2. Stage 1에 `horizontal_flip`, `brightness_contrast`, `jpeg_compression` 정도의 기본 transform부터 추가한다
3. Stage 1에 `gaussian_blur` 또는 `gauss_noise`를 추가해 화질 열화 내성을 본다
4. 필요 시 Stage 1에 `downscale` 또는 약한 `motion_blur`를 추가한다
5. 비스듬한 촬영이 많다고 판단되면 Stage 1에 약한 `perspective`를 마지막으로 실험한다
6. Stage 2는 `brightness_contrast + jpeg_compression + bbox_jitter`부터 시작한다
7. Stage 2에서 어려운 샘플이 저대비 위주면 `clahe`, 품질 저하 위주면 `downscale`, 흔들림 위주면 약한 `motion_blur`를 순차 추가한다

## 운영 권장안

- 실험 비교 시에는 한 번에 한두 개 transform만 바꾸고 나머지는 고정한다
- `downscale + compression + blur`를 동시에 크게 올리는 조합은 피한다
- Stage 2에서는 flip과 강한 geometry transform은 기본 정책에서 제외한다
