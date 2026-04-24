# 데이터셋 배포 규칙

## 배경
학습 중 데이터셋이 변경되면 실험 결과의 재현성이 무너집니다.
어떤 데이터로 학습했는지 추적이 불가능해지므로 아래 규칙을 반드시 준수합니다.

## 데이터셋 버전 관리

| 버전 | 내용 | 상태 |
|------|------|------|
| v1.0 | raw + external 통합 (단일 클래스 pill 기준) | 완료 (2026-04-22) |
| v1.1 | Stage 1용 수정본 | 완료 (2026-04-24) |
| v2.0 | 클래스 추가/수정 시 새 버전으로 분리 | 예정 |

## Freeze 규칙

데이터셋 Freeze 선언 이후 아래 항목 변경 금지:
- 클래스 추가 / 삭제 금지
- Train / Val / Test split 변경 금지
- Annotation 수정 금지
- 이미지 추가 / 삭제 금지

변경이 필요한 경우 반드시 새 버전(v2.0)으로 분리하여 관리합니다.

## 데이터셋 배포 프로세스

### 업로더 (DE 소원)
1. 데이터셋 압축 대상 경로 지정
2. zip 압축 진행
3. 구글 드라이브에서 기존 버전 테이블 다운로드
4. 버전 정보 기입 후 버전 테이블 업로드
5. zip 파일 업로드

### 다운로더 (EL 도혁)
1. 실험 전 버전 테이블 다운로드
2. 대상 버전 정보를 config.yaml에 반영
3. config.yaml 갱신 완료
4. 학습 프로세스 진행

### 버전 테이블 포맷 (CSV)
```csv
version, filename
0, 20260422_dataset_v1.0.zip
1, 20260424_dataset_v1.1.zip
```

### 주의사항
- zip 파일명은 날짜 기반으로 작성 (중복 방지)
- 버전 테이블은 구글 드라이브 API로 자동 공유
- 학습 시작 전 반드시 최신 버전 테이블 확인

## 학습 시 데이터셋 명시 규칙

모든 실험 config.yaml에 사용 데이터셋 버전 반드시 명시:

```yaml
data:
  yaml: data/processed/dataset_v1.0/dataset.yaml
  version: "v1.0"
  description: "raw + external 통합, 단일 클래스 pill"
```

## 현재 데이터셋 구조

```
data/
├── raw/          # 원본 데이터 (56클래스, 수정 금지)
├── external/     # 외부 데이터 (371클래스)
└── processed/
    └── dataset_v1.0/   # Freeze 완료 데이터셋
        ├── train/
        ├── val/
        ├── test/
        └── dataset.yaml
```

## Stage별 데이터 사용 기준

| Stage | 사용 데이터 | 클래스 | 비고 |
|-------|-----------|--------|------|
| Stage 1 (YOLO) | raw + external | pill (단일 클래스) | bbox 탐지만 담당 |
| Stage 2 (Classifier) | raw + external | 브랜드명 (371 클래스) | bbox 크롭 후 분류 |

## Stage 2 크롭 규칙

- 학습 시: raw 원본 annotation bbox 기준 자동 크롭
- 추론 시: Stage 1 예측 bbox 기준 크롭
- external 데이터도 동일하게 bbox 기준 크롭으로 통일

## 데이터셋 다운로드 링크

> 호정님 자동화 프로그램으로 관리되는 구글 드라이브 공유 폴더입니다.
> 공유 URL은 보안상 디코 채널에서 EL (도혁)에게 문의하세요.

| 버전 | 파일명 | 업로드일 | 비고 |
|------|--------|---------|------|
| v1.0 | yolo_dataset_v1.zip | 2026-04-22 | raw + external 통합, 단일 클래스 pill, nc=1 |
| v1.1 | yolo_dataset_v1.1.zip | 2026-04-24 | Stage 1용 수정본 |

>  최신 버전 다운로드 코드는 gdrive_downloader.ipynb 참고 (notebooks/ 폴더)

## Stage별 진행 가능 조건

- Stage 1은 클래스 수 확정과 무관하게 단일 클래스 pill 기준으로 즉시 진행 가능하다
- Stage 2는 클래스 매핑 확정 이후 최종 학습 기준을 고정한다
- 클래스 매핑 확정 완료 (2026-04-24, 371클래스, DE 소원)
- Stage 2 학습 시작 가능 조건 충족