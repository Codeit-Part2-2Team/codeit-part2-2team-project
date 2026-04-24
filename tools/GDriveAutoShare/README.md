# GDriveAutoShare

Google Drive를 이용한 데이터셋 배포 자동화 도구입니다.

---

## 개발 동기 — 수동 ZIP 배포의 문제

ML 프로젝트에서 데이터셋은 수시로 갱신됩니다. 갱신할 때마다 ZIP으로 압축해 메신저·이메일·USB로 배포하는 방식은 다음 문제를 내포합니다.

| 문제 | 구체적 위험 |
|------|-----------|
| **버전 추적 불가** | `dataset_final2_진짜최종.zip` 같은 파일명만으로 어느 팀원이 어떤 버전을 쓰는지 알 수 없음 |
| **무결성 보장 없음** | 전송 중 파일이 손상되어도 발견할 방법이 없음 |
| **배포 이력 없음** | 누가 언제 어떤 버전을 받아 사용 중인지 추적 불가 |
| **중복 배포** | 이미 배포된 것과 동일한 파일을 재업로드해도 인식하지 못함 |
| **대용량 전송 실패** | 수 GB 파일을 이메일이나 메신저로 전송 시 크기 제한·중단 위험 |

**GDriveAutoShare**는 이 문제들을 한 번에 해결합니다.

- OAuth 2.0 인증으로 안전하게 Google Drive에 업로드
- SHA-256 체크섬으로 파일 무결성 자동 검증
- 버전 기록(VersionRecord JSON)을 Drive에 자동 기록
- 동일 SHA-256 감지로 중복 업로드 방지
- 재개 가능한 청크 업로드로 대용량 파일 안정 전송
- 공유 URL을 토스트 알림으로 즉시 제공

---

## 시스템 구성

```
배포자 (Data Engineer)                 수신자 (ML Engineer / Researcher)
┌──────────────────────┐               ┌──────────────────────────────────────┐
│  GDriveAutoShare.exe │  ──업로드──▶  │  Google Drive                        │
│  (Windows GUI 앱)    │               │  ├── 📁 데이터셋 ZIP 파일            │
└──────────────────────┘               │  └── 📁 버전 기록 폴더               │
                                       │       └── v_YYYYMMDD_*.json ×n       │
                                       └──────────┬───────────────────────────┘
                                                  │ API Key + 폴더 공유 URL
                                       ┌──────────▼───────────────────────────┐
                                       │  gdrive_downloader.py / .ipynb       │
                                       │  (수신자가 로컬 또는 Colab에서 실행) │
                                       └──────────────────────────────────────┘
```

---

## 1. GDriveAutoShare.zip — 업로더 앱

### 1-1. ZIP 파일 내용

[`GDriveAutoShare.zip`](https://drive.google.com/file/d/1fvrCTxXyF5rWe0r96tyhuyhJ5z3bm-Gf/view?usp=drive_link)을 압축 해제하면 다음 파일이 포함되어 있습니다.

```
GDriveAutoShare/
├── GDriveAutoShare.exe      # 실행 파일 (자체 포함 .NET 8, 별도 설치 불필요)
├── credential.json          # Google Cloud OAuth 2.0 클라이언트 인증 정보 : 팀에 요청
├── *.dll                    # 런타임 의존 라이브러리
└── ...
```

실행 전에 `credential.json`을 팀에 요청하여 위에서 설명한 폴더 위치에 넣어 둡니다.

> **주의** `credential.json`에는 OAuth 클라이언트 시크릿이 포함되어 있습니다.  
> 이 파일을 공개 저장소나 제3자에게 공유하지 마세요.

첫 실행 후 앱 실행 폴더에 `appsettings.json`이 자동 생성됩니다.  
이 파일에는 Drive 폴더 ID, 암호화된 리프레시 토큰 등 런타임 설정이 저장됩니다.

### 1-2. 설치 방법

1. `GDriveAutoShare.zip`을 원하는 폴더에 압축 해제합니다.
2. `GDriveAutoShare.exe`를 실행합니다.
3. Windows Defender SmartScreen 경고가 나타나면 **"추가 정보" → "실행"**을 클릭합니다.  
   *(자체 서명 인증서가 없는 개인 배포 앱이기 때문)*

> **시스템 요구사항:** Windows 10/11 x64 (별도 .NET 설치 불필요 — self-contained 빌드)

### 1-3. 초기 인증

최초 실행 또는 토큰 만료 시 인증이 필요합니다.

1. 앱 상단 **[인증]** 버튼을 클릭합니다.
2. 기본 브라우저가 열리며 Google 계정 선택 화면이 표시됩니다.
3. 업로드에 사용할 Google 계정으로 로그인하고 Drive 접근 권한을 허용합니다.
4. 브라우저에 `Authentication successful` 메시지가 표시되면 창을 닫아도 됩니다.
5. 앱 상태가 **인증됨**으로 변경되고 이메일 주소가 표시됩니다.

리프레시 토큰은 Windows DPAPI로 암호화된 뒤 `appsettings.json`에 저장됩니다.  
이후 재실행 시에는 자동으로 토큰을 갱신하므로 재인증이 필요 없습니다.

### 1-4. 업로드 대상 경로 설정

1. **[경로 추가]** 버튼을 클릭해 업로드할 폴더 또는 파일을 선택합니다.
2. 선택한 경로가 목록에 추가됩니다. 필요 없는 항목은 선택 후 **[제거]**를 클릭합니다.
3. 경로 목록은 `appsettings.json`에 자동 저장되어 다음 실행 시 복원됩니다.

### 1-5. 공유 모드 선택

| 모드 | 설명 | 수신자 조건 |
|------|------|-----------|
| **링크 공유** | `AnyoneWithLink` — URL만 있으면 누구나 접근 | 별도 Google 로그인 불필요 |
| **이메일 공유** | `SpecificEmails` — 지정한 이메일 주소만 접근 | Google 계정 로그인 필요 |

> 링크 공유 모드에서만 `gdrive_downloader.py / .ipynb`가 API 키만으로 동작합니다.  
> 이메일 공유 모드를 사용하는 경우 수신자 쪽에서 OAuth 인증이 별도로 필요합니다.

### 1-6. 업로드 실행

1. **[업로드]** 버튼을 클릭합니다.
2. 진행 상황이 프로그레스 바와 상태 메시지로 표시됩니다.

   ```
   압축 중... → SHA-256 계산 중... → 업로드 중 (n%) → 공유 설정 중... → 완료
   ```

3. 완료 시 화면 우하단에 토스트 알림이 표시되며, 공유 URL이 클립보드에 복사됩니다.
4. **중복 감지**: 이전 업로드와 SHA-256이 동일하면 업로드를 건너뛰고 기존 URL을 반환합니다.

### 1-7. 업로드 재개

네트워크 중단 등으로 업로드가 실패하더라도, 다시 **[업로드]** 버튼을 클릭하면  
서버 측 오프셋을 조회해 중단 지점부터 이어서 업로드합니다.

---

## 2. gdrive_downloader.py — Python 다운로더

로컬 Python 환경에서 직접 실행하거나 VS Code의 Jupyter 셀로 실행하는 스크립트입니다.

### 2-1. 사전 준비

#### 의존 패키지 설치

```bash
pip install requests tqdm tabulate
```

#### Google Cloud API 키 발급

수신자는 Drive API에 접근할 API 키가 필요합니다.  
(배포자가 미리 발급해 공유하거나, 수신자가 직접 생성합니다.)

1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. GDriveAutoShare와 동일한 프로젝트 선택
3. **APIs & Services → Credentials → + CREATE CREDENTIALS → API key**
4. 생성된 API 키를 복사합니다. *(공개 공유 파일 접근이므로 키 제한 설정 불필요)*

### 2-2. 파일 구성

스크립트는 `# %% [N]` 마커로 논리적 섹션이 구분되어 있으며,  
VS Code에서는 각 섹션이 Jupyter 셀처럼 독립 실행됩니다.

| 셀 | 역할 |
|----|------|
| `[1]` 설정 | `FOLDER_URL`, `API_KEY`, `OUTPUT_DIR` 입력 |
| `[2]` 임포트 | 표준 라이브러리 및 서드파티 패키지 로드 |
| `[3]` 함수 정의 | Drive API 호출, 버전 테이블 출력, ZIP 다운로드 함수 |
| `[4]` 버전 불러오기 | 버전 기록 폴더에서 모든 VersionRecord 조회 후 테이블 출력 |
| `[5]` 다운로드 | `SELECTED_INDEX`로 버전 지정 후 다운로드 및 SHA-256 검증 |

### 2-3. 사용 방법

#### 설정 입력 (셀 [1])

```python
FOLDER_URL = "https://drive.google.com/drive/folders/1Abc...xyz?usp=sharing"
             # 배포자에게 받은 버전 기록 폴더 공유 URL

API_KEY    = "AIza..."
             # 발급받은 Google Cloud API 키

OUTPUT_DIR = "."
             # 다운로드 저장 경로 (기본값: 현재 디렉터리)
```

#### 실행 방법

**독립 스크립트 실행:**
```bash
python gdrive_downloader.py
```

**VS Code에서 셀 단위 실행:**
1. `gdrive_downloader.py`를 VS Code에서 엽니다.
2. Python 확장과 Jupyter 확장이 설치되어 있으면 셀 경계에 `Run Cell` 버튼이 표시됩니다.
3. 셀 [1]에 설정을 입력한 뒤 순서대로 실행합니다.

#### 버전 목록 확인 (셀 [4] 실행 결과 예시)

```
3개 버전 발견:

| # | 버전 | 업로드 일시 (로컬)  | 업로더                  | 크기     | SHA256   | 설명         |
|---|------|---------------------|-------------------------|----------|----------|--------------|
| 0 | 1    | 2026-04-20 11:30    | dataeng@company.com     | 1.2 GB   | a1b2c3d4 | 초기 배포     |
| 1 | 2    | 2026-04-22 15:45    | dataeng@company.com     | 1.3 GB   | e5f6g7h8 | 레이블 수정   |
| 2 | 3    | 2026-04-24 09:10    | dataeng@company.com     | 1.4 GB   | i9j0k1l2 | 증강 데이터   |
```

#### 버전 선택 및 다운로드 (셀 [5])

```python
SELECTED_INDEX = -1   # 최신 버전 (기본값)
                  # 0  → 가장 오래된 버전
                  # 2  → 테이블 # 열의 2번 버전
```

다운로드 완료 후 SHA-256이 자동으로 검증됩니다.  
불일치 시 다운로드된 파일이 삭제되고 오류 메시지가 출력됩니다.

---

## 3. gdrive_downloader.ipynb — Jupyter Notebook 다운로더

Google Colab 또는 로컬 Jupyter 환경에서 실행하는 노트북입니다.  
`.py` 파일과 동일한 기능을 제공하되, **Colab Google Drive 자동 마운트** 기능이 추가되어 있습니다.

### 3-1. 환경별 사용 방법

#### Google Colab (추천 — 별도 설치 불필요)

1. [Google Colab](https://colab.research.google.com/)에서 **파일 → 노트북 업로드**로 `gdrive_downloader.ipynb`를 업로드합니다.
2. 런타임 메뉴에서 **모두 실행** 또는 셀을 순서대로 실행합니다.

#### 로컬 Jupyter 환경

```bash
pip install notebook requests tqdm tabulate
jupyter notebook gdrive_downloader.ipynb
```

### 3-2. 셀 구성

| 셀 번호 | 제목 | 역할 |
|---------|------|------|
| 0 | 준비 | Colab 환경 여부 감지 (`is_colab` 플래그) |
| 1 | 의존 패키지 설치 | `pip install` 자동 실행 (환경 불문하고 최신 버전 설치) |
| 2 | 환경 변수 설정 | `API_KEY`, `FOLDER_URL`, `OUTPUT_DIR` 입력; Colab이면 Google Drive 자동 마운트 |
| 3 | 패키지 설정 | 임포트 및 Drive API 기본 URL 상수 정의 |
| 4 | 기능 함수 정의 | `.py`와 동일한 함수 세트 (폴더 ID 추출, 목록 조회, 다운로드) |
| 5 | 버전 기록 불러오기 | API 호출 → 테이블 출력 |
| 6 | 버전 선택 및 다운로드 | `SELECTED_INDEX` 지정 → 다운로드 → SHA-256 검증 |

### 3-3. Colab 전용 동작

셀 0에서 `is_colab = True`가 감지되면 셀 2에서 다음 동작이 추가됩니다.

```python
from google.colab import drive
drive.mount("/content/drive")
OUTPUT_DIR = "/content/drive/MyDrive/GDriveAutoShare_Downloads"
```

Colab 세션에 Google Drive가 마운트되고, 다운로드된 파일이 **사용자의 My Drive** 안에 저장됩니다.  
이를 통해 Colab 세션이 종료되어도 파일이 유지됩니다.

### 3-4. 설정 입력 (셀 2)

```python
API_KEY    = "AIza..."   # Google Cloud API 키
FOLDER_URL = "https://drive.google.com/drive/folders/..."  # 배포자에게 받은 URL
```

이후 셀을 순서대로 실행하면 버전 목록이 표시되고, 마지막 셀에서 다운로드가 시작됩니다.

---

## 자주 묻는 질문

**Q. "Drive API 오류 403" 메시지가 출력됩니다.**  
A. API 키가 잘못되었거나, Drive API v3가 해당 Cloud 프로젝트에서 활성화되지 않은 경우입니다.  
Cloud Console → APIs & Services → Enabled APIs에서 `Google Drive API`를 활성화하세요.

**Q. "다운로드 실패 403" 오류가 납니다.**  
A. ZIP 파일 또는 버전 기록 폴더가 `링크가 있는 모든 사용자`로 공유되지 않은 상태입니다.  
배포자에게 공유 모드를 확인하거나, 이메일 공유 모드인 경우 OAuth 인증을 추가해야 합니다.

**Q. SHA-256 불일치 오류가 발생했습니다.**  
A. 다운로드 중 파일이 손상되었을 가능성이 높습니다. 다시 실행하면 파일을 재다운로드합니다.  
재시도에도 동일 오류가 발생하면 배포자에게 문의하세요.

**Q. appsettings.json이 없다는 오류가 납니다.**  
A. 이 파일은 첫 실행 시 자동 생성됩니다. `GDriveAutoShare.exe`를 한 번 실행하고 인증을 완료하면 생성됩니다.

**Q. Windows Defender가 앱을 차단합니다.**  
A. "추가 정보 → 실행"을 클릭하세요. 코드 서명 인증서 없이 배포하는 개인 앱에서 발생하는 정상적인 경고입니다.
