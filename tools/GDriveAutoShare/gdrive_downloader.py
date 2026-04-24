# gdrive_downloader.py
#
# 실행 방법:
#   독립 실행  : python gdrive_downloader.py
#   VS Code    : 파일 열기 — # %% 마커가 Jupyter 셀로 자동 인식됨
#   Jupyter    : VS Code → File → "Export Current Python File as Jupyter Notebook"
#
# 의존 패키지:
#   pip install requests tqdm tabulate
#
# 사전 준비 — Google Cloud API 키 (GDriveAutoShare 앱과 동일 프로젝트):
#   Cloud Console → APIs & Services → Credentials → + CREATE CREDENTIALS → API key
#   공개 공유 파일에만 접근하므로 키 제한 설정 불필요.
#
# 공유 모드 주의:
#   버전 기록 폴더와 ZIP 파일이 "링크가 있는 모든 사용자" 로 공유된 경우에만 동작.
#   특정 이메일 공유 모드일 경우 OAuth 인증이 별도로 필요.

# %% [markdown]
# ## 설정
# 아래 셀 [1]의 `FOLDER_URL`과 `API_KEY`를 입력한 뒤 전체 셀을 실행하세요.

# %% [1] 설정 — 실행 전 반드시 입력
FOLDER_URL = ""  # GDriveAutoShare 버전 기록 폴더 공유 URL
# 예: "https://drive.google.com/drive/folders/1Abc...xyz?usp=sharing"

API_KEY = ""  # Drive API 접근 권한이 있는 Google Cloud API 키

OUTPUT_DIR = "."  # 다운로드 저장 위치 (상대 경로 또는 절대 경로)

# %% [2] 임포트
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from tabulate import tabulate as _tabulate

    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False

_DRIVE_API = "https://www.googleapis.com/drive/v3"

# %% [3] 함수 정의


def extract_folder_id(url_or_id: str) -> str:
    """Drive 공유 URL 또는 폴더 ID 문자열에서 폴더 ID를 반환."""
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(f"폴더 ID를 파싱할 수 없습니다: {url_or_id!r}")


def _get_json(path: str, api_key: str, **params) -> dict:
    resp = requests.get(
        f"{_DRIVE_API}/{path}",
        params={"key": api_key, **params},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Drive API 오류 {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def list_version_records(folder_id: str, api_key: str) -> list[dict]:
    """버전 기록 폴더에서 VersionRecord JSON 파일을 전부 읽어 파싱."""
    q = f"'{folder_id}' in parents and mimeType='application/json' and trashed=false"
    files, page_token = [], None

    while True:
        params = dict(
            q=q,
            fields="nextPageToken,files(id,name)",
            orderBy="createdTime",
            pageSize=1000,
        )
        if page_token:
            params["pageToken"] = page_token
        data = _get_json("files", api_key, **params)
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    records = []
    for f in files:
        try:
            record = _get_json(f"files/{f['id']}", api_key, alt="media")
            records.append(record)
        except Exception as exc:
            print(f"[경고] {f['name']} 건너뜀: {exc}", file=sys.stderr)

    records.sort(key=lambda r: r.get("uploadedAtUtc", ""))
    return records


def _fmt_size(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def _fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16]


def print_version_table(records: list[dict]) -> None:
    """버전 기록 테이블을 터미널에 출력."""
    headers = ["#", "버전", "업로드 일시 (로컬)", "업로더", "크기", "SHA256", "설명"]
    rows = [
        [
            i,
            r.get("version", "?"),
            _fmt_ts(r.get("uploadedAtLocal") or r.get("uploadedAtUtc", "")),
            r.get("uploaderEmail", ""),
            _fmt_size(r.get("fileSizeBytes", 0)),
            (r.get("sha256") or "")[:8],
            r.get("description", ""),
        ]
        for i, r in enumerate(records)
    ]

    if _HAS_TABULATE:
        print(_tabulate(rows, headers=headers, tablefmt="github"))
    else:
        col_w = [4, 4, 20, 28, 8, 8, 20]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("-" * sum(col_w + [2] * len(col_w)))
        for row in rows:
            print(fmt.format(*[str(v) for v in row]))


def download_zip(record: dict, api_key: str, output_dir: str | Path = ".") -> Path:
    """
    VersionRecord에 해당하는 ZIP 파일을 다운로드.

    다운로드 완료 후 SHA-256 검증을 수행하며, 저장된 파일 경로를 반환.
    """
    file_id = record.get("zipFileId", "")
    if not file_id:
        raise ValueError("레코드에 zipFileId가 없습니다.")

    targets = record.get("targets", [])
    stem = Path(targets[0]).stem if targets else "dataset"
    dest = Path(output_dir) / f"{stem}_v{record.get('version', 'unknown')}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"{_DRIVE_API}/files/{file_id}"
    expected_size = record.get("fileSizeBytes", 0)

    with requests.get(
        url,
        params={"alt": "media", "key": api_key},
        stream=True,
        timeout=60,
    ) as resp:
        if not resp.ok:
            raise RuntimeError(
                f"다운로드 실패 {resp.status_code}: {resp.text[:200]}\n"
                "'링크가 있는 모든 사용자'로 공유되었는지 확인하세요."
            )
        total = int(resp.headers.get("Content-Length", 0)) or expected_size
        sha = hashlib.sha256()

        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, ncols=80
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                sha.update(chunk)
                bar.update(len(chunk))

    digest = sha.hexdigest()
    expected_sha = record.get("sha256", "")
    if expected_sha and digest.lower() != expected_sha.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 불일치 — 파일이 손상되었을 수 있습니다.\n"
            f"  예상값: {expected_sha}\n"
            f"  실제값: {digest}"
        )

    print(f"✓  {dest}  ({_fmt_size(dest.stat().st_size)})")
    return dest


# %% [4] 버전 기록 불러오기

if not FOLDER_URL or not API_KEY:
    raise SystemExit("셀 [1]의 FOLDER_URL과 API_KEY를 입력한 뒤 다시 실행하세요.")

folder_id = extract_folder_id(FOLDER_URL)
print(f"폴더 ID : {folder_id}")
print("버전 기록을 가져오는 중...\n")

records = list_version_records(folder_id, API_KEY)

if not records:
    raise SystemExit(
        "버전 기록을 찾을 수 없습니다. FOLDER_URL과 공유 권한을 확인하세요."
    )

print(f"{len(records)}개 버전 발견:\n")
print_version_table(records)

# %% [5] 버전 선택 및 다운로드
# SELECTED_INDEX로 다운로드할 버전을 지정:
#   0  → 가장 오래된 버전
#   -1 → 최신 버전 (기본값)
#   n  → 위 테이블의 # 열 번호

SELECTED_INDEX = -1

selected = records[SELECTED_INDEX]
print(
    f"\nv{selected['version']} 다운로드 중 · "
    f"{_fmt_size(selected['fileSizeBytes'])} · "
    f"업로더: {selected.get('uploaderEmail', '알 수 없음')}\n"
)
saved_path = download_zip(selected, API_KEY, OUTPUT_DIR)
