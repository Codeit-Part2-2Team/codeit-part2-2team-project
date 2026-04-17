# main 브랜치 보호 규칙 설정 가이드

GitHub UI에서 직접 설정합니다 (Settings → Branches → Branch protection rules).

## 설정 방법

1. GitHub 레포지토리 → **Settings** 탭
2. 좌측 메뉴 **Branches** 클릭
3. **Add branch ruleset** 또는 **Add rule** 클릭
4. **Branch name pattern**: `main`

## 필수 활성화 항목

| 항목 | 설정값 | 이유 |
|------|--------|------|
| Require a pull request before merging | ✅ ON | Direct push 금지 |
| Required number of approvals | **1** | 최소 1인 리뷰 |
| Dismiss stale pull request approvals when new commits are pushed | ✅ ON | 리뷰 후 추가 커밋 방지 |
| Require status checks to pass before merging | ✅ ON (CI 설정 후) | 코드 품질 게이트 |
| Do not allow bypassing the above settings | ✅ ON | 관리자 포함 예외 없음 |

## 주의

- 팀장(호정)도 Direct push 불가 → PR 경유 필수
- 긴급 수정 시에도 PR 생성 후 셀프 머지 허용 (단, 코멘트 필수)
