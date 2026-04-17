# data/

실제 데이터 파일은 `.gitignore`로 Git 관리에서 제외됩니다.
로컬 / Colab / Kaggle Kernel 환경에서 직접 배치하세요.

```
data/
├── raw/          # 원본 데이터 (절대 수정 금지)
├── processed/    # 전처리·변환 완료 데이터
└── external/     # 외부 보조 데이터
```

## ⚠️ 금지 데이터

대회 규정상 아래 데이터는 사용 불가합니다.

- AI Hub `TL_2_조합.zip`
- AI Hub `TS_2_조합.zip`
