# src/predict_rf.py

import pandas as pd
import joblib
from pathlib import Path
from typing import Sequence


def predict_rf(
    data_path: str = "data/features.csv",
    model_path: str = "model_rf.pkl",
    out_path: str = "data/pred_12.csv",
    pred_month: int = 11,
    feature_cols: Sequence[str] = ("count_t", "count_t-1", "count_t-2", "count_t-3"),
    out_col: str = "count",
) -> Path:
    """
    학습된 RandomForest 모델로 특정 월(pred_month)을 입력으로
    다음 달(out_col) 예측 결과를 저장한다.

    - model_rf.pkl이
      1) RandomForestRegressor 단독 저장이든
      2) {"model": rf, "oob_r2": ...} dict 형태든
      모두 호환되도록 처리
    """
    # ----------------------------
    # 데이터 로드
    # ----------------------------
    df = pd.read_csv(data_path)

    # ----------------------------
    # 모델 로드 (OOB 적용 전/후 호환)
    # ----------------------------
    saved = joblib.load(model_path)
    if isinstance(saved, dict):
        model = saved["model"]
    else:
        model = saved

    # ----------------------------
    # 예측 대상 월 필터링
    # ----------------------------
    pred_df = df[df["month"] == pred_month].copy()
    if pred_df.empty:
        raise ValueError(
            f"pred_month={pred_month}에 해당하는 행이 없습니다. "
            "features 생성/월 선택을 확인하세요."
        )

    # ----------------------------
    # feature 컬럼 검증
    # ----------------------------
    missing = set(feature_cols) - set(pred_df.columns)
    if missing:
        raise KeyError(
            f"예측 입력에 필요한 feature 컬럼이 없습니다: {sorted(missing)}"
        )

    X_pred = pred_df[list(feature_cols)]

    # ----------------------------
    # 예측 수행
    # ----------------------------
    print(f"[INFO] 예측 대상 격자 수: {len(X_pred)} (month={pred_month})")
    y_pred = model.predict(X_pred)

    pred_df[out_col] = y_pred

    # ----------------------------
    # 결과 저장
    # ----------------------------
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    pred_df[["grid_id", out_col]].to_csv(out_p, index=False)
    print(f"[DONE] 예측 결과 저장: {out_p}")

    return out_p


def main():
    predict_rf()


if __name__ == "__main__":
    main()