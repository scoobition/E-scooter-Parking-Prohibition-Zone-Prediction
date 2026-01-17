import pandas as pd
import joblib
from pathlib import Path
from typing import Sequence


# Predict next month counts using trained RF model
def predict_rf(
    data_path: str = "data/features.csv",
    model_path: str = "model_rf.pkl",
    out_path: str = "data/pred_12.csv",
    pred_month: int = 11,
    feature_cols: Sequence[str] = ("count_t", "count_t-1", "count_t-2"),
    out_col: str = "count",
) -> Path:
    df = pd.read_csv(data_path)
    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle  # Handle wrapped model

    pred_df = df[df["month"] == pred_month].copy()
    if pred_df.empty:
        raise ValueError(f"pred_month={pred_month}에 해당하는 행이 없습니다. features 생성/월 선택을 확인하세요.")

    missing = set(feature_cols) - set(pred_df.columns)
    if missing:
        raise KeyError(f"예측 입력에 필요한 feature 컬럼이 없습니다: {sorted(missing)}")

    X_pred = pred_df[list(feature_cols)]
    print(f"[INFO] 예측 대상 격자 수: {len(X_pred)} (month={pred_month})")

    pred_df[out_col] = model.predict(X_pred)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    pred_df[["grid_id", out_col]].to_csv(out_p, index=False)
    print(f"[DONE] 예측 결과 저장: {out_p}")
    return out_p


def main():
    predict_rf()


if __name__ == "__main__":
    main()