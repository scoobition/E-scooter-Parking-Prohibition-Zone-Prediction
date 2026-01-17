import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from typing import Sequence


def train_rf(
    data_path: str = "data/features.csv",
    model_path: str = "model_rf.pkl",
    train_months: Sequence[int] = (4,5,6,7,8,9,10),
    feature_cols: Sequence[str] = ("count_t", "count_t-1", "count_t-2", "count_t-3"),
    n_estimators: int = 500,
    max_depth: int = 7,
    random_state: int = 42,
    max_features = 2
) -> Path:
    """RandomForestRegressor 학습 후 모델 저장.
    + OOB 기반 내부 성능 평가 포함
    """
    df = pd.read_csv(data_path)

    missing = set(["month", "grid_id", "count_t", *feature_cols]) - set(df.columns)
    if missing:
        raise KeyError(f"features.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    train = df[df["month"].isin(list(train_months))].copy()
    if train.empty:
        raise ValueError(f"train_months={list(train_months)}에 해당하는 학습 데이터가 없습니다.")

    # y = 다음 달 count_t
    train["y"] = train.groupby("grid_id")["count_t"].shift(-1)
    train = train.dropna(subset=["y", *feature_cols]).copy()

    X = train[list(feature_cols)]
    y = train["y"].astype(float)

    # =========================
    # RandomForest + OOB
    # =========================
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        max_features=max_features,
        n_jobs=-1,
        min_samples_leaf=2,
        oob_score=True,     # ⭐ 추가
        bootstrap=True,     # ⭐ 반드시 필요
    )
    model.fit(X, y)

    # =========================
    # OOB 성능 (R²)
    # =========================
    oob_r2 = float(model.oob_score_)

    # =========================
    # 모델 + OOB 점수 저장
    # =========================
    out_p = Path(model_path)
    joblib.dump(
        {
            "model": model,
            "oob_r2": oob_r2,
        },
        out_p,
    )

    print(f"[DONE] 모델 저장: {out_p}")
    print(f"[INFO] OOB R2 score: {oob_r2:.4f}")

    return out_p


def main():
    train_rf()


if __name__ == "__main__":
    main()