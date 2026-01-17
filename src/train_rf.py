import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from typing import Sequence, List


def train_rf(
    data_path: str = "data/features.csv",
    model_path: str = "model_rf.pkl",
    train_months: Sequence[int] = (3,4,5,6,7,8,9,10),
    feature_cols: Sequence[str] = ("count_t", "count_t-1", "count_t-2"),
    n_estimators: int = 300,
    max_depth: int = 8,
    random_state: int = 42,
) -> Path:
    """RandomForestRegressor 학습 후 모델 저장.

    - 학습 데이터: month in train_months
    - 타깃 y: 같은 grid_id에서 '다음 달' count_t (shift(-1))
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

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    out_p = Path(model_path)
    joblib.dump(model, out_p)
    print(f"[DONE] 모델 저장: {out_p}")
    return out_p


def main():
    train_rf()


if __name__ == "__main__":
    main()
