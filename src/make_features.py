import pandas as pd
from pathlib import Path
from typing import Iterable, List, Sequence


def make_lag_features(df: pd.DataFrame, lags: Sequence[int] = (1, 2)) -> pd.DataFrame:
    """grid_id별 월별 count로 lag feature 생성.

    입력 df는 최소 컬럼: [month, grid_id, count]
    출력 df는 count_t, count_t-1, count_t-2 ... 컬럼 포함.
    """
    required = {"month", "grid_id", "count"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"make_lag_features() 입력 df에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df = df.sort_values(["grid_id", "month"]).copy()
    df["count_t"] = df["count"].astype(float)

    for lag in lags:
        df[f"count_t-{lag}"] = df.groupby("grid_id")["count_t"].shift(lag)

    # 과거 정보 없는 행 제거 (lags 전부 존재해야 학습/예측 입력으로 사용 가능)
    lag_cols = [f"count_t-{lag}" for lag in lags]
    df = df.dropna(subset=["count_t", *lag_cols]).copy()
    return df


def make_features(
    in_path: str = "data/predata.csv",
    out_path: str = "data/features.csv",
    lags: Sequence[int] = (1, 2),
) -> Path:
    """predata.csv -> features.csv 생성 (lag feature 추가)."""
    in_p = Path(in_path)
    if not in_p.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {in_path}")

    df = pd.read_csv(in_p)
    df_feat = make_lag_features(df, lags=lags)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_p, index=False)

    print(f"[DONE] features 저장: {out_p} (rows={len(df_feat)})")
    return out_p


def main():
    make_features()


if __name__ == "__main__":
    main()
