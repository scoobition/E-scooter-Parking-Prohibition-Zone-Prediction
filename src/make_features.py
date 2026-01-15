import pandas as pd

def make_lag_features(df, lags=[1, 2]):
    df = df.sort_values(["grid_id", "month"]).copy()
    df["count_t"] = df["count"]

    for lag in lags:
        df[f"count_t-{lag}"] = df.groupby("grid_id")["count"].shift(lag)

    # 과거 정보 없는 행 제거
    df = df.dropna(subset=["count_t", "count_t-1", "count_t-2"])

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/predata.csv")

    df_feat = make_lag_features(df)

    # 확인용 출력 (처음 몇 줄)
    print(df_feat.head())

    df_feat.to_csv("data/features.csv", index=False)