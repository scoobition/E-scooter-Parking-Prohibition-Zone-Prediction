import os
import pandas as pd

def load_months(input_dir="original_data", months=(7,8,9,10,11)) -> pd.DataFrame:
    dfs = []
    for m in months:
        path = os.path.join(input_dir, f"{m}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 파일이 없습니다.")
        df = pd.read_csv(path)
        df["month"] = m
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
