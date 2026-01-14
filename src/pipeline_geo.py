import os
from src.io_loader import load_months
from src.preprocess import clean_address
from src.google_geocode import fill_cache_for_addresses

def geo(
    input_dir="original_data",
    months=(7,8,9,10,11),
    out_path="data/after.csv",
    cache_path="data/geocode_cache.csv",
    sleep_sec=0.05,
):
    os.makedirs("data", exist_ok=True)

    df = load_months(input_dir, months)
    if "주소" not in df.columns:
        raise KeyError("입력 CSV에 '주소' 컬럼이 없습니다.")

    print(f"[INFO] 전체 행 수: {len(df)}")

    df["주소_clean"] = df["주소"].apply(clean_address)

    unique_addrs = df["주소_clean"].unique()
    print(f"[INFO] unique 주소 수: {len(unique_addrs)}")

    cache = fill_cache_for_addresses(unique_addrs, cache_path=cache_path, sleep_sec=sleep_sec)

    merged = df.merge(cache, on="주소_clean", how="left")

    out = merged[["month", "lat", "lon"]]
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    fail = out["lat"].isna().sum()
    print(f"[DONE] after.csv 저장 완료: {out_path}")
    print(f"[INFO] 지오코딩 실패: {fail}/{len(out)} ({fail/len(out)*100:.2f}%)")
