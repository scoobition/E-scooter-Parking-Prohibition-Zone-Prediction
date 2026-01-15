# src/result.py
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from src.preprocess import clean_address
from src.google_geocode import fill_cache_for_addresses
from src.grid import make_predata_and_meta_csv
from src.viz_grid_map import make_grid_heatmap_html


def make_result(
    input_dir: str = "original_data",
    filename: str = "12.csv",
    out_data_dir: str = "data",
    out_map_dir: str = "map",
    cache_path: str = "data/geocode_cache_result.csv",
    sleep_sec: float = 0.05,
    opacity: float = 0.5,
    max_cells: int | None = 20000,
):
    """
    original_dataa/12.csv 1개 파일을 대상으로:
    1) 지오코딩 (캐시 활용)
    2) 200m 격자화(predata/meta)
    3) 히트맵 HTML 생성
    출력 파일명 규칙:
      - 입력 파일 stem(예: 12) + "_result" 를 기본으로 사용
    """

    in_path = Path(input_dir) / filename
    if not in_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {in_path}")

    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(out_map_dir, exist_ok=True)

    stem = in_path.stem  # "12"
    month = int(stem) if stem.isdigit() else 12  # 혹시 파일명이 숫자가 아니어도 최소 12로

    # -------------------------
    # 1) CSV 로드 + month 부여
    # -------------------------
    df = pd.read_csv(in_path)
    if "주소" not in df.columns:
        raise KeyError("입력 CSV에 '주소' 컬럼이 없습니다.")

    df["month"] = month
    df["주소_clean"] = df["주소"].apply(clean_address)

    unique_addrs = df["주소_clean"].dropna().astype(str).unique()
    print(f"[INFO] input rows: {len(df)}")
    print(f"[INFO] unique addresses: {len(unique_addrs)}")

    # -------------------------
    # 2) 지오코딩(캐시)
    # -------------------------
    cache = fill_cache_for_addresses(
        unique_addrs,
        cache_path=cache_path,
        sleep_sec=sleep_sec,
    )

    merged = df.merge(cache, on="주소_clean", how="left")
    out_geo = merged[["month", "lat", "lon"]].copy()

    geo_out_path = Path(out_data_dir) / f"{stem}_result.csv"
    out_geo.to_csv(geo_out_path, index=False, encoding="utf-8-sig")

    fail = out_geo["lat"].isna().sum()
    print(f"[DONE] geocoded saved: {geo_out_path}")
    print(f"[INFO] geocode fail: {fail}/{len(out_geo)} ({(fail/len(out_geo)*100 if len(out_geo) else 0):.2f}%)")

    # -------------------------
    # 3) 격자화(predata/meta)
    # -------------------------
    predata_path = Path(out_data_dir) / f"{stem}_result_predata.csv"
    meta_path = Path(out_data_dir) / f"{stem}_result_grid_meta.csv"

    make_predata_and_meta_csv(
        input_csv=str(geo_out_path),
        predata_csv=str(predata_path),
        meta_csv=str(meta_path),
    )

        # -------------------------
    # 3-1) predata_12.csv 추가 저장
    # -------------------------
    simple_predata_path = Path(out_data_dir) / f"predata_{month}.csv"

    df_pre = pd.read_csv(predata_path)
    df_pre = df_pre[["month", "grid_id", "count"]]

    df_pre.to_csv(simple_predata_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] simple predata saved: {simple_predata_path}")

    # -------------------------
    # 4) 히트맵(html)
    # -------------------------
    html_out_path = Path(out_map_dir) / f"{stem}_result_grid_heatmap_200m_{month}.html"
    make_grid_heatmap_html(
        month=month,
        predata_csv=str(predata_path),
        meta_csv=str(meta_path),
        out_html=str(html_out_path),
        opacity=opacity,
        max_cells=max_cells,
    )

    print(f"[DONE] heatmap saved: {html_out_path}")

    return {
        "geo_csv": str(geo_out_path),
        "predata_csv": str(predata_path),
        "meta_csv": str(meta_path),
        "heatmap_html": str(html_out_path),
    }
