# src/grid.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pyproj import Transformer

# =========================
# 설정값
# =========================
CELL_SIZE_M = 200
SRC_CRS = "EPSG:4326"   # lat/lon (WGS84)
DST_CRS = "EPSG:5179"   # meter (Korea 2000 / Unified CS)

def _get_transformer():
    # always_xy=True 이면 입력이 (lon, lat) 순서!
    return Transformer.from_crs(SRC_CRS, DST_CRS, always_xy=True)

# =========================
# 1) 격자 컬럼 추가
# =========================
def add_grid_columns(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    입력 df (month, lat, lon)에
    x_m, y_m, grid_x, grid_y, grid_id 를 추가한 df를 반환
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError("입력 df에 lat/lon 컬럼이 필요합니다.")

    out = df.copy()
    out = out.dropna(subset=[lat_col, lon_col]).copy()

    transformer = _get_transformer()
    lon = out[lon_col].to_numpy(dtype=float)
    lat = out[lat_col].to_numpy(dtype=float)

    x_m, y_m = transformer.transform(lon, lat)
    out["x_m"] = x_m
    out["y_m"] = y_m

    out["grid_x"] = np.floor(out["x_m"] / CELL_SIZE_M).astype(np.int64)
    out["grid_y"] = np.floor(out["y_m"] / CELL_SIZE_M).astype(np.int64)
    out["grid_id"] = out["grid_x"].astype(str) + "_" + out["grid_y"].astype(str)

    return out

# =========================
# 2) predata 만들기 (month, grid_id, count)
# =========================
def build_predata(
    df_grid: pd.DataFrame,
    month_col: str = "month",
    grid_id_col: str = "grid_id",
) -> pd.DataFrame:
    for c in [month_col, grid_id_col]:
        if c not in df_grid.columns:
            raise KeyError(f"입력 df에 '{c}' 컬럼이 필요합니다.")

    predata = (
        df_grid
        .groupby([month_col, grid_id_col])
        .size()
        .reset_index(name="count")
        .sort_values([month_col, "count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return predata

# =========================
# 3) grid_meta 만들기 (grid_id, grid_x, grid_y, center_x_m, center_y_m)
# =========================
def build_grid_meta(
    df_grid: pd.DataFrame,
    grid_id_col: str = "grid_id",
) -> pd.DataFrame:
    for c in ["grid_x", "grid_y", grid_id_col]:
        if c not in df_grid.columns:
            raise KeyError(f"입력 df에 '{c}' 컬럼이 필요합니다.")

    meta = (
        df_grid[[grid_id_col, "grid_x", "grid_y"]]
        .drop_duplicates()
        .copy()
        .sort_values(["grid_x", "grid_y"])
        .reset_index(drop=True)
    )

    meta["center_x_m"] = (meta["grid_x"].to_numpy(dtype=float) + 0.5) * CELL_SIZE_M
    meta["center_y_m"] = (meta["grid_y"].to_numpy(dtype=float) + 0.5) * CELL_SIZE_M

    # 컬럼 순서 고정
    meta = meta[[grid_id_col, "grid_x", "grid_y", "center_x_m", "center_y_m"]]
    return meta

# =========================
# 4) 전체 파이프라인: after.csv -> predata.csv + grid_meta.csv 저장
# =========================
def make_predata_and_meta_csv(
    input_csv: str = "data/after.csv",
    predata_csv: str = "data/predata.csv",
    meta_csv: str = "data/grid_meta.csv",
):
    """
    after.csv (month, lat, lon) →
    200m 격자화 →
    predata.csv (month, grid_id, count) 저장 +
    grid_meta.csv (grid_id, grid_x, grid_y, center_x_m, center_y_m) 저장
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"{input_csv} 파일이 없습니다.")

    # 출력 폴더 보장
    os.makedirs(os.path.dirname(predata_csv), exist_ok=True)
    os.makedirs(os.path.dirname(meta_csv), exist_ok=True)

    df = pd.read_csv(input_csv)

    # 격자화
    df_grid = add_grid_columns(df)

    # 저장 1) predata
    predata = build_predata(df_grid)
    predata.to_csv(predata_csv, index=False)

    # 저장 2) grid_meta
    meta = build_grid_meta(df_grid)
    meta.to_csv(meta_csv, index=False)

    print(f"[INFO] predata 저장 완료: {predata_csv} (rows={len(predata)})")
    print(f"[INFO] grid_meta 저장 완료: {meta_csv} (rows={len(meta)})")
