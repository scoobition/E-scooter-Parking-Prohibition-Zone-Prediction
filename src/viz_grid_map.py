# src/viz_grid_map.py
import os
import numpy as np
import pandas as pd
import folium
from typing import Optional
from pyproj import Transformer


CELL_SIZE_M = 200
SRC_CRS = "EPSG:4326"   # lat/lon
DST_CRS = "EPSG:5179"   # meter

def _red_color_from_count(count: float, vmin: float, vmax: float) -> str:
    import math

    if vmax <= vmin:
        t = 1.0
    else:
        t = (math.log(count + 1) - math.log(vmin + 1)) / (
            math.log(vmax + 1) - math.log(vmin + 1)
        )
        t = max(0.0, min(1.0, t))

    gb = int(200 - t * 200)   # 진한 빨강
    return f"#{255:02x}{gb:02x}{gb:02x}"


def make_grid_heatmap_html(
    month: int,
    predata_csv: str = "data/predata.csv",
    meta_csv: str = "data/grid_meta.csv",
    out_html: Optional[str] = None,
    opacity: float = 0.35,
    max_cells: Optional[int] = 20000,
):
    """
    month에 해당하는 격자들을 지도에 사각형(200m x 200m)으로 표시하고,
    count가 많을수록 더 진한 빨강으로 채움(반투명).
    결과는 html로 저장.
    """

    if out_html is None:
        out_html = f"data/grid_heatmap_200m_{month}.html"

    if not os.path.exists(predata_csv):
        raise FileNotFoundError(f"{predata_csv} 없음")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"{meta_csv} 없음")

    pre = pd.read_csv(predata_csv)
    meta = pd.read_csv(meta_csv)

    # month 필터
    pre_m = pre[pre["month"] == month].copy()
    if pre_m.empty:
        raise ValueError(f"predata에 month={month} 데이터가 없습니다.")

    # merge: month,grid_id,count + (grid_x,grid_y 등)
    df = pre_m.merge(meta, on="grid_id", how="left")
    if df["grid_x"].isna().any():
        missing = df[df["grid_x"].isna()]["grid_id"].head(5).tolist()
        raise ValueError(f"grid_meta에 없는 grid_id가 있습니다. 예: {missing}")

    # 너무 많은 격자는 folium 렌더링이 느릴 수 있음
    if max_cells is not None and len(df) > max_cells:
        df = df.sort_values("count", ascending=False).head(max_cells).copy()

    # 색 범위
    vmin = float(df["count"].min())
    vmax = float(df["count"].max())

    # EPSG:5179(m) -> EPSG:4326(latlon) 변환기
    to_latlon = Transformer.from_crs(DST_CRS, SRC_CRS, always_xy=True)

    # 지도 중심(데이터 평균 중심점)
    center_lon, center_lat = to_latlon.transform(
        df["center_x_m"].to_numpy(dtype=float).mean(),
        df["center_y_m"].to_numpy(dtype=float).mean()
    )

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    legend_html = f"""
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.92);
        padding: 12px 14px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-size: 13px;
        line-height: 1.4;
    ">
        <div style="font-weight: 700; margin-bottom: 6px;">
            {month}월 · 200m 격자 히트맵
        </div>
        <div style="font-size: 12px; color: #444;">
            <div>색상: <b>진할수록 많음</b></div>
            <div>건수 범위: {int(vmin)} ~ {int(vmax)}</div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


    # 각 격자 사각형 그리기
    half = CELL_SIZE_M / 2.0

    for _, r in df.iterrows():
        cx = float(r["center_x_m"])
        cy = float(r["center_y_m"])

        # 200m 격자 사각형의 4개 코너(미터)
        minx, maxx = cx - half, cx + half
        miny, maxy = cy - half, cy + half

        # folium Rectangle은 lat/lon bounds 필요:
        # SW(minx,miny) / NE(maxx,maxy)
        sw_lon, sw_lat = to_latlon.transform(minx, miny)
        ne_lon, ne_lat = to_latlon.transform(maxx, maxy)

        color = _red_color_from_count(float(r["count"]), vmin, vmax)

        folium.Rectangle(
            bounds=[[sw_lat, sw_lon], [ne_lat, ne_lon]],
            color=None,             # 테두리 없애기
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=f'grid_id={r["grid_id"]} | count={int(r["count"])}',
        ).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"[INFO] saved: {out_html}")


def run_grid_heatmap(
    month: int,
    predata_csv: str = "data/predata.csv",
    meta_csv: str = "data/grid_meta.csv",
    out_html: Optional[str] = None,
    opacity: float = 0.35,
    max_cells: Optional[int] = 20000,
):
    """
    메인에서 한 줄로 호출하려고 만든 래퍼 함수.
    내부적으로 make_grid_heatmap_html() 호출.
    """
    make_grid_heatmap_html(
        month=month,
        predata_csv=predata_csv,
        meta_csv=meta_csv,
        out_html=out_html,
        opacity=opacity,
        max_cells=max_cells,
    )

def make_monthly_heatmaps(
    months=None,  # None이면 predata에 있는 month 전부 자동
    predata_csv: str = "data/predata.csv",
    meta_csv: str = "data/grid_meta.csv",
    out_dir: str = "map",
    opacity: float = 0.5,
    max_cells: Optional[int] = 20000,
):
    """
    여러 월에 대해 격자 히트맵 HTML을 일괄 생성해서 out_dir에 저장.
    out_dir 예: "map"  -> map/grid_heatmap_200m_7.html 이런 식으로 생성됨
    """
    if not os.path.exists(predata_csv):
        raise FileNotFoundError(f"{predata_csv} 없음")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"{meta_csv} 없음")

    os.makedirs(out_dir, exist_ok=True)

    pre = pd.read_csv(predata_csv)

    # months를 지정 안 하면, predata에 있는 month 전부 사용
    if months is None:
        months = sorted(pre["month"].dropna().unique().tolist())

    for m in months:
        out_html = os.path.join(out_dir, f"grid_heatmap_200m_{int(m)}.html")
        make_grid_heatmap_html(
            month=int(m),
            predata_csv=predata_csv,
            meta_csv=meta_csv,
            out_html=out_html,
            opacity=opacity,
            max_cells=max_cells,
        )
