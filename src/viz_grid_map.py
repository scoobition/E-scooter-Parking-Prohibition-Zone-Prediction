# src/viz_grid_map.py
import os
import math
import pandas as pd
import folium
from typing import Optional
from pyproj import Transformer

CELL_SIZE_M = 200
SRC_CRS = "EPSG:4326"
DST_CRS = "EPSG:5179"


def _red_color_from_value(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 1.0
    else:
        t = (math.log(value + 1) - math.log(vmin + 1)) / (
            math.log(vmax + 1) - math.log(vmin + 1)
        )
        t = max(0.0, min(1.0, t))
    gb = int(220 - 200 * t)
    return f"#{255:02x}{gb:02x}{gb:02x}"


def make_grid_heatmap_html(
    *,
    month: Optional[int] = None,
    predata_csv: str = "data/predata.csv",
    value_csv: Optional[str] = None,
    value_col: str = "count",
    meta_csv: str = "data/grid_meta.csv",
    out_html: Optional[str] = None,
    title: Optional[str] = None,
    opacity: float = 0.4,
    max_cells: Optional[int] = 20000,
    show_top10: bool = True,
):
    if value_csv is None and month is None:
        raise ValueError("month 또는 value_csv 중 하나는 반드시 필요합니다.")

    meta = pd.read_csv(meta_csv)

    # =========================
    # 데이터 로드
    # =========================
    if value_csv:
        df_val = pd.read_csv(value_csv)
        df = df_val[["grid_id", value_col]].copy()
        df.rename(columns={value_col: "value"}, inplace=True)
        map_title = title or f"{value_col} 기반 시각화"
    else:
        pre = pd.read_csv(predata_csv)
        df_m = pre[pre["month"] == month]
        df = df_m[["grid_id", "count"]].copy()
        df.rename(columns={"count": "value"}, inplace=True)
        map_title = title or f"{month}월 실제 견인 발생"

    df = df.merge(meta, on="grid_id", how="left")
    df = df.dropna(subset=["center_x_m", "center_y_m", "value"])

    if max_cells and len(df) > max_cells:
        df = df.sort_values("value", ascending=False).head(max_cells)

    vmin, vmax = df["value"].min(), df["value"].max()

    # =========================
    # 지도 생성
    # =========================
    to_latlon = Transformer.from_crs(DST_CRS, SRC_CRS, always_xy=True)
    lon, lat = to_latlon.transform(df["center_x_m"].mean(), df["center_y_m"].mean())

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    # =========================
    # 범례
    # =========================
    legend_html = f"""
    <div style="position:fixed; top:20px; right:20px; z-index:9999;
                background:rgba(255,255,255,0.92); padding:12px;
                border-radius:10px; font-size:13px;">
      <b>{map_title}</b><br>
      진할수록 많음<br>
      범위: {vmin:.2f} ~ {vmax:.2f}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # =========================
    # Top10 박스
    # =========================
    if show_top10:
        top10 = df.sort_values("value", ascending=False).head(10)
        rows = ""
        for i, r in enumerate(top10.itertuples(), 1):
            rows += f"{i}. {r.grid_id} ({r.value:.2f})<br>"

        top10_html = f"""
        <div style="position:fixed; top:150px; right:20px; z-index:9999;
                    background:rgba(255,255,255,0.92); padding:12px;
                    border-radius:10px; font-size:12px; max-width:260px;">
          <b>Top-10 격자</b><br>
          {rows}
        </div>
        """
        m.get_root().html.add_child(folium.Element(top10_html))

    # =========================
    # 격자 렌더링
    # =========================
    half = CELL_SIZE_M / 2
    for r in df.itertuples():
        color = _red_color_from_value(r.value, vmin, vmax)
        minx, maxx = r.center_x_m - half, r.center_x_m + half
        miny, maxy = r.center_y_m - half, r.center_y_m + half
        sw_lon, sw_lat = to_latlon.transform(minx, miny)
        ne_lon, ne_lat = to_latlon.transform(maxx, maxy)

        folium.Rectangle(
            bounds=[[sw_lat, sw_lon], [ne_lat, ne_lon]],
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            weight=0,
            tooltip=f"{r.grid_id}: {r.value:.2f}",
        ).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"[DONE] saved: {out_html}")