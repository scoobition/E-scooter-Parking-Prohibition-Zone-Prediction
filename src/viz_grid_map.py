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

def _diverging_color_from_residual(residual: float, vabsmax: float) -> str:
    """
    residual = pred - real
    - residual > 0 : 과대예측 (빨강)
    - residual < 0 : 과소예측 (파랑)
    - 0 근처 : 흰색
    """
    if vabsmax <= 0:
        t = 0.0
    else:
        t = abs(residual) / vabsmax
        t = max(0.0, min(1.0, t))

    # t=0 -> 흰색(255), t=1 -> 진한 색(55 정도)
    fade = int(255 - 200 * t)  # 255 -> 55

    if residual >= 0:
        # 빨강 계열: (255, fade, fade)
        return f"#{255:02x}{fade:02x}{fade:02x}"
    else:
        # 파랑 계열: (fade, fade, 255)
        return f"#{fade:02x}{fade:02x}{255:02x}"


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
    # ✅ 추가: 색 스케일 고정(실제/예측 지도 동일 진하기)
    scale_vmin: Optional[float] = None,
    scale_vmax: Optional[float] = None,
):
    if value_csv is None and month is None:
        raise ValueError("month 또는 value_csv 중 하나는 반드시 필요합니다.")
    if out_html is None:
        raise ValueError("out_html은 반드시 필요합니다.")

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

    # ✅ 데이터 자체 범위(참고용)
    vmin_data = float(df["value"].min())
    vmax_data = float(df["value"].max())

    # ✅ 실제/예측 스케일 통일용 범위
    vmin = float(scale_vmin) if scale_vmin is not None else vmin_data
    vmax = float(scale_vmax) if scale_vmax is not None else vmax_data

    # 안전장치
    if vmax <= vmin:
        vmax = vmin + 1e-9

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
                border-radius:10px; font-size:13px; line-height:1.35;">
      <b>{map_title}</b><br>
      진할수록 많음<br>
      <span style="font-size:12px;">
        색 스케일: {vmin:.2f} ~ {vmax:.2f}<br>
        데이터 범위: {vmin_data:.2f} ~ {vmax_data:.2f}
      </span>
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
                    border-radius:10px; font-size:12px; max-width:260px; line-height:1.35;">
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
        # ✅ vmin/vmax를 스케일로 사용
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

    # out_html 경로 생성
    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    m.save(out_html)
    print(f"[DONE] saved: {out_html}")

def make_grid_error_heatmap_html(
    *,
    real_csv: str,
    pred_csv: str,
    value_col: str = "count",
    meta_csv: str = "data/grid_meta.csv",
    out_html: str,
    title: str = "오차지도 (예측 - 실제)",
    opacity: float = 0.45,
    max_cells: Optional[int] = 20000,
    # ✅ 오차 스케일(색 진하기) 고정하고 싶으면 사용
    scale_absmax: Optional[float] = None,
    show_top10: bool = True,
):
    """
    오차지도: residual = pred - real
    - 빨강: 과대예측(예측이 더 큼)
    - 파랑: 과소예측(예측이 더 작음)
    """

    meta = pd.read_csv(meta_csv)

    df_real = pd.read_csv(real_csv)[["grid_id", value_col]].rename(columns={value_col: "real"})
    df_pred = pd.read_csv(pred_csv)[["grid_id", value_col]].rename(columns={value_col: "pred"})

    df = df_real.merge(df_pred, on="grid_id", how="inner").dropna()
    df["residual"] = df["pred"].astype(float) - df["real"].astype(float)

    df = df.merge(meta, on="grid_id", how="left")
    df = df.dropna(subset=["center_x_m", "center_y_m", "residual"])

    # 너무 많으면 |오차| 큰 것 위주로 제한(오차지도 목적에 맞음)
    if max_cells and len(df) > max_cells:
        df = df.reindex(df["residual"].abs().sort_values(ascending=False).head(max_cells).index)

    absmax_data = float(df["residual"].abs().max()) if len(df) else 0.0
    absmax = float(scale_absmax) if scale_absmax is not None else absmax_data
    if absmax <= 0:
        absmax = 1e-9

    # 지도 중심
    to_latlon = Transformer.from_crs(DST_CRS, SRC_CRS, always_xy=True)
    lon, lat = to_latlon.transform(df["center_x_m"].mean(), df["center_y_m"].mean())

    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    # 범례
    legend_html = f"""
    <div style="position:fixed; top:20px; right:20px; z-index:9999;
                background:rgba(255,255,255,0.92); padding:12px;
                border-radius:10px; font-size:13px; line-height:1.35;">
      <b>{title}</b><br>
      빨강: 과대예측 / 파랑: 과소예측<br>
      <span style="font-size:12px;">
        색 스케일(|오차|): 0 ~ {absmax:.2f}<br>
        데이터 최대 |오차|: {absmax_data:.2f}
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Top-10 (|오차| 큰 격자)
    if show_top10 and len(df):
        top10 = df.reindex(df["residual"].abs().sort_values(ascending=False).head(10).index)
        rows = ""
        for i, r in enumerate(top10.itertuples(), 1):
            rows += f"{i}. {r.grid_id} (diff {r.residual:+.2f}, real {r.real:.2f}, pred {r.pred:.2f})<br>"

        top10_html = f"""
        <div style="position:fixed; top:170px; right:20px; z-index:9999;
                    background:rgba(255,255,255,0.92); padding:12px;
                    border-radius:10px; font-size:12px; max-width:320px; line-height:1.35;">
          <b>|오차| Top-10 격자</b><br>
          {rows}
        </div>
        """
        m.get_root().html.add_child(folium.Element(top10_html))

    # 격자 렌더
    half = CELL_SIZE_M / 2
    for r in df.itertuples():
        color = _diverging_color_from_residual(r.residual, absmax)

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
            tooltip=f"{r.grid_id} | real={r.real:.2f}, pred={r.pred:.2f}, diff={r.residual:+.2f}",
        ).add_to(m)

    out_dir = os.path.dirname(out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    m.save(out_html)
    print(f"[DONE] saved: {out_html}")
