import os
import pandas as pd
import requests
from pyproj import Transformer
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple, List


# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

PRED_PATH = "data/pred_12.csv"
META_PATH = "data/grid_meta.csv"
OUT_PATH = "data/top10_with_address.csv"


# Transform from meter-based CRS to lat/lon
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)


# Load prediction and grid metadata
def load_and_merge(pred_path: str = PRED_PATH, meta_path: str = META_PATH) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(meta_path)
    df = pred.merge(meta, on="grid_id", how="left")
    df = df.dropna(subset=["center_x_m", "center_y_m", "count"]).copy()
    return df


# Convert grid center to lat/lon
def to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
    lon, lat = transformer.transform(x_m, y_m)
    return float(lat), float(lon)


# Reverse geocode coordinates to address
def reverse_geocode(lat: float, lon: float, api_key: str, timeout: float = 8.0) -> Optional[str]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": api_key, "language": "ko"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK":
            return None
        results = data.get("results") or []
        if not results:
            return None
        return results[0].get("formatted_address")
    except Exception:
        return None


# Reverse geocode top-N predicted grids
def reverse_geocode_top10(
    pred_path: str = PRED_PATH,
    meta_path: str = META_PATH,
    out_path: str = OUT_PATH,
    topn: int = 10,
) -> Path:
    df = load_and_merge(pred_path, meta_path)

    top = df.sort_values("count", ascending=False).head(topn).copy()
    top = top.reset_index(drop=True)

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY 환경변수가 필요합니다. (.env 또는 환경변수에 설정)"
        )

    lats: List[float] = []
    lons: List[float] = []
    addrs: List[Optional[str]] = []

    for _, row in top.iterrows():
        lat, lon = to_latlon(float(row["center_x_m"]), float(row["center_y_m"]))
        lats.append(lat)
        lons.append(lon)
        addrs.append(reverse_geocode(lat, lon, GOOGLE_API_KEY))

    top["lat"] = lats
    top["lon"] = lons
    top["address"] = addrs

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    top[["grid_id", "count", "lat", "lon", "address"]].to_csv(out_p, index=False)

    print("\n[TOP GRID ADDRESS SAVED]")
    print(top[["grid_id", "count", "address"]])

    print(f"[DONE] top{topn} 주소 결과 저장: {out_p}")
    return out_p


def main():
    reverse_geocode_top10()


if __name__ == "__main__":
    main()