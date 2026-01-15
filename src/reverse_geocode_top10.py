import os
import pandas as pd
import requests
from pyproj import Transformer
from dotenv import load_dotenv
from pathlib import Path

# =========================
# .env 강제 로드 (프로젝트 루트)
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if GOOGLE_API_KEY is None:
    raise RuntimeError(f"GOOGLE_MAPS_API_KEY not found in {ENV_PATH}")


# =========================
# 파일 경로
# =========================
PRED_PATH = "data/pred_12.csv"
META_PATH = "data/grid_meta.csv"


# =========================
# 좌표 변환기 (EPSG:5179 → WGS84)
# =========================
transformer = Transformer.from_crs(
    "EPSG:5179", "EPSG:4326", always_xy=True
)


# =========================
# 데이터 로드
# =========================
def load_and_merge():
    pred = pd.read_csv(PRED_PATH)
    meta = pd.read_csv(META_PATH)

    df = pred.merge(meta, on="grid_id", how="left")
    df = df.dropna(subset=["center_x_m", "center_y_m", "pred_12"]).copy()
    return df


# =========================
# 좌표 변환
# =========================
def to_latlon(x, y):
    lon, lat = transformer.transform(x, y)
    return lat, lon


# =========================
# Google Reverse Geocoding
# =========================
def google_reverse_geocode(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": GOOGLE_API_KEY,
        "language": "ko"
    }

    r = requests.get(url, params=params, timeout=5)
    if r.status_code != 200:
        return None

    data = r.json()
    if not data.get("results"):
        return None

    # 가장 상세한 주소 하나 사용
    return data["results"][0]["formatted_address"]


# =========================
# 메인 로직
# =========================
def main():
    df = load_and_merge()

    # Top-10 격자 추출
    top10 = df.sort_values("pred_12", ascending=False).head(10).copy()
    top10 = top10.reset_index(drop=True)
    top10.index += 1

    lats, lons, addresses = [], [], []

    for x, y in zip(top10["center_x_m"], top10["center_y_m"]):
        lat, lon = to_latlon(x, y)
        addr = google_reverse_geocode(lat, lon)

        lats.append(lat)
        lons.append(lon)
        addresses.append(addr)

    top10["lat"] = lats
    top10["lon"] = lons
    top10["address"] = addresses

    print("\n[TOP 10 HIGH-RISK GRIDS WITH ADDRESS]")
    print(top10[["grid_id", "pred_12", "address"]])


# =========================
# 실행
# =========================
if __name__ == "__main__":
    main()
