# src/google_geocode.py
from dotenv import load_dotenv
load_dotenv()

import os
import time
import pandas as pd
import requests
from typing import Optional, Tuple


# 0) Load API keys
def _get_google_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY (또는 GOOGLE_API_KEY) 환경변수가 필요합니다.")
    return key


def _get_juso_key() -> str:
    return os.getenv("JUSO_CONFM_KEY", "")


# 1) JUSO address lookup
def jibun_to_roadaddr(keyword_addr: str, confm_key: str, timeout=20) -> Optional[str]:
    if not keyword_addr or not confm_key:
        return None

    url = "https://www.juso.go.kr/addrlink/addrLinkApi.do"
    params = {
        "confmKey": confm_key,
        "currentPage": 1,
        "countPerPage": 5,
        "keyword": keyword_addr,
        "resultType": "json",
        "firstSort": "location",
    }

    try:
        r = requests.get(url, params=params, timeout=timeout)
    except Exception:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    results = data.get("results", {})
    common = results.get("common", {})
    if common.get("errorCode") != "0":
        print("[JUSO FAIL]", common.get("errorCode"), common.get("errorMessage"), "| keyword =", keyword_addr)
        return None

    juso_list = results.get("juso", [])
    if not juso_list:
        return None

    road = juso_list[0].get("roadAddr")
    return road or None


# 2) Google geocoding
def google_geocode_one(
    address: str,
    api_key: str,
    timeout=20,
    region="kr",
    language="ko",
) -> Tuple[Optional[float], Optional[float]]:
    if not address:
        return None, None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key,
        "region": region,
        "language": language,
    }

    try:
        r = requests.get(url, params=params, timeout=timeout)
    except Exception:
        return None, None

    if r.status_code != 200:
        print("[GOOGLE HTTP FAIL]", r.status_code, "| query =", address, "| body =", r.text[:200])
        return None, None

    try:
        data = r.json()
    except Exception:
        return None, None

    status = data.get("status")
    if status != "OK":
        err = data.get("error_message")
        if status != "ZERO_RESULTS":
            print("[GOOGLE FAIL]", status, "| query =", address, "| err =", err)
        return None, None

    results = data.get("results", [])
    if not results:
        return None, None

    try:
        loc = results[0]["geometry"]["location"]
        return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None, None


# 3) Cache I/O
def load_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)

        # Backfill missing columns
        if "roadAddr" not in cache.columns:
            cache["roadAddr"] = pd.NA
        for col in ["주소_clean", "lat", "lon"]:
            if col not in cache.columns:
                cache[col] = pd.NA

        cache = cache.drop_duplicates("주소_clean", keep="last")
    else:
        cache = pd.DataFrame(columns=["주소_clean", "roadAddr", "lat", "lon"])

    return cache


def save_cache(cache: pd.DataFrame, cache_path: str):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    cache.to_csv(cache_path, index=False, encoding="utf-8-sig")


# 4) Geocode with road-address fallback
def geocode_with_roadaddr_fallback(addr: str, api_key: str, confm_key: str):
    road = jibun_to_roadaddr(addr, confm_key) if confm_key else None

    # Debug first few cases
    if not hasattr(geocode_with_roadaddr_fallback, "_dbg"):
        geocode_with_roadaddr_fallback._dbg = 0
    if geocode_with_roadaddr_fallback._dbg < 5:
        print("[DEBUG] raw  =", addr)
        print("[DEBUG] road =", road)
        geocode_with_roadaddr_fallback._dbg += 1

    if road:
        lat, lon = google_geocode_one(road, api_key)
        if lat is not None:
            return road, lat, lon

    lat, lon = google_geocode_one(addr, api_key)
    return road, lat, lon


# 5) Fill geocode cache
def fill_cache_for_addresses(
    unique_addrs,
    cache_path="data/geocode_cache.csv",
    sleep_sec=0.05,
    print_every=200,
    retry_unknown_error=2,
):
    api_key = _get_google_key()
    confm_key = _get_juso_key()

    cache = load_cache(cache_path)

    # Skip cached addresses
    cache_map = set(cache["주소_clean"].astype(str).tolist())

    need = [a for a in unique_addrs if str(a) not in cache_map]
    print(f"[INFO] 새로 처리할 주소 수: {len(need)}")

    new_rows = []
    for i, addr in enumerate(need, 1):
        addr = str(addr).strip()

        road = lat = lon = None
        for t in range(retry_unknown_error + 1):
            road, lat, lon = geocode_with_roadaddr_fallback(addr, api_key, confm_key)
            if lat is not None:
                break
            time.sleep(min(1.0, sleep_sec * (2 ** t)))

        new_rows.append({"주소_clean": addr, "roadAddr": road, "lat": lat, "lon": lon})

        if print_every and i % print_every == 0:
            ok = sum(1 for r in new_rows if r["lat"] is not None)
            print(f"[INFO] processing {i}/{len(need)} (ok so far: {ok})")

        time.sleep(sleep_sec)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates("주소_clean", keep="last")
        save_cache(cache, cache_path)

    fail = cache["lat"].isna().sum()
    total = len(cache)
    print(f"[INFO] 지오코딩 실패(unique 기준): {fail}/{total} ({(fail/total*100 if total else 0):.2f}%)")

    return cache