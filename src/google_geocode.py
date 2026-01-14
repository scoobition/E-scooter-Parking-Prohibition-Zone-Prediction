# src/google_geocode.py
from dotenv import load_dotenv
load_dotenv()

import os
import time
import pandas as pd
import requests
from typing import Optional, Tuple


# -------------------------
# 0) 환경변수 키 읽기
# -------------------------
def _get_google_key() -> str:
    """
    Google Geocoding API Key
    - .env에 GOOGLE_MAPS_API_KEY=... 형태로 저장 추천
    """
    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY (또는 GOOGLE_API_KEY) 환경변수가 필요합니다.")
    return key


def _get_juso_key() -> str:
    """
    (선택) 정부 도로명주소 검색 API 승인키
    - 있으면 지번/불완전 주소 → 도로명주소로 한번 정리한 뒤 구글 지오코딩 정확도가 올라갈 수 있습니다.
    - 없으면 원문 주소만으로 바로 구글 지오코딩합니다.
    """
    return os.getenv("JUSO_CONFM_KEY", "")


# -------------------------
# 1) 정부 도로명주소 검색 API
#    (지번/키워드 → 도로명주소 roadAddr)
# -------------------------
def jibun_to_roadaddr(keyword_addr: str, confm_key: str, timeout=20) -> Optional[str]:
    """
    keyword_addr(지번 포함 가능)를 넣으면 가장 적절한 도로명주소(roadAddr) 반환.
    없으면 None.
    """
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


# -------------------------
# 2) Google Geocoding
# -------------------------
def google_geocode_one(
    address: str,
    api_key: str,
    timeout=20,
    region="kr",
    language="ko",
) -> Tuple[Optional[float], Optional[float]]:
    """
    Google Geocoding API (server-side)
    - endpoint: https://maps.googleapis.com/maps/api/geocode/json
    - result: (lat, lon)
    """
    if not address:
        return None, None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key,
        "region": region,     # 결과를 한국 쪽으로 유도
        "language": language, # 응답 언어(좌표에는 영향 거의 없지만 디버그에 도움)
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
        # 상태코드 참고: OK, ZERO_RESULTS, OVER_QUERY_LIMIT, REQUEST_DENIED, INVALID_REQUEST, UNKNOWN_ERROR 등
        err = data.get("error_message")
        if status != "ZERO_RESULTS":  # ZERO_RESULTS는 흔하니 너무 시끄럽지 않게
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


# -------------------------
# 3) 캐시 I/O
# -------------------------
def load_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        # 예전 캐시에 roadAddr 컬럼이 없을 수 있어서 보정
        if "roadAddr" not in cache.columns:
            cache["roadAddr"] = pd.NA
        # 필수 컬럼 보정
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


# -------------------------
# 4) (핵심) 지번 → 도로명 변환 후 지오코딩
# -------------------------
def geocode_with_roadaddr_fallback(addr: str, api_key: str, confm_key: str):
    """
    1) (선택) 정부 API로 roadAddr(도로명) 얻기
    2) roadAddr로 구글 지오코딩
    3) 실패 시 원문 주소로 구글 지오코딩(백업)
    """
    road = jibun_to_roadaddr(addr, confm_key) if confm_key else None

    # ✅ 디버그(처음 몇 개만)
    if not hasattr(geocode_with_roadaddr_fallback, "_dbg"):
        geocode_with_roadaddr_fallback._dbg = 0
    if geocode_with_roadaddr_fallback._dbg < 5:
        print("[DEBUG] raw  =", addr)
        print("[DEBUG] road =", road)
        geocode_with_roadaddr_fallback._dbg += 1

    # 1차: 도로명으로 구글 지오코딩
    if road:
        lat, lon = google_geocode_one(road, api_key)
        if lat is not None:
            return road, lat, lon

    # 2차(백업): 원문 주소로 구글 지오코딩
    lat, lon = google_geocode_one(addr, api_key)
    return road, lat, lon


def fill_cache_for_addresses(
    unique_addrs,
    cache_path="data/geocode_cache.csv",
    sleep_sec=0.05,
    print_every=200,
    retry_unknown_error=2,
):
    """
    unique_addrs: df["주소_clean"].unique() 같은 iterable
    - 캐시에 없는 주소만 추가로 지오코딩
    - 결과는 cache DF(주소_clean, roadAddr, lat, lon)
    """
    api_key = _get_google_key()
    confm_key = _get_juso_key()

    cache = load_cache(cache_path)

    # 이미 처리된 주소는 스킵
    cache_map = set(cache["주소_clean"].astype(str).tolist())

    need = [a for a in unique_addrs if str(a) not in cache_map]
    print(f"[INFO] 새로 처리할 주소 수: {len(need)}")

    new_rows = []
    for i, addr in enumerate(need, 1):
        addr = str(addr).strip()

        # UNKNOWN_ERROR 같은 케이스는 약간 재시도하면 성공하는 경우가 있음
        road = lat = lon = None
        for t in range(retry_unknown_error + 1):
            road, lat, lon = geocode_with_roadaddr_fallback(addr, api_key, confm_key)
            if lat is not None:
                break
            # 재시도 텀(점진적으로 증가)
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
