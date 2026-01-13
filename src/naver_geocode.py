import os
import time
import pandas as pd
import requests

def _get_keys():
    cid = os.getenv("p6w7jxc5tg")
    csec = os.getenv("1USKRYn9jRP6EqsA8aOQ5b0ElkrRtc0e1HtDRBDz")
    if not cid or not csec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다.")
    return cid, csec

def naver_geocode_one(address: str, cid: str, csec: str):
    if not address:
        return None, None

    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": cid,
        "X-NCP-APIGW-API-KEY": csec,
    }
    params = {"query": address}

    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        return None, None

    data = r.json()
    addrs = data.get("addresses", [])
    if not addrs:
        return None, None

    try:
        return float(addrs[0]["y"]), float(addrs[0]["x"])  # lat, lon
    except:
        return None, None

def load_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path).drop_duplicates("주소_clean")
    else:
        cache = pd.DataFrame(columns=["주소_clean", "lat", "lon"])
    return cache

def save_cache(cache: pd.DataFrame, cache_path: str):
    cache.to_csv(cache_path, index=False, encoding="utf-8-sig")

def fill_cache_for_addresses(unique_addrs, cache_path="data/geocode_cache.csv", sleep_sec=0.05):
    cid, csec = _get_keys()

    cache = load_cache(cache_path)
    cache_map = dict(zip(cache["주소_clean"], zip(cache["lat"], cache["lon"])))

    need = [a for a in unique_addrs if a not in cache_map]
    print(f"[INFO] 새로 지오코딩할 주소 수: {len(need)}")

    new_rows = []
    for i, addr in enumerate(need, 1):
        lat, lon = naver_geocode_one(addr, cid, csec)
        new_rows.append({"주소_clean": addr, "lat": lat, "lon": lon})
        if i % 200 == 0:
            print(f"[INFO] geocoding {i}/{len(need)}")
        time.sleep(sleep_sec)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache = cache.drop_duplicates("주소_clean", keep="last")
        save_cache(cache, cache_path)

    return cache
