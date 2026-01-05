import pandas as pd
from geopy.geocoders import Nominatim
import osmnx as ox
from scipy.spatial import cKDTree
import numpy as np
import time

# 원본 데이터 로드. 파일이 없을 경우 대비.
try:
    df = pd.read_csv("25_first_half.csv")
except FileNotFoundError:
    print("Error: '25_first_half.csv' not found. Please make sure the source data file exists.")
    df = pd.DataFrame(columns=["구정보", "주소"])

def limiting():
    """'노원구', '성북구', '강북구' 데이터만 필터링합니다."""
    region_list = ["노원구", "성북구", "강북구"]
    filtered_df = df[df["구정보"].isin(region_list)].reset_index(drop=True)
    print(f"limiting() - 필터링된 민원 데이터 수: {len(filtered_df)}") # Debug print
    return filtered_df

# Nominatim 지오코더 설정
geolocator = Nominatim(user_agent="e-scooter-project", timeout=10)

def geocode_address(address, retries=3):
    """
    주소를 좌표로 변환합니다. 변환 실패 시 None을 반환합니다.
    (참고: Nominatim은 대량 요청 시 속도가 느리고 제한될 수 있습니다.)
    """
    for i in range(retries):
        try:
            # print(f"주소 변환 시도 ({i + 1}/{retries}): {address}") # Suppress verbose prints during normal run
            loc = geolocator.geocode(address)
            if loc:
                # print(" -> 성공!") # Suppress verbose prints during normal run
                return loc.longitude, loc.latitude
            else:
                # print(" -> 실패: 주소를 찾을 수 없음") # Suppress verbose prints during normal run
                pass
        except Exception as e:
            # print(f" -> 실패: 오류 발생 ({e})") # Suppress verbose prints during normal run
            time.sleep(1) # 오류 발생 시 잠시 대기 후 재시도
    # print(f" -> 최종 실패: {address}") # Suppress verbose prints during normal run
    return None, None

def get_pois_from_osm(districts):
    """
    지정된 구(districts) 목록에 대해 OpenStreetMap에서 횡단보도와 버스정류장 POI를 가져옵니다.
    """
    print(f"\nOSM에서 {', '.join(districts)} 지역의 POI를 수집합니다...")
    tags = {"highway": ["crossing", "bus_stop"]}
    try:
        gdf_pois = ox.features_from_place(
            [f'{d}, Seoul, South Korea' for d in districts], tags
        )
        # Point 객체만 필터링
        gdf_pois = gdf_pois[gdf_pois.geometry.type == 'Point']
        print(f"get_pois_from_osm() - 총 {len(gdf_pois)}개의 POI (횡단보도, 버스정류장)를 수집했습니다.")
        return gdf_pois
    except Exception as e:
        print(f"OSM에서 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def find_nearest_poi_coords(complaint_coords, poi_kdtree, poi_points):
    """
    주어진 민원 좌표에 가장 가까운 POI의 좌표를 찾습니다.
    """
    dist, idx = poi_kdtree.query(complaint_coords, k=1)
    nearest_poi_point = poi_points[idx]
    return nearest_poi_point.x, nearest_poi_point.y

def set_data():
    """
    1. 민원 데이터를 특정 구로 필터링합니다.
    2. 해당 구의 횡단보도/버스정류장 POI를 OSM에서 가져옵니다.
    3. 각 민원 주소를 지오코딩하여 좌표를 얻습니다.
    4. 각 민원에 대해 가장 가까운 POI의 좌표를 찾습니다.
    5. 이 POI 좌표들을 DBSCAN에 사용할 데이터로 저장합니다.
    """
    # 1. 민원 데이터 필터링
    region_list = ["노원구", "성북구", "강북구"]
    complaints_df = limiting()
    if complaints_df.empty:
        print(f"set_data() - 분석할 데이터가 없습니다. ({', '.join(region_list)} 지역 데이터 확인)")
        empty_df = pd.DataFrame(columns=['경도', '위도'])
        output_path = "data/processed_data.csv"
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    # 2. OSM에서 POI 데이터 가져오기
    poi_gdf = get_pois_from_osm(region_list)
    if poi_gdf.empty:
        print("set_data() - POI 데이터를 가져오지 못해 분석을 중단합니다.")
        empty_df = pd.DataFrame(columns=['경도', '위도'])
        output_path = "data/processed_data.csv"
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    # 3. 민원 주소 지오코딩
    print("\n민원 주소를 좌표로 변환합니다...")
    original_complaint_count = len(complaints_df)
    complaints_df[["민원_경도", "민원_위도"]] = complaints_df["주소"].apply(
        lambda x: pd.Series(geocode_address(x))
    )
    complaints_df.dropna(subset=["민원_경도", "민원_위도"], inplace=True)
    print(f"geocode_address() - 원래 민원 {original_complaint_count}건 중 좌표로 변환 성공한 민원 수: {len(complaints_df)}건")
    
    if complaints_df.empty:
        print("set_data() - 좌표로 변환할 수 있는 민원 주소가 없어 분석을 중단합니다.")
        empty_df = pd.DataFrame(columns=['경도', '위도'])
        output_path = "data/processed_data.csv"
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    print(f"\n좌표로 변환된 민원 {len(complaints_df)}건에 대해 최근접 POI를 찾습니다.")

    # 4. 가장 가까운 POI 찾기 (KDTree 사용)
    poi_points = list(poi_gdf.geometry)
    poi_coords_for_tree = np.array([(p.y, p.x) for p in poi_points]) # (lat, lon)
    
    # Handle case where there are no valid POI coordinates for KDTree
    if poi_coords_for_tree.size == 0:
        print("set_data() - KDTree를 생성할 POI 좌표가 없습니다. 분석을 중단합니다.")
        empty_df = pd.DataFrame(columns=['경도', '위도'])
        output_path = "data/processed_data.csv"
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    poi_kdtree = cKDTree(poi_coords_for_tree)
    
    nearest_poi_coords_list = []
    for idx, row in complaints_df.iterrows(): # Iterate through complaints_df
        complaint_coords = (row['민원_위도'], row['민원_경도'])
        try:
            nearest_poi_coords_list.append(find_nearest_poi_coords(complaint_coords, poi_kdtree, poi_points))
        except Exception as e:
            print(f"find_nearest_poi_coords() - 민원 좌표 ({complaint_coords})에 대한 최근접 POI 찾기 실패: {e}")
            # Optionally, you can add None or some placeholder if a POI cannot be found
            # For now, we'll just skip adding to the list. If all fail, list will be empty.

    print(f"find_nearest_poi_coords() - 찾은 최근접 POI 좌표 수 (중복 포함): {len(nearest_poi_coords_list)}")
    
    # 5. POI 좌표를 새 데이터프레임으로 저장
    result_df = pd.DataFrame(nearest_poi_coords_list, columns=['경도', '위도'])
    original_unique_count = len(result_df)
    result_df.drop_duplicates(inplace=True)
    print(f"set_data() - 중복 제거 전 POI 좌표 수: {original_unique_count}, 중복 제거 후 유니크한 POI 좌표 수: {len(result_df)}")

    if result_df.empty:
        print("set_data() - 최종적으로 저장할 POI 좌표가 없습니다.")
        output_path = "data/processed_data.csv"
        empty_df = pd.DataFrame(columns=['경도', '위도'])
        empty_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return

    output_path = "data/processed_data.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"DBSCAN을 위한 최종 데이터가 '{output_path}'에 저장되었습니다. 총 {len(result_df)}개.")
