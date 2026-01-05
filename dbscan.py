from data import set_data
import numpy as np
import hdbscan

def apply_dbscan(df):
    print(df)

    # 결측치 제거
    df = df.dropna(subset=["경도", "위도"]).reset_index(drop=True)

    # 좌표 추출 (경도, 위도)
    coords = df[["경도", "위도"]].values

    # haversine 거리 사용 → radian 변환 필수
    coords_rad = np.radians(coords)

    # HDBSCAN 적용
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,   # 클러스터 최소 크기 (작게 = 많이 생김)
        min_samples=1,        # 낮출수록 노이즈 감소
        metric="haversine"
    )

    df["dbscan_label"] = clusterer.fit_predict(coords_rad)

    # 중간 저장
    df.to_csv("data/dbscan.csv", index=False, encoding="utf-8-sig")

    # 클러스터 통계 출력
    labels = df["dbscan_label"]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print("HDBSCAN 클러스터 수:", n_clusters)
    print("노이즈 수:", n_noise)

    return df
