import numpy as np
import pandas as pd

# 1. 데이터 가져오기
def load_file(csv_path):
    df = pd.read_csv(csv_path)

    # 6,7열에 존재하는 모든 경도/위도 저장
    X = df.iloc[:, [6,7]].values
    return X

# 2. 초기 중심점 셋팅
def init_centroids(X, K, seed=42): # 매번 같은 결과를 위해 seed=42 설정
    np.random.seed(seed)
    index = np.random.choice(len(X), K, replace=False) # 중복X
    centroid = X[index]

    return centroid

# 3. 그룹 설정을 위한 (X - 중심점) 거리 계산
def compute_distances(X, centroid):

    n = X.shape[0]
    K = centroid.shape[0]

    distance = np.zeros((n, K)) # 거리 초기화

    # 각 점에 대해 모든 중심점과의 거리 계산
    for i in range(n):
        for j in range(K):
            distance[i, j] = np.linalg.norm(X[i] - centroid[j])

    return distance

# 4. 그룹 설정 (가장 가까운 중심점)
def assign_labels(distance):
    n = distance.shape[0]
    k = distance.shape[1]

    label = np.zeros(n, dtype=int)

    for i in range(n):
        min_dist = distance[i][0]
        min_index = 0;

        for j in range(1,k):
            if distance[i][j] < min_dist:
                min_dist = distance[i][j]
                min_index = j
        label[i] = min_index

    return label

# 5. 중심점 갱신 (각 그룹의 평균 좌표 계산)
def update_centroid(X, label, K, origin):
    new_centroid = np.zeros((K, X.shape[1]))

    for k in range(K):
        sum_x = 0
        sum_y = 0
        count = 0

        for i in range(X.shape[0]):
            if label[i] == k:
                sum_x += X[i][0]
                sum_y += X[i][1]
                count += 1
        if count > 0:
            new_centroid[k][0] = sum_x / count
            new_centroid[k][1] = sum_y / count
        else:
            new_centroid[k] = origin[k]

    return new_centroid

# 6. 3-5 과정 반복
def kmeans(X, K, max_iter=100):
    centroid = init_centroids(X, K)

    for _ in range(max_iter):
        distance = compute_distances(X, centroid)
        label = assign_labels(distance)
        new_centroid = update_centroid(X, label, K, centroid)

        # 수렴 체크 (중심점이 안 변하면 종료)
        if np.allclose(centroid, new_centroid):
            break

        centroid = new_centroid

    return label, centroid