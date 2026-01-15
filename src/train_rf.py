import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/features.csv"
MODEL_PATH = "model_rf.pkl"

if __name__ == "__main__":
    # 1. 데이터 로드
    df = pd.read_csv(DATA_PATH)

    # 2. 학습에 쓸 월만 선택 (9, 10월)
    train = df[df["month"].isin([9, 10])].copy()

    # 3. 정답(y) = 다음 달 count
    train["y"] = train.groupby("grid_id")["count"].shift(-1)

    # 4. 결측 제거
    train = train.dropna(subset=["count_t", "count_t-1", "count_t-2", "y"])

    # 5. X, y 분리
    X = train[["count_t", "count_t-1", "count_t-2"]]
    y = train["y"]

    print(f"[INFO] 학습 샘플 수: {len(X)}")

    # 6. 랜덤 포레스트 모델 정의
    model = RandomForestRegressor(
        n_estimators=300, # 트리 개수
        max_depth=8, # 트리 깊이 제한
        random_state=42,
        n_jobs=-1
    )

    # 7. 학습
    model.fit(X, y)

    # 8. 모델 저장
    joblib.dump(model, MODEL_PATH)
    print(f"[DONE] 모델 저장 완료: {MODEL_PATH}")
