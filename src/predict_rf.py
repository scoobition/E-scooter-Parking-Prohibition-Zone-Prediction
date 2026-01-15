import pandas as pd
import joblib

DATA_PATH = "data/features.csv"
MODEL_PATH = "model_rf.pkl"
OUT_PATH = "data/pred_12.csv"

if __name__ == "__main__":
    # 1. 데이터 로드
    df = pd.read_csv(DATA_PATH)

    # 2. 학습된 모델 로드
    model = joblib.load(MODEL_PATH)

    # 3. 12월 예측에 쓸 입력 (month == 11)
    pred_df = df[df["month"] == 11].copy()

    # 결측 남아있으면 제거
    pred_df = pred_df.dropna(
        subset=["count_t", "count_t-1", "count_t-2"]
    )

    # 4. 입력 X 구성
    X_pred = pred_df[["count_t", "count_t-1", "count_t-2"]]

    print(f"[INFO] 12월 예측 대상 격자 수: {len(X_pred)}")

    # 5. 예측
    pred_df["pred_12"] = model.predict(X_pred)

    # 6. 결과 저장
    pred_df[["grid_id", "pred_12"]].to_csv(
        OUT_PATH, index=False
    )

    print(f"[DONE] 12월 예측 결과 저장: {OUT_PATH}")
