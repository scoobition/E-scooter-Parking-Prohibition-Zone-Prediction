# main.py
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# =========================
# import
# =========================
from src.pipeline_geo import geo
from src.grid import make_predata_and_meta_csv
from src.make_features import make_features
from src.train_rf import train_rf
from src.predict_rf import predict_rf
from src.reverse_geocode_top10 import reverse_geocode_top10
from src.viz_grid_map import make_grid_heatmap_html, make_grid_error_heatmap_html



# =========================
# PIPELINES
# =========================
def geo_pipeline():
    """
    원본 CSV → 지오코딩 → after.csv
    """
    print("\n=== GEO PIPELINE ===")
    geo()


def grid_pipeline():
    """
    after.csv → predata.csv + grid_meta.csv
    """
    print("\n=== GRID PIPELINE ===")
    make_predata_and_meta_csv()


def ml_pipeline():
    """
    학습 + 예측 파이프라인
    """
    print("\n=== ML PIPELINE ===")
    make_features()
    train_rf()
    predict_rf()

def _print_similarity_and_error(real_csv: str, pred_csv: str, value_col: str = "count", topk: int = 10):
    real = pd.read_csv(real_csv)[["grid_id", value_col]].rename(columns={value_col: "real"})
    pred = pd.read_csv(pred_csv)[["grid_id", value_col]].rename(columns={value_col: "pred"})

    df = real.merge(pred, on="grid_id", how="inner").dropna()
    if len(df) == 0:
        print("[WARN] 실제/예측 공통 grid_id가 없습니다.")
        return

    y = df["real"].to_numpy(dtype=float)
    yhat = df["pred"].to_numpy(dtype=float)

    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # 오차율(MAPE): real=0인 곳은 제외
    denom_mask = np.abs(y) > 1e-12
    if np.any(denom_mask):
        mape = float(np.mean(np.abs((yhat[denom_mask] - y[denom_mask]) / y[denom_mask])) * 100.0)
    else:
        mape = float("nan")

    # 유사도(상관계수)
    if np.std(y) < 1e-12 or np.std(yhat) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y, yhat)[0, 1])

    # Top-K 겹침(랭킹 유사)
    top_real = set(df.sort_values("real", ascending=False).head(topk)["grid_id"].tolist())
    top_pred = set(df.sort_values("pred", ascending=False).head(topk)["grid_id"].tolist())
    jaccard = float(len(top_real & top_pred) / max(1, len(top_real | top_pred)))

    print("\n===== PRED vs REAL (12월) 유사도/오차 =====")
    print(f"- 공통 격자 수: {len(df)}")
    print(f"- MAE : {mae:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- 오차율(MAPE): {mape:.2f}%")
    print(f"- 피어슨 상관계수: {corr:.4f}")
    print(f"- Top-{topk} Jaccard(겹침 유사도): {jaccard:.4f}")
    print("========================================\n")

def map_pipeline():
    print("\n=== MAP PIPELINE ===")

    # 1️⃣ 7~11월 실제
    for m in [7, 8, 9, 10, 11]:
        make_grid_heatmap_html(
            month=m,
            out_html=f"map/grid_heatmap_200m_{m}.html",
        )

    # ✅ 12월 실제/예측 공통 스케일 계산
    real_csv = "data/predata_12.csv"
    pred_csv = "data/pred_12.csv"

    df_real = pd.read_csv(real_csv)[["grid_id", "count"]].rename(columns={"count": "real"})
    df_pred = pd.read_csv(pred_csv)[["grid_id", "count"]].rename(columns={"count": "pred"})

    # 공통 스케일은 두 값 전체에서 min/max를 잡는게 가장 직관적
    combined = pd.concat([df_real["real"], df_pred["pred"]], axis=0).dropna()
    scale_vmin = float(combined.min())
    scale_vmax = float(combined.max())

    # ✅ 터미널에 유사도/오차율 출력
    _print_similarity_and_error(real_csv, pred_csv, value_col="count", topk=10)

    # 2️⃣ 실제 12월 (✅ 동일 스케일 적용)
    make_grid_heatmap_html(
        value_csv=real_csv,
        value_col="count",
        title="12월 실제 견인 발생",
        out_html="map/real_12.html",
        show_top10=True,
        scale_vmin=scale_vmin,
        scale_vmax=scale_vmax,
    )

    # 3️⃣ 예측 12월 (✅ 동일 스케일 적용)
    make_grid_heatmap_html(
        value_csv=pred_csv,
        value_col="count",
        title="12월 견인 위험 예측",
        out_html="map/pred_12.html",
        show_top10=True,
        scale_vmin=scale_vmin,
        scale_vmax=scale_vmax,
    )

    # 4️⃣ 오차지도(예측-실제)
    make_grid_error_heatmap_html(
        real_csv=real_csv,
        pred_csv=pred_csv,
        value_col="count",
        title="12월 오차지도 (예측 - 실제)",
        out_html="map/error_12.html",
        show_top10=True,
        # scale_absmax=??  # 필요하면 고정 가능 (예: 20)
    )



def analysis_pipeline():
    """
    예측 결과 분석용 (Top-10 주소)
    """
    print("\n=== ANALYSIS PIPELINE ===")
    reverse_geocode_top10()


# =========================
# MAIN BRANCH
# =========================
if __name__ == "__main__":

    """
    실행 전략:
    - 처음 한 번만: geo_pipeline(), grid_pipeline()
    - 이후 반복 실험: ml_pipeline() + map_pipeline()
    """
    def printing():
        print("=======================")
        print("1. geocoding pipeline")
        print("2. grid pipeline")
        print("3. model testing")
        print("4. result analization")
        print("5. vizualization")
        print("6. all")
        print("7. exit")
        print("=======================")
        command = input("원하시는 작업의 번호를 눌러주세요: ")
        return command
    command = printing()
    while(command != "7"):
        if command == "1":
            geo_pipeline()
            command = printing()
        elif command == "2":
            grid_pipeline()
            command = printing()
        elif command == "3":
            ml_pipeline()
            command = printing()
        elif command == "4":
            analysis_pipeline()
            command = printing()
        elif command == "5":
            map_pipeline()
            command = printing()
        elif command == "6":
            geo_pipeline()
            grid_pipeline()
            ml_pipeline()
            analysis_pipeline()
            map_pipeline()
            command = printing()
        else:
            print("please enter the right number of the command")
            command = printing()
