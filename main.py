# main.py
from dotenv import load_dotenv
from pathlib import Path

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
from src.viz_grid_map import make_grid_heatmap_html


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


def map_pipeline():
    print("\n=== MAP PIPELINE ===")

    # 1️⃣ 7~11월 실제
    for m in [7, 8, 9, 10, 11]:
        make_grid_heatmap_html(
            month=m,
            out_html=f"map/grid_heatmap_200m_{m}.html",
            # show_top10=False,   # 과거 데이터는 Top10 생략 추천
        )

    # 2️⃣ 실제 12월
    make_grid_heatmap_html(
        value_csv="data/predata_12.csv",
        value_col="count",
        title="12월 실제 견인 발생",
        out_html="map/real_12.html",
        show_top10=True,
    )

    # 3️⃣ 예측 12월
    make_grid_heatmap_html(
        value_csv="data/pred_12.csv",
        value_col="count",
        title="12월 견인 위험 예측",
        out_html="map/pred_12.html",
        show_top10=True,
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

    # ===== 1. 최초 1회 =====
    # geo_pipeline()
    # grid_pipeline()

    # ===== 2. 모델 실험 =====
    ml_pipeline()

    # ===== 3. 결과 분석 =====
    analysis_pipeline()

    # ===== 4. 시각화 =====
    map_pipeline()