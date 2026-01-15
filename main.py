from dotenv import load_dotenv
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.pipeline_geo import geo
from src.grid import make_predata_and_meta_csv
from src.viz_grid_map import make_monthly_heatmaps
from src.result import make_result
from src.make_features import make_features
from src.train_rf import train_rf
from src.predict_rf import predict_rf
from src.visualize_pred import visualize_pred
from src.reverse_geocode_top10 import reverse_geocode_top10


def ml_pipeline():
    """ML 파이프라인: features -> train -> predict"""
    make_features()
    train_rf()
    predict_rf()
    visualize_pred(show=True)  # 원하면 주석 해제
    reverse_geocode_top10()   # 원하면 주석 해제

if __name__ == "__main__":
    print("\n=== GEO / GRID PIPELINE ===")
    # geo()
    # make_predata_and_meta_csv()
    # make_monthly_heatmaps(opacity=0.5, out_dir="map")
    # make_result()
    ml_pipeline()