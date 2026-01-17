# main.py
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# Import pipelines
from src.pipeline_geo import geo
from src.grid import make_predata_and_meta_csv
from src.make_features import make_features
from src.train_rf import train_rf
from src.predict_rf import predict_rf
from src.reverse_geocode_top10 import reverse_geocode_top10
from src.viz_grid_map import make_grid_heatmap_html, make_grid_error_heatmap_html


# Evaluate prediction error (MAE / RMSE)
def error_check(
    real_csv="data/predata_12.csv",
    pred_csv="data/pred_12.csv",
):
    print("\n=== ERROR CHECK (MAE / RMSE) ===")

    df_real = pd.read_csv(real_csv)[["grid_id", "count"]].rename(columns={"count": "real"})
    df_pred = pd.read_csv(pred_csv)[["grid_id", "count"]].rename(columns={"count": "pred"})

    df = df_real.merge(df_pred, on="grid_id", how="inner")
    df = df.dropna(subset=["real", "pred"])

    y_true = df["real"].values
    y_pred = df["pred"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE  (Mean Absolute Error): {mae:.3f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
    }


# Run geocoding pipeline
def geo_pipeline():
    print("\n=== GEO PIPELINE ===")
    geo()


# Run grid generation pipeline
def grid_pipeline():
    print("\n=== GRID PIPELINE ===")
    make_predata_and_meta_csv()


# Run training and prediction pipeline
def ml_pipeline():
    print("\n=== ML PIPELINE ===")
    make_features()
    train_rf()
    predict_rf()


# Run map visualization pipeline
def map_pipeline():
    print("\n=== MAP PIPELINE ===")

    # Render monthly actual maps
    for m in [1,2,3,4,5,6,7, 8, 9, 10, 11]:
        make_grid_heatmap_html(
            month=m,
            out_html=f"map/grid_heatmap_200m_{m}.html",
        )

    real_csv = "data/predata_12.csv"
    pred_csv = "data/pred_12.csv"

    df_real = pd.read_csv(real_csv)[["grid_id", "count"]].rename(columns={"count": "real"})
    df_pred = pd.read_csv(pred_csv)[["grid_id", "count"]].rename(columns={"count": "pred"})

    combined = pd.concat([df_real["real"], df_pred["pred"]], axis=0).dropna()
    scale_vmin = float(combined.min())
    scale_vmax = float(combined.max())

    # Render actual December map
    make_grid_heatmap_html(
        value_csv=real_csv,
        value_col="count",
        title="12월 실제 견인 발생",
        out_html="map/real_12.html",
        show_top10=True,
        scale_vmin=scale_vmin,
        scale_vmax=scale_vmax,
    )

    # Render predicted December map
    make_grid_heatmap_html(
        value_csv=pred_csv,
        value_col="count",
        title="12월 견인 위험 예측",
        out_html="map/pred_12.html",
        show_top10=True,
        scale_vmin=scale_vmin,
        scale_vmax=scale_vmax,
    )

    # Render residual map (prediction - actual)
    make_grid_error_heatmap_html(
        real_csv=real_csv,
        pred_csv=pred_csv,
        value_col="count",
        title="12월 오차지도 (예측 - 실제)",
        out_html="map/error_12.html",
        show_top10=True,
    )


# Run analysis pipeline
def analysis_pipeline():
    print("\n=== ANALYSIS PIPELINE ===")
    reverse_geocode_top10()


# CLI entry point
if __name__ == "__main__":

    # Print menu and read command
    def printing():
        print("=======================")
        print("1. geocoding pipeline")
        print("2. grid pipeline")
        print("3. model testing")
        print("4. result analization")
        print("5. vizualization")
        print("6. error check")
        print("7. all")
        print("8. exit")
        print("=======================")
        command = input("원하시는 작업의 번호를 눌러주세요: ")
        return command

    command = printing()
    while(command != "8"):
        if command == "1":
            # geo_pipeline()
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
            error_check()
            command = printing()
        elif command == "7":
            geo_pipeline()
            grid_pipeline()
            ml_pipeline()
            analysis_pipeline()
            map_pipeline()
            error_check()
            command = printing()
        else:
            print("please enter the right number of the command")
            command = printing()