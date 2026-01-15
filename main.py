from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.pipeline_geo import geo
from src.grid import make_predata_and_meta_csv
from src.viz_grid_map import make_monthly_heatmaps
from src.result import make_result

if __name__ == "__main__":
    # geo()
    # make_predata_and_meta_csv()
    # make_monthly_heatmaps(opacity=0.5, out_dir="map")
    make_result()