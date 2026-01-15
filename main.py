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

def run_step(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[ERROR] 파이프라인 중단")
        sys.exit(1)

if __name__ == "__main__":
    print("\n=== GEO / GRID PIPELINE ===")
    # geo()
    # make_predata_and_meta_csv()
    # make_monthly_heatmaps(opacity=0.5, out_dir="map")
    # make_result()

    print("\n=== ML PIPELINE ===")
    run_step([sys.executable, "src/make_features.py"])
    run_step([sys.executable, "src/train_rf.py"])
    run_step([sys.executable, "src/predict_rf.py"])

    print("\n[DONE] 전체 파이프라인 완료")
    
