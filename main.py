from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.pipeline_geo import geo
from src.grid import make_predata_and_meta_csv

if __name__ == "__main__":
    # geo()
    make_predata_and_meta_csv()