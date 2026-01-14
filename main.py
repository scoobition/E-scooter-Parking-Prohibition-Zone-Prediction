from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src.pipeline_geo import geo

if __name__ == "__main__":
    geo()
