import re
import pandas as pd


# Normalize address string
def clean_address(addr) -> str:
    if pd.isna(addr):
        return ""
    addr = str(addr).strip()
    addr = re.sub(r"\(.*?\)", "", addr)   # Remove parentheses
    addr = re.sub(r"\s+", " ", addr)      # Normalize whitespace
    return addr.strip()
