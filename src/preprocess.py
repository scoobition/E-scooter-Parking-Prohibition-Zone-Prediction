import re
import pandas as pd

def clean_address(addr) -> str:
    if pd.isna(addr):
        return ""
    addr = str(addr).strip()
    addr = re.sub(r"\(.*?\)", "", addr)   # 괄호 제거
    addr = re.sub(r"\s+", " ", addr)      # 공백 정리
    return addr.strip()
