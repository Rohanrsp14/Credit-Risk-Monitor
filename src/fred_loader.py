"""Pull macro credit-risk indicator series from FRED.

Series:
- DRCCLACBS: Delinquency Rate on Credit Card Loans, All Commercial Banks (SA)
- CORCACBS: Charge-Off Rate on Consumer Loans, All Commercial Banks (SA)

Source: Federal Reserve Economic Data (FRED), https://fred.stlouisfed.org/
Requires FRED_API_KEY in .env (see .env.example).
"""

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
import os

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FRED_SERIES_PATH = PROCESSED_DIR / "fred_series.csv"

SERIES_IDS = ["DRCCLACBS", "CORCACBS"]


def load_fred_series():
    """Pull SERIES_IDS from FRED and save as a combined wide CSV."""
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Set it in a .env file (see .env.example)."
        )

    fred = Fred(api_key=api_key)

    series_data = {}
    for series_id in SERIES_IDS:
        print(f"Fetching FRED series {series_id}...")
        series_data[series_id] = fred.get_series(series_id)

    df = pd.DataFrame(series_data)
    df.index.name = "date"
    df = df.reset_index()

    print(f"FRED series shape: {df.shape}")
    print(df.head())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FRED_SERIES_PATH, index=False)
    print(f"Saved FRED series to {FRED_SERIES_PATH}")

    return df


if __name__ == "__main__":
    load_fred_series()
