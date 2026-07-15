"""Load and clean HMDA and CFPB source data.

Data sources:
- HMDA 2022 LAR, Texas: data/raw/hmda_2022_tx.csv
  (https://ffiec.cfpb.gov/data-publication/)
- CFPB Consumer Complaint Database, payday/personal/consumer loan products:
  data/raw/cfpb_complaints.csv
  (https://www.consumerfinance.gov/data-research/consumer-complaints/)
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

HMDA_RAW_PATH = RAW_DIR / "hmda_2022_tx.csv"
HMDA_CLEAN_PATH = PROCESSED_DIR / "hmda_clean.csv"

CFPB_RAW_PATH = RAW_DIR / "cfpb_complaints.csv"
CFPB_CLEAN_PATH = PROCESSED_DIR / "complaints_clean.csv"

# action_taken: 1 = loan originated, 3 = application denied
HMDA_ACTION_TAKEN_KEEP = [1, 3]


def load_hmda():
    """Load HMDA LAR data, report shape/nulls, filter to originated/denied, save."""
    df = pd.read_csv(HMDA_RAW_PATH, low_memory=False)

    print(f"HMDA raw shape: {df.shape}")
    print(f"HMDA columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print("HMDA null counts (columns with at least 1 null):")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0].sort_values(ascending=False))

    df_clean = df[df["action_taken"].isin(HMDA_ACTION_TAKEN_KEEP)].copy()
    print(
        f"HMDA filtered to action_taken in {HMDA_ACTION_TAKEN_KEEP}: "
        f"{df_clean.shape[0]} rows (from {df.shape[0]})"
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(HMDA_CLEAN_PATH, index=False)
    print(f"Saved cleaned HMDA data to {HMDA_CLEAN_PATH}")

    return df_clean


def load_cfpb():
    """Load CFPB complaints data, report shape/product counts, save."""
    df = pd.read_csv(CFPB_RAW_PATH, low_memory=False)

    print(f"CFPB raw shape: {df.shape}")
    print("CFPB Product value counts:")
    print(df["Product"].value_counts())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CFPB_CLEAN_PATH, index=False)
    print(f"Saved cleaned CFPB data to {CFPB_CLEAN_PATH}")

    return df


if __name__ == "__main__":
    load_hmda()
    load_cfpb()
