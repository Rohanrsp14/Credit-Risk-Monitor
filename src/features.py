"""Feature engineering and risk segmentation on cleaned HMDA data.

Input:  data/processed/hmda_clean.csv (produced by src/data_loader.py)
Output: data/processed/hmda_features.csv

Adds: approved (target), dti_bucket, income_tier, loan_size_bucket,
risk_segment, is_minority, county_fips.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
HMDA_CLEAN_PATH = PROCESSED_DIR / "hmda_clean.csv"
HMDA_FEATURES_PATH = PROCESSED_DIR / "hmda_features.csv"

# --- dti_bucket -------------------------------------------------------
# HMDA reports debt_to_income_ratio as either a bucketed string
# ("<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%", "Exempt") or an
# exact integer percentage string (36-49) for the 36-49% range only.
# "50%-60%" straddles our 43-50 / >50 boundary; grouped into >50 since
# its floor (50) sits at that boundary.
DTI_BUCKET_ORDER = ["<20", "20-<30", "30-<36", "36-<43", "43-50", ">50", "exempt/NA"]

DTI_STRING_MAP = {
    "<20%": "<20",
    "20%-<30%": "20-<30",
    "30%-<36%": "30-<36",
    "50%-60%": ">50",
    ">60%": ">50",
    "Exempt": "exempt/NA",
}


def _bucket_dti(value):
    if pd.isna(value):
        return "exempt/NA"
    if value in DTI_STRING_MAP:
        return DTI_STRING_MAP[value]
    try:
        pct = int(value)
    except (TypeError, ValueError):
        return "exempt/NA"
    if pct < 36:
        return "30-<36"
    if pct < 43:
        return "36-<43"
    return "43-50"


# --- income_tier --------------------------------------------------------
# income is reported by HMDA in thousands of dollars.
INCOME_TIER_ORDER = ["low", "moderate", "middle", "upper"]


def _bucket_income(value):
    if pd.isna(value):
        return np.nan
    if value < 40:
        return "low"
    if value < 75:
        return "moderate"
    if value < 120:
        return "middle"
    return "upper"


# --- loan_size_bucket -----------------------------------------------------
# Cutoffs anchored near the 2022 conforming loan limit (~$647,200):
# small/medium/large span the bulk of the distribution, jumbo ~ non-conforming.
LOAN_SIZE_ORDER = ["small", "medium", "large", "jumbo"]
LOAN_SIZE_BINS = [-np.inf, 150_000, 350_000, 650_000, np.inf]


def _bucket_loan_size(series):
    return pd.cut(series, bins=LOAN_SIZE_BINS, labels=LOAN_SIZE_ORDER)


# --- risk_segment ---------------------------------------------------------
# Points-based combination of income_tier and dti_bucket. Higher income and
# lower DTI both push toward "prime"; unknown DTI (exempt/NA) is treated as
# neutral risk rather than assumed good or bad.
#
# "<20" is deliberately NOT given the max DTI score. Investigation showed
# this bucket's approval rate (65.1%) is well below "20-<30" through "43-50"
# (81-85%) in this dataset, driven by a different loan_purpose mix (more
# home-improvement/other-purpose, less home-purchase) and collateral-driven
# denials rather than debt burden. We score it the same as exempt/NA
# (unreliable signal) rather than tuning it to match the observed approval
# rate, which would make risk_segment circularly derived from the target.
INCOME_POINTS = {"low": 1, "moderate": 2, "middle": 3, "upper": 4}
DTI_POINTS = {
    "<20": 1,
    "20-<30": 3,
    "30-<36": 2,
    "36-<43": 1,
    "43-50": 0,
    ">50": -1,
    "exempt/NA": 1,
}


def _risk_segment(income_tier, dti_bucket):
    if pd.isna(income_tier):
        return np.nan
    score = INCOME_POINTS[income_tier] + DTI_POINTS[dti_bucket]
    if score >= 6:
        return "prime"
    if score >= 3:
        return "near-prime"
    return "subprime"


def _county_fips(census_tract):
    if pd.isna(census_tract):
        return np.nan
    tract_str = str(int(census_tract)).zfill(11)
    return tract_str[:5]


def build_features():
    df = pd.read_csv(HMDA_CLEAN_PATH, low_memory=False)
    print(f"Loaded hmda_clean.csv: {df.shape}")

    df["approved"] = (df["action_taken"] == 1).astype(int)

    df["dti_bucket"] = pd.Categorical(
        df["debt_to_income_ratio"].apply(_bucket_dti),
        categories=DTI_BUCKET_ORDER,
        ordered=True,
    )

    df["income_tier"] = pd.Categorical(
        df["income"].apply(_bucket_income),
        categories=INCOME_TIER_ORDER,
        ordered=True,
    )

    df["loan_size_bucket"] = _bucket_loan_size(df["loan_amount"])

    df["risk_segment"] = pd.Categorical(
        [
            _risk_segment(it, db)
            for it, db in zip(df["income_tier"], df["dti_bucket"])
        ],
        categories=["subprime", "near-prime", "prime"],
        ordered=True,
    )

    # "Race Not Available" and "Free Form Text Only" mean race is unknown,
    # not minority — folding them into is_minority=1 would distort any
    # fair-lending disparity metric built on this flag (~20% of rows).
    RACE_UNKNOWN = {"Race Not Available", "Free Form Text Only"}
    RACE_NON_MINORITY = {"White", "Joint"}

    def _is_minority(race):
        if race in RACE_UNKNOWN:
            return np.nan
        return 0 if race in RACE_NON_MINORITY else 1

    df["is_minority"] = df["derived_race"].apply(_is_minority)

    df["county_fips"] = df["census_tract"].apply(_county_fips)

    print("\n=== risk_segment value counts ===")
    print(df["risk_segment"].value_counts(dropna=False))

    print("\n=== approval rate by risk_segment ===")
    print(df.groupby("risk_segment", observed=True)["approved"].mean())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HMDA_FEATURES_PATH, index=False)
    print(f"\nSaved features to {HMDA_FEATURES_PATH}")

    return df


if __name__ == "__main__":
    build_features()
