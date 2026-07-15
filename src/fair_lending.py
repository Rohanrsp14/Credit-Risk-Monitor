"""Fair lending parity analysis: approval rate disparity by race category.

Input:  data/processed/hmda_features.csv
Output: outputs/fair_lending_report.csv

Applies the four-fifths rule (a group's approval rate below 80% of the
highest-approval group's rate is flagged as a potential adverse-impact
signal) commonly used as a screening heuristic in fair lending analysis.
This is a screening flag, not a legal determination — any real disparity
finding needs controls for income, DTI, loan size, etc. before it can
support a causal or compliance claim (see CLAUDE.md never-overclaim rule).
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

HMDA_FEATURES_PATH = PROCESSED_DIR / "hmda_features.csv"
REPORT_PATH = OUTPUTS_DIR / "fair_lending_report.csv"

EXCLUDED_RACE_CATEGORIES = ["Race Not Available", "Free Form Text Only"]
FOUR_FIFTHS_THRESHOLD = 0.8


def build_report():
    df = pd.read_csv(HMDA_FEATURES_PATH, low_memory=False)
    print(f"Loaded hmda_features.csv: {df.shape}")

    df = df[~df["derived_race"].isin(EXCLUDED_RACE_CATEGORIES)].copy()
    print(f"Excluded {EXCLUDED_RACE_CATEGORIES}: {len(df):,} rows remain")

    report = (
        df.groupby("derived_race")["approved"]
        .agg(applications="count", approvals="sum", approval_rate="mean")
        .reset_index()
        .rename(columns={"derived_race": "race_category"})
        .sort_values("approval_rate", ascending=False)
        .reset_index(drop=True)
    )

    highest_rate = report["approval_rate"].max()
    report["disparity_ratio_vs_highest"] = report["approval_rate"] / highest_rate
    report["four_fifths_flag"] = report["disparity_ratio_vs_highest"] < FOUR_FIFTHS_THRESHOLD

    print("\n=== Fair lending report: approval rate by race category ===")
    print(report.to_string(index=False))

    flagged = report[report["four_fifths_flag"]]
    if len(flagged):
        print(f"\n{len(flagged)} group(s) below the four-fifths threshold ({FOUR_FIFTHS_THRESHOLD:.0%} of highest):")
        for _, row in flagged.iterrows():
            print(
                f"  - {row['race_category']}: approval rate {row['approval_rate']:.1%} "
                f"({row['disparity_ratio_vs_highest']:.1%} of highest-rate group)"
            )
    else:
        print("\nNo groups fall below the four-fifths threshold.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    print(f"\nSaved fair lending report to {REPORT_PATH}")

    return report


if __name__ == "__main__":
    build_report()
