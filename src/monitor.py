"""Monitoring engine: cohort delinquency projection + model score-drift (PSI).

Input:  data/processed/fred_series.csv, models/champion_lr.pkl,
        models/challenger_xgb.pkl, data/processed/hmda_features.csv
Output: outputs/cohort_delinquency.csv

IMPORTANT — cohort_delinquency.csv is a PROJECTION, not observed data.
HMDA is a single cross-sectional snapshot with no loan-level performance
history, so there is no real 12-month delinquency curve to report. The
values here are a documented linear ramp from 0 to
(latest FRED DRCCLACBS rate x risk-tier multiplier), included so the
monitoring framework has a placeholder curve shape to slot real
performance data into later. Every input (FRED rate, multipliers) is
real and cited; nothing here is a fabricated loan record. See CLAUDE.md
never-do rules on synthetic data.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_features, compute_psi, load_and_prepare  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

FRED_SERIES_PATH = PROCESSED_DIR / "fred_series.csv"
CHAMPION_PATH = MODELS_DIR / "champion_lr.pkl"
CHALLENGER_PATH = MODELS_DIR / "challenger_xgb.pkl"
COHORT_OUTPUT_PATH = OUTPUTS_DIR / "cohort_delinquency.csv"

# Risk-tier multipliers applied to the FRED DRCCLACBS (Delinquency Rate on
# Credit Card Loans, All Commercial Banks) baseline as a proxy for relative
# consumer-loan delinquency risk by tier.
TIER_MULTIPLIERS = {"subprime": 2.8, "near-prime": 1.4, "prime": 0.6}
N_MONTHS = 12

PSI_YELLOW_THRESHOLD = 0.1
PSI_RED_THRESHOLD = 0.25


def get_fred_baseline():
    fred = pd.read_csv(FRED_SERIES_PATH, parse_dates=["date"])
    latest = fred.dropna(subset=["DRCCLACBS"]).sort_values("date").iloc[-1]
    print(f"FRED DRCCLACBS baseline: {latest['DRCCLACBS']:.2f}% (as of {latest['date'].date()})")
    return latest["DRCCLACBS"], latest["date"]


def build_cohort_curves(baseline_rate, baseline_date):
    rows = []
    for tier, multiplier in TIER_MULTIPLIERS.items():
        terminal_rate = baseline_rate * multiplier
        for month in range(1, N_MONTHS + 1):
            rows.append(
                {
                    "risk_segment": tier,
                    "month": month,
                    "projected_delinquency_rate_pct": terminal_rate * (month / N_MONTHS),
                    "fred_baseline_rate_pct": baseline_rate,
                    "fred_baseline_date": baseline_date.date(),
                    "tier_multiplier": multiplier,
                    "terminal_rate_pct": terminal_rate,
                    "methodology": "linear_projection_not_observed_data",
                }
            )
    curves = pd.DataFrame(rows)

    print("\n=== Cohort delinquency projection (illustrative, NOT observed data) ===")
    print(
        curves.pivot(index="month", columns="risk_segment", values="projected_delinquency_rate_pct")[
            list(TIER_MULTIPLIERS.keys())
        ].round(3)
    )

    return curves


def monitor_score_drift():
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = build_features(df)

    print("\n=== Model score-drift (PSI) monitoring ===")
    results = []
    for name, path in [("Champion (Logistic Regression)", CHAMPION_PATH), ("Challenger (XGBoost)", CHALLENGER_PATH)]:
        with open(path, "rb") as f:
            model = pickle.load(f)
        train_scores = model.predict_proba(X_train)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        psi = compute_psi(train_scores, test_scores)

        if psi > PSI_RED_THRESHOLD:
            alert = "RED ALERT"
        elif psi > PSI_YELLOW_THRESHOLD:
            alert = "YELLOW ALERT"
        else:
            alert = "stable"

        print(f"{name}: PSI = {psi:.4f} -> {alert}")
        results.append({"model": name, "psi": psi, "alert": alert})

    return pd.DataFrame(results)


def run():
    baseline_rate, baseline_date = get_fred_baseline()
    curves = build_cohort_curves(baseline_rate, baseline_date)

    drift_results = monitor_score_drift()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(COHORT_OUTPUT_PATH, index=False)
    print(f"\nSaved cohort delinquency projection to {COHORT_OUTPUT_PATH}")

    return curves, drift_results


if __name__ == "__main__":
    run()
