"""Champion/challenger credit-approval models: Logistic Regression vs XGBoost.

Input:  data/processed/hmda_features.csv
Output: models/champion_lr.pkl, models/challenger_xgb.pkl,
        outputs/model_evaluation.csv

Target: approved (1 = originated, 0 = denied), from HMDA action_taken.

NOTE ON is_minority: included as a model feature per this session's spec.
In a real underwriting system, using protected-class status as a
decisioning input would violate fair lending law (ECOA/Reg B). This is
only appropriate here because this is an analytical/portfolio framework,
never a production underwriting tool (see CLAUDE.md never-do rules).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

HMDA_FEATURES_PATH = PROCESSED_DIR / "hmda_features.csv"
CHAMPION_PATH = MODELS_DIR / "champion_lr.pkl"
CHALLENGER_PATH = MODELS_DIR / "challenger_xgb.pkl"
EVAL_PATH = OUTPUTS_DIR / "model_evaluation.csv"
DECILE_LIFT_PATH = OUTPUTS_DIR / "decile_lift.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
INCOME_CAP_PCTL = 0.99
LOAN_AMOUNT_CAP_PCTL = 0.99

DTI_BUCKET_ORDER = ["<20", "20-<30", "30-<36", "36-<43", "43-50", ">50", "exempt/NA"]
INCOME_TIER_ORDER = ["low", "moderate", "middle", "upper"]
LOAN_SIZE_ORDER = ["small", "medium", "large", "jumbo"]
RISK_SEGMENT_ORDER = ["subprime", "near-prime", "prime"]

# county_fips target encoding smoothing: counties with fewer than ~min_samples
# loans shrink strongly toward the global approval rate rather than using
# their own (noisy, small-sample) mean.
COUNTY_TE_MIN_SAMPLES = 20
COUNTY_TE_SMOOTHING = 10

FEATURE_COLS = [
    "dti_bucket_ord",
    "income_tier_ord",
    "loan_size_bucket_ord",
    "risk_segment_ord",
    "is_minority_filled",
    "county_fips_te",
    "income_capped",
    "loan_amount_capped",
]


def _ordinal_encode(series, order):
    mapping = {cat: i for i, cat in enumerate(order)}
    return series.map(mapping)


def _fit_county_target_encoder(county_train, y_train):
    stats = pd.DataFrame({"county_fips": county_train, "approved": y_train})
    grouped = stats.groupby("county_fips")["approved"].agg(["mean", "count"])
    global_mean = y_train.mean()
    weight = 1 / (
        1 + np.exp(-(grouped["count"] - COUNTY_TE_MIN_SAMPLES) / COUNTY_TE_SMOOTHING)
    )
    grouped["smoothed"] = global_mean * (1 - weight) + grouped["mean"] * weight
    return grouped["smoothed"], global_mean


def load_and_prepare():
    df = pd.read_csv(HMDA_FEATURES_PATH, low_memory=False)
    print(f"Loaded hmda_features.csv: {df.shape}")

    n_before = len(df)
    df = df.dropna(subset=["risk_segment"]).copy()
    print(
        f"Dropped {n_before - len(df):,} rows with missing risk_segment "
        f"(missing income) -> {len(df):,} rows remain"
    )

    df["county_fips"] = df["county_fips"].fillna("UNKNOWN").astype(str)

    approved_n = df["approved"].sum()
    denied_n = len(df) - approved_n
    approval_rate = df["approved"].mean()
    print(
        f"\nClass balance: approved={approved_n:,} ({approval_rate:.1%}), "
        f"denied={denied_n:,} ({1 - approval_rate:.1%}) "
        f"-> imbalance ratio {approved_n / denied_n:.2f}:1"
    )

    return df


def build_features(df):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["risk_segment"],
    )
    train_df = train_df.copy()
    test_df = test_df.copy()
    print(f"\nTrain/test split: {len(train_df):,} / {len(test_df):,} (stratified on risk_segment)")

    # Cap income / loan_amount at the 99th percentile, threshold fit on train only.
    income_cap = train_df["income"].quantile(INCOME_CAP_PCTL)
    loan_amount_cap = train_df["loan_amount"].quantile(LOAN_AMOUNT_CAP_PCTL)
    print(f"income capped at 99th pctl (train) = {income_cap:.1f} (thousands)")
    print(f"loan_amount capped at 99th pctl (train) = {loan_amount_cap:,.0f}")

    for d in (train_df, test_df):
        d["income_capped"] = d["income"].clip(upper=income_cap)
        d["loan_amount_capped"] = d["loan_amount"].clip(upper=loan_amount_cap)

        d["dti_bucket_ord"] = _ordinal_encode(d["dti_bucket"], DTI_BUCKET_ORDER)
        d["income_tier_ord"] = _ordinal_encode(d["income_tier"], INCOME_TIER_ORDER)
        d["loan_size_bucket_ord"] = _ordinal_encode(d["loan_size_bucket"], LOAN_SIZE_ORDER)
        d["risk_segment_ord"] = _ordinal_encode(d["risk_segment"], RISK_SEGMENT_ORDER)

        # is_minority is NaN when derived_race is unknown (~20% of rows) —
        # encoded as -1, a distinct "unknown" level rather than assumed 0.
        d["is_minority_filled"] = d["is_minority"].fillna(-1)

    county_encoding, global_mean = _fit_county_target_encoder(
        train_df["county_fips"], train_df["approved"]
    )
    for d in (train_df, test_df):
        d["county_fips_te"] = d["county_fips"].map(county_encoding).fillna(global_mean)

    X_train, y_train = train_df[FEATURE_COLS], train_df["approved"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["approved"]

    return X_train, X_test, y_train, y_test


def train_champion(X_train, y_train):
    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            # L2 is scikit-learn's default penalty; passing it explicitly is
            # deprecated as of sklearn 1.8.
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    champion = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    champion.fit(X_train, y_train)
    return champion


def train_challenger(X_train, y_train):
    challenger = xgb.XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    challenger.fit(X_train, y_train)
    return challenger


def compute_psi(expected, actual, bins=10):
    """Population Stability Index between two score distributions."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, test_scores)
    gini = 2 * auc - 1
    ks_stat = ks_2samp(test_scores[y_test == 1], test_scores[y_test == 0]).statistic * 100
    psi = compute_psi(train_scores, test_scores)

    print(f"\n=== {name} ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Gini: {gini:.4f}")
    print(f"KS statistic: {ks_stat:.2f}")
    print(f"PSI (train vs test): {psi:.4f}")

    return {
        "model": name,
        "auc_roc": auc,
        "gini": gini,
        "ks_statistic": ks_stat,
        "psi": psi,
    }


def decile_lift_table(y_test, test_scores, name):
    df = pd.DataFrame({"y": y_test.values, "score": test_scores})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")

    table = (
        df.groupby("decile")
        .agg(count=("y", "size"), approved=("y", "sum"), approval_rate=("y", "mean"))
        .sort_index(ascending=False)
        .reset_index(drop=True)
    )
    table.index = range(1, len(table) + 1)
    table.index.name = "decile_rank"  # 1 = highest predicted approval score

    overall_rate = df["y"].mean()
    table["lift"] = table["approval_rate"] / overall_rate
    table.insert(0, "model", name)

    print(f"\n=== {name}: decile lift table (1 = highest predicted score) ===")
    print(table)

    return table


def run():
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = build_features(df)

    champion = train_champion(X_train, y_train)
    challenger = train_challenger(X_train, y_train)

    champion_metrics = evaluate_model(champion, X_train, y_train, X_test, y_test, "Champion (Logistic Regression)")
    challenger_metrics = evaluate_model(challenger, X_train, y_train, X_test, y_test, "Challenger (XGBoost)")

    champion_decile = decile_lift_table(y_test, champion.predict_proba(X_test)[:, 1], "Champion (Logistic Regression)")
    challenger_decile = decile_lift_table(
        y_test, challenger.predict_proba(X_test)[:, 1], "Challenger (XGBoost)"
    )

    comparison = pd.DataFrame([champion_metrics, challenger_metrics])
    print("\n=== Champion vs Challenger comparison ===")
    print(comparison.to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_PATH, "wb") as f:
        pickle.dump(champion, f)
    with open(CHALLENGER_PATH, "wb") as f:
        pickle.dump(challenger, f)
    print(f"\nSaved models to {CHAMPION_PATH} and {CHALLENGER_PATH}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(EVAL_PATH, index=False)
    print(f"Saved evaluation metrics to {EVAL_PATH}")

    decile_lift = pd.concat([champion_decile, challenger_decile]).reset_index()
    decile_lift.to_csv(DECILE_LIFT_PATH, index=False)
    print(f"Saved decile lift tables to {DECILE_LIFT_PATH}")

    return comparison


if __name__ == "__main__":
    run()
