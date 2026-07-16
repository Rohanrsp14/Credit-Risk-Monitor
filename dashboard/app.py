"""Streamlit dashboard for the Credit Risk Monitor framework.

Reads pre-computed outputs from outputs/ — these are committed to the repo
(see .gitignore) so the dashboard works on Streamlit Cloud out of the box.
If outputs/model_evaluation.csv is missing (e.g. a fresh local clone with
raw data present but no pipeline run yet), setup.py runs automatically.
Note: on Streamlit Cloud that auto-run cannot succeed from a cold start —
data/raw/hmda_2022_tx.csv (525MB) is gitignored and can never be present
there — so the committed outputs/ files are the real fix for deployment;
the auto-run is a local-dev convenience only.

This dashboard visualizes an analytical/portfolio framework, not a
production underwriting system — see CLAUDE.md never-do/always-do rules.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Reference palette (see CLAUDE.md / dataviz conventions): status colors for
# ordinal risk tiers and pass/fail flags, fixed categorical slots elsewhere.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
CATEGORICAL_BLUE = "#2a78d6"
CATEGORICAL_ORANGE = "#eb6834"

RISK_SEGMENT_ORDER = ["prime", "near-prime", "subprime"]
RISK_SEGMENT_COLORS = {"prime": STATUS_GOOD, "near-prime": STATUS_WARNING, "subprime": STATUS_CRITICAL}
MODEL_COLORS = {
    "Champion (Logistic Regression)": CATEGORICAL_BLUE,
    "Challenger (XGBoost)": CATEGORICAL_ORANGE,
}

st.set_page_config(page_title="Credit Risk Monitor", layout="wide")

MODEL_EVAL_PATH = OUTPUTS_DIR / "model_evaluation.csv"
if not MODEL_EVAL_PATH.exists():
    with st.spinner("Running pipeline setup..."):
        sys.path.insert(0, str(BASE_DIR))
        import setup as pipeline_setup

        try:
            pipeline_setup.run()
        except Exception as e:
            st.error(
                "Pipeline setup failed — this dashboard expects committed files under "
                "outputs/ (see .gitignore) rather than regenerating from raw data on "
                "Streamlit Cloud, since data/raw/hmda_2022_tx.csv (525MB) can't be "
                f"present there.\n\nUnderlying error: {e}"
            )
            st.stop()


@st.cache_data
def load_applications_summary():
    df = pd.read_csv(OUTPUTS_DIR / "dashboard_summary.csv")
    return int(df["total_applications"].iloc[0]), float(df["approval_rate"].iloc[0])


@st.cache_data
def load_csv(name):
    return pd.read_csv(OUTPUTS_DIR / name)


st.title("Credit Risk Monitor — Consumer Installment Lending Framework")
st.caption(
    "Analytical / portfolio framework for regional consumer installment lending risk. "
    "Not a production underwriting system."
)

REQUIRED_FILES = {
    "outputs/dashboard_summary.csv": OUTPUTS_DIR / "dashboard_summary.csv",
    "outputs/model_evaluation.csv": OUTPUTS_DIR / "model_evaluation.csv",
    "outputs/decile_lift.csv": OUTPUTS_DIR / "decile_lift.csv",
    "outputs/cohort_delinquency.csv": OUTPUTS_DIR / "cohort_delinquency.csv",
    "outputs/fair_lending_report.csv": OUTPUTS_DIR / "fair_lending_report.csv",
    "outputs/psi_monitoring.csv": OUTPUTS_DIR / "psi_monitoring.csv",
    "outputs/texas_risk_heatmap.html": OUTPUTS_DIR / "texas_risk_heatmap.html",
}
missing = [label for label, path in REQUIRED_FILES.items() if not path.exists()]
if missing:
    st.error(
        "Missing required output files even after pipeline setup:\n\n"
        + "\n".join(f"- {m}" for m in missing)
    )
    st.stop()

total_apps, approval_rate = load_applications_summary()
model_eval = load_csv("model_evaluation.csv")
decile_lift = load_csv("decile_lift.csv")
cohort = load_csv("cohort_delinquency.csv")
fair_lending = load_csv("fair_lending_report.csv")
psi_monitoring = load_csv("psi_monitoring.csv")

minority_gap = fair_lending["approval_rate"].max() - fair_lending["approval_rate"].min()
psi_alert = "ALERT" if (psi_monitoring["alert"] != "stable").any() else "Stable"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Applications", f"{total_apps:,}")
col2.metric("Approval Rate", f"{approval_rate:.1%}")
col3.metric(
    "Minority Disparity Gap",
    f"{minority_gap:.1%}",
    help="Highest minus lowest approval rate across race categories — see Fair Lending tab.",
)
col4.metric(
    "PSI Status",
    psi_alert,
    help="Population Stability Index drift monitoring, train vs test score distributions, both models.",
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Model Performance", "Cohort Tracking", "Fair Lending", "Texas Heatmap"]
)

with tab1:
    st.subheader("Champion vs Challenger")
    st.dataframe(
        model_eval.style.format(
            {"auc_roc": "{:.4f}", "gini": "{:.4f}", "ks_statistic": "{:.2f}", "psi": "{:.4f}"}
        ),
        width='stretch',
    )

    st.subheader("Decile Lift")
    fig = px.line(
        decile_lift,
        x="decile_rank",
        y="lift",
        color="model",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        labels={"decile_rank": "Decile (1 = highest predicted score)", "lift": "Lift"},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="#898781")
    st.plotly_chart(fig, width='stretch')
    with st.expander("View decile lift data"):
        st.dataframe(decile_lift, width='stretch')

with tab2:
    st.subheader("12-Month Cohort Delinquency Projection")
    st.warning(
        "This is an illustrative projection, not observed loan performance data. "
        "HMDA is a cross-sectional snapshot with no loan-level performance history. "
        "Curve = linear ramp from 0 to (FRED DRCCLACBS baseline x risk-tier multiplier)."
    )
    fig = px.line(
        cohort,
        x="month",
        y="projected_delinquency_rate_pct",
        color="risk_segment",
        category_orders={"risk_segment": RISK_SEGMENT_ORDER},
        color_discrete_map=RISK_SEGMENT_COLORS,
        labels={
            "month": "Month Since Origination",
            "projected_delinquency_rate_pct": "Projected Delinquency Rate (%)",
            "risk_segment": "Risk Segment",
        },
    )
    st.plotly_chart(fig, width='stretch')
    with st.expander("View cohort projection data"):
        st.dataframe(cohort, width='stretch')

with tab3:
    st.subheader("Approval Rate by Race Category")
    fl = fair_lending.sort_values("approval_rate", ascending=False).copy()
    fl["status"] = fl["four_fifths_flag"].map({True: "Below four-fifths threshold", False: "Above threshold"})
    threshold = fl["approval_rate"].max() * 0.8

    fig = px.bar(
        fl,
        x="race_category",
        y="approval_rate",
        color="status",
        color_discrete_map={"Below four-fifths threshold": STATUS_CRITICAL, "Above threshold": STATUS_GOOD},
        labels={"race_category": "Race Category", "approval_rate": "Approval Rate", "status": ""},
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#0b0b0b",
        annotation_text="Four-fifths threshold",
        annotation_position="top left",
    )
    fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None)
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "Excludes 'Race Not Available' and 'Free Form Text Only'. Unadjusted raw rates — "
        "not controlled for income, DTI, or loan size. Screening flag, not a compliance determination."
    )
    st.dataframe(fair_lending, width='stretch')

with tab4:
    st.subheader("Texas County Approval Rate Heatmap")
    st.caption("Red = low approval rate, green = high approval rate. Hover a county for details.")
    st.iframe(OUTPUTS_DIR / "texas_risk_heatmap.html", height=650)

with st.sidebar:
    st.header("Dataset Info")
    st.markdown(
        "**HMDA 2022 LAR, Texas**  \n"
        "Home Mortgage Disclosure Act Loan Application Register, filtered to "
        "originated/denied applications.  \n"
        "[ffiec.cfpb.gov/data-publication](https://ffiec.cfpb.gov/data-publication/)"
    )
    st.markdown(
        "**CFPB Consumer Complaints**  \n"
        "Payday/personal/consumer loan complaints.  \n"
        "[consumerfinance.gov/data-research/consumer-complaints]"
        "(https://www.consumerfinance.gov/data-research/consumer-complaints/)"
    )
    st.markdown(
        "**FRED**  \n"
        "DRCCLACBS (credit card delinquency rate) and CORCACBS (consumer loan "
        "charge-off rate), All Commercial Banks.  \n"
        "[fred.stlouisfed.org](https://fred.stlouisfed.org/)"
    )
    st.divider()
    st.caption(
        "Portfolio / analytical framework, not a production underwriting system. "
        "See CLAUDE.md for full data provenance and never-do/always-do rules."
    )
