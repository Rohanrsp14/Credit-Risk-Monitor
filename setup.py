"""Pipeline setup script: regenerates data/processed and outputs artifacts.

Runs the full pipeline in dependency order:
  data_loader -> fred_loader -> features -> models -> monitor -> fair_lending -> choropleth

fred_loader is required before monitor (monitor reads data/processed/fred_series.csv).

Requires data/raw/hmda_2022_tx.csv and data/raw/cfpb_complaints.csv to be
present locally — both are gitignored (too large for GitHub, see
.gitignore). This script is a local/dev convenience for regenerating
everything from scratch. On Streamlit Cloud, the small outputs/*.csv and
outputs/*.html files are committed directly instead, since the raw data
can never be present there for this script to run against.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))


def run():
    from choropleth import run as run_choropleth
    from data_loader import load_cfpb, load_hmda
    from fair_lending import build_report as run_fair_lending
    from features import build_features
    from fred_loader import load_fred_series
    from models import run as run_models
    from monitor import run as run_monitor

    print("=== [1/7] data_loader ===")
    load_hmda()
    load_cfpb()

    print("=== [2/7] fred_loader ===")
    load_fred_series()

    print("=== [3/7] features ===")
    build_features()

    print("=== [4/7] models ===")
    run_models()

    print("=== [5/7] monitor ===")
    run_monitor()

    print("=== [6/7] fair_lending ===")
    run_fair_lending()

    print("=== [7/7] choropleth ===")
    run_choropleth()

    print("Pipeline complete.")


if __name__ == "__main__":
    run()
