# Credit Risk Monitor

## Project Purpose

A credit risk monitoring framework for consumer installment lenders operating
at the Regional Finance tier (mid-size, brick-and-mortar-plus-digital
installment/personal lenders serving near-prime and subprime borrowers). The
goal is to combine public mortgage-lending data (HMDA) and consumer complaint
data (CFPB) into a framework that surfaces credit risk signals, complaint
trends, and underwriting/fair-lending patterns relevant to this lender tier.

This is a portfolio/analytical project, not a production underwriting system.
Nothing built here should be presented as, or used as, an actual credit
decisioning tool.

## Datasets

### `data/raw/hmda_2022_tx.csv`
- **Source**: CFPB/FFIEC HMDA (Home Mortgage Disclosure Act) Loan Application
  Register (LAR), 2022 activity year, filtered to Texas (`state_code == TX`).
- **Size**: ~502 MB, 1,388,355 rows (including header).
- **Grain**: One row per mortgage loan application/action.
- **Key columns**: `activity_year`, `lei`, `derived_msa-md`, `state_code`,
  `county_code`, `census_tract`, `derived_loan_product_type`,
  `derived_ethnicity`, `derived_race`, `derived_sex`, `action_taken`,
  `loan_type`, `loan_purpose`, `loan_amount`, `interest_rate`, `rate_spread`,
  `loan_term`, `property_value`, `income`, `debt_to_income_ratio`,
  `applicant_credit_score_type`, plus applicant/co-applicant ethnicity, race,
  and sex detail columns (`applicant_ethnicity-1..5`, etc.).
- **Note**: HMDA covers *mortgage* lending, not installment/payday loans.
  It's used here as a proxy for regional credit risk, income, and
  underwriting patterns in Texas, and for fair-lending-style analysis
  (disparities by race/ethnicity/sex), not as direct data on the lender's
  own loan book.
- **Public source**: https://ffiec.cfpb.gov/data-publication/

### `data/raw/cfpb_complaints.csv`
- **Source**: CFPB Consumer Complaint Database, filtered to
  payday/personal/consumer installment loan products.
- **Size**: ~75 MB, 219,592 rows (including header).
- **Grain**: One row per consumer complaint.
- **Columns**: `Date received`, `Product`, `Sub-product`, `Issue`,
  `Sub-issue`, `Consumer complaint narrative`, `Company public response`,
  `Company`, `State`, `ZIP code`, `Tags`, `Submitted via`,
  `Date sent to company`, `Company response to consumer`,
  `Timely response?`, `Complaint ID`.
- **Public source**: https://www.consumerfinance.gov/data-research/consumer-complaints/

Both datasets are public government data. Raw CSVs are gitignored (too large
for the repo and redistributable directly from source) — see `.gitignore`.

## Folder Structure

```
credit-risk-monitor/
├── data/
│   ├── raw/            # Original source CSVs (gitignored, not committed)
│   └── processed/      # Cleaned/derived datasets produced by src/ scripts
├── src/                # Reusable Python modules (data loading, features, models)
├── models/             # Trained model artifacts (gitignored, not committed)
├── dashboard/           # Streamlit dashboard app
├── notebooks/          # Exploratory Jupyter notebooks
├── tests/              # Unit/integration tests
├── outputs/            # All generated charts, reports, exports (gitignored)
├── .env.example        # Template for required environment variables
├── requirements.txt
└── CLAUDE.md
```

## Never Do

- **Never generate or use synthetic/fabricated data** to fill gaps, pad
  samples, or simulate results. If data is missing, say so — do not invent
  plausible-looking rows or numbers.
- **Never hardcode API keys, tokens, or secrets** in source files, notebooks,
  or commits. All secrets go in `.env` (gitignored) and are referenced via
  `python-dotenv`. `.env.example` documents required variables with
  placeholder values only.
- **Never overclaim model accuracy or predictive power.** Always report
  metrics with their limitations (sample size, class imbalance, out-of-time
  vs. in-sample, proxy-data caveats since HMDA is mortgage data being used
  as a regional proxy, not the lender's actual portfolio). Do not present
  correlational findings as causal.
- **Never commit raw data files or model binaries** to git (see
  `.gitignore`). Never commit `data/raw/*.csv`, `models/*.pkl`, or contents
  of `outputs/`.

## Always Do

- **Cite data sources** on every chart, report, and notebook that uses HMDA
  or CFPB data — note the dataset, vintage/year, and any filters applied
  (e.g., "HMDA 2022 LAR, Texas, filtered to action_taken=1").
- **All charts/visualizations must be exportable** (PNG/HTML/SVG as
  appropriate) — not just rendered inline in a notebook.
- **All generated outputs (charts, reports, exports, model metrics) go into
  `outputs/`**, not scattered across the repo.
- **Document assumptions and filters explicitly** in code and notebooks —
  especially any filtering, deduplication, or exclusion logic applied to
  the raw datasets.

## Environment

- Python 3.12.4, pip 24.0, Windows 11.
- Dependencies managed via `requirements.txt`, installed into a local `venv`
  (gitignored).
- FRED API key (via `fredapi`) required for macro/economic indicator data —
  see `.env.example`.

## Build Order (in progress)

This project is being built incrementally, session by session. See git
history and commit messages for what's been completed. Do not skip ahead to
data loading, feature engineering, or modeling until explicitly instructed —
each phase is built and confirmed before the next begins.
