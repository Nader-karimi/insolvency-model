# Joint Liquidity–Credit Stress Testing Dashboard

This repository contains the Python source code and dependencies for the interactive **Structural Insolvency Stress Test** dashboard associated with the manuscript. It illustrates the effects of combined asset-value, liquidity, and credit-risk shocks on a bank's balance sheet and risk-state trajectory.

## Online dashboard

**Interactive application:** https://insolvency-model.onrender.com/

The application is hosted on Render. On a free hosting instance, the first visit after a period of inactivity may take additional time to load. The repository can also be run locally using the instructions below.

## Repository files

- `app.py` — Streamlit dashboard, illustrative stress-test calculations, and both interactive Plotly charts.
- `requirements.txt` — Python package requirements.

## Run locally

Use a compatible Python 3 installation. From the repository directory, run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will display a local URL (usually `http://localhost:8501`) that can be opened in a web browser.

## Default illustrative scenario

The dashboard provides editable inputs. Its default values are:

| Input | Default value |
| --- | ---: |
| Initial equity (W0) | 8,000 million USD |
| High-quality liquid assets (HQLA) | 15,000 million USD |
| Marketable assets (unencumbered) | 20,000 million USD |
| Illiquid assets (loan book) | 57,000 million USD |
| Short-term liabilities | 25,000 million USD |
| Long-term liabilities | 67,000 million USD |
| Stable cash inflows (SCI) | 8,000 million USD |
| Short-term cash outflows (SCO) | 14,000 million USD |
| Initial non-performing loan ratio (d0) | 0.05 |
| Initial NSFR buffer (above 100%) | 0.05 |
| Asset-value shock | -25% |
| Endogenous feedback parameter (phi) | 0.00005 |
| Repo / fire-sale haircut | 0.20 |

To reproduce the default figures, start the dashboard and leave these controls unchanged.

## Dashboard visualizations

1. **Balance Sheet Trajectory under Joint Stress** — plots the initial and final equity and liquidity-buffer states, together with regions corresponding to solvency and liquidity conditions.
2. **State-Space Dynamics & Insolvency Boundary** — plots the bank's liquidity-risk/credit-risk state trajectory against a model-inspired boundary.

**Computational clarification:** The boundary displayed by the dashboard is drawn from the parameterized `surrogate_function` and threshold defined in `app.py`. This script does **not** numerically solve the HJB partial differential equation. Interpretation as an exact HJB-derived boundary would require a separate derivation or validation. The dashboard is an interactive illustration of the stress mechanism, with all implementation details visible in the source code.

## Reproducibility and archival reference

For a stable reference in a manuscript, cite a tagged release or a specific Git commit as well as the repository's main URL. The historical source version restored for the interactive dashboard can be viewed at:

https://github.com/Nader-karimi/insolvency-model/tree/bdf919549f11bdc2fdbe31703f4a51f9715695ee

The live deployment may change if new commits are deployed. The commit-specific link remains tied to the referenced code version.
