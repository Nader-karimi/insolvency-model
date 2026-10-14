# Absorbing-boundary bank-insolvency HJB and operational dashboard

This repository is the complete reproducibility package for the three-state,
finite-horizon bank-survival model. It contains the HJB solver, action engine,
held-out surrogate, numerical verification, public-data software test, 22
automated tests, and a Render-ready Persian dashboard.

The operational states are:

- `w = equity / minimum_required_equity`;
- `lambda = 30_day_NCOF / HQLA = 1 / LCR`;
- `d = NPL / gross_loans`.

The minimum-equity face `w=1` and loss of feasible LCR/NSFR controls are
absorbing. The output is a finite-horizon, model-implied boundary-crossing
probability—not a legal insolvency declaration or supervisory rating.

## Verified production artifacts

- production grid: `25 × 29 × 23 = 16,675` states;
- 17 asset-sale controls and 30 time steps;
- eight-node correlated cubature and trilinear semi-Lagrangian interpolation;
- probability bounds pass and 100% adjacent-grid monotonicity in all states;
- refined reference: 23,925 states, 19 controls, 60 time steps;
- held-out surrogate: 13,340 train / 3,335 test, RMSE `0.0256`,
  `R²=0.9959`, AUC `0.9998`, boundary FNR `0.15%`;
- automated suite: `22 passed`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
pytest -q
python generate_operational_outputs.py
flask --app app run
```

Open <http://127.0.0.1:5000>. A production-style command is:

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

## Bank-facing API

`POST /api/assess` accepts all monetary inputs in one common unit:

```json
{
  "name": "Example bank",
  "total_assets": 1500000,
  "equity": 80000,
  "minimum_required_equity": 60000,
  "hqla": 120000,
  "net_cash_outflow_30d": 115000,
  "available_stable_funding": 660000,
  "required_stable_funding": 600000,
  "gross_loans": 900000,
  "nonperforming_loans": 90000,
  "max_additional_borrowing": 80000,
  "max_sellable_assets": 120000,
  "target_pd": 0.20,
  "mode": "institution_specific"
}
```

Institution-specific mode re-solves the HJB using the submitted stable-funding
slack and intervention capacities. The response includes PD/survival, region,
breaches, optimal sale/borrowing, immediate gaps, and capital/HQLA/NCOF/NPL
actions needed to reach the target.

## Deploy on Render

1. Put this directory at the root of a GitHub repository.
2. In Render, create a Blueprint/Web Service from that repository.
3. `render.yaml` installs dependencies, starts Gunicorn, and checks `/health`.
4. Replace `USERNAME` in the paper and `CITATION.cff` with the final GitHub path.

The intended dashboard address is <https://insolvency-model.onrender.com/>.

## Repository map

- `operational_model.py`: three-state survival solver, feasibility theorem, bank assessment, exit actions
- `generate_operational_outputs.py`: grids, reference solve, surrogate, sensitivity, figures, samples
- `app.py`, `templates/index.html`: Persian dashboard and JSON API
- `model_outputs_v4/`: full survival grid, boundaries, validation, convergence, sensitivities, samples
- `paper_assets/v4/`: 11 publication-quality figures
- `tests/`: solver, API, pipeline, legacy comparison, and stress tests
- `data/`: six-bank public snapshot and line-level provenance
- `hjb_model.py`: preserved stationary two-state continuation model from the earlier arXiv design
- `DATA_DICTIONARY.md`: input/output definitions and limitations

## Public six-bank panel

The repository also contains 48 separate-statement observations for six Iranian
listed banks over Jalali years 1397–1404. This layer tests provenance,
extraction, plotting, and application logic. Public annual accounting statements
do not reconstruct official HQLA, stressed NCOF, ASF, RSF, NPL, regulatory
capital, or bank PD; the public mapping is therefore explicitly labeled a proxy.
The publication figures retain the original ticker fields internally but display
standardized English bank names and Gregorian fiscal year-ends (2019-2026).

## Model-risk and licensing note

Before high-stakes use, estimate parameters on the bank's own histories, perform
rolling out-of-sample validation and independent code review, approve the local
regulatory mapping, and establish model governance. Choose a software license
and confirm data-redistribution permissions before public release.
