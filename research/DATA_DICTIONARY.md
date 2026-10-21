# Data and numerical-output dictionary

All values are **illustrative model inputs or public accounting proxies**, not observed legal bank defaults. The bank-specific inputs to the three-state HJB use a common monetary unit. The saved synthetic-bank examples use million Iranian rials, whereas the existing separate Streamlit dashboard uses million USD. These are different scenarios and should never be merged or interpreted as the same model run.

## State definitions and constraints

| Field | Definition | Interpretation limit |
|---|---|---|
| `w` | Equity / minimum required equity | Model absorption when `w <= 1`; not a legal insolvency order |
| `lambda` | 30-day net cash outflow / HQLA | Inverse LCR from compliant inputs only |
| `d` | NPL / gross loans | Ratio interpretation applies locally; the geometric diffusion is not globally bounded by one |
| `a` | Asset sale / HQLA | Nonnegative sale intensity, capacity bounded |
| `b` | Additional borrowing / HQLA | Nonnegative borrowing intensity, capacity bounded |
| `s0` | normalized stable-funding slack | A local linear NSFR approximation, not an official Basel ratio |
| `PD` | `1 - V(0,w,lambda,d)` | Controlled one-year *model boundary-crossing probability* |

Institution-specific inputs required for a future external prediction study include official HQLA, stressed NCOF, ASF/RSF factors, NPL loan migrations, regulatory minimum equity and realized distress/breach outcomes. These inputs/outcomes are **not** represented by the public CODAL-derived panel.

## Saved model-output files

| File | Contents |
|---|---|
| `model_outputs_v4/operational_survival_solution.npz` | State grids, solved survival cube, optimal controls, feasibility indicator, illustrative parameters |
| `model_outputs_v4/operational_survival_grid.csv` | 16,675 HJB states, survival/PD/policies/feasibility |
| `model_outputs_v4/operational_boundaries.csv` | Capital boundaries at two chosen illustrative model-PD cutoffs |
| `model_outputs_v4/operational_convergence.csv` | Coarse, medium, production, time-refined and numerical reference comparisons |
| `model_outputs_v4/operational_surrogate_validation.csv` | Internal HJB-grid holdout used to test approximation, **not historical default prediction** |
| `model_outputs_v4/operational_sensitivity.csv` | Fresh HJB evaluations under illustrative parameter shifts |
| `model_outputs_v4/stress_trajectory.csv` | Deterministic model stress path and separate illustrative intervention |
| `model_outputs_v4/sample_bank_inputs.csv` | Two synthetic balance sheets |
| `model_outputs_v4/sample_bank_assessments.json` | Synthetic bank model-conditional outputs and action calculations |
| `model_outputs_v4/operational_summary.json` | Solver parameters, diagnostics and surrogate statistics |
| `audit/numerical_controls_time.csv` | Independent fixed-grid control/time checks |
| `audit/numerical_domain.csv` | Independent domain and mesh sensitivity checks |
| `audit/coupled_refinement.csv` | Independent simultaneous refinement at three grids |

## CODAL-derived public panel

`data/processed/bank_year_metrics.csv` contains six named listed banks over Jalali 1397–1404, with 48 bank-year rows. `cash_bank_assets_to_deposits`, `loan_to_deposit_proxy`, and `equity_to_assets` are **accounting proxies** and are not regulatory LCR, NPL share, or capital adequacy. Figure 2 uses only the **34 cross-field-consistent records** of `cash_bank_assets_to_deposits` (14/48 records disagree with `cash_and_bank_assets / customer_deposits` and are omitted, not fixed); it is neither HJB PD nor an actual bank-default probability. `data/processed/qc_checks.csv` holds extraction quality-control flags.

`data/raw/report_index_selected.csv`, `data/raw/source_line_items.csv`, and `data/raw/statement_values.csv` provide source IDs, line-level extraction and provenance. The original statement PDFs are not included. Confirm redistribution rights before uploading the source-extract CSVs publicly.

## Figure 6–7 scenario

The exact **separate HJB** state trajectory, physical conversion, exogenous shocks and month-9 intervention are in `figure6_7_scenario.json` and `model_outputs_v4/stress_trajectory.csv`. Figure 7 is a two-dimensional projection over a fixed-equity slice; probabilities at path points are evaluated at their true varying three-dimensional states. Neither panel uses the current Streamlit default inputs or the Streamlit heuristic boundary.
