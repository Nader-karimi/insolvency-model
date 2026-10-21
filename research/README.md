# Reproducible three-state bank-insolvency HJB research code

This **research reproducibility** folder accompanies the IREF manuscript. The repository root has a newly integrated Streamlit application (`../app.py`), and its dedicated numerical HJB tab loads the solution archive from this folder. The root app and its support modules are part of the same upload package and intentionally replace the old three-file Streamlit repository. No Flask code is required.

## Scope
The model is a conditional, illustrative one-year absorbing-boundary HJB with state `(w, lambda, d)`. Its output is **not a calibrated prediction of actual legal bank failure**. The Streamlit dashboard has two distinct analysis modes: the historical balance-sheet scenario includes a labeled heuristic boundary, while the numerical HJB tab reads this folder’s actual HJB solution grid. The two are never identified as the same state or probability measure. Figures 6 and 7 in the paper are independently generated from the **saved three-state HJB solution** and the synthetic path defined in `figure6_7_scenario.json`. They are not screenshots and do **not** reuse the Streamlit default bank inputs.

## Install and verify
From the repository root:

```sh
cd research
python -m pip install -r requirements-research.txt
python -m pytest -q
python verify_released_outputs.py
python reproduce_figure6.py
python reproduce_figure7.py
```

## Recreate all article figures and tables from code

```sh
cd research
python generate_operational_outputs.py
python reproduce_figure6.py
python reproduce_figure7.py
python reproduce_dashboard_default.py
python verify_released_outputs.py
```

`generate_operational_outputs.py` regenerates `model_outputs_v4` (production solver, refined reference, held-out surrogate, sensitivities, synthetic assessments) and `paper_assets/v4` (eleven manuscript figures and a supplementary dashboard-only scenario visualization). It invokes the independent Figure 6/7 plotting scripts. **The dashboard-only illustration is not included in the manuscript.** The manuscript figures are those directly relevant to the HJB mathematical and numerical contribution. Re-solving the HJB may take minutes; the reference arrays are included for independent verification without recomputation.

Re-run the additional independent reviewer checks:

```sh
python audit/run_independent_checks.py
python audit/reproduce_coupled_refinement.py
```

The latter is a genuine set of three re-solves; numerical stabilization is **not** claimed. Its existing results are recorded in `audit/coupled_refinement.csv`.

## Data and external validation limitations

`data/processed/` contains 48 bank-year entries for six public-statement banks (Jalali 1397–1404), together with quality-control records. Figure 2 plots this accounting ratio for the **34 of 48** public bank-year records whose stored ratio matches the quotient of recorded numerator and denominator; 14 inconsistent records are left as gaps rather than silently corrected. Run `python research/audit/public_panel_completeness.py` for the exact affected bank-years and audit counts. The 48/48 basic accounting and extraction QC result does **not** imply 48/48 derived-proxy verification; these are distinct checks. No model PD is constructed from the public panel. `data/raw/` contains **source identifiers and extracted line-level data only**, not the underlying third-party PDF financial statements. Bank-specific regulatory inputs (HQLA, stressed NCOF, ASF/RSF, regulatory capital and official NPL series) and observed bank-failure labels are not supplied. Do not use these accounting proxies to claim real-bank PD estimation or external failure prediction performance.

## Figure numbering (submitted manuscript)

| Figure in manuscript | Python-generated image |
|---|---|
| 1 | `paper_assets/v4/01_model_workflow.png` |
| 2 | `paper_assets/v4/11_public_six_bank_panel.png` |
| 3 | `paper_assets/v4/02_hjb_risk_slices.png` |
| 4 | `paper_assets/v4/03_insolvency_boundary_3d.png` |
| 5 | `paper_assets/v4/04_optimal_policy_regimes.png` |
| 6 | `paper_assets/v4/05_phase_portrait_and_stress_path.png` (reproduce_figure6.py) |
| 7 | `paper_assets/v4/06_dual_balance_sheet_state_map.png` (reproduce_figure7.py) |
| 8 | `paper_assets/v4/08_numerical_convergence.png` |
| 9 | `paper_assets/v4/07_surrogate_validation.png` |
| 10 | `paper_assets/v4/09_parameter_sensitivity.png` |
| 11 | `paper_assets/v4/10_exit_action_plan.png` |

### Quantitative tables

`model_outputs_v4/operational_summary.json` underpins the parameters, diagnostics and synthetic assessments; `operational_convergence.csv` underpins the reference-grid table; `operational_surrogate_validation.csv` underpins the held-out surrogate table; `operational_sensitivity.csv` underpins sensitivity plots; `audit/` holds additional independent robustness computations. Some theoretical/definitional tables are manuscript text and have no empirical data file.

## Licensing / deployment

Please check permission before redistribution of statement-derived files and obtain any repository license separately. This folder is the numeric research component of the integrated Streamlit repository. Replacing the root Streamlit app does not overwrite the research solver or its saved arrays. The online dashboard URL is intentionally omitted from the manuscript, and the archived solver remains the authority for the paper’s numerical results. Pin a GitHub release and specific commit before updating the manuscript's code availability statement.

## Additional numerical reviewer diagnostics

`audit/matched_domain_sensitivity.py` appends log-spaced nodes at artificial domain edges while holding every original interior grid node unchanged. `audit/analytical_benchmark.py` compares a constant-liquidity/credit, zero-feedback special case with the exact reflection-principle first-passage formula. Both scripts emit machine-readable CSV files, and the manuscript explicitly reports the observed residual grid sensitivity.

## Supplemental dashboard-only illustration (not a manuscript figure)

`reproduce_dashboard_default.py` calls the same `Scenario` and `stress` functions used in the main `app.py`. The complete default input vector, accounting outputs and historical scenario-score coefficients are archived in `dashboard_default_scenario_results.json`. This is distinct from HJB Figure 7 and may never be described as the solved HJB probability boundary.

## Extended Reviewer 1 numerical audit

`audit/consistent_refinement.py` independently re-solves the HJB on four progressively finer grids, with time-step counts 16, 24, 36, and 42 and control counts 13, 17, 21, and 23. The largest squared log-grid spacing to time-step ratio `h^2/dt` decreases from approximately 0.7900 to 0.4593. The output `audit/consistent_refinement.csv` records five fixed comparison states. **It does not certify asymptotic convergence:** the last two grids still differ by up to 0.03710 in model PD. The alternative 24/48/96-step sequence, isolated time/control/spatial checks, matched-boundary test and independent 1D analytical benchmark are retained so the remaining numerical sensitivity can be assessed rather than concealed.

### Additional public-panel ratio audit

`python research/audit/public_panel_completeness.py` writes `audit/public_panel_completeness.json` containing the independently checked 34/48 cross-field ratio identities, the 14 exclusions and the relevant bank-year keys. Use the audit before interpreting or regenerating Figure 2. The underlying extracted amounts remain unchanged.

## Model-internal explanation of synthetic risk diagnoses

`python research/counterfactual_risk_attribution.py` produces `model_outputs_v4/model_counterfactual_risk_attribution.csv` from the released three-state numerical HJB solution and synthetic balance-sheet inputs. The numerical risk-attribution table in the revised manuscript is generated from this CSV. It changes one initial coordinate at a time and retains the original HJB optimization; it neither estimates observed bank-failure prediction nor attributes causes of a historical default.
