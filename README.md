# Joint Liquidity–Credit Stress Testing: integrated dashboard and HJB research code

This repository includes a **Streamlit** interactive dashboard and an independent, reproducible **three-state finite-horizon HJB** computation for the manuscript *Model-Implied Bank Insolvency Regions under Joint Liquidity and Credit Risk: An Absorbing-Boundary HJB Approach* (IREF-D-26-09861).

## Fast start

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The deployed Render service must use Streamlit, **not** the former Flask `gunicorn app:app` command:

```
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

All original default **scenario values** were preserved (see `research/streamlit_default_inputs.json`). Their two-stage output is a balance-sheet illustration; the original dashed contour is a **heuristic**, not an exact HJB insolvency boundary.

## Dashboard views

The sidebar has two sets of inputs. The **balance-sheet scenario** uses initial equity, HQLA, marketable/loan-book assets, liabilities, scenario inflows/outflows, NPL fraction, scenario NSFR buffer, percentage asset shock, reduced-form feedback coefficient and sale haircut. It generates:

1. Accounting equity-versus-liquidity buffer map, including economic quadrant labels;
2. Original two-stage credit-versus-SCO/SCI **heuristic** map (correctly labeled);
3. Equity-loss waterfall showing shock and funding costs;
4. Asset-shock sensitivity curve and a borrowing/sale/unmet-gap summary.

The independently labeled **numerical HJB** inputs require the minimum required equity, 30-day **NCOF**, HJB eligible HQLA, and NPL ratio. They generate:

1. One-year model-implied default-probability contour at a fixed capital ratio;
2. HJB capital requirement for a selected conditional probability level (two-dimensional heatmap and interactive three-dimensional boundary surface);
3. Model optimal-policy regimes (borrowing / asset sales / no action / infeasibility);
4. Probability-versus-capital curve;
5. One-factor model-risk explanatory diagnostic (10% separate capital, liquidity, and credit shifts);
6. Local joint LCR/NSFR feasible-control intensities.

All charts have figure-specific English interpretation below them. The additional one-factor panel is a model-state what-if, not causal attribution or an empirical default backtest. **These two input systems are not interchangeable.** In particular, the scenario SCO/SCI ratio is NOT the HJB state `NCOF/HQLA`. The HJB plots read the actual numerical solution from `research/model_outputs_v4/operational_survival_solution.npz` rather than the historical algebraic surrogate contour.

The HJB archive uses a fixed illustrative calibration and fixed reference control limits. Changing bank HJB **state inputs** updates the plot and model PD, but **does not re-estimate parameters** or change the reference-control problem. For production use, solve the HJB with approved institution-specific parameters and actual supervisory inputs. When the input is outside the grid, no interior PD is extrapolated; an immediately absorbing state is separately identified.

## Replication of all paper outputs

The numerical research code is in `research/`. See `research/README.md` for the complete figure mapping and commands. From the repository root:

```sh
python research/generate_operational_outputs.py
python research/reproduce_figure6.py
python research/reproduce_figure7.py
python research/verify_released_outputs.py
python research/counterfactual_risk_attribution.py
python -m pytest -q tests research/tests/test_operational_model.py
python tests/ui_static_smoke.py
```

The generator may take several minutes on a small CPU. It re-solves HJB reference, refinement and sensitivity problems and refreshes all 11 manuscript figure files in `research/paper_assets/v4/`. Reference solution and CSVs are included, so visiting the dashboard does **not** launch a slow numerical re-solve on every interaction.

### Input units

All scenario monetary defaults are in **millions USD**, as on the original public interface. The research paper's standalone HJB demonstration uses **million rials** for its separate reproducible physical-balance-sheet figure scenario, and uses dimensionless states inside the HJB. The choice of currency unit never changes a dimensionless state if numerator and denominator use the same unit. Do not merge the two illustrative scenarios or quote the USD baseline as a CODAL-observed bank.

### Scientific limitations

- Neither sample is an empirical calibration for any named bank.
- The six-bank CODAL-derived panel documents data provenance and proxy transformations; it does not supply official LCR, NSFR, NPL or bank-failure event labels.
- External bank-default prediction has **not** been validated.
- Comparison/viscosity uniqueness holds under the stated comparison principle; the numerical error is not certified. Some tested space–time refinements show material differences, as documented in `research/audit/`.
- The interactive chart descriptions deliberately avoid portraying policy regimes or model risk thresholds as regulatory recommendations.
- No online dashboard URL is necessary in the paper. The repository URL alone may be included in the code-availability statement after these files are uploaded and tagged.

## Minimal repository organization

```
app.py                  # Streamlit dashboard (root entry point)
scenario_engine.py      # original balance-sheet stress mechanism, independently testable
hjb_dashboard.py        # interpolation, plot arrays and domain control
requirements.txt
render.yaml             # optional blueprint for Render
research/               # solver, reproducible data/figures, and numerical audit
  operational_model.py
  generate_operational_outputs.py
  reproduce_figure6.py
  reproduce_figure7.py
  model_outputs_v4/
  paper_assets/v4/
  data/
  audit/
tests/
  test_dashboard.py
  ui_static_smoke.py
```

No environment secrets, authentication credentials, deployed endpoint tokens, obsolete Flask API, or LaTeX editorial checklists are included.

## Upload to existing Render service

1. Download this repository ZIP and **upload the files at repository root** (`app.py`, `requirements.txt`, `research/`, etc.), not in an extra enclosing folder. The new `app.py` **replaces** the old Streamlit `app.py` while retaining default scenario calculations and adding HJB tabs.
2. In Render, make sure the existing Web Service is connected to this same GitHub repository and `main` branch. Verify `Build Command = pip install -r requirements.txt`; set `Start Command = streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`. Leave the service's existing instance and domain intact. Editing `render.yaml` does not automatically change an already-created service's settings.
3. Commit the files, wait for deploy success, and open the domain. Verify both tabs and the ten charts; check `research/model_outputs_v4/operational_survival_solution.npz` is in the repository (a valid relative file path).
4. Run the tests locally; tag a GitHub release so the manuscript points to an immutable code version. **No GitHub push, Render deployment, or browser-level production test was performed by this ZIP-creation process.**

## Licenses, confidentiality and redistribution

Before making the repository public, verify reuse rights for extracted CODAL records, author permissions for the manuscript and the license for the software. Third-party PDF statements are not included in the archive.

### Article's default stress illustration (two panels)

The optional figure corresponding to the dashboard's original default stress scenario is reproducible with `python research/reproduce_dashboard_default.py`. Its inputs and outputs are in `research/dashboard_default_scenario_results.json`. Unlike HJB risk maps, the right-panel scenario score threshold is illustrative and **not a mathematical HJB insolvency boundary**. The independent HJB solution is preserved in the separately labeled tab.

**CODAL cross-field quality check:** The six-bank panel passes 48/48 basic statement identity checks, but its stored cash/bank-assets-to-deposits ratio disagrees with the recorded amount fields in 14/48 observations. Figure 2 omits those 14 rather than silently treating the proxy as verified. See `research/audit/public_panel_completeness.json` for the affected records.
