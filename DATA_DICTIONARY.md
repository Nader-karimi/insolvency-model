# Data and model-output dictionary

## 1. Operational bank input

All monetary values must use one common unit. The Persian UI recommends million
Iranian rials, but normalization makes the model unit invariant.

| Field | Validation | Meaning |
|---|---|---|
| `name` | text, max 120 chars in API | bank/scenario label |
| `total_assets` | finite, `> 0` | total assets |
| `equity` | finite, `0 <= equity <= total_assets` | current book/regulatory equity selected by governance |
| `minimum_required_equity` | finite, `> 0` | absorbing minimum-equity level |
| `hqla` | finite, `> 0` | eligible haircut-adjusted HQLA |
| `net_cash_outflow_30d` | finite, `> 0` | Basel-consistent 30-day stressed net cash outflow |
| `available_stable_funding` | finite, `>= 0` | ASF |
| `required_stable_funding` | finite, `> 0` | RSF |
| `gross_loans` | finite, `> 0` | gross loan book |
| `nonperforming_loans` | finite, `0 <= NPL <= gross_loans` | NPL stock under approved definition |
| `max_additional_borrowing` | finite, `>= 0` | incremental borrowing capacity over the horizon |
| `max_sellable_assets` | finite, `>= 0` | assets operationally/legal available for sale |
| `target_pd` | `0 < target_pd < 1` | target used by remediation search |
| `mode` | `institution_specific` or `standard` | fresh capacity solve or precomputed surface |

Derived states and regulatory ratios:

- `wealth_ratio = equity / minimum_required_equity`;
- `liquidity_state = net_cash_outflow_30d / hqla = 1 / LCR`;
- `credit_state = nonperforming_loans / gross_loans`;
- `LCR = hqla / net_cash_outflow_30d`;
- `NSFR = available_stable_funding / required_stable_funding`.

## 2. Assessment response

| Object/field | Definition |
|---|---|
| `default_probability`, `survival_probability` | one-year model probability interpolated from the solved HJB |
| `region` | `safe`, `warning`, `insolvency_region`, or `critical` |
| `breaches` | current equity, LCR, NSFR, and control-feasibility flags |
| `optimal_policy` | sale/borrowing intensities relative to HQLA and monetary amounts |
| `minimum_regulatory_actions` | immediate HQLA gap, stable-funding increase, minimum feasible controls |
| `capital_only` | capital target and injection reaching target PD, or null if unavailable in domain |
| `liquidity_only` | target inverse-LCR plus equivalent HQLA increase/NCOF reduction |
| `credit_only` | target NPL ratio and NPL reduction |
| `combined_minimum_effort` | feasible capital/HQLA/NPL combination minimizing the documented normalized score |
| `domain_flags` | whether raw input is outside a solved state range |
| `solve` | mode, grid shape, control count, and unit convention |

`insolvency_region` is a model threshold (`PD >= 50%` by default), not legal
insolvency. `critical` means equity at/below the selected minimum or no feasible
control within the capacity assumptions.

## 3. `model_outputs_v4`

| File | Contents |
|---|---|
| `operational_survival_solution.npz` | grids, survival cube, policies, feasibility, parameters |
| `operational_survival_grid.csv` | 16,675 state rows with PD, survival, policies, feasibility |
| `operational_boundaries.csv` | generalized capital boundary for PD 20% and 50% |
| `operational_convergence.csv` | coarse, medium, production, time-refined, and reference comparisons |
| `operational_surrogate_validation.csv` | HJB/surrogate probability, error, partition, classifications for every state |
| `operational_sensitivity.csv` | fresh HJB results for parameter low/base/high variants |
| `sample_bank_inputs.csv` | baseline and adverse input templates |
| `sample_bank_assessments.json` | complete expected assessment outputs |
| `stress_trajectory.csv` | deterministic stress/remediation plotting path |
| `operational_summary.json` | parameters, diagnostics, validation metrics, and sample results |

## 4. Public six-bank snapshot

The public layer has 48 annual observations: six listed Iranian banks, Jalali
years 1397–1404, separate/parent financial statements, monetary unit million
rials.

| Field | Definition |
|---|---|
| `symbol`, `jalali_year` | bank ticker and Jalali fiscal year |
| `total_assets`, `total_liabilities`, `total_equity` | reported values |
| `customer_funding` | deposits plus investment-depositor rights when disclosed |
| `gross_loans_proxy` | government/non-government loans and receivables or reported total |
| `cash_and_bank_assets` | cash plus bank/central-bank balances and receivables |
| `cash_bank_assets_to_deposits` | structural liquidity proxy; **not LCR/NSFR** |
| `loan_to_deposit_proxy` | exposure proxy; **not NPL/PD** |
| `equity_to_assets` | book ratio; **not regulatory capital adequacy** |
| `qc_pass` | required fields, positive denominators, and balance tolerance |

The legacy public mapping in `bank_year_hjb_mapping.csv` is retained only to
test display/application behavior. Its `hjb_risk_index` is a normalized proxy,
not an official probability or cross-bank ranking.

## 5. Provenance

`data/raw/report_index_selected.csv` retains report ID, dates, audit flag, scope,
public PDF URL, selection reason, and SHA-256. `source_line_items.csv` links each
extracted value to source text and PDF hash. Source terms and redistribution
permissions must be reviewed before public release.
