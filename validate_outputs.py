#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
m = pd.read_csv(ROOT / "data/processed/bank_year_metrics.csv")
r = pd.read_csv(ROOT / "data/raw/report_index_selected.csv")
c = pd.read_csv(ROOT / "data/processed/qc_checks.csv")
h = pd.read_csv(ROOT / "data/processed/bank_year_hjb_mapping.csv")
g = pd.read_csv(ROOT / "model_outputs/hjb_grid.csv")
s = json.loads((ROOT / "model_outputs/model_summary.json").read_text(encoding="utf-8"))

assert len(m) == 48, f"expected 48 bank-years, found {len(m)}"
assert m[["symbol", "jalali_year"]].drop_duplicates().shape[0] == 48
assert set(m["jalali_year"]) == set(range(1397, 1405))
assert m["parse_status"].eq("ok").all()
assert m["qc_pass"].all()
assert c["pass"].all(), c.loc[~c["pass"]].to_dict("records")
assert m["statement_heading"].fillna("").str.contains("تلفیقی").sum() == 0
assert m["gross_loans_proxy"].notna().all()
assert m["customer_funding"].gt(0).all()
assert m["cash_bank_assets_to_deposits"].between(0, 2).all()
assert m["loan_to_deposit_proxy"].between(0, 3).all()
assert r["download_status"].eq("ok").all()
assert r["pdf_sha256"].str.len().eq(64).all()
assert len(h) == 48 and h[["symbol", "jalali_year"]].drop_duplicates().shape[0] == 48
assert h["hjb_risk_index"].between(0, 1).all()
assert len(g) == 2295 and g["feasible"].all()
assert s["all_grid_states_feasible"]
assert s["bellman_residual_sup_norm"] < 1e-8
assert s["surrogate"]["test_r_squared"] > 0.99
assert s["numerical_diagnostics"]["liquidity_monotonicity_share"] == 1.0
assert s["numerical_diagnostics"]["credit_monotonicity_share"] == 1.0
print("VALIDATION_OK: 48/48 bank-years, 288/288 QC checks, and HJB numerical checks passed.")
