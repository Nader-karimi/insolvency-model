"""Reproducible scenario explanations; not a bank-default validation test."""
from pathlib import Path
import sys, json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.counterfactual_risk_attribution import counterfactual_rows
from research.operational_model import load_solution
from hjb_dashboard import local_counterfactuals
ROOT=Path(__file__).resolve().parents[1]

def test_adverse_counterfactual_monotonic_and_replicable():
    model=load_solution(ROOT/'research/model_outputs_v4/operational_survival_solution.npz')
    cases=json.loads((ROOT/'research/model_outputs_v4/sample_bank_assessments.json').read_text())
    rows=counterfactual_rows(model,cases['baseline']['states'],cases['adverse']['states'])
    assert len(rows)==5
    assert np.isclose(rows[0]['model_probability'],cases['adverse']['default_probability'],atol=1e-10)
    assert np.isclose(rows[-1]['model_probability'],cases['baseline']['default_probability'],atol=1e-10)
    assert all(0<=row['model_probability']<=1 for row in rows)
    assert all(row['model_probability']<=rows[0]['model_probability']+1e-12 for row in rows[1:])

def test_live_dashboard_counterfactuals():
    model=load_solution(ROOT/'research/model_outputs_v4/operational_survival_solution.npz')
    rows=local_counterfactuals(model,1.2,1.1,.10)
    assert len(rows)==4
    assert all(row['pd'] is not None for row in rows)
    assert all(row['pd'] <= rows[0]['pd']+1e-11 for row in rows[1:])
    assert local_counterfactuals(model,5.1,1.1,.10)[0]['pd'] is None
