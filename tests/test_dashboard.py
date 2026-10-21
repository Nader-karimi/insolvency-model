"""Lightweight deterministic tests independent of the Streamlit runtime."""
from pathlib import Path
from scenario_engine import Scenario, stress
from hjb_dashboard import risk_slice,policy_slice,boundary_surface,in_domain,pd_at
from research.operational_model import load_solution,minimum_feasible_controls
import numpy as np

SOL=load_solution(Path(__file__).parents[1]/'research/model_outputs_v4/operational_survival_solution.npz')

def test_public_defaults_reproducible():
    s=stress(Scenario())
    assert abs(s['E2']+12061.25)<1e-8
    assert abs(s['d_final']-.3109375)<1e-10
    assert abs(s['lambda_final']-2.240625)<1e-10
    assert s['sale']<=Scenario().marketable*(1+Scenario().shock)
    assert s['unmet_liquidity_gap']>=0

def test_state_and_pd_contract():
    assert in_domain(SOL,1.8,.82,.05)
    assert pd_at(SOL,1.8,.82,.05) is not None
    assert pd_at(SOL,1,.82,.05)==1.
    assert pd_at(SOL,5.,.82,.05) is None
    assert pd_at(SOL,1.5,4,.05)==1.  # control infeasibility, not clipping

def test_slice_matches_center_point():
    w=float(SOL.wealth_grid[10]);j=8;k=10
    z=risk_slice(SOL,w)[2]
    assert abs(z[k,j]-pd_at(SOL,w,float(SOL.liquidity_grid[j]),float(SOL.credit_grid[k])))<1e-10

def test_policy_and_threshold_shapes():
    assert policy_slice(SOL,1.8)[1].shape==(23,29)
    z=boundary_surface(SOL,.5)
    assert z.shape==(23,29)
    assert np.nanmin(z)>=1

def test_feasibility_independent():
    assert minimum_feasible_controls(.9,SOL.parameters)['feasible']
    assert not minimum_feasible_controls(4.,SOL.parameters)['feasible']


def test_shock_monotonic_loss():
    mild=stress(Scenario(shock=-.1))
    severe=stress(Scenario(shock=-.3))
    assert mild['E2'] > severe['E2']
