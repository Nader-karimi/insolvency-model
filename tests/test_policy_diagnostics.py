"""Check reported controls against their mathematical constraints and units."""
from pathlib import Path
import numpy as np
from policy_diagnostics import (policy_diagnostic,feasible_pair_at_state,
                                control_slacks,zero_borrowing_alternative)
from research.operational_model import load_solution,minimum_feasible_controls

ROOT=Path(__file__).resolve().parents[1]
SOL=load_solution(ROOT/'research/model_outputs_v4/operational_survival_solution.npz')


def test_adverse_case_minimum_pair_has_binding_constraints():
    p=SOL.parameters
    lam=22500/15000
    minimum=minimum_feasible_controls(lam,p)
    assert minimum['feasible']
    assert np.isclose(minimum['asset_sale'],4/15,atol=1e-12)
    assert np.isclose(minimum['borrowing'],43/150,atol=1e-12)
    assert all(v>=-1e-10 for v in control_slacks(lam,p,minimum['asset_sale'],minimum['borrowing']).values())


def test_nearest_grid_policy_is_not_presented_as_exact_optimizer():
    lam=1.5;state=(8000/7000,lam,.18)
    x=policy_diagnostic(SOL,*state)
    assert x is not None
    node=x['grid_state']
    gp=x['grid_policy']
    exact=x['candidate']
    assert gp is not None and exact is not None
    assert np.isclose(node[1], SOL.liquidity_grid[np.argmin(abs(SOL.liquidity_grid-lam))])
    assert not np.isclose(node[1],lam)  # critical: nearest-node control differs from user input
    assert min(gp['margins'].values())>=-1e-10
    assert min(exact['margins'].values())>=-1e-10
    assert np.isclose(exact['borrowing'],lam-1-SOL.parameters.sale_liquidity_yield*exact['sale'])
    assert exact['borrowing'] > gp['borrowing']


def test_less_borrowing_may_require_more_asset_sales():
    p=SOL.parameters; lam=1.5
    min_pair=minimum_feasible_controls(lam,p)
    zero=zero_borrowing_alternative(lam,p)
    assert zero is not None
    assert np.isclose(zero['borrowing'],0)
    assert zero['sale']>min_pair['asset_sale']
    assert all(x>=-1e-10 for x in zero['margins'].values())


def test_refuse_impossible_controls_or_extrapolated_policy():
    p=SOL.parameters
    assert feasible_pair_at_state(4.0,p,0.5) is None
    assert policy_diagnostic(SOL,5.0,1.5,.18) is None
    assert policy_diagnostic(SOL,1.1,4.0,.18) is None
    assert zero_borrowing_alternative(3.5,p) is None


def test_low_risk_default_has_zero_action():
    x=policy_diagnostic(SOL,8000/4500,12300/15000,.05)
    assert x['grid_policy'] is not None
    assert np.isclose(x['grid_policy']['sale'],0.0)
    assert np.isclose(x['grid_policy']['borrowing'],0.0)
    assert np.isclose(x['candidate']['sale'],0.0)
    assert np.isclose(x['candidate']['borrowing'],0.0)
