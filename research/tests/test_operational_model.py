"""Independent invariants for the operational three-state survival engine."""
from __future__ import annotations

import numpy as np
import pytest

from operational_model import (
    BankInputs,
    OperationalParameters,
    evaluate_bank,
    load_solution,
    minimum_feasible_controls,
    save_solution,
    solve_survival_hjb,
)


@pytest.fixture(scope="module")
def solution():
    return solve_survival_hjb(n_wealth=11, n_liquidity=13, n_credit=11, n_controls=9)


@pytest.fixture()
def bank():
    return BankInputs(
        name="test bank", total_assets=1_500_000, equity=80_000,
        minimum_required_equity=60_000, hqla=120_000,
        net_cash_outflow_30d=115_000, available_stable_funding=660_000,
        required_stable_funding=600_000, gross_loans=900_000,
        nonperforming_loans=90_000, max_additional_borrowing=80_000,
        max_sellable_assets=120_000,
    )


def test_bank_input_validation_rejects_impossible_npl(bank):
    bad = BankInputs(**{**bank.__dict__, "nonperforming_loans": bank.gross_loans + 1})
    with pytest.raises(ValueError, match="cannot exceed"):
        bad.validate()


def test_minimum_feasible_controls_satisfy_both_constraints():
    p = OperationalParameters(stable_funding_buffer=-0.05, max_borrowing=0.7, max_asset_sale=1.0)
    result = minimum_feasible_controls(1.45, p)
    assert result["feasible"]
    a, b = result["asset_sale"], result["borrowing"]
    assert 1 + b + p.sale_liquidity_yield * a >= 1.45 - 1e-10
    assert p.stable_funding_buffer + p.sale_nsfr_relief * a - p.borrowing_nsfr_weight * b >= -1e-10


def test_solution_is_probability_and_absorbing(solution):
    v = solution.survival_probability
    assert np.isfinite(v).all()
    assert np.min(v) >= 0 and np.max(v) <= 1
    assert np.allclose(v[0], 0)


def test_risk_has_economic_monotonicity(solution):
    pd = solution.default_probability
    assert np.all(np.diff(pd, axis=0) <= 1e-12)  # more capital lowers risk
    assert np.all(np.diff(pd, axis=1) >= -1e-12)  # inverse-LCR raises risk
    assert np.all(np.diff(pd, axis=2) >= -1e-12)  # NPL ratio raises risk


def test_evaluation_returns_actions_and_disclaimer(solution, bank):
    result = evaluate_bank(bank, solution, target_pd=0.20)
    assert 0 <= result["default_probability"] <= 1
    assert result["region"] in {"safe", "warning", "insolvency_region", "critical"}
    assert "combined_minimum_effort" in result["exit_actions_to_target_pd"]
    assert "not a legal insolvency" in result["interpretation_warning"]


def test_solution_roundtrip(tmp_path, solution):
    path = tmp_path / "solution.npz"
    save_solution(solution, path)
    restored = load_solution(path)
    assert np.allclose(restored.survival_probability, solution.survival_probability)
    assert restored.parameters == solution.parameters


def test_bank_outside_grid_never_receives_clipped_prediction():
    import pytest
    from research.operational_model import BankInputs, load_solution, evaluate_bank
    from pathlib import Path
    sol = load_solution(Path(__file__).resolve().parents[1] / "model_outputs_v4/operational_survival_solution.npz")
    bank = BankInputs(total_assets=1000, equity=120, minimum_required_equity=60,
        hqla=100, net_cash_outflow_30d=85, available_stable_funding=120,
        required_stable_funding=100, gross_loans=600, nonperforming_loans=300,
        max_additional_borrowing=20, max_sellable_assets=20)
    with pytest.raises(ValueError, match="outside the validated HJB grid"):
        evaluate_bank(bank, sol)
