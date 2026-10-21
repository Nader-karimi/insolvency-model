"""Finite-horizon three-state bank survival HJB and action engine.

The operational state is (w, lambda, d):
  w       = book equity / minimum required equity,
  lambda  = 30-day net cash outflow / HQLA (= inverse LCR),
  d       = non-performing loans / gross loans.

The module distinguishes a model-implied insolvency region from a legal or
supervisory insolvency determination. Monetary inputs use one common unit
(million rials in the dashboard) and are normalized internally.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BankInputs:
    total_assets: float
    equity: float
    minimum_required_equity: float
    hqla: float
    net_cash_outflow_30d: float
    available_stable_funding: float
    required_stable_funding: float
    gross_loans: float
    nonperforming_loans: float
    max_additional_borrowing: float
    max_sellable_assets: float
    name: str = "Input bank"

    def validate(self) -> None:
        positive = {
            "total_assets": self.total_assets,
            "minimum_required_equity": self.minimum_required_equity,
            "hqla": self.hqla,
            "net_cash_outflow_30d": self.net_cash_outflow_30d,
            "required_stable_funding": self.required_stable_funding,
            "gross_loans": self.gross_loans,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = {
            "equity": self.equity,
            "available_stable_funding": self.available_stable_funding,
            "nonperforming_loans": self.nonperforming_loans,
            "max_additional_borrowing": self.max_additional_borrowing,
            "max_sellable_assets": self.max_sellable_assets,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.nonperforming_loans > self.gross_loans:
            raise ValueError("nonperforming_loans cannot exceed gross_loans")
        if self.equity > self.total_assets:
            raise ValueError("equity cannot exceed total_assets")

    @property
    def wealth_ratio(self) -> float:
        return self.equity / self.minimum_required_equity

    @property
    def liquidity_state(self) -> float:
        return self.net_cash_outflow_30d / self.hqla

    @property
    def credit_state(self) -> float:
        return self.nonperforming_loans / self.gross_loans

    @property
    def lcr(self) -> float:
        return self.hqla / self.net_cash_outflow_30d

    @property
    def nsfr(self) -> float:
        return self.available_stable_funding / self.required_stable_funding


@dataclass(frozen=True)
class OperationalParameters:
    horizon_years: float = 1.0
    time_steps: int = 30
    mu_liquidity: float = 0.03
    sigma_liquidity: float = 0.30
    mu_credit: float = 0.02
    sigma_credit: float = 0.25
    liquidity_credit_correlation: float = 0.35
    fire_sale_credit_feedback: float = 0.10
    asset_return: float = 0.04
    liquidity_cost: float = 0.08
    credit_loss_severity: float = 0.65
    borrowing_cost: float = 0.08
    sale_discount: float = 0.04
    market_illiquidity: float = 0.20
    wealth_volatility: float = 0.22
    sale_liquidity_yield: float = 0.80
    stable_funding_buffer: float = 0.10
    borrowing_nsfr_weight: float = 1.00
    sale_nsfr_relief: float = 0.70
    max_borrowing: float = 0.70
    max_asset_sale: float = 1.00
    warning_pd: float = 0.20
    insolvency_region_pd: float = 0.50

    def validate(self) -> None:
        if self.horizon_years <= 0 or self.time_steps < 2:
            raise ValueError("positive horizon and at least two time steps are required")
        if not -0.999 < self.liquidity_credit_correlation < 0.999:
            raise ValueError("liquidity_credit_correlation must be in (-1,1)")
        if min(self.sigma_liquidity, self.sigma_credit, self.wealth_volatility) < 0:
            raise ValueError("volatilities must be nonnegative")
        if min(self.max_borrowing, self.max_asset_sale) < 0:
            raise ValueError("control bounds must be nonnegative")
        if not 0 < self.warning_pd < self.insolvency_region_pd < 1:
            raise ValueError("probability thresholds must satisfy 0<warning<insolvency<1")


@dataclass
class SurvivalSolution:
    wealth_grid: np.ndarray
    liquidity_grid: np.ndarray
    credit_grid: np.ndarray
    survival_probability: np.ndarray
    optimal_asset_sale: np.ndarray
    optimal_borrowing: np.ndarray
    feasible: np.ndarray
    parameters: OperationalParameters
    quadrature: str

    @property
    def default_probability(self) -> np.ndarray:
        return np.clip(1.0 - self.survival_probability, 0.0, 1.0)


def parameters_from_bank(bank: BankInputs, base: OperationalParameters | None = None) -> OperationalParameters:
    """Translate bank capacity inputs into the dimensionless control problem."""

    bank.validate()
    base = base or OperationalParameters()
    # NSFR slack is dimensionless. Controls are intensities relative to HQLA.
    nsfr_slack = float(np.clip(bank.nsfr - 1.0, -0.75, 2.0))
    max_borrowing = float(np.clip(bank.max_additional_borrowing / bank.hqla, 0.0, 3.0))
    max_sale = float(np.clip(bank.max_sellable_assets / bank.hqla, 0.0, 4.0))
    params = replace(
        base,
        stable_funding_buffer=nsfr_slack,
        max_borrowing=max_borrowing,
        max_asset_sale=max_sale,
    )
    params.validate()
    return params


def minimum_feasible_controls(
    liquidity_state: float,
    params: OperationalParameters,
) -> dict[str, float | bool]:
    """Closed-form minimum sale and borrowing satisfying LCR/NSFR/bounds."""

    gap = max(0.0, float(liquidity_state) - 1.0)
    eta = params.sale_liquidity_yield
    wa, wb = params.sale_nsfr_relief, params.borrowing_nsfr_weight
    candidates = [0.0]
    if eta > 0:
        candidates.append(max(0.0, gap - params.max_borrowing) / eta)
    elif gap > params.max_borrowing:
        return {"feasible": False, "asset_sale": np.nan, "borrowing": np.nan}
    if wa > 0:
        candidates.append(max(0.0, -params.stable_funding_buffer / wa))
    elif params.stable_funding_buffer < 0:
        return {"feasible": False, "asset_sale": np.nan, "borrowing": np.nan}
    denominator = wa + wb * eta
    if denominator > 0:
        candidates.append(max(0.0, wb * gap - params.stable_funding_buffer) / denominator)
    asset_sale = max(candidates)
    borrowing = max(0.0, gap - eta * asset_sale)
    feasible = bool(
        asset_sale <= params.max_asset_sale + 1e-12
        and borrowing <= params.max_borrowing + 1e-12
        and params.stable_funding_buffer + wa * asset_sale - wb * borrowing >= -1e-10
        and 1.0 + borrowing + eta * asset_sale >= liquidity_state - 1e-10
    )
    return {
        "feasible": feasible,
        "asset_sale": float(asset_sale) if feasible else np.nan,
        "borrowing": float(borrowing) if feasible else np.nan,
    }


def _trilinear(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    points: np.ndarray,
    absorbing_wealth: bool = True,
) -> np.ndarray:
    """Vectorized trilinear interpolation with an absorbing lower wealth edge."""

    g0, g1, g2 = grids
    p0, p1, p2 = points[:, 0], points[:, 1], points[:, 2]
    absorbed = p0 <= g0[0] + 1e-14 if absorbing_wealth else np.zeros(len(points), dtype=bool)
    p0 = np.clip(p0, g0[0], g0[-1])
    p1 = np.clip(p1, g1[0], g1[-1])
    p2 = np.clip(p2, g2[0], g2[-1])
    i0 = np.clip(np.searchsorted(g0, p0, side="right") - 1, 0, len(g0) - 2)
    i1 = np.clip(np.searchsorted(g1, p1, side="right") - 1, 0, len(g1) - 2)
    i2 = np.clip(np.searchsorted(g2, p2, side="right") - 1, 0, len(g2) - 2)
    t0 = (p0 - g0[i0]) / (g0[i0 + 1] - g0[i0])
    t1 = (p1 - g1[i1]) / (g1[i1 + 1] - g1[i1])
    t2 = (p2 - g2[i2]) / (g2[i2 + 1] - g2[i2])
    result = np.zeros(len(points), dtype=float)
    for b0, b1, b2 in product((0, 1), repeat=3):
        weight = (
            (t0 if b0 else 1 - t0)
            * (t1 if b1 else 1 - t1)
            * (t2 if b2 else 1 - t2)
        )
        result += weight * values[i0 + b0, i1 + b1, i2 + b2]
    result[absorbed] = 0.0
    return result


def interpolate_solution(
    solution: SurvivalSolution,
    points: np.ndarray,
    field: str = "default_probability",
) -> np.ndarray:
    values = getattr(solution, field)
    if callable(values):
        values = values()
    return _trilinear(
        (solution.wealth_grid, solution.liquidity_grid, solution.credit_grid),
        np.asarray(values), np.asarray(points, dtype=float), absorbing_wealth=(field == "survival_probability"),
    )


def solve_survival_hjb(
    params: OperationalParameters | None = None,
    n_wealth: int = 25,
    n_liquidity: int = 29,
    n_credit: int = 23,
    n_controls: int = 17,
    domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((1.0,4.0),(0.40,3.50),(0.01,0.35)),
) -> SurvivalSolution:
    """Solve the finite-horizon survival HJB by a monotone semi-Lagrangian scheme."""

    params = params or OperationalParameters()
    params.validate()
    wealth_grid = np.geomspace(*domain[0], n_wealth)
    liquidity_grid = np.geomspace(*domain[1], n_liquidity)
    credit_grid = np.geomspace(*domain[2], n_credit)
    log_grids = (np.log(wealth_grid), np.log(liquidity_grid), np.log(credit_grid))
    zz, xx, yy = np.meshgrid(*log_grids, indexing="ij")
    z0, x0, y0 = zz.ravel(), xx.ravel(), yy.ravel()
    lambda_state, credit_state = np.exp(x0), np.exp(y0)
    n_states = len(z0)
    controls = np.linspace(0.0, params.max_asset_sale, n_controls)

    borrowing = np.empty((n_controls, n_liquidity), dtype=float)
    feasible = np.empty((n_controls, n_liquidity), dtype=bool)
    for k, sale in enumerate(controls):
        b = np.maximum(0.0, liquidity_grid - 1.0 - params.sale_liquidity_yield * sale)
        okay = (
            (b <= params.max_borrowing + 1e-12)
            & (params.stable_funding_buffer + params.sale_nsfr_relief * sale
               - params.borrowing_nsfr_weight * b >= -1e-12)
        )
        borrowing[k] = b
        feasible[k] = okay
    feasible_state = feasible.any(axis=0)

    shape = (n_wealth, n_liquidity, n_credit)
    survival = np.ones(shape, dtype=float)
    survival[0, :, :] = 0.0
    survival[:, ~feasible_state, :] = 0.0
    optimal_sale = np.full(shape, np.nan)
    optimal_borrowing = np.full(shape, np.nan)

    dt = params.horizon_years / params.time_steps
    sqrt_dt = np.sqrt(dt)
    corr = params.liquidity_credit_correlation
    # Symmetric degree-three cubature: exact first two moments, eight nodes.
    shocks = []
    for zw, zl, zi in product((-1.0, 1.0), repeat=3):
        zd = corr * zl + np.sqrt(1.0 - corr**2) * zi
        shocks.append((zw, zl, zd, 1.0 / 8.0))

    state_liquidity_index = np.tile(np.repeat(np.arange(n_liquidity), n_credit), n_wealth)
    grids = (wealth_grid, liquidity_grid, credit_grid)
    for _ in range(params.time_steps):
        best = np.full(n_states, -1.0)
        best_sale = np.full(n_states, np.nan)
        best_borrow = np.full(n_states, np.nan)
        for k, sale in enumerate(controls):
            b_state = borrowing[k, state_liquidity_index]
            feasible_flat = feasible[k, state_liquidity_index]
            drift_w = (
                params.asset_return
                - params.liquidity_cost * np.maximum(lambda_state - 1.0, 0.0)
                - params.credit_loss_severity * credit_state
                - params.borrowing_cost * b_state
                - params.sale_discount * sale
                - 0.5 * params.market_illiquidity * sale**2
            )
            expected = np.zeros(n_states)
            for zw, zl, zd, probability in shocks:
                next_log_w = z0 + (drift_w - 0.5 * params.wealth_volatility**2) * dt \
                    + params.wealth_volatility * sqrt_dt * zw
                next_log_l = x0 + (params.mu_liquidity - 0.5 * params.sigma_liquidity**2) * dt \
                    + params.sigma_liquidity * sqrt_dt * zl
                next_log_d = y0 + (
                    params.mu_credit + params.fire_sale_credit_feedback * sale
                    - 0.5 * params.sigma_credit**2
                ) * dt + params.sigma_credit * sqrt_dt * zd
                points = np.column_stack([
                    np.exp(next_log_w), np.exp(next_log_l), np.exp(next_log_d),
                ])
                expected += probability * _trilinear(grids, survival, points)
            expected[~feasible_flat] = -1.0
            improve = expected > best
            best[improve] = expected[improve]
            best_sale[improve] = sale
            best_borrow[improve] = b_state[improve]
        survival = np.clip(best.reshape(shape), 0.0, 1.0)
        survival[0, :, :] = 0.0
        survival[:, ~feasible_state, :] = 0.0
        optimal_sale = best_sale.reshape(shape)
        optimal_borrowing = best_borrow.reshape(shape)

    feasible_grid = np.broadcast_to(feasible_state[None, :, None], shape).copy()
    return SurvivalSolution(
        wealth_grid=wealth_grid,
        liquidity_grid=liquidity_grid,
        credit_grid=credit_grid,
        survival_probability=survival,
        optimal_asset_sale=optimal_sale,
        optimal_borrowing=optimal_borrowing,
        feasible=feasible_grid,
        parameters=params,
        quadrature="eight-node correlated Gaussian cubature",
    )


def _first_target(candidates: np.ndarray, probabilities: np.ndarray, target: float) -> float | None:
    indices = np.flatnonzero(probabilities <= target)
    return float(candidates[indices[0]]) if len(indices) else None


def _policy_at(solution: SurvivalSolution, point: np.ndarray) -> tuple[float, float]:
    sale = _trilinear(
        (solution.wealth_grid, solution.liquidity_grid, solution.credit_grid),
        np.nan_to_num(solution.optimal_asset_sale), point[None, :], absorbing_wealth=False,
    )[0]
    borrowing = _trilinear(
        (solution.wealth_grid, solution.liquidity_grid, solution.credit_grid),
        np.nan_to_num(solution.optimal_borrowing), point[None, :], absorbing_wealth=False,
    )[0]
    return float(sale), float(borrowing)


def evaluate_bank(
    bank: BankInputs,
    solution: SurvivalSolution,
    target_pd: float | None = None,
) -> dict[str, Any]:
    """Classify the bank and calculate one-dimensional and combined exit actions."""

    bank.validate()
    params = solution.parameters
    target_pd = params.warning_pd if target_pd is None else float(target_pd)
    if not 0 < target_pd < 1:
        raise ValueError("target_pd must be in (0,1)")
    raw_point = np.array([bank.wealth_ratio, bank.liquidity_state, bank.credit_state], dtype=float)
    # A point beyond the solved numerical range must not be silently clipped and
    # reported as a bank-specific HJB probability or feasible remediation.
    solved_bounds = (
        (solution.wealth_grid[0], solution.wealth_grid[-1]),
        (solution.liquidity_grid[0], solution.liquidity_grid[-1]),
        (solution.credit_grid[0], solution.credit_grid[-1]),
    )
    if any(not (lo <= value <= hi) for value, (lo, hi) in zip(raw_point, solved_bounds)):
        raise ValueError("Bank state lies outside the validated HJB grid; no probability or remediation can be reported without re-solving the model on a covering domain")
    point = np.array([
        np.clip(raw_point[0], solution.wealth_grid[0], solution.wealth_grid[-1]),
        np.clip(raw_point[1], solution.liquidity_grid[0], solution.liquidity_grid[-1]),
        np.clip(raw_point[2], solution.credit_grid[0], solution.credit_grid[-1]),
    ])
    pd_current = float(interpolate_solution(solution, point[None, :], "default_probability")[0])
    feasibility = minimum_feasible_controls(bank.liquidity_state, params)
    sale_ratio, borrow_ratio = _policy_at(solution, point)
    hard_equity_breach = bank.equity <= bank.minimum_required_equity
    lcr_breach, nsfr_breach = bank.lcr < 1.0, bank.nsfr < 1.0
    control_infeasible = not bool(feasibility["feasible"])
    regulatory_breach = hard_equity_breach or lcr_breach or nsfr_breach or control_infeasible

    if hard_equity_breach or control_infeasible:
        region = "critical"
        label = "Hard threshold/control infeasibility"
    elif pd_current >= params.insolvency_region_pd:
        region = "insolvency_region"
        label = "Model-implied insolvency region"
    elif regulatory_breach or pd_current >= params.warning_pd:
        region = "warning"
        label = "Warning/regulatory remediation region"
    else:
        region = "safe"
        label = "Model-implied safe region"

    # Capital-only action.
    w_candidates = np.linspace(max(point[0], 1.0), solution.wealth_grid[-1], 240)
    w_points = np.column_stack([
        w_candidates,
        np.full_like(w_candidates, point[1]),
        np.full_like(w_candidates, point[2]),
    ])
    w_pd = interpolate_solution(solution, w_points, "default_probability")
    target_w = _first_target(w_candidates, w_pd, target_pd)
    capital_injection = None if target_w is None else max(
        0.0, target_w * bank.minimum_required_equity - bank.equity,
    )

    # Liquidity-only action: lower lambda by adding HQLA or reducing NCOF.
    l_candidates = np.linspace(point[1], solution.liquidity_grid[0], 240)
    l_points = np.column_stack([
        np.full_like(l_candidates, point[0]), l_candidates,
        np.full_like(l_candidates, point[2]),
    ])
    l_pd = interpolate_solution(solution, l_points, "default_probability")
    target_l = _first_target(l_candidates, l_pd, target_pd)
    hqla_increase = None if target_l is None else max(
        0.0, bank.net_cash_outflow_30d / target_l - bank.hqla,
    )
    ncof_reduction = None if target_l is None else max(
        0.0, bank.net_cash_outflow_30d - target_l * bank.hqla,
    )

    # Credit-only action.
    d_candidates = np.linspace(point[2], solution.credit_grid[0], 240)
    d_points = np.column_stack([
        np.full_like(d_candidates, point[0]),
        np.full_like(d_candidates, point[1]), d_candidates,
    ])
    d_pd = interpolate_solution(solution, d_points, "default_probability")
    target_d = _first_target(d_candidates, d_pd, target_pd)
    npl_reduction = None if target_d is None else max(
        0.0, bank.nonperforming_loans - target_d * bank.gross_loans,
    )

    # Combined least-normalized-effort action on the frozen parameter surface.
    wc = np.linspace(max(point[0], 1.0), min(solution.wealth_grid[-1], max(point[0], 1.0) + 1.5), 15)
    lc = np.linspace(point[1], solution.liquidity_grid[0], 17)
    dc = np.linspace(point[2], solution.credit_grid[0], 15)
    W, L, D = np.meshgrid(wc, lc, dc, indexing="ij")
    candidate_points = np.column_stack([W.ravel(), L.ravel(), D.ravel()])
    candidate_pd = interpolate_solution(solution, candidate_points, "default_probability")
    capital = np.maximum(0.0, candidate_points[:, 0] * bank.minimum_required_equity - bank.equity)
    hqla = np.maximum(0.0, bank.net_cash_outflow_30d / candidate_points[:, 1] - bank.hqla)
    npl = np.maximum(0.0, bank.nonperforming_loans - candidate_points[:, 2] * bank.gross_loans)
    score = (
        capital / max(bank.equity, bank.minimum_required_equity, 1.0)
        + hqla / max(bank.hqla, 1.0)
        + npl / max(bank.nonperforming_loans, 0.01 * bank.gross_loans, 1.0)
    )
    eligible = candidate_pd <= target_pd
    combined = None
    if eligible.any():
        idx = int(np.argmin(np.where(eligible, score, np.inf)))
        combined = {
            "target_default_probability": float(candidate_pd[idx]),
            "target_wealth_ratio": float(candidate_points[idx, 0]),
            "target_liquidity_state": float(candidate_points[idx, 1]),
            "target_credit_state": float(candidate_points[idx, 2]),
            "capital_injection": float(capital[idx]),
            "hqla_increase": float(hqla[idx]),
            "npl_reduction": float(npl[idx]),
            "normalized_effort_score": float(score[idx]),
        }

    stable_funding_increase = max(
        0.0, bank.required_stable_funding - bank.available_stable_funding,
    )
    immediate_hqla_gap = max(0.0, bank.net_cash_outflow_30d - bank.hqla)
    domain_flags = {
        "wealth_clipped": bool(raw_point[0] < solution.wealth_grid[0] or raw_point[0] > solution.wealth_grid[-1]),
        "liquidity_clipped": bool(raw_point[1] < solution.liquidity_grid[0] or raw_point[1] > solution.liquidity_grid[-1]),
        "credit_clipped": bool(raw_point[2] < solution.credit_grid[0] or raw_point[2] > solution.credit_grid[-1]),
    }
    return {
        "bank_name": bank.name,
        "model_horizon_years": params.horizon_years,
        "states": {
            "wealth_ratio": bank.wealth_ratio,
            "liquidity_state": bank.liquidity_state,
            "credit_state": bank.credit_state,
        },
        "regulatory_metrics": {
            "lcr": bank.lcr,
            "nsfr": bank.nsfr,
            "equity_to_minimum": bank.wealth_ratio,
        },
        "default_probability": pd_current,
        "survival_probability": 1.0 - pd_current,
        "region": region,
        "region_label": label,
        "regulatory_breach": regulatory_breach,
        "breaches": {
            "equity": hard_equity_breach,
            "lcr": lcr_breach,
            "nsfr": nsfr_breach,
            "control_infeasible": control_infeasible,
        },
        "optimal_policy": {
            "asset_sale_ratio_to_hqla": sale_ratio,
            "borrowing_ratio_to_hqla": borrow_ratio,
            "asset_sale_amount": sale_ratio * bank.hqla,
            "borrowing_amount": borrow_ratio * bank.hqla,
        },
        "minimum_regulatory_actions": {
            "immediate_hqla_gap": immediate_hqla_gap,
            "stable_funding_increase": stable_funding_increase,
            "minimum_feasible_asset_sale": feasibility["asset_sale"] * bank.hqla
                if feasibility["feasible"] else None,
            "minimum_feasible_borrowing": feasibility["borrowing"] * bank.hqla
                if feasibility["feasible"] else None,
        },
        "exit_actions_to_target_pd": {
            "target_pd": target_pd,
            "capital_only": {
                "target_wealth_ratio": target_w,
                "capital_injection": capital_injection,
            },
            "liquidity_only": {
                "target_liquidity_state": target_l,
                "hqla_increase": hqla_increase,
                "net_cash_outflow_reduction": ncof_reduction,
            },
            "credit_only": {
                "target_credit_state": target_d,
                "npl_reduction": npl_reduction,
            },
            "combined_minimum_effort": combined,
        },
        "domain_flags": domain_flags,
        "interpretation_warning": (
            "Model-implied screening result; not a legal insolvency declaration, "
            "supervisory rating, or investment recommendation. Re-estimate parameters "
            "and independently validate before high-stakes use."
        ),
    }


def save_solution(solution: SurvivalSolution, path: str | Path) -> None:
    path = Path(path)
    np.savez_compressed(
        path,
        wealth_grid=solution.wealth_grid,
        liquidity_grid=solution.liquidity_grid,
        credit_grid=solution.credit_grid,
        survival_probability=solution.survival_probability,
        optimal_asset_sale=solution.optimal_asset_sale,
        optimal_borrowing=solution.optimal_borrowing,
        feasible=solution.feasible,
        parameter_names=np.array(list(asdict(solution.parameters).keys())),
        parameter_values=np.array(list(asdict(solution.parameters).values()), dtype=float),
    )


def load_solution(path: str | Path) -> SurvivalSolution:
    data = np.load(Path(path), allow_pickle=False)
    params = OperationalParameters(**dict(zip(data["parameter_names"].tolist(), data["parameter_values"].tolist())))
    return SurvivalSolution(
        wealth_grid=data["wealth_grid"],
        liquidity_grid=data["liquidity_grid"],
        credit_grid=data["credit_grid"],
        survival_probability=data["survival_probability"],
        optimal_asset_sale=data["optimal_asset_sale"],
        optimal_borrowing=data["optimal_borrowing"],
        feasible=data["feasible"].astype(bool),
        parameters=params,
        quadrature="eight-node correlated Gaussian cubature",
    )
