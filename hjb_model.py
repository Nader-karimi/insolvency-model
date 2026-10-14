"""Numerical core for the liquidity--credit bank insolvency model.

The module implements the corrected stationary HJB described in the paper.
It is intentionally self-contained: regulatory constraints are represented by
the feasible-control set rather than by an unspecified penalty term.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata


@dataclass(frozen=True)
class ModelParameters:
    """Parameters of the stylised, dimensionless HJB calibration."""

    discount_rate: float = 0.08
    mu_liquidity: float = 0.02
    sigma_liquidity: float = 0.22
    mu_credit: float = 0.005
    sigma_credit: float = 0.20
    liquidity_credit_correlation: float = 0.35
    fire_sale_credit_feedback: float = 0.12
    asset_return: float = 0.09
    liquidity_cost: float = 0.015
    credit_loss_severity: float = 0.70
    borrowing_cost: float = 0.04
    sale_discount: float = 0.03
    market_illiquidity: float = 0.25
    wealth_volatility: float = 0.12
    initial_hqla: float = 0.65
    outflow_scale: float = 0.40
    sale_liquidity_yield: float = 0.80
    stable_funding_buffer: float = 0.18
    borrowing_nsfr_weight: float = 1.00
    sale_nsfr_relief: float = 0.70
    max_borrowing: float = 0.70
    max_asset_sale: float = 0.80
    time_step: float = 0.25
    wealth_buffer_ratio: float = 1.35

    def validate(self) -> None:
        if self.discount_rate <= 0 or self.time_step <= 0:
            raise ValueError("discount_rate and time_step must be positive")
        if not -0.999 < self.liquidity_credit_correlation < 0.999:
            raise ValueError("correlation must lie strictly between -1 and 1")
        if min(self.max_borrowing, self.max_asset_sale) <= 0:
            raise ValueError("control bounds must be positive")
        if self.wealth_buffer_ratio <= 1:
            raise ValueError("wealth_buffer_ratio must exceed one")


@dataclass
class HJBSolution:
    liquidity_grid: np.ndarray
    credit_grid: np.ndarray
    value_adjustment: np.ndarray
    optimal_asset_sale: np.ndarray
    optimal_borrowing: np.ndarray
    feasible: np.ndarray
    iterations: int
    policy_changes: int
    bellman_residual: float
    parameters: ModelParameters

    @property
    def risk_distance(self) -> np.ndarray:
        """Certainty-equivalent log distance from a low-risk reference state."""

        interp = RegularGridInterpolator(
            (self.liquidity_grid, self.credit_grid), self.value_adjustment,
            bounds_error=False, fill_value=None,
        )
        safe_lambda = max(float(self.liquidity_grid.min()), 0.60)
        safe_credit = max(float(self.credit_grid.min()), 0.02)
        safe_value = float(interp([[safe_lambda, safe_credit]])[0])
        return -self.parameters.discount_rate * (self.value_adjustment - safe_value)

    @property
    def risk_index(self) -> np.ndarray:
        raw = self.risk_distance
        lo, hi = float(np.nanmin(raw)), float(np.nanmax(raw))
        return np.clip((raw - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _bilinear_transitions(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    next_x: np.ndarray,
    next_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened neighbour indices and weights for bilinear interpolation."""

    nx, ny = len(x_grid), len(y_grid)
    x = np.clip(next_x, x_grid[0], x_grid[-1])
    y = np.clip(next_y, y_grid[0], y_grid[-1])
    ix = np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, ny - 2)
    tx = (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    ty = (y - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    indices = np.column_stack([
        ix * ny + iy,
        (ix + 1) * ny + iy,
        ix * ny + (iy + 1),
        (ix + 1) * ny + (iy + 1),
    ]).astype(np.int32)
    weights = np.column_stack([
        (1 - tx) * (1 - ty),
        tx * (1 - ty),
        (1 - tx) * ty,
        tx * ty,
    ])
    return indices, weights


def _transition_tensors(
    params: ModelParameters,
    liquidity_grid: np.ndarray,
    credit_grid: np.ndarray,
    controls: np.ndarray,
    quadrature_order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute correlated Gaussian semi-Lagrangian transitions."""

    x_grid, y_grid = np.log(liquidity_grid), np.log(credit_grid)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    x0, y0 = xx.ravel(), yy.ravel()
    nodes, one_d_weights = np.polynomial.hermite.hermgauss(quadrature_order)
    nodes = np.sqrt(2.0) * nodes
    one_d_weights = one_d_weights / np.sqrt(np.pi)
    shock_pairs = [(u, v, wu * wv) for u, wu in zip(nodes, one_d_weights)
                   for v, wv in zip(nodes, one_d_weights)]

    n_controls, n_states = len(controls), len(x0)
    neighbours = 4 * len(shock_pairs)
    indices = np.empty((n_controls, n_states, neighbours), dtype=np.int32)
    weights = np.empty((n_controls, n_states, neighbours), dtype=np.float64)
    dt, sqrt_dt = params.time_step, np.sqrt(params.time_step)
    corr = params.liquidity_credit_correlation

    for k, asset_sale in enumerate(controls):
        cursor = 0
        for z1, z2, probability in shock_pairs:
            z_credit = corr * z1 + np.sqrt(1.0 - corr**2) * z2
            next_x = (
                x0 + (params.mu_liquidity - 0.5 * params.sigma_liquidity**2) * dt
                + params.sigma_liquidity * sqrt_dt * z1
            )
            next_y = (
                y0 + (params.mu_credit + params.fire_sale_credit_feedback * asset_sale
                      - 0.5 * params.sigma_credit**2) * dt
                + params.sigma_credit * sqrt_dt * z_credit
            )
            idx, wgt = _bilinear_transitions(x_grid, y_grid, next_x, next_y)
            indices[k, :, cursor:cursor + 4] = idx
            weights[k, :, cursor:cursor + 4] = probability * wgt
            cursor += 4
    return indices, weights


def _control_rewards(
    params: ModelParameters,
    liquidity_grid: np.ndarray,
    credit_grid: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute HJB flow reward, minimum borrowing and feasibility."""

    liquidity, credit = np.meshgrid(liquidity_grid, credit_grid, indexing="ij")
    liquidity, credit = liquidity.ravel(), credit.ravel()
    n_controls, n_states = len(controls), len(liquidity)
    reward = np.full((n_controls, n_states), -np.inf)
    borrowing = np.full((n_controls, n_states), np.nan)
    feasible = np.zeros((n_controls, n_states), dtype=bool)

    for k, asset_sale in enumerate(controls):
        lcr_gap = (
            params.outflow_scale * liquidity - params.initial_hqla
            - params.sale_liquidity_yield * asset_sale
        )
        minimum_borrowing = np.maximum(0.0, lcr_gap)
        nsfr_capacity = (
            params.stable_funding_buffer + params.sale_nsfr_relief * asset_sale
        ) / params.borrowing_nsfr_weight
        okay = minimum_borrowing <= np.minimum(params.max_borrowing, nsfr_capacity) + 1e-12
        flow = (
            params.asset_return
            - params.liquidity_cost * liquidity
            - params.credit_loss_severity * credit
            - params.borrowing_cost * minimum_borrowing
            - params.sale_discount * asset_sale
            - 0.5 * params.market_illiquidity * asset_sale**2
            - 0.5 * params.wealth_volatility**2
        ) / params.discount_rate
        reward[k, okay] = params.time_step * flow[okay]
        borrowing[k, okay] = minimum_borrowing[okay]
        feasible[k, okay] = True
    return reward, borrowing, feasible


def solve_hjb(
    params: ModelParameters | None = None,
    n_liquidity: int = 51,
    n_credit: int = 45,
    n_controls: int = 41,
    max_policy_iterations: int = 80,
    tolerance: float = 1e-9,
) -> HJBSolution:
    """Solve the discounted stationary HJB by discrete policy iteration."""

    params = params or ModelParameters()
    params.validate()
    liquidity_grid = np.geomspace(0.40, 4.00, n_liquidity)
    credit_grid = np.geomspace(0.01, 0.25, n_credit)
    controls = np.linspace(0.0, params.max_asset_sale, n_controls)
    reward, borrowing, feasible = _control_rewards(
        params, liquidity_grid, credit_grid, controls,
    )
    if not feasible.any(axis=0).all():
        raise RuntimeError("The control set is empty at one or more grid states")

    indices, weights = _transition_tensors(
        params, liquidity_grid, credit_grid, controls,
    )
    n_states = n_liquidity * n_credit
    beta = np.exp(-params.discount_rate * params.time_step)
    policy = np.argmax(reward, axis=0)
    value = np.zeros(n_states)
    final_changes = n_states

    for iteration in range(1, max_policy_iterations + 1):
        chosen_idx = indices[policy, np.arange(n_states)]
        chosen_weights = weights[policy, np.arange(n_states)]
        rows = np.repeat(np.arange(n_states), chosen_idx.shape[1])
        transition = csr_matrix(
            (chosen_weights.ravel(), (rows, chosen_idx.ravel())),
            shape=(n_states, n_states),
        )
        chosen_reward = reward[policy, np.arange(n_states)]
        value = spsolve(eye(n_states, format="csr") - beta * transition, chosen_reward)

        continuation = np.empty_like(reward)
        for k in range(n_controls):
            continuation[k] = np.sum(weights[k] * value[indices[k]], axis=1)
        q_value = reward + beta * continuation
        new_policy = np.argmax(q_value, axis=0)
        final_changes = int(np.count_nonzero(new_policy != policy))
        bellman = np.max(q_value, axis=0)
        residual = float(np.max(np.abs(value - bellman)))
        policy = new_policy
        if final_changes == 0 and residual < tolerance:
            break
    else:
        iteration = max_policy_iterations

    selected_sale = controls[policy]
    selected_borrowing = borrowing[policy, np.arange(n_states)]
    shape = (n_liquidity, n_credit)
    return HJBSolution(
        liquidity_grid=liquidity_grid,
        credit_grid=credit_grid,
        value_adjustment=value.reshape(shape),
        optimal_asset_sale=selected_sale.reshape(shape),
        optimal_borrowing=selected_borrowing.reshape(shape),
        feasible=feasible.any(axis=0).reshape(shape),
        iterations=iteration,
        policy_changes=final_changes,
        bellman_residual=residual,
        parameters=params,
    )


def extract_boundary(solution: HJBSolution) -> pd.DataFrame:
    """Interpolate the HJB insolvency boundary for a fixed equity buffer."""

    threshold = np.log(solution.parameters.wealth_buffer_ratio)
    distance = solution.risk_distance
    rows: list[dict[str, float]] = []
    for j, credit in enumerate(solution.credit_grid):
        column = distance[:, j]
        crossings = np.flatnonzero(column >= threshold)
        if len(crossings) == 0:
            boundary = np.nan
        elif crossings[0] == 0:
            boundary = float(solution.liquidity_grid[0])
        else:
            i = int(crossings[0])
            x0, x1 = solution.liquidity_grid[i - 1:i + 1]
            y0, y1 = column[i - 1:i + 1]
            boundary = float(x0 + (threshold - y0) * (x1 - x0) / max(y1 - y0, 1e-12))
        rows.append({
            "credit_state": float(credit),
            "liquidity_boundary": boundary,
            "risk_threshold": float(threshold),
            "wealth_buffer_ratio": solution.parameters.wealth_buffer_ratio,
        })
    return pd.DataFrame(rows)


def _surrogate(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w0, w1, w2, w3, p, q, r, s = params
    return w0 + w1 * x**p + w2 * y**q + w3 * x**r * y**s


def fit_surrogate(solution: HJBSolution) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit and hold out a nonlinear analytical approximation to HJB risk."""

    ll, dd = np.meshgrid(solution.liquidity_grid, solution.credit_grid, indexing="ij")
    target = solution.risk_index
    lambda_c, credit_c = float(np.median(solution.liquidity_grid)), float(np.median(solution.credit_grid))
    x, y = (ll / lambda_c).ravel(), (dd / credit_c).ravel()
    z = target.ravel()
    ii, jj = np.indices(target.shape)
    test = ((ii + 2 * jj) % 5 == 0).ravel()
    train = ~test
    initial = np.array([0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 1.0, 1.0])
    lower = np.array([-1.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
    upper = np.array([1.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0])
    result = least_squares(
        lambda theta: _surrogate(theta, x[train], y[train]) - z[train],
        initial, bounds=(lower, upper), max_nfev=20000,
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    prediction = _surrogate(result.x, x, y)
    error = prediction - z
    threshold = np.log(solution.parameters.wealth_buffer_ratio)
    true_label = solution.risk_distance.ravel() >= threshold
    # Map the CE threshold into the [0,1] index used by the surrogate.
    raw = solution.risk_distance
    normalized_threshold = (threshold - float(raw.min())) / max(float(raw.max() - raw.min()), 1e-12)
    predicted_label = prediction >= normalized_threshold
    test_labels = true_label[test]
    test_scores = prediction[test]
    positive_n = int(test_labels.sum())
    negative_n = int((~test_labels).sum())
    score_ranks = rankdata(test_scores)
    auc = float(
        (score_ranks[test_labels].sum() - positive_n * (positive_n + 1) / 2)
        / max(positive_n * negative_n, 1)
    )
    false_negative_rate = float(
        np.sum(test & true_label & ~predicted_label) / max(np.sum(test & true_label), 1)
    )
    test_rmse = float(np.sqrt(np.mean(error[test] ** 2)))
    test_mae = float(np.mean(np.abs(error[test])))
    test_max = float(np.max(np.abs(error[test])))
    test_r2 = float(1.0 - np.sum(error[test] ** 2) / np.sum((z[test] - z[test].mean()) ** 2))
    names = ["w0", "w1", "w2", "w3", "p", "q", "r", "s"]
    summary: dict[str, Any] = {
        "lambda_c": lambda_c,
        "credit_c": credit_c,
        "parameters": dict(zip(names, [float(v) for v in result.x])),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_max_absolute_error": test_max,
        "test_r_squared": test_r2,
        "test_boundary_auc": auc,
        "test_boundary_accuracy": float(np.mean(predicted_label[test] == true_label[test])),
        "false_negative_rate_at_boundary": false_negative_rate,
        "normalized_boundary_threshold": float(normalized_threshold),
        "optimizer_success": bool(result.success),
    }
    table = pd.DataFrame({
        "liquidity_state": ll.ravel(),
        "credit_state": dd.ravel(),
        "hjb_risk_index": z,
        "surrogate_risk_index": prediction,
        "error": error,
        "partition": np.where(test, "test", "train"),
        "hjb_insolvent": true_label,
        "surrogate_insolvent": predicted_label,
    })
    return summary, table


def map_bank_proxies(metrics: pd.DataFrame, solution: HJBSolution) -> pd.DataFrame:
    """Map public balance-sheet proxies to the model state space transparently.

    These transformations are prototype mappings, not measured LCR or NPL.
    They must be replaced with supervisory cash-flow and loan-performance data
    before the dashboard is used for bank-level regulatory decisions.
    """

    frame = metrics.copy()
    liquidity_ratio = frame["cash_bank_assets_to_deposits"].astype(float)
    exposure_ratio = frame["loan_to_deposit_proxy"].astype(float)
    liquidity_anchor = float(liquidity_ratio.median())
    exposure_anchor = float(exposure_ratio.median())
    frame["liquidity_state_proxy"] = np.clip(
        liquidity_anchor / liquidity_ratio,
        solution.liquidity_grid.min(), solution.liquidity_grid.max(),
    )
    frame["credit_state_proxy"] = np.clip(
        0.05 * exposure_ratio / exposure_anchor,
        solution.credit_grid.min(), solution.credit_grid.max(),
    )
    points = frame[["liquidity_state_proxy", "credit_state_proxy"]].to_numpy()
    outputs = {
        "hjb_risk_distance": solution.risk_distance,
        "hjb_risk_index": solution.risk_index,
        "optimal_asset_sale": solution.optimal_asset_sale,
        "optimal_borrowing": solution.optimal_borrowing,
    }
    for name, values in outputs.items():
        interp = RegularGridInterpolator(
            (solution.liquidity_grid, solution.credit_grid), values,
            bounds_error=False, fill_value=None,
        )
        frame[name] = interp(points)
    threshold = np.log(solution.parameters.wealth_buffer_ratio)
    frame["conditional_boundary_breach"] = frame["hjb_risk_distance"] >= threshold
    frame["assumed_wealth_to_minimum_ratio"] = solution.parameters.wealth_buffer_ratio
    frame["proxy_mapping_note"] = (
        "Public-data prototype: liquidity=median(H/D)/(H/D); "
        "credit=0.05*(L/D)/median(L/D). Not official LCR or NPL."
    )
    keep = [
        "symbol", "jalali_year", "report_id", "is_audited",
        "cash_bank_assets_to_deposits", "loan_to_deposit_proxy", "equity_to_assets",
        "liquidity_state_proxy", "credit_state_proxy", "hjb_risk_distance",
        "hjb_risk_index", "optimal_asset_sale", "optimal_borrowing",
        "conditional_boundary_breach", "assumed_wealth_to_minimum_ratio",
        "proxy_mapping_note",
    ]
    return frame[keep]


def solution_grid_frame(solution: HJBSolution) -> pd.DataFrame:
    ll, dd = np.meshgrid(solution.liquidity_grid, solution.credit_grid, indexing="ij")
    return pd.DataFrame({
        "liquidity_state": ll.ravel(),
        "credit_state": dd.ravel(),
        "value_adjustment_g": solution.value_adjustment.ravel(),
        "risk_distance": solution.risk_distance.ravel(),
        "risk_index": solution.risk_index.ravel(),
        "optimal_asset_sale": solution.optimal_asset_sale.ravel(),
        "optimal_borrowing": solution.optimal_borrowing.ravel(),
        "feasible": solution.feasible.ravel(),
    })


def serializable_parameters(params: ModelParameters) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(params).items()}
