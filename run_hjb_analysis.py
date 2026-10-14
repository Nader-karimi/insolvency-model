"""Run the HJB solver, surrogate validation, bank mapping and paper figures."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from hjb_model import (
    ModelParameters,
    extract_boundary,
    fit_surrogate,
    map_bank_proxies,
    serializable_parameters,
    solution_grid_frame,
    solve_hjb,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "model_outputs"
FIGURES = ROOT / "paper_assets" / "hjb"

BANK_LABELS = {
    "وبملت": "Bank Mellat",
    "وتجارت": "Bank Tejarat",
    "وبصادر": "Bank Saderat Iran",
    "وپاسار": "Bank Pasargad",
    "وپارس": "Parsian Bank",
    "ونوین": "Bank Eghtesad Novin",
}


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_figures(solution, surrogate: pd.DataFrame, banks: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    ll, dd = np.meshgrid(solution.liquidity_grid, solution.credit_grid, indexing="ij")
    boundary_level = np.log(solution.parameters.wealth_buffer_ratio)

    plt.figure(figsize=(8.2, 5.8))
    contour = plt.contourf(ll, dd, solution.risk_distance, levels=24, cmap="magma")
    plt.colorbar(contour, label="certainty-equivalent risk distance")
    plt.contour(ll, dd, solution.risk_distance, levels=[boundary_level], colors="cyan", linewidths=2)
    plt.xlabel("liquidity-risk state, lambda")
    plt.ylabel("credit-risk state, d")
    plt.title("Numerical HJB risk surface and insolvency boundary")
    _savefig(FIGURES / "hjb_risk_surface.png")

    plt.figure(figsize=(8.2, 5.8))
    policy = plt.contourf(ll, dd, solution.optimal_asset_sale, levels=20, cmap="viridis")
    plt.colorbar(policy, label="optimal normalized asset-sale rate")
    plt.contour(ll, dd, solution.optimal_borrowing, levels=[0.01, 0.15, 0.30, 0.50], colors="white", linewidths=0.8)
    plt.xlabel("liquidity-risk state, lambda")
    plt.ylabel("credit-risk state, d")
    plt.title("Optimal fire-sale policy; white lines show borrowing")
    _savefig(FIGURES / "optimal_policy_map.png")

    pivot_true = surrogate.pivot(index="liquidity_state", columns="credit_state", values="hjb_risk_index")
    pivot_fit = surrogate.pivot(index="liquidity_state", columns="credit_state", values="surrogate_risk_index")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True, sharey=True)
    for ax, matrix, title in zip(axes, [pivot_true, pivot_fit], ["HJB", "Analytical surrogate"]):
        item = ax.contourf(matrix.index, matrix.columns, matrix.to_numpy().T, levels=20, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("liquidity-risk state")
    axes[0].set_ylabel("credit-risk state")
    fig.colorbar(item, ax=axes.ravel().tolist(), label="normalized risk index", shrink=0.90)
    fig.subplots_adjust(wspace=0.12, right=0.90)
    plt.savefig(FIGURES / "surrogate_contour_comparison.png", dpi=220, bbox_inches="tight")
    plt.close()

    test = surrogate.loc[surrogate["partition"].eq("test")]
    plt.figure(figsize=(6.2, 5.6))
    plt.scatter(test["hjb_risk_index"], test["surrogate_risk_index"], s=13, alpha=0.65)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("HJB risk index")
    plt.ylabel("surrogate risk index")
    plt.title("Held-out validation of the surrogate")
    _savefig(FIGURES / "surrogate_holdout_scatter.png")

    plt.figure(figsize=(9.4, 5.6))
    unknown_symbols = sorted(set(banks["symbol"]) - set(BANK_LABELS))
    if unknown_symbols:
        raise ValueError(f"Missing English bank labels for: {unknown_symbols}")
    for symbol in sorted(BANK_LABELS, key=lambda item: BANK_LABELS[item]):
        group = banks.loc[banks["symbol"].eq(symbol)].sort_values("fiscal_year")
        plt.plot(
            group["fiscal_year"], group["hjb_risk_index"], marker="o",
            linewidth=1.6, label=BANK_LABELS[symbol],
        )
    plt.xticks(sorted(banks["fiscal_year"].astype(int).unique()))
    plt.xlabel("Fiscal year-end (Gregorian)")
    plt.ylabel("prototype HJB risk index")
    plt.title("Six-bank public-data proxy mapped to the HJB surface")
    plt.legend(ncol=3, frameon=False)
    _savefig(FIGURES / "bank_hjb_risk_trends.png")


def _solution_error(reference, candidate) -> dict[str, float]:
    ll, dd = np.meshgrid(reference.liquidity_grid, reference.credit_grid, indexing="ij")
    points = np.column_stack([ll.ravel(), dd.ravel()])
    interpolator = RegularGridInterpolator(
        (candidate.liquidity_grid, candidate.credit_grid), candidate.risk_distance,
        bounds_error=False, fill_value=None,
    )
    error = interpolator(points) - reference.risk_distance.ravel()
    return {
        "rmse_vs_primary": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_vs_primary": float(np.max(np.abs(error))),
    }


def numerical_diagnostics(primary, params: ModelParameters) -> tuple[dict, pd.DataFrame]:
    """Run grid/time-step checks and verify regulatory residuals."""

    coarse = solve_hjb(params, n_liquidity=31, n_credit=27, n_controls=25)
    medium = solve_hjb(params, n_liquidity=41, n_credit=36, n_controls=33)
    half_dt = solve_hjb(
        replace(params, time_step=params.time_step / 2),
        n_liquidity=41, n_credit=36, n_controls=33,
    )
    rows = []
    for label, solution in [("coarse", coarse), ("medium", medium), ("half_time_step", half_dt)]:
        rows.append({
            "run": label,
            "liquidity_points": len(solution.liquidity_grid),
            "credit_points": len(solution.credit_grid),
            "time_step": solution.parameters.time_step,
            "policy_iterations": solution.iterations,
            "bellman_residual": solution.bellman_residual,
            **_solution_error(primary, solution),
        })
    convergence = pd.DataFrame(rows)

    ll, _ = np.meshgrid(primary.liquidity_grid, primary.credit_grid, indexing="ij")
    a, b = primary.optimal_asset_sale, primary.optimal_borrowing
    lcr_margin = (
        params.initial_hqla + b + params.sale_liquidity_yield * a
        - params.outflow_scale * ll
    )
    nsfr_margin = (
        params.stable_funding_buffer + params.sale_nsfr_relief * a
        - params.borrowing_nsfr_weight * b
    )
    d_lambda = np.diff(primary.risk_distance, axis=0)
    d_credit = np.diff(primary.risk_distance, axis=1)
    diagnostics = {
        "minimum_lcr_feasibility_margin": float(lcr_margin.min()),
        "minimum_nsfr_feasibility_margin": float(nsfr_margin.min()),
        "share_positive_asset_sales": float(np.mean(a > 1e-12)),
        "share_positive_borrowing": float(np.mean(b > 1e-12)),
        "liquidity_monotonicity_share": float(np.mean(d_lambda >= -1e-8)),
        "credit_monotonicity_share": float(np.mean(d_credit >= -1e-8)),
        "coarse_rmse_vs_primary": float(convergence.loc[convergence.run.eq("coarse"), "rmse_vs_primary"].iloc[0]),
        "medium_rmse_vs_primary": float(convergence.loc[convergence.run.eq("medium"), "rmse_vs_primary"].iloc[0]),
        "half_time_step_rmse_vs_primary": float(convergence.loc[convergence.run.eq("half_time_step"), "rmse_vs_primary"].iloc[0]),
    }
    return diagnostics, convergence


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    params = ModelParameters()
    solution = solve_hjb(params)
    diagnostics, convergence = numerical_diagnostics(solution, params)
    boundary = extract_boundary(solution)
    surrogate_summary, surrogate = fit_surrogate(solution)
    metrics = pd.read_csv(DATA / "processed" / "bank_year_metrics.csv", encoding="utf-8-sig")
    banks = map_bank_proxies(metrics, solution)
    fiscal_dates = metrics[["symbol", "jalali_year", "fiscal_year_gregorian"]].copy()
    fiscal_dates["fiscal_year"] = pd.to_datetime(
        fiscal_dates["fiscal_year_gregorian"], errors="raise"
    ).dt.year
    banks = banks.merge(
        fiscal_dates[["symbol", "jalali_year", "fiscal_year"]],
        on=["symbol", "jalali_year"], how="left", validate="one_to_one",
    )
    grid = solution_grid_frame(solution)

    grid.to_csv(OUTPUTS / "hjb_grid.csv", index=False, encoding="utf-8-sig")
    grid[["liquidity_state", "credit_state", "optimal_asset_sale", "optimal_borrowing"]].to_csv(
        OUTPUTS / "optimal_policy.csv", index=False, encoding="utf-8-sig",
    )
    boundary.to_csv(OUTPUTS / "hjb_boundary.csv", index=False, encoding="utf-8-sig")
    surrogate.to_csv(OUTPUTS / "surrogate_validation.csv", index=False, encoding="utf-8-sig")
    banks.to_csv(DATA / "processed" / "bank_year_hjb_mapping.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(OUTPUTS / "numerical_convergence.csv", index=False, encoding="utf-8-sig")

    summary = {
        "method": "stationary discounted semi-Lagrangian HJB with discrete policy iteration",
        "grid_shape": [len(solution.liquidity_grid), len(solution.credit_grid)],
        "grid_points": int(len(solution.liquidity_grid) * len(solution.credit_grid)),
        "control_points": 41,
        "policy_iterations": solution.iterations,
        "final_policy_changes": solution.policy_changes,
        "bellman_residual_sup_norm": solution.bellman_residual,
        "all_grid_states_feasible": bool(solution.feasible.all()),
        "risk_distance_min": float(solution.risk_distance.min()),
        "risk_distance_max": float(solution.risk_distance.max()),
        "boundary_grid_rows": int(boundary["liquidity_boundary"].notna().sum()),
        "bank_observations": int(len(banks)),
        "conditional_bank_boundary_breaches": int(banks["conditional_boundary_breach"].sum()),
        "parameters": serializable_parameters(params),
        "surrogate": surrogate_summary,
        "numerical_diagnostics": diagnostics,
        "interpretation_warning": (
            "The six-bank mapping uses public balance-sheet proxies and is not an official "
            "LCR, NSFR, NPL, probability-of-default or supervisory classification."
        ),
    }
    (OUTPUTS / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    make_figures(solution, surrogate, banks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
