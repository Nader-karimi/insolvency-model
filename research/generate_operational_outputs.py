"""Generate the v4 three-state HJB outputs, validation tables, and paper figures."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import rankdata

from operational_model import (
    BankInputs,
    OperationalParameters,
    SurvivalSolution,
    _trilinear,
    evaluate_bank,
    interpolate_solution,
    save_solution,
    solve_survival_hjb,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_outputs_v4"
FIG = ROOT / "paper_assets" / "v4"
DATA = ROOT / "data"

NAVY = "#0B1F3A"
BLUE = "#2F6BFF"
CYAN = "#17B6C8"
GREEN = "#19A974"
AMBER = "#F4A62A"
CORAL = "#EF5B5B"
PURPLE = "#7457D9"
LIGHT = "#F4F7FB"
GRID = "#D9E2EF"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#9AA9BD",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": GRID,
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def savefig(name: str, fig=None) -> None:
    fig = fig or plt.gcf()
    fig.savefig(FIG / name, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def solution_frame(solution: SurvivalSolution) -> pd.DataFrame:
    w, l, d = np.meshgrid(
        solution.wealth_grid, solution.liquidity_grid, solution.credit_grid,
        indexing="ij",
    )
    return pd.DataFrame({
        "wealth_to_minimum": w.ravel(),
        "liquidity_state_inverse_lcr": l.ravel(),
        "credit_state_npl_ratio": d.ravel(),
        "survival_probability": solution.survival_probability.ravel(),
        "default_probability": solution.default_probability.ravel(),
        "optimal_asset_sale_to_hqla": solution.optimal_asset_sale.ravel(),
        "optimal_borrowing_to_hqla": solution.optimal_borrowing.ravel(),
        "control_set_feasible": solution.feasible.ravel(),
    })


def boundary_surface(solution: SurvivalSolution, threshold: float) -> pd.DataFrame:
    rows = []
    pd_cube = solution.default_probability
    for j, liquidity in enumerate(solution.liquidity_grid):
        for k, credit in enumerate(solution.credit_grid):
            column = pd_cube[:, j, k]
            safe = np.flatnonzero(column <= threshold)
            if len(safe) == 0:
                boundary = np.nan
            elif safe[0] == 0:
                boundary = float(solution.wealth_grid[0])
            else:
                i = int(safe[0])
                p0, p1 = column[i - 1], column[i]
                w0, w1 = solution.wealth_grid[i - 1:i + 1]
                boundary = float(w0 + (threshold - p0) * (w1 - w0) / (p1 - p0))
            rows.append({
                "liquidity_state": float(liquidity),
                "credit_state": float(credit),
                "wealth_boundary": boundary,
                "default_probability_threshold": threshold,
            })
    return pd.DataFrame(rows)


def _poly_features(w: np.ndarray, l: np.ndarray, d: np.ndarray) -> np.ndarray:
    x, y, z = np.log(w), np.log(l), np.log(d / 0.05)
    return np.column_stack([
        np.ones_like(x), x, y, z,
        x**2, y**2, z**2, x*y, x*z, y*z,
        x**3, y**3, z**3,
        x**2*y, x**2*z, y**2*x, y**2*z, z**2*x, z**2*y, x*y*z,
    ])


def fit_operational_surrogate(solution: SurvivalSolution) -> tuple[dict, pd.DataFrame, np.ndarray]:
    frame = solution_frame(solution)
    w = frame["wealth_to_minimum"].to_numpy()
    l = frame["liquidity_state_inverse_lcr"].to_numpy()
    d = frame["credit_state_npl_ratio"].to_numpy()
    target = frame["default_probability"].to_numpy()
    features = _poly_features(w, l, d)
    iw, il, id_ = np.indices(solution.default_probability.shape)
    test = ((3 * iw + 2 * il + id_) % 5 == 0).ravel()
    train = ~test

    def logistic(theta, X):
        score = np.clip(X @ theta, -30, 30)
        return 1.0 / (1.0 + np.exp(-score))

    result = least_squares(
        lambda theta: logistic(theta, features[train]) - target[train],
        np.zeros(features.shape[1]), max_nfev=8000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    prediction = logistic(result.x, features)
    error = prediction - target
    labels = target >= solution.parameters.insolvency_region_pd
    test_labels = labels[test]
    ranks = rankdata(prediction[test])
    n_pos, n_neg = int(test_labels.sum()), int((~test_labels).sum())
    auc = float((ranks[test_labels].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))
    pred_labels = prediction >= solution.parameters.insolvency_region_pd
    false_negative = float(
        np.sum(test & labels & ~pred_labels) / max(np.sum(test & labels), 1)
    )
    metrics = {
        "feature_count": int(features.shape[1]),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "test_rmse": float(np.sqrt(np.mean(error[test] ** 2))),
        "test_mae": float(np.mean(np.abs(error[test]))),
        "test_max_absolute_error": float(np.max(np.abs(error[test]))),
        "test_r_squared": float(1 - np.sum(error[test] ** 2) / np.sum((target[test] - target[test].mean()) ** 2)),
        "test_boundary_auc": auc,
        "test_boundary_accuracy": float(np.mean(pred_labels[test] == labels[test])),
        "test_false_negative_rate": false_negative,
        "optimizer_success": bool(result.success),
        "coefficients": [float(v) for v in result.x],
    }
    validation = frame[[
        "wealth_to_minimum", "liquidity_state_inverse_lcr", "credit_state_npl_ratio",
    ]].copy()
    validation["hjb_default_probability"] = target
    validation["surrogate_default_probability"] = prediction
    validation["error"] = error
    validation["partition"] = np.where(test, "test", "train")
    validation["hjb_insolvency_region"] = labels
    validation["surrogate_insolvency_region"] = pred_labels
    return metrics, validation, result.x


def compare_solution(reference: SurvivalSolution, candidate: SurvivalSolution) -> dict[str, float]:
    w, l, d = np.meshgrid(
        reference.wealth_grid, reference.liquidity_grid, reference.credit_grid,
        indexing="ij",
    )
    points = np.column_stack([w.ravel(), l.ravel(), d.ravel()])
    candidate_pd = interpolate_solution(candidate, points, "default_probability")
    error = candidate_pd - reference.default_probability.ravel()
    return {
        "rmse_vs_reference": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error_vs_reference": float(np.max(np.abs(error))),
    }


def run_numerical_diagnostics(primary: SurvivalSolution, params: OperationalParameters) -> tuple[dict, pd.DataFrame]:
    refined_time = replace(params, time_steps=60)
    reference = solve_survival_hjb(refined_time, 29, 33, 25, 19)
    runs = [
        ("coarse", solve_survival_hjb(refined_time, 17, 21, 17, 13)),
        ("medium", solve_survival_hjb(refined_time, 21, 25, 19, 15)),
        ("production", primary),
        ("time_refined", solve_survival_hjb(refined_time, 25, 29, 23, 17)),
        ("reference", reference),
    ]
    rows = []
    for label, sol in runs:
        rows.append({
            "run": label,
            "wealth_points": len(sol.wealth_grid),
            "liquidity_points": len(sol.liquidity_grid),
            "credit_points": len(sol.credit_grid),
            "time_steps": sol.parameters.time_steps,
            "state_points": int(np.prod(sol.default_probability.shape)),
            **compare_solution(reference, sol),
        })
    table = pd.DataFrame(rows)
    pd_cube = primary.default_probability
    diagnostics = {
        "probability_min": float(pd_cube.min()),
        "probability_max": float(pd_cube.max()),
        "wealth_monotonicity_share": float(np.mean(np.diff(pd_cube, axis=0) <= 1e-10)),
        "liquidity_monotonicity_share": float(np.mean(np.diff(pd_cube, axis=1) >= -1e-10)),
        "credit_monotonicity_share": float(np.mean(np.diff(pd_cube, axis=2) >= -1e-10)),
        "feasible_state_share": float(primary.feasible.mean()),
        "probability_bounds_pass": bool((pd_cube >= -1e-12).all() and (pd_cube <= 1 + 1e-12).all()),
    }
    return diagnostics, table


def sample_banks() -> tuple[BankInputs, BankInputs]:
    baseline = BankInputs(
        name="Illustrative bank - baseline",
        total_assets=1_000_000,
        equity=80_000,
        minimum_required_equity=60_000,
        hqla=120_000,
        net_cash_outflow_30d=115_000,
        available_stable_funding=748_000,
        required_stable_funding=680_000,
        gross_loans=600_000,
        nonperforming_loans=60_000,
        max_additional_borrowing=84_000,
        max_sellable_assets=120_000,
    )
    adverse = BankInputs(
        name="Illustrative bank - adverse",
        total_assets=920_000,
        equity=66_000,
        minimum_required_equity=60_000,
        hqla=92_000,
        net_cash_outflow_30d=155_000,
        available_stable_funding=640_000,
        required_stable_funding=680_000,
        gross_loans=600_000,
        nonperforming_loans=108_000,
        max_additional_borrowing=64_000,
        max_sellable_assets=92_000,
    )
    return baseline, adverse


def make_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    boxes = [
        (0.4, 4.7, 2.5, 1.25, "BANK INPUTS", "Equity, HQLA, NCOF\nASF/RSF, loans, NPL", "#E8F0FF"),
        (3.35, 4.7, 2.5, 1.25, "NORMALIZATION", "w = E/Emin\nlambda = NCOF/HQLA; d = NPL/L", "#E6F8FA"),
        (6.3, 4.7, 2.5, 1.25, "3D SURVIVAL HJB", "Absorbing boundary at w=1\nLCR/NSFR feasible controls", "#FFF3D9"),
        (9.25, 4.7, 2.3, 1.25, "VALIDATION", "Grid/time refinement\nmonotonicity and bounds", "#F1EAFE"),
        (2.0, 1.6, 2.8, 1.35, "RISK DIAGNOSIS", "PD and survival horizon\nsafe / warning / model region", "#EAF8F1"),
        (5.05, 1.6, 2.8, 1.35, "OPTIMAL POLICY", "Borrowing and asset sales\nregulatory feasibility", "#FFF0E9"),
        (8.1, 1.6, 2.8, 1.35, "EXIT PLAN", "Capital, HQLA, NPL reduction\nminimum-effort combination", "#FCEAEC"),
    ]
    for x, y, w, h, title, body, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                               linewidth=1.3, edgecolor=NAVY, facecolor=color)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h*.68, title, ha="center", va="center", color=NAVY, weight="bold", fontsize=10.5)
        ax.text(x+w/2, y+h*.34, body, ha="center", va="center", color="#34445C", fontsize=8.5, linespacing=1.35)
    for x1, y1, x2, y2 in [
        (2.9,5.32,3.35,5.32),(5.85,5.32,6.3,5.32),(8.8,5.32,9.25,5.32),
        (7.55,4.7,3.4,2.95),(7.55,4.7,6.45,2.95),(7.55,4.7,9.5,2.95),
    ]:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,
                                     linewidth=1.3,color="#52657E",connectionstyle="arc3,rad=0"))
    ax.text(6, 6.6, "Reproducible bank-insolvency monitoring architecture", ha="center",
            fontsize=17, weight="bold", color=NAVY)
    ax.text(6, 6.22, "Mathematical model, numerical verification and bank-facing remediation in one auditable workflow",
            ha="center", fontsize=10, color="#52657E")
    savefig("01_model_workflow.png", fig)


def plot_risk_slices(solution: SurvivalSolution) -> None:
    wealth_levels = [1.15, 1.50, 2.00]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharex=True, sharey=True)
    L, D = np.meshgrid(solution.liquidity_grid, solution.credit_grid, indexing="ij")
    last = None
    for ax, wealth in zip(axes, wealth_levels):
        points = np.column_stack([
            np.full(L.size, wealth), L.ravel(), D.ravel(),
        ])
        pd_slice = interpolate_solution(solution, points, "default_probability").reshape(L.shape)
        last = ax.contourf(L, D, pd_slice, levels=np.linspace(0,1,21), cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.contour(L, D, pd_slice, levels=[.2,.5], colors=[AMBER,CORAL], linewidths=[1.5,2.2])
        ax.axvline(1.0, color=NAVY, ls="--", lw=1, alpha=.8)
        ax.set_title(f"Equity buffer  w = {wealth:.2f}")
        ax.set_xlabel("Liquidity state  λ = NCOF / HQLA")
        ax.grid(False)
    axes[0].set_ylabel("Credit state  d = NPL / gross loans")
    cbar = fig.colorbar(last, ax=axes, fraction=.025, pad=.025)
    cbar.set_label("One-year model default probability")
    fig.suptitle("Three-state HJB risk map: capital changes the liquidity-credit boundary", color=NAVY, fontsize=15, weight="bold")
    fig.subplots_adjust(top=.82, wspace=.08)
    savefig("02_hjb_risk_slices.png", fig)


def plot_boundary_surface(solution: SurvivalSolution, boundary: pd.DataFrame) -> None:
    matrix = boundary.pivot(index="liquidity_state", columns="credit_state", values="wealth_boundary")
    L, D = np.meshgrid(matrix.index.to_numpy(), matrix.columns.to_numpy(), indexing="ij")
    W = matrix.to_numpy()
    fig = plt.figure(figsize=(10.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(L, D, W, cmap="viridis", edgecolor="none", alpha=.92, antialiased=True)
    ax.plot_wireframe(L, D, W, rstride=4, cstride=4, color="white", linewidth=.35, alpha=.55)
    ax.set_xlabel("Liquidity state λ")
    ax.set_ylabel("Credit state d")
    ax.set_zlabel("Required equity buffer W / Wmin")
    ax.set_title("Model-implied 50% default-probability boundary", color=NAVY, pad=18)
    ax.view_init(elev=27, azim=-128)
    fig.colorbar(surface, ax=ax, shrink=.62, pad=.10, label="Boundary wealth ratio")
    savefig("03_insolvency_boundary_3d.png", fig)


def plot_policy_map(solution: SurvivalSolution, wealth: float = 1.35) -> None:
    L, D = np.meshgrid(solution.liquidity_grid, solution.credit_grid, indexing="ij")
    points = np.column_stack([np.full(L.size, wealth), L.ravel(), D.ravel()])
    pd_slice = interpolate_solution(solution, points, "default_probability").reshape(L.shape)
    sale = _trilinear((solution.wealth_grid, solution.liquidity_grid, solution.credit_grid),
                      np.nan_to_num(solution.optimal_asset_sale), points, False).reshape(L.shape)
    borrow = _trilinear((solution.wealth_grid, solution.liquidity_grid, solution.credit_grid),
                        np.nan_to_num(solution.optimal_borrowing), points, False).reshape(L.shape)
    feasible = np.array([
        solution.feasible[:, np.argmin(abs(solution.liquidity_grid-l)), :].any()
        for l in L.ravel()
    ]).reshape(L.shape)
    regime = np.zeros(L.shape, dtype=int)
    regime[borrow > .01] = 1
    regime[sale > .01] = 2
    regime[~feasible] = 3
    cmap = ListedColormap([GREEN, BLUE, AMBER, "#4D5666"])
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    mesh = ax.pcolormesh(L, D, regime, shading="auto", cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4), alpha=.88)
    ax.contour(L, D, pd_slice, levels=[.2,.5], colors=["white",CORAL], linewidths=[1.6,2.2])
    ax.axvline(1, color=NAVY, ls="--", lw=1.2)
    ax.set_xlabel("Liquidity state λ = inverse LCR")
    ax.set_ylabel("Credit state d")
    ax.set_title(f"Optimal regulatory-control regimes at w = {wealth:.2f}", color=NAVY)
    handles = [plt.Line2D([0],[0],marker='s',ls='',color=c,markersize=10,label=l) for c,l in zip(
        [GREEN,BLUE,AMBER,"#4D5666"],["No active action","Borrowing","Asset sale / mixed","Infeasible control set"])]
    handles += [plt.Line2D([0],[0],color="white",lw=2,label="PD = 20%"),plt.Line2D([0],[0],color=CORAL,lw=2,label="PD = 50%")]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    ax.grid(False)
    savefig("04_optimal_policy_regimes.png", fig)


def deterministic_trajectory(solution: SurvivalSolution) -> pd.DataFrame:
    dt = 1/12
    rows = []
    states = {
        "managed": np.array([1.80, .82, .055], dtype=float),
        "remediated": np.array([1.80, .82, .055], dtype=float),
    }
    for month in range(13):
        for label, state in states.items():
            pd_now = float(interpolate_solution(solution, state[None,:], "default_probability")[0])
            sale = float(_trilinear((solution.wealth_grid,solution.liquidity_grid,solution.credit_grid),
                                    np.nan_to_num(solution.optimal_asset_sale),state[None,:],False)[0])
            borrow = float(_trilinear((solution.wealth_grid,solution.liquidity_grid,solution.credit_grid),
                                      np.nan_to_num(solution.optimal_borrowing),state[None,:],False)[0])
            rows.append({"month":month,"path":label,"wealth":state[0],"liquidity":state[1],"credit":state[2],"pd":pd_now,"sale":sale,"borrowing":borrow})
        if month == 12:
            break
        for label, state in states.items():
            w,l,d = state
            if month == 3:
                l *= 1.45
                d += .012
            if month == 7:
                d += .065
                l *= 1.10
            if label == "remediated" and month == 8:
                w += .38
                l *= .78
                d *= .82
            point=np.array([w,l,d])
            sale=float(_trilinear((solution.wealth_grid,solution.liquidity_grid,solution.credit_grid),np.nan_to_num(solution.optimal_asset_sale),point[None,:],False)[0])
            borrow=float(_trilinear((solution.wealth_grid,solution.liquidity_grid,solution.credit_grid),np.nan_to_num(solution.optimal_borrowing),point[None,:],False)[0])
            p=solution.parameters
            drift=p.asset_return-p.liquidity_cost*max(l-1,0)-p.credit_loss_severity*d-p.borrowing_cost*borrow-p.sale_discount*sale-.5*p.market_illiquidity*sale**2
            w=max(1.001,w*np.exp((drift-.5*p.wealth_volatility**2)*dt))
            l=l*np.exp((p.mu_liquidity-.5*p.sigma_liquidity**2)*dt)
            d=d*np.exp((p.mu_credit+p.fire_sale_credit_feedback*sale-.5*p.sigma_credit**2)*dt)
            states[label]=np.array([w,l,d])
    return pd.DataFrame(rows)


# Figures 6 and 7 use dedicated scripts driven by the saved 3D HJB solution.

def plot_surrogate(validation: pd.DataFrame, metrics: dict) -> None:
    test=validation[validation.partition.eq("test")]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    axes[0].scatter(test.hjb_default_probability,test.surrogate_default_probability,s=8,alpha=.28,color=BLUE,rasterized=True)
    axes[0].plot([0,1],[0,1],"--",color=NAVY,lw=1.2)
    axes[0].set_xlabel("HJB default probability"); axes[0].set_ylabel("Analytical surrogate")
    axes[0].set_title("Held-out prediction")
    bins=np.linspace(0,1,11); ids=np.digitize(test.surrogate_default_probability,bins)-1
    x=[];y=[];n=[]
    for i in range(10):
        mask=ids==i
        if mask.any():
            x.append(test.loc[mask,"surrogate_default_probability"].mean()); y.append(test.loc[mask,"hjb_default_probability"].mean()); n.append(mask.sum())
    axes[1].plot([0,1],[0,1],"--",color=NAVY,lw=1.2)
    axes[1].plot(x,y,"o-",color=CORAL,lw=2.2,ms=6)
    for xx,yy,nn in zip(x,y,n): axes[1].annotate(str(nn),(xx,yy),xytext=(4,5),textcoords="offset points",fontsize=7,color="#52657E")
    axes[1].set_xlabel("Mean surrogate probability"); axes[1].set_ylabel("Mean HJB probability")
    axes[1].set_title("Calibration by probability decile")
    fig.suptitle(f"Operational surrogate validation — test RMSE {metrics['test_rmse']:.3f}, AUC {metrics['test_boundary_auc']:.3f}",color=NAVY,fontsize=14,weight="bold")
    fig.subplots_adjust(top=.83,wspace=.25)
    savefig("07_surrogate_validation.png",fig)


def plot_convergence(convergence: pd.DataFrame) -> None:
    spatial=convergence[convergence.run.isin(["coarse","medium","time_refined","reference"])]
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4))
    plotted=spatial.loc[spatial.rmse_vs_reference>0]
    axes[0].plot(plotted.state_points,plotted.rmse_vs_reference,"o-",color=BLUE,lw=2.3,ms=7)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("State-grid points"); axes[0].set_ylabel("RMSE versus refined reference")
    axes[0].set_title("Spatial refinement")
    for _,r in spatial.loc[spatial.rmse_vs_reference>0].iterrows(): axes[0].annotate(r.run,(r.state_points,r.rmse_vs_reference),xytext=(6,6),textcoords="offset points")
    palette=[BLUE,CYAN,CORAL,PURPLE,"#2FAE82"]
    axes[1].bar(convergence.run,convergence.max_absolute_error_vs_reference,color=palette[:len(convergence)])
    axes[1].set_ylabel("Maximum absolute error")
    axes[1].set_title("Worst-case discretization difference")
    axes[1].tick_params(axis="x",rotation=15)
    fig.suptitle("Numerical verification of the three-state HJB",color=NAVY,fontsize=14,weight="bold")
    fig.subplots_adjust(top=.82,wspace=.25)
    savefig("08_numerical_convergence.png",fig)


def sensitivity_analysis(params: OperationalParameters, sample: BankInputs) -> pd.DataFrame:
    rows=[]
    variants={
        "Correlation": [("Low",replace(params,liquidity_credit_correlation=0.0)),("Base",params),("High",replace(params,liquidity_credit_correlation=.70))],
        "Liquidity volatility": [("Low",replace(params,sigma_liquidity=.20)),("Base",params),("High",replace(params,sigma_liquidity=.40))],
        "Fire-sale feedback": [("Low",replace(params,fire_sale_credit_feedback=0.0)),("Base",params),("High",replace(params,fire_sale_credit_feedback=.20))],
    }
    for family,items in variants.items():
        for level,p in items:
            sol=solve_survival_hjb(p,17,21,17,13)
            assessment=evaluate_bank(sample,sol)
            b=boundary_surface(sol,.50)
            rows.append({
                "parameter_family":family,"level":level,
                "sample_default_probability":assessment["default_probability"],
                "mean_boundary_wealth":float(b.wealth_boundary.mean()),
                "max_boundary_wealth":float(b.wealth_boundary.max()),
            })
    return pd.DataFrame(rows)


def plot_sensitivity(table: pd.DataFrame) -> None:
    families=table.parameter_family.unique()
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.2),sharey=True)
    colors=[CYAN,BLUE,CORAL]
    for ax,family in zip(axes,families):
        g=table[table.parameter_family.eq(family)]
        ax.plot(g.level,100*g.sample_default_probability,"o-",color=BLUE,lw=2.3,ms=7,label="Example-bank PD")
        ax2=ax.twinx(); ax2.plot(g.level,g.mean_boundary_wealth,"s--",color=CORAL,lw=1.8,ms=6,label="Mean boundary")
        ax.set_title(family); ax.set_xlabel("Parameter level")
        ax.grid(axis="y")
        if ax is axes[0]: ax.set_ylabel("One-year default probability (%)")
        if ax is axes[-1]: ax2.set_ylabel("Mean required equity buffer")
        else: ax2.set_yticklabels([])
    fig.suptitle("Sensitivity of risk and the insolvency boundary",color=NAVY,fontsize=14,weight="bold")
    fig.subplots_adjust(top=.80,wspace=.25)
    savefig("09_parameter_sensitivity.png",fig)


def plot_exit_actions(assessment: dict) -> None:
    actions=assessment["exit_actions_to_target_pd"]
    values=[
        actions["capital_only"]["capital_injection"],
        actions["liquidity_only"]["hqla_increase"],
        actions["credit_only"]["npl_reduction"],
    ]
    labels=["Capital injection","HQLA increase","NPL reduction"]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    plotted=[0 if v is None else v for v in values]
    bars=axes[0].barh(labels,plotted,color=[BLUE,CYAN,CORAL])
    axes[0].invert_yaxis(); axes[0].set_xlabel("Million rials")
    axes[0].set_title("Single-action requirement to reach target PD")
    for bar,v in zip(bars,values):
        label="Not sufficient alone" if v is None else f"{v:,.0f}"
        axes[0].text(bar.get_width()+max(plotted+[1])*.02,bar.get_y()+bar.get_height()/2,label,va="center",fontsize=8.5)
    combo=actions["combined_minimum_effort"] or {"capital_injection":0,"hqla_increase":0,"npl_reduction":0}
    combo_vals=[combo["capital_injection"],combo["hqla_increase"],combo["npl_reduction"]]
    axes[1].bar(["Capital","HQLA","NPL cure"],combo_vals,color=[BLUE,CYAN,CORAL])
    axes[1].set_ylabel("Million rials"); axes[1].set_title("Minimum normalized-effort combination")
    for i,v in enumerate(combo_vals): axes[1].text(i,v+max(combo_vals+[1])*.02,f"{v:,.0f}",ha="center",fontsize=8.5)
    fig.suptitle("Actionable exit plan generated from the monotone HJB surface",color=NAVY,fontsize=14,weight="bold")
    fig.subplots_adjust(top=.82,wspace=.32)
    savefig("10_exit_action_plan.png",fig)


def plot_public_bank_panel() -> None:
    """Descriptive accounting proxy only; no HJB-derived bank PD or ranking."""
    frame = pd.read_csv(DATA / "processed" / "bank_year_metrics.csv", encoding="utf-8-sig")
    key = "cash_bank_assets_to_deposits"
    # Independent recomputation: stored ratio and source-record fields must agree.
    # Keep missing years as gaps instead of silently correcting dubious raw records.
    ratio = frame["cash_and_bank_assets"] / frame["customer_deposits"]
    valid = (frame["customer_deposits"] > 0) & np.isfinite(ratio) & \
        np.isclose(frame[key], ratio, atol=1e-9, rtol=1e-9)
    n_dropped = int((~valid).sum())
    frame.loc[~valid, key] = np.nan
    bank_labels = {"وبصادر": "Saderat Iran", "وبملت": "Mellat", "وتجارت": "Tejarat",
                   "ونوین": "Eghtesad Novin", "وپارس": "Parsian", "وپاسار": "Pasargad"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for symbol, group in frame.groupby("symbol", sort=False):
        group = group.sort_values("jalali_year")
        axes[0].plot(group.jalali_year, group[key], "o-", lw=1.8, ms=4,
                     label=bank_labels.get(symbol, symbol))
    axes[0].set_xlabel("Jalali fiscal year")
    axes[0].set_ylabel("Cash and bank assets / customer deposits")
    axes[0].set_title("Public accounting proxy by bank and year")
    axes[0].legend(ncol=3, fontsize=8)
    latest = frame.loc[frame.jalali_year.eq(frame.jalali_year.max()), ["symbol", key]].copy()
    latest["english_name"] = latest.symbol.map(bank_labels)
    latest = latest.sort_values("english_name")
    axes[1].barh(latest.english_name, latest[key], color="#536B8A")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Same accounting proxy (not LCR)")
    axes[1].set_title("Latest available statements, alphabetical order")
    fig.suptitle(f"Public accounting proxy (ratio-consistent rows only; {n_dropped} omitted) | not regulatory LCR or PD",
                 color=NAVY, fontsize=14, weight="bold")
    fig.subplots_adjust(top=.82, wspace=.35)
    savefig("11_public_six_bank_panel.png", fig)

def main() -> None:
    setup_style(); OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    params=OperationalParameters(stable_funding_buffer=.10,max_borrowing=.70,max_asset_sale=1.00)
    primary=solve_survival_hjb(params,25,29,23,17)
    save_solution(primary,OUT/"operational_survival_solution.npz")
    grid=solution_frame(primary); grid.to_csv(OUT/"operational_survival_grid.csv",index=False,encoding="utf-8-sig")
    b50=boundary_surface(primary,.50); b20=boundary_surface(primary,.20)
    boundaries=pd.concat([b50,b20],ignore_index=True); boundaries.to_csv(OUT/"operational_boundaries.csv",index=False,encoding="utf-8-sig")
    surrogate_metrics,surrogate_validation,coefficients=fit_operational_surrogate(primary)
    surrogate_validation.to_csv(OUT/"operational_surrogate_validation.csv",index=False,encoding="utf-8-sig")
    diagnostics,convergence=run_numerical_diagnostics(primary,params)
    convergence.to_csv(OUT/"operational_convergence.csv",index=False,encoding="utf-8-sig")
    baseline,adverse=sample_banks()
    baseline_assessment=evaluate_bank(baseline,primary)
    adverse_assessment=evaluate_bank(adverse,primary)
    (OUT/"sample_bank_inputs.csv").write_text(pd.DataFrame([asdict(baseline),asdict(adverse)]).to_csv(index=False),encoding="utf-8")
    (OUT/"sample_bank_assessments.json").write_text(json.dumps({"baseline":baseline_assessment,"adverse":adverse_assessment},indent=2),encoding="utf-8")
    trajectory=deterministic_trajectory(primary); trajectory.to_csv(OUT/"stress_trajectory.csv",index=False,encoding="utf-8-sig")
    sensitivity=sensitivity_analysis(params,baseline); sensitivity.to_csv(OUT/"operational_sensitivity.csv",index=False,encoding="utf-8-sig")
    summary={
        "model":"finite-horizon three-state survival HJB with absorbing minimum-equity boundary",
        "state_definition":{"w":"equity/minimum required equity","lambda":"30-day NCOF/HQLA (inverse LCR)","d":"NPL/gross loans"},
        "grid_shape":list(primary.default_probability.shape),
        "grid_points":int(primary.default_probability.size),
        "controls":17,
        "parameters":asdict(params),
        "numerical_diagnostics":diagnostics,
        "surrogate":surrogate_metrics,
        "baseline_assessment":baseline_assessment,
        "adverse_assessment":adverse_assessment,
        "interpretation_warning":"Research model output, not a legal insolvency declaration or supervisory rating.",
    }
    (OUT/"operational_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    make_workflow_figure(); plot_risk_slices(primary); plot_boundary_surface(primary,b50)
    plot_policy_map(primary)
    plot_surrogate(surrogate_validation,surrogate_metrics); plot_convergence(convergence)
    plot_sensitivity(sensitivity); plot_exit_actions(baseline_assessment); plot_public_bank_panel()
    from reproduce_figure6 import build_figure as build_figure6
    from reproduce_figure7 import build_figure as build_figure7
    build_figure6(); build_figure7()
    from reproduce_dashboard_default import generate as build_dashboard_default_figure
    build_dashboard_default_figure()
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
