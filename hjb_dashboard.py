"""Pure numerical dashboard helpers: never extrapolate or relabel heuristic scores as HJB PD."""
from __future__ import annotations
import numpy as np
from research.operational_model import interpolate_solution, minimum_feasible_controls


def in_domain(solution, w:float, lam:float, d:float)->bool:
    return (solution.wealth_grid[0] <= w <= solution.wealth_grid[-1] and
            solution.liquidity_grid[0] <= lam <= solution.liquidity_grid[-1] and
            solution.credit_grid[0] <= d <= solution.credit_grid[-1])


def pd_at(solution, w:float, lam:float, d:float):
    if w <= 1 or not minimum_feasible_controls(lam,solution.parameters)['feasible']:
        return 1.0  # model absorbing event, including when outside grid
    if not in_domain(solution,w,lam,d):
        return None
    return float(interpolate_solution(solution,np.array([[w,lam,d]]),'default_probability')[0])


def risk_slice(solution,wealth:float):
    L,D=np.meshgrid(solution.liquidity_grid,solution.credit_grid,indexing='xy')
    p=np.column_stack([np.full(L.size,wealth),L.ravel(),D.ravel()])
    z=interpolate_solution(solution,p,'default_probability').reshape(L.shape)
    return L,D,z


def policy_slice(solution,wealth:float):
    """Nearest wealth grid surface to avoid interpolating categorical policy labels."""
    i=int(np.argmin(abs(solution.wealth_grid-wealth)))
    sale=solution.optimal_asset_sale[i].T
    borrow=solution.optimal_borrowing[i].T
    feas=solution.feasible[i].T
    region=np.where(~feas, 3, np.where(sale>1e-9,2,np.where(borrow>1e-9,1,0)))
    return float(solution.wealth_grid[i]),region,sale,borrow


def boundary_surface(solution,alpha=.5):
    """Discrete generalized w threshold; NaN means target not reached in finite grid."""
    pd=solution.default_probability
    values=np.full((len(solution.credit_grid),len(solution.liquidity_grid)),np.nan)
    for i in range(len(solution.liquidity_grid)):
        for j in range(len(solution.credit_grid)):
            good=np.flatnonzero(pd[:,i,j]<=alpha)
            if good.size:values[j,i]=solution.wealth_grid[good[0]]
    return values


def risk_vs_capital(solution, lam:float,d:float):
    grid=solution.wealth_grid
    vals=[pd_at(solution,float(w),lam,d) for w in grid]
    return grid,np.array([np.nan if p is None else p for p in vals])


def intervention_path(solution,w:float,lam:float,d:float,steps=12,scale=0.25):
    """Comparative accounting intervention along one dimension, not a simulated control path."""
    wealth=np.linspace(w, min(solution.wealth_grid[-1],w+scale),steps)
    return wealth,np.array([pd_at(solution,float(v),lam,d) for v in wealth],dtype=float)


def local_counterfactuals(solution, w: float, lam: float, d: float, size: float = .10):
    """One-factor initial-state what-if (10% by default), using fixed reference HJB.

    None means a proposed scenario is outside the solved numerical domain.
    These are not causal contributions, implementable strategies, or historical forecasts.
    """
    if not (0 < size < 1):
        raise ValueError('size must lie in (0,1)')
    base=pd_at(solution,w,lam,d)
    options=[('Current state',w,lam,d),
             ('More equity (+10%)',w*(1+size),lam,d),
             ('Lower inverse LCR (-10%)',w,lam*(1-size),d),
             ('Lower NPL (-10%)',w,lam,d*(1-size))]
    return [{'label':label,'pd':pd_at(solution,ww,ll,dd),
             'delta_percentage_points':(100*(pd_at(solution,ww,ll,dd)-base) if base is not None and pd_at(solution,ww,ll,dd) is not None else None)}
            for label,ww,ll,dd in options]
