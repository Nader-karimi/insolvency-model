"""Audit-ready interpretations of a stored, finite-grid HJB policy.

A stored policy is optimized at its grid state, not at arbitrary new input
coordinates. Exact-input constrained controls are feasible projections only;
no new HJB optimization or institutional recommendation is claimed.
"""
from __future__ import annotations

from math import isfinite
import numpy as np
from research.operational_model import minimum_feasible_controls
from hjb_dashboard import in_domain


def control_slacks(lam, params, sale, borrowing):
    """Feasibility margins (negative means constraint violation)."""
    return {
        'liquidity': float(1.0 + borrowing + params.sale_liquidity_yield * sale - lam),
        'stable_funding': float(params.stable_funding_buffer + params.sale_nsfr_relief * sale
                                - params.borrowing_nsfr_weight * borrowing),
        'sale_headroom': float(params.max_asset_sale - sale),
        'borrowing_headroom': float(params.max_borrowing - borrowing),
    }


def feasible_pair_at_state(lam, params, sale):
    """Keep the given sale intensity and choose its smallest liquidity-feasible borrowing.

    Returns no pair if stable funding or capacity would be violated. This is
    an algebraic feasibility check, not an optimization of survival probability.
    """
    sale = float(sale)
    if not isfinite(sale) or not (0 <= sale <= params.max_asset_sale):
        return None
    borrowing = float(max(0.0, lam - 1.0 - params.sale_liquidity_yield * sale))
    margins = control_slacks(lam,params,sale,borrowing)
    if any(x < -1e-9 for x in margins.values()):
        return None
    return {'sale': sale, 'borrowing': borrowing, 'margins': margins}


def policy_diagnostic(solution, w, lam, d):
    """Policy at the nearest three-dimensional grid node, with an exact-input feasible pair.

    The proposed pair keeps the grid-optimal sale (or raises it to the
    mathematical minimum sale if needed), then computes the associated
    minimum borrowing for the user's exact liquidity state. It is *not*
    the exact-state optimal HJB control.
    """
    params=solution.parameters
    if not in_domain(solution,w,lam,d) or w <= 1:
        return None
    minimum=minimum_feasible_controls(lam,params)
    if not minimum['feasible']:
        return None
    axes=(solution.wealth_grid,solution.liquidity_grid,solution.credit_grid)
    ix=tuple(int(np.argmin(np.abs(g-z))) for g,z in zip(axes,(w,lam,d)))
    point=tuple(float(g[i]) for g,i in zip(axes,ix))
    grid_sale=float(solution.optimal_asset_sale[ix])
    grid_borrow=float(solution.optimal_borrowing[ix])
    if not all(map(isfinite,(grid_sale,grid_borrow))) or not bool(solution.feasible[ix]):
        return {'grid_state':point,'grid_policy':None,'candidate':None,'minimum':minimum}
    grid_margins=control_slacks(point[1],params,grid_sale,grid_borrow)
    grid_policy={'sale':grid_sale,'borrowing':grid_borrow,'margins':grid_margins}
    sale=max(grid_sale,float(minimum['asset_sale']))
    candidate=feasible_pair_at_state(lam,params,sale)
    return {'grid_state':point,'grid_policy':grid_policy,'candidate':candidate,'minimum':minimum}


def zero_borrowing_alternative(lam,params):
    """Least sale needed if no borrowing is allowed; None if capacity blocks it."""
    gap=max(0.0,float(lam)-1.0)
    if params.sale_liquidity_yield <= 0 and gap > 0:
        return None
    sale=gap/params.sale_liquidity_yield if params.sale_liquidity_yield>0 else 0.0
    if params.sale_nsfr_relief>0:
        sale=max(sale,-params.stable_funding_buffer/params.sale_nsfr_relief)
    elif params.stable_funding_buffer<0:
        return None
    proposal=feasible_pair_at_state(lam,params,max(sale,0.0))
    return proposal if proposal is not None and proposal['borrowing']<=1e-9 else None
