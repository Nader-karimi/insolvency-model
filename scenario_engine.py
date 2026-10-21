"""Self-contained, faithful replica of the existing public illustrative stress mechanism.

This balance-sheet shock calculation is separate from the three-state HJB solver.
It is a scenario illustration, not a bank-default PD estimate or a Basel calculation.
"""
from dataclasses import dataclass, asdict
from math import isfinite


@dataclass(frozen=True)
class Scenario:
    equity: float = 8000.0
    hqla: float = 15000.0
    marketable: float = 20000.0
    loan_book: float = 57000.0
    short_liabilities: float = 25000.0
    long_liabilities: float = 67000.0
    inflows: float = 8000.0
    outflows: float = 14000.0
    npl_ratio: float = 0.05
    nsfr_buffer: float = 0.05
    shock: float = -0.25
    phi: float = 0.00005
    haircut: float = 0.20

    def validate(self):
        for name, value in asdict(self).items():
            if not isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('equity','hqla','marketable','loan_book','short_liabilities',
                     'long_liabilities','inflows','outflows'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} cannot be negative')
        if not 0 <= self.npl_ratio <= 1:
            raise ValueError('npl_ratio must be in [0,1]')
        if not 0 <= self.nsfr_buffer <= .5:
            raise ValueError('nsfr_buffer must be in [0,0.5]')
        if not -0.5 <= self.shock <= 0:
            raise ValueError('shock must be in [-.5,0]')
        if not 0 <= self.haircut < 1 or self.phi < 0:
            raise ValueError('invalid haircut or feedback coefficient')
        if self.inflows <= 0 or self.outflows <= 0:
            raise ValueError('positive scenario inflows and outflows are required')


def stress(s: Scenario) -> dict:
    """Replicate the existing GitHub Streamlit illustration, exposing residuals.

    In the original code ``at`` is desired sales, while ``actual_at`` is executed;
    we report them separately to avoid displaying an infeasible sale as an action.
    """
    s.validate()
    delta_m, delta_j = s.marketable*s.shock, s.loan_book*s.shock
    e1 = s.equity + delta_m + delta_j
    m1 = s.marketable*(1+s.shock)
    margin_call = abs(delta_m)*.5 + abs(delta_j)*.1
    need = s.short_liabilities + margin_call
    available = s.hqla + s.inflows
    gap_initial = need - available
    gap = max(0.0,gap_initial)
    borrowing = min(gap, max(0.0,s.nsfr_buffer*s.long_liabilities))
    gap = max(0.0, gap-borrowing)
    desired_sale = gap/(1-s.haircut)
    sale = min(desired_sale, m1)
    gap = max(0.0,gap - sale*(1-s.haircut))
    funding_cost = borrowing*.05 + sale*s.haircut
    e2 = e1-funding_cost
    final_liquid = max(0.0, available + borrowing + sale*(1-s.haircut)-need)
    feedback = s.phi*sale
    d2 = min(1.0,s.npl_ratio + abs(s.shock)*.4 + feedback)
    lam_initial = s.outflows/s.inflows
    lam_final = (s.outflows + margin_call)/s.inflows
    return {
        'E0': s.equity, 'E1': e1, 'E2': e2, 'C0':s.hqla, 'C2':final_liquid,
        'd0':s.npl_ratio, 'd_final':d2, 'lambda_initial':lam_initial,
        'lambda_final':lam_final, 'margin_call':margin_call,
        'initial_funding_gap':gap_initial, 'unmet_liquidity_gap':gap,
        'borrow': borrowing, 'sale':sale, 'requested_sale':desired_sale,
        'cost':funding_cost, 'feedback':feedback, 'shock_loss':delta_m+delta_j,
        'base_credit_increase':abs(s.shock)*.4, 'liquidity_need':need,
        'available_liquidity':available,
    }
