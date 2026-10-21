"""An independent zero-liquidity/credit-volatility log-wealth benchmark.

For fixed feasible lambda<1 and d>0, controls a=b=0 are optimal when
phi=0, all action costs positive, and lambda/d are deterministic constants.
The survival probability is the reflection-principle closed form for a drifted
Brownian motion stopped at log(w)=0. We compare at *actual grid nodes* to
avoid errors from interpolating the fixed non-diffusing lambda/d coordinates.
"""
from dataclasses import replace
from pathlib import Path
from math import erf, exp, log, sqrt
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operational_model import OperationalParameters, solve_survival_hjb

def Phi(x): return .5*(1+erf(x/sqrt(2)))

def exact_survival(w, lam, d, p):
    x=log(w);T=p.horizon_years;sig=p.wealth_volatility
    drift=p.asset_return-p.liquidity_cost*max(lam-1.,0)-p.credit_loss_severity*d -sig**2/2
    return Phi((x+drift*T)/(sig*sqrt(T))) - exp(-2*drift*x/sig**2)*Phi((-x+drift*T)/(sig*sqrt(T)))

rows=[]
for nw,nt in [(13,12),(21,24),(33,48),(45,96),(65,96),(89,128)]:
    p=replace(OperationalParameters(),time_steps=nt, sigma_liquidity=0.0,
       sigma_credit=0.0,mu_liquidity=0.,mu_credit=0.,fire_sale_credit_feedback=0.)
    sol=solve_survival_hjb(p,n_wealth=nw,n_liquidity=9,n_credit=9,n_controls=3)
    il=int(np.argmin(abs(sol.liquidity_grid-.82)))
    id_=int(np.argmin(abs(sol.credit_grid-.05)))
    l=float(sol.liquidity_grid[il]); d=float(sol.credit_grid[id_]);
    for target in [1.15,1.35,1.65,2.0]:
        w=float(target)
        numeric=float(np.interp(w,sol.wealth_grid,sol.survival_probability[:,il,id_]))
        exact=float(exact_survival(w,l,d,p))
        rows.append({'n_wealth':nw,'time_steps':nt,'w':w,'lambda':l,'d':d,
              'numeric_pd':1-numeric,'exact_pd':1-exact,'absolute_error':abs(numeric-exact)})
    print(nw,nt,[round(a['absolute_error'],5) for a in rows[-4:]],flush=True)
out=ROOT/'audit/analytical_benchmark.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print('written',out)
