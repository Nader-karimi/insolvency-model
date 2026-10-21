"""True domain-only test: logarithmic grid spacing and all inner grid nodes are held fixed.

Append geometric nodes at nonabsorbing outer boundaries. Separately append to
both lower/upper liquidity and credit boundaries. The original interior nodes
then align exactly and allow direct, interpolation-free comparison. A critical
absorbing equity face at wealth=1 is never moved.
"""
from dataclasses import replace
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operational_model import OperationalParameters,solve_survival_hjb
p=replace(OperationalParameters(),time_steps=24)
bounds=((1,4),(.4,3.5),(.01,.35));shape=(21,25,19);k=4
ratios=[(hi/lo)**(1/(n-1)) for (lo,hi),n in zip(bounds,shape)]
def upper_domain(k):
    return tuple((lo,hi*r**k) for (lo,hi),r in zip(bounds,ratios))
def both_domain(k):
    # We cannot move the absorbing wealth lower edge, unlike the other two faces.
    return ((1,bounds[0][1]*ratios[0]**k),
            (bounds[1][0]/ratios[1]**k,bounds[1][1]*ratios[1]**k),
            (bounds[2][0]/ratios[2]**k,bounds[2][1]*ratios[2]**k))
sol0=solve_survival_hjb(p,*shape,17,domain=bounds)
rows=[]
for name,domain,ns,offset in [
 ('matched_upper',upper_domain(k),tuple(n+k for n in shape),(0,0,0)),
 ('matched_both',both_domain(k),(shape[0]+k,shape[1]+2*k,shape[2]+2*k),(0,k,k))]:
 sol=solve_survival_hjb(p,*ns,17,domain=domain)
 a,b,c=offset
 assert np.allclose(sol.wealth_grid[:shape[0]],sol0.wealth_grid,atol=1e-12)
 assert np.allclose(sol.liquidity_grid[b:b+shape[1]],sol0.liquidity_grid,atol=1e-12)
 assert np.allclose(sol.credit_grid[c:c+shape[2]],sol0.credit_grid,atol=1e-12)
 arr=sol.default_probability[:shape[0],b:b+shape[1],c:c+shape[2]]
 mask=(sol0.wealth_grid[:,None,None]>=1.15)&(sol0.wealth_grid[:,None,None]<=3.4)&(sol0.liquidity_grid[None,:,None]>=.6)&(sol0.liquidity_grid[None,:,None]<=2.0)&(sol0.credit_grid[None,None,:]>=.04)&(sol0.credit_grid[None,None,:]<=.25)
 diff=abs(arr-sol0.default_probability)[mask]
 row={'run':name,'shape':str(ns),'time_steps':p.time_steps,'domain':str(domain),
  'max_abs_interior_pd_diff':float(np.max(diff)),
  'mean_abs_interior_pd_diff':float(np.mean(diff))}
 print(row,flush=True);rows.append(row)
rows.insert(0,{'run':'base','shape':str(shape),'time_steps':p.time_steps,'domain':str(bounds),
 'max_abs_interior_pd_diff':0.,'mean_abs_interior_pd_diff':0.})
pd.DataFrame(rows).to_csv(ROOT/'audit/matched_domain_sensitivity.csv',index=False)
