"""Asymptotically admissible coupled refinements: h^2 / dt decreases.

All solutions are independently re-solved at fixed illustrative parameters,
all results are probabilities at interior points, no post-hoc smoothing.
"""
from dataclasses import replace
from pathlib import Path
import sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operational_model import OperationalParameters,solve_survival_hjb,interpolate_solution
pts=np.array([[1.2,.82,.06],[1.35,.96,.1],[1.55,1.2,.12],[1.85,1.35,.15],[2.2,1.6,.1]])
rows=[]
for name,shape,steps,ctrl in [
    ('C1',(17,21,17),16,13),
    ('C2',(25,29,23),24,17),
    ('C3',(33,37,29),36,21),
    ('C4',(41,45,35),42,23),
]:
    t=time.monotonic()
    sol=solve_survival_hjb(replace(OperationalParameters(),time_steps=steps),*shape,ctrl)
    vals=interpolate_solution(sol,pts)
    dlogs=[np.log(x[1]/x[0])/(n-1) for x,n in zip([(1,4),(.4,3.5),(.01,.35)],shape)]
    row={'run':name,'grid':'x'.join(map(str,shape)),'time_steps':steps,'controls':ctrl,
      'h2_over_dt':max(dlogs)**2*steps,'elapsed_seconds':time.monotonic()-t,
      **{f'pd_{i+1}':float(v) for i,v in enumerate(vals)}}
    print(row,flush=True);rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'audit/consistent_refinement.csv',index=False)
pd.DataFrame(rows).to_csv(ROOT/'audit/consistent_refinement.csv',index=False)
