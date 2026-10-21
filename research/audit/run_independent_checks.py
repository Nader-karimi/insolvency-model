"""Re-run independent HJB sensitivity checks. Usage: python audit/run_independent_checks.py"""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operational_model import OperationalParameters,solve_survival_hjb,interpolate_solution
p=OperationalParameters(time_steps=24)
points=np.array([[1.20,.82,.06],[1.35,.96,.10],[1.55,1.20,.12],[1.85,1.35,.15],[2.20,1.60,.10]])
rows=[]
for nc,nt,grid in [(9,24,(17,21,17)),(17,24,(17,21,17)),(25,24,(17,21,17)),
                   (17,12,(17,21,17)),(17,48,(17,21,17)),(17,24,(21,25,19))]:
 sol=solve_survival_hjb(replace(p,time_steps=nt),*grid,nc)
 vals=interpolate_solution(sol,points)
 rows.append(dict(controls=nc,time_steps=nt,grid='x'.join(map(str,grid)),
      **{f'pd_{i}':float(v) for i,v in enumerate(vals)}))
pd.DataFrame(rows).to_csv(ROOT/'audit/numerical_controls_time_rerun.csv',index=False)
rows=[]
for name,domain,grid in [
 ('baseline',((1,4),(.4,3.5),(.01,.35)),(21,25,19)),
 ('expanded_upper',((1,5),(.4,4.2),(.01,.45)),(25,27,21)),
 ('expanded_both',((1,5),(.3,4.2),(.005,.45)),(25,29,24))]:
 sol=solve_survival_hjb(p,*grid,17,domain=domain)
 vals=interpolate_solution(sol,points)
 rows.append(dict(name=name,domain=str(domain),grid='x'.join(map(str,grid)),
     **{f'pd_{i}':float(v) for i,v in enumerate(vals)}))
pd.DataFrame(rows).to_csv(ROOT/'audit/numerical_domain_rerun.csv',index=False)
print('Re-run complete: two CSVs written to audit/')
