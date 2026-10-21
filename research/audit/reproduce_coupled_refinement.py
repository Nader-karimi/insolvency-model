"""Reproduce Section 8.4 joint space-time-control sensitivity (~1-2 min)."""
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import pandas as pd
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from operational_model import OperationalParameters, solve_survival_hjb, interpolate_solution
pts=np.array([[1.2,.82,.06],[1.35,.96,.10],[1.55,1.2,.12],[1.85,1.35,.15],[2.2,1.6,.10]])
rows=[]
for name,shape,steps,controls in [
 ('coupled_coarse',(17,21,17),24,13),
 ('coupled_medium',(25,29,23),48,17),
 ('coupled_fine',(33,37,29),96,21),
]:
 t=time.time();s=solve_survival_hjb(replace(OperationalParameters(),time_steps=steps),*shape,controls)
 vals=interpolate_solution(s,pts,'default_probability')
 r=dict(run=name,nw=shape[0],nl=shape[1],nd=shape[2],time_steps=steps,controls=controls,
        elapsed_s=round(time.time()-t,2),**{f'PD_at_x{i+1}':round(float(v),6) for i,v in enumerate(vals)})
 print(r,flush=True); rows.append(r)
pd.DataFrame(rows).to_csv(root/'audit/coupled_refinement_rerun.csv',index=False)
