"""Quick non-empirical integrity checks on the provided reference artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from operational_model import load_solution, interpolate_solution
ROOT=Path(__file__).resolve().parent
sol=load_solution(ROOT/'model_outputs_v4/operational_survival_solution.npz')
assert sol.survival_probability.shape == (25,29,23)
assert np.isfinite(sol.survival_probability).all()
assert sol.survival_probability.min()>=0 and sol.survival_probability.max()<=1
assert np.allclose(sol.survival_probability[0],0)
for axis,sgn in [(0,-1),(1,1),(2,1)]:
    diff=np.diff(sol.default_probability,axis=axis)*sgn
    assert diff.min()>=-1e-10,(axis,float(diff.min()))
data=pd.read_csv(ROOT/'data/processed/bank_year_metrics.csv')
assert len(data)==48
assert data[['symbol','jalali_year']].drop_duplicates().shape[0]==48
s=json.loads((ROOT/'model_outputs_v4/operational_summary.json').read_text(encoding='utf-8'))
assert s['grid_points']==16675
con=pd.read_csv(ROOT/'model_outputs_v4/operational_convergence.csv')
assert {'coarse','medium','production','time_refined','reference'}==set(con['run'])
assert len(pd.read_csv(ROOT/'model_outputs_v4/operational_surrogate_validation.csv'))==16675
cou=pd.read_csv(ROOT/'audit/coupled_refinement.csv')
assert len(cou)==3
for c in [f'PD_at_x{i}' for i in range(1,6)]:
 assert cou[c].between(0,1).all()
for f in ['05_phase_portrait_and_stress_path.png','06_dual_balance_sheet_state_map.png']:
 assert (ROOT/'paper_assets/v4'/f).stat().st_size>10000
print('VERIFIED: 3-state output shapes, monotonicity, probability bounds, provenance-panel row count, base tables and joint checks.')
