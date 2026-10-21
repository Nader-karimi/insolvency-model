"""Reproduce model-internal one-factor counterfactuals for TWO synthetic banks.

The output is a decomposition of the numerical HJB function's sensitivity,
NOT an empirical/causal attribution or an observed bank default probability.
Usage: python research/counterfactual_risk_attribution.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.operational_model import load_solution, interpolate_solution
BASE=Path(__file__).resolve().parent

def counterfactual_rows(solution,base,adverse):
    bw,bl,bd=(base[k] for k in ('wealth_ratio','liquidity_state','credit_state'))
    aw,al,ad=(adverse[k] for k in ('wealth_ratio','liquidity_state','credit_state'))
    cases=[
       ('Adverse reference', (aw,al,ad)),
       ('Improve capital only', (bw,al,ad)),
       ('Improve liquidity only', (aw,bl,ad)),
       ('Improve credit only', (aw,al,bd)),
       ('Improve all three', (bw,bl,bd)),
    ]
    ref=float(interpolate_solution(solution,np.array([cases[0][1]]),'default_probability')[0])
    rows=[]
    for label,state in cases:
        v=float(interpolate_solution(solution,np.array([state]),'default_probability')[0])
        rows.append({'scenario':label,'w':float(state[0]),'lambda':float(state[1]),'d':float(state[2]),
                     'model_probability':v,'risk_reduction_percentage_points':100*(ref-v)})
    return rows

def main():
    solution=load_solution(BASE/'model_outputs_v4/operational_survival_solution.npz')
    j=json.loads((BASE/'model_outputs_v4/sample_bank_assessments.json').read_text())
    rows=counterfactual_rows(solution,j['baseline']['states'],j['adverse']['states'])
    out=BASE/'model_outputs_v4/model_counterfactual_risk_attribution.csv'
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(out)
    for r in rows:print(r['scenario'],round(r['model_probability'],6),round(r['risk_reduction_percentage_points'],3))

if __name__=='__main__': main()
