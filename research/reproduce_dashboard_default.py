"""Reproduce the two *scenario dashboard* panels using its published default inputs.

This script intentionally does NOT solve the HJB, and does not label the
heuristic score threshold as a legal or model-implied HJB failure boundary.
The exact same Scenario/stress engine powers the dashboard's first tab.
Run from repository root: python research/reproduce_dashboard_default.py
"""
from __future__ import annotations
import json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scenario_engine import Scenario,stress


def heuristic_score(lam,d):
    # Exact coefficients of the old public app.py display score.
    return (0.4*(np.asarray(lam)/1.5)**1.5
            +0.6*(np.asarray(d)/0.20)**1.2
            +0.8*(np.asarray(lam)/1.5)**1.1*(np.asarray(d)/0.20)**1.1)


def generate(output=None, json_output=None):
    s=Scenario()
    r=stress(s)
    # Metadata makes the operational and HJB scenarios unambiguously distinct.
    settings={**asdict(s),'default_results':r,'score_coefficients':
      {'w1':0.4,'w2':0.6,'w3':0.8,'p':1.5,'q':1.2,'r':1.1,'s':1.1,'lambda_c':1.5,'d_c':0.20,'score_threshold':1.0},
      'interpretation':'Illustrative scenario score, NOT the HJB solution or official regulatory insolvency'}
    output=Path(output or ROOT/'research/paper_assets/v4/12_dashboard_default_scenario.png')
    json_output=Path(json_output or ROOT/'research/dashboard_default_scenario_results.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    json_output.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':9,'axes.grid':False,'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12.1,4.6),constrained_layout=True)
    ax=axes[0]
    xlo=min(-16000,r['E2']*1.25);xhi=13000; ylo=-1900;yhi=21000
    ax.axvspan(xlo,0,facecolor='#fce5e7',alpha=.86,zorder=0)
    ax.fill_between([0,xhi],0,yhi,facecolor='#e2f6e8',zorder=0)
    ax.fill_between([0,xhi],ylo,0,facecolor='#fff0db',zorder=0)
    ax.plot([r['E0'],r['E2']],[r['C0'],r['C2']],'-o',lw=2.5,c='#2455a4',markersize=6,zorder=3)
    ax.scatter([r['E0']],[r['C0']],c='#12845b',s=60,zorder=4)
    ax.scatter([r['E2']],[r['C2']],c='#bd373a',s=60,zorder=4)
    ax.annotate('Initial',xy=(r['E0'],r['C0']),xytext=(-44,12),textcoords='offset points',fontsize=9)
    ax.annotate('Stressed',xy=(r['E2'],r['C2']),xytext=(-15,14),textcoords='offset points',fontsize=9)
    ax.axhline(0,c='#555',lw=.9);ax.axvline(0,c='#555',lw=.9)
    ax.text(2200,17800,'Positive equity\nand liquid buffer',color='#1a7044',fontsize=9)
    ax.text(-15100,17800,'Negative equity',color='#9d3539',fontsize=9)
    ax.text(2300,-1350,'Liquidity deficit',color='#9f681e',fontsize=8)
    ax.set(xlim=(xlo,xhi),ylim=(ylo,yhi),xlabel='Equity (million USD)',ylabel='Liquid buffer (million USD)',title='(a) Balance-sheet path under the default shock')
    ax.ticklabel_format(axis='x',style='plain');ax.grid(False)
    ax=axes[1]
    lv=np.linspace(.5,4.,240);dv=np.linspace(0,.4,220)
    LG,DG=np.meshgrid(lv,dv)
    score=heuristic_score(LG,DG)
    ax.contourf(LG,DG,score,levels=[-1e5,1,1e5],colors=['#dff0fa','#e7ebf2'],alpha=1.)
    ax.contour(LG,DG,score,levels=[1.0],colors=['#252525'],linewidths=1.65,linestyles='dashed')
    ax.plot([r['lambda_initial'],r['lambda_final']],[r['d0'],r['d_final']],'-o',color='#20478f',lw=2.6,markersize=6)
    ax.scatter([r['lambda_initial']],[r['d0']],s=55,color='#2457d7',zorder=5)
    ax.scatter([r['lambda_final']],[r['d_final']],s=55,color='#c43b32',zorder=5)
    ax.annotate('Initial',xy=(r['lambda_initial'],r['d0']),xytext=(-30,-20),textcoords='offset points')
    ax.annotate('Stressed',xy=(r['lambda_final'],r['d_final']),xytext=(9,-12),textcoords='offset points')
    ax.text(.72,.325,'Heuristic score < 1',fontsize=9,color='#335e7c')
    ax.text(2.75,.35,'Score >= 1',fontsize=9,color='#4b526b')
    ax.set(xlim=(.5,4.),ylim=(0,.4),xlabel='Scenario liquidity proxy  SCO / SCI',ylabel='Scenario NPL ratio',title='(b) Scenario path and heuristic score threshold')
    ax.grid(False)
    fig.savefig(output,dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return output,r

if __name__=='__main__':
    img,res=generate()
    print(img)
    print('initial/final equity',res['E0'],res['E2'],'liquidity',res['C0'],res['C2'],'lambda',res['lambda_initial'],res['lambda_final'],'credit',res['d0'],res['d_final'])
