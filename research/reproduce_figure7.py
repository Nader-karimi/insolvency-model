"""Produce two-panel Figure 7 using saved three-state HJB results (no dashboard rendering)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from operational_model import load_solution, interpolate_solution

ROOT=Path(__file__).resolve().parent
SOL=ROOT/'model_outputs_v4/operational_survival_solution.npz'
TRAJ=ROOT/'model_outputs_v4/stress_trajectory.csv'
TARGET=ROOT/'paper_assets/v4/06_dual_balance_sheet_state_map.png'

def build_figure(target=TARGET):
    solution=load_solution(SOL)
    path=pd.read_csv(TRAJ,encoding='utf-8-sig')
    g=path[path['path'].eq('remediated')].sort_values('month').copy()
    eq_min=60_000.0; ncof=115_000.0
    equity=(g['wealth'].to_numpy()-1.0)*eq_min
    surplus=ncof*(1.0/g['liquidity'].to_numpy()-1.0)
    W=1.45
    lam=np.linspace(.65,2.30,130)
    credit=np.linspace(.035,.24,110)
    L,D=np.meshgrid(lam,credit)
    points=np.c_[np.full(L.size,W),L.ravel(),D.ravel()]
    pd_slice=interpolate_solution(solution,points,'default_probability').reshape(L.shape)
    true_pd=interpolate_solution(solution,g[['wealth','liquidity','credit']].to_numpy(),'default_probability')

    plt.rcdefaults(); plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
        'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white'})
    fig, axes=plt.subplots(1,2,figsize=(13.5,5.3),facecolor='white',layout='constrained')
    ax=axes[0]
    xmin,xmax=min(float(equity.min())-4000,-1000),max(float(equity.max())+4000,5000)
    ymin,ymax=min(float(surplus.min())-4000,-5000),max(float(surplus.max())+4000,5000)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    # Four quadrants, defined purely by sign of the illustrated surplus metrics.
    import matplotlib.patches as patches
    for x,y,w,h,color in [
        (0,0,xmax,ymax,'#e7f7ee'),(xmin,0,-xmin,ymax,'#fff2e4'),
        (xmin,ymin,-xmin,-ymin,'#fdeced'),(0,ymin,xmax,-ymin,'#fff2e4')]:
        if w>0 and h>0:ax.add_patch(patches.Rectangle((x,y),w,h,color=color,zorder=0))
    ax.axhline(0,color='#30343a',lw=1);ax.axvline(0,color='#30343a',lw=1)
    ax.plot(equity,surplus,'-',lw=2.0,color='#2367d8',zorder=3)
    sc=ax.scatter(equity,surplus,c=g.month,cmap='viridis',s=47,zorder=4,edgecolor='white',lw=.5)
    for m in [0,4,8,9,12]:
        row=g[g.month.eq(m)]
        if row.empty:continue
        j=int(np.flatnonzero(g.month.eq(m))[0]);ax.annotate(f'M{m}',(equity[j],surplus[j]),xytext=({0: (7,7),4: (8,7),8: (-20,-13),9:(8,-15),12:(-27,8)}[m]),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Equity buffer, (w − 1) × minimum equity (million rials)')
    ax.set_ylabel('Illustrative liquidity surplus, HQLA − NCOF (million rials)')
    ax.set_title('(a) Balance-sheet state under assumed shocks')
    cb=fig.colorbar(sc,ax=ax,shrink=.75,pad=.01)
    cb.set_label('Scenario month')

    ax=axes[1]
    cs=ax.contourf(L,D,pd_slice,levels=np.linspace(0,1,17),cmap='RdYlGn_r',vmin=0,vmax=1,extend='neither')
    for level,clr in [(0.2,'#ad7628'),(0.5,'#9e3033')]:
        if pd_slice.min()<level<pd_slice.max():
            c=ax.contour(L,D,pd_slice,levels=[level],colors=[clr],linewidths=[1.5],linestyles=['--'])
            ax.clabel(c,fmt={level:f'{int(level*100)}% at fixed w'},inline=True,fontsize=7)
    ax.plot(g.liquidity,g.credit,color='#193a71',lw=1.9,zorder=3)
    ss=ax.scatter(g.liquidity,g.credit,c=true_pd,cmap='RdYlGn_r',vmin=0,vmax=1,s=60,
        edgecolors='white',linewidths=.55,zorder=4)
    for m in [0,4,8,9,12]:
        row=g[g.month.eq(m)]
        if row.empty:continue
        j=int(np.flatnonzero(g.month.eq(m))[0]); ax.annotate(f'M{m}',(g.liquidity.iloc[j],g.credit.iloc[j]),xytext=({0:(7,6),4:(8,-13),8:(8,8),9:(-16,-14),12:(10,-2)}[m]),textcoords='offset points',fontsize=8)
    ax.set(xlim=(.65,2.30),ylim=(.035,.24),xlabel='Liquidity state λ = NCOF / HQLA',ylabel='Credit state d = NPL / gross loans')
    ax.set_title('(b) Path projection and model PD (HJB slice at w = 1.45)')
    cb=fig.colorbar(cs,ax=ax,shrink=.75,pad=.01)
    cb.set_label('Model PD on fixed-w slice')
    fig.suptitle('Illustrative joint-stress path in physical and HJB state coordinates',fontweight='bold',fontsize=13)
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return target

if __name__=='__main__': print(build_figure())
