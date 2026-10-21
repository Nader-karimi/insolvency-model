"""Regenerate Figure 6 directly from the saved HJB surface and illustrative path."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operational_model import load_solution, interpolate_solution, _trilinear
ROOT=Path(__file__).resolve().parent
TARGET=ROOT/'paper_assets/v4/05_phase_portrait_and_stress_path.png'

def build_figure(target=TARGET):
 sol=load_solution(ROOT/'model_outputs_v4/operational_survival_solution.npz')
 tr=pd.read_csv(ROOT/'model_outputs_v4/stress_trajectory.csv',encoding='utf-8-sig')
 p=sol.parameters
 L,D=np.meshgrid(np.linspace(.65,2.25,30),np.linspace(.035,.24,24))
 pts=np.c_[np.full(L.size,1.45),L.ravel(),D.ravel()]
 pdv=interpolate_solution(sol,pts).reshape(L.shape)
 sale=_trilinear((sol.wealth_grid,sol.liquidity_grid,sol.credit_grid),
                  np.nan_to_num(sol.optimal_asset_sale),pts,False).reshape(L.shape)
 U=p.mu_liquidity*L;V=(p.mu_credit+p.fire_sale_credit_feedback*sale)*D
 mag=np.hypot(U,V)
 plt.rcdefaults(); plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.facecolor':'white','figure.facecolor':'white'})
 fig,axs=plt.subplots(1,2,figsize=(13,5.2),layout='constrained')
 cf=axs[0].contourf(L,D,pdv,levels=np.linspace(0,1,17),cmap='RdYlGn_r')
 axs[0].quiver(L[::2,::2],D[::2,::2],U[::2,::2]/np.maximum(mag[::2,::2],1e-10),V[::2,::2]/np.maximum(mag[::2,::2],1e-10),
       color='#344256',width=.0032,alpha=.64,scale=32)
 for val,color in [(0.2,'#b47b26'),(0.5,'#9e3033')]:
  if pdv.min()<val<pdv.max():
   c=axs[0].contour(L,D,pdv,levels=[val],colors=[color],linewidths=[1.3],linestyles='--')
   axs[0].clabel(c,fontsize=7,fmt={val:f'{val:.0%}'})
 axs[0].set(xlabel='Inverse LCR λ',ylabel='NPL share d',title='(a) Controlled drift, with HJB PD at w = 1.45')
 cb=fig.colorbar(cf,ax=axs[0],fraction=.035,pad=.02);cb.set_label('Model PD')
 for key,color in [('managed','#6d56bf'),('remediated','#16a5ad')]:
  g=tr[tr.path.eq(key)].sort_values('month')
  axs[1].plot(g.liquidity,g.credit,'o-',color=color,ms=3.1,lw=1.8,label=key.capitalize())
  for m in ([0,4,8,12] if key=='remediated' else [9,12]):
   row=g[g.month.eq(m)]
   if row.empty: continue
   r=row.iloc[0]
   shift= {0:(-24,8),4:(8,-12),8:(8,-7),9:(5,9),12:(9,12)}[m]
   axs[1].annotate(f'M{m}',(r.liquidity,r.credit),xytext=shift,textcoords='offset points',fontsize=8,color=color)
 axs[1].set(xlabel='Inverse LCR λ',ylabel='NPL share d',title='(b) Prescribed stress and illustrative intervention')
 axs[1].legend(loc='upper left',frameon=False)
 axs[1].set_xlim(.76,1.43);axs[1].set_ylim(.046,.155)
 fig.suptitle('HJB-controlled drift and synthetic stress trajectory',fontsize=13,fontweight='bold')
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(target,dpi=240,bbox_inches='tight',facecolor='white');plt.close(fig)
 return target
if __name__=='__main__':print(build_figure())
