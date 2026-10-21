"""Streamlit interface for the research-grade three-state HJB and illustrative stress model.

The original public default scenario is preserved. Crucially, the stylized
balance-sheet simulator is not misidentified as the numerical HJB. All HJB
surfaces are read directly from the released NPZ solution, not a toy contour.
"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scenario_engine import Scenario,stress
from hjb_dashboard import (in_domain,pd_at,risk_slice,policy_slice,
                           boundary_surface,risk_vs_capital,local_counterfactuals)
from research.operational_model import (load_solution,minimum_feasible_controls)

ROOT=Path(__file__).resolve().parent
SOLUTION_FILE=ROOT/'research/model_outputs_v4/operational_survival_solution.npz'

st.set_page_config(page_title='Joint Liquidity–Credit Stress Testing', layout='wide')
st.title('Joint Liquidity–Credit Stress Testing Dashboard')
st.caption('Interactive balance-sheet scenarios and model-implied three-state HJB insolvency regions')
st.info('Research illustration only. HJB probabilities are conditional on the released illustrative calibration, not validated bank-failure forecasts, official LCR/NSFR measurements, or legal insolvency determinations.')

@st.cache_resource(show_spinner='Loading verified HJB solution ...')
def get_solution():
    return load_solution(SOLUTION_FILE)

try:
    solution=get_solution()
except (OSError, ValueError, KeyError) as e:
    st.error(f'Numerical HJB solution could not be loaded: {e}. Confirm the research/model_outputs_v4 folder was uploaded.')
    st.stop()

with st.sidebar:
    st.header('1. Initial Balance Sheet')
    st.caption('Common units: **millions USD**, default public stress scenario')
    c1,c2=st.columns(2)
    with c1:
        E=st.number_input('Initial equity',value=8000.0,min_value=0.0,step=500.0)
        H=st.number_input('HQLA',value=15000.0,min_value=0.0,step=500.0)
        M=st.number_input('Marketable assets',value=20000.0,min_value=0.0,step=500.0)
    with c2:
        J=st.number_input('Loan book',value=57000.0,min_value=0.0,step=500.0)
        S=st.number_input('Short-term liabilities',value=25000.0,min_value=0.0,step=500.0)
        LL=st.number_input('Long-term liabilities',value=67000.0,min_value=0.0,step=500.0)
    st.header('2. Cash Flow & Scenario')
    SCI=st.number_input('Stable cash inflows (scenario)',value=8000.0,min_value=1.0,step=500.0)
    SCO=st.number_input('Short-term cash outflows (scenario)',value=14000.0,min_value=1.0,step=500.0)
    d0=st.number_input('Initial NPL ratio',value=.05,min_value=0.0,max_value=1.0,step=.01,format='%.3f')
    s0=st.slider('Scenario stable-funding buffer',0.0,.5,.05,step=.01)
    shock_percent=st.slider('Asset-value shock (%)',-50.0,0.0,-25.0,step=1.0)
    phi=st.number_input('Scenario feedback parameter',value=.00005,min_value=0.0,max_value=.01,step=.00001,format='%.5f')
    hc=st.number_input('Fire-sale haircut',value=.2,min_value=0.0,max_value=.9,step=.05,format='%.2f')
    with st.expander('3. Numerical HJB state (separate regulatory inputs)',expanded=False):
        st.caption('These inputs define an independently normalized HJB state. No official regulatory ratios are inferred from the simplified scenario inputs above.')
        Emin=st.number_input('Minimum required equity',min_value=1.0,value=4500.0,step=100.0)
        NCOF=st.number_input('30-day net cash outflow (NCOF)',min_value=1.0,value=12300.0,step=500.0)
        hd=st.number_input('HJB NPL / gross loans',value=.05,min_value=.01,max_value=.35,step=.01,format='%.3f')
        lcr_hqla=st.number_input('HJB eligible HQLA',value=15000.0,min_value=1.0,step=500.0)
        alpha=st.select_slider('Model risk threshold for boundary',options=[.20,.30,.40,.50,.60],value=.50)

s=Scenario(equity=E,hqla=H,marketable=M,loan_book=J,
           short_liabilities=S,long_liabilities=LL,inflows=SCI,outflows=SCO,
           npl_ratio=d0,nsfr_buffer=s0,shock=shock_percent/100,phi=phi,haircut=hc)
try:
    result=stress(s)
except ValueError as exc:
    st.error(f'Invalid scenario: {exc}')
    st.stop()

w=E/Emin
lam=NCOF/lcr_hqla
d=hd
model_pd=pd_at(solution,w,lam,d)
calibration=solution.parameters

# Helpers
PLOT=dict(template='plotly_white',margin=dict(l=35,r=18,t=35,b=35),height=440,
          font=dict(family='Arial, sans-serif',size=12))
def fig_theme(fig,title,x,y):
    fig.update_layout(title=title,xaxis_title=x,yaxis_title=y,**PLOT)
    return fig

def explanation(text):
    st.markdown(f'**Interpretation.** {text}')

def money(x):
    return f'{x:,.0f}'


tab_s,tab_h,tab_c=st.tabs(['Balance-sheet stress','Three-state HJB risk maps','Validation & definitions'])
with tab_s:
    st.subheader('Scenario results — linked to the editable inputs')
    a,b,c,e=st.columns(4)
    a.metric('Initial equity',money(result['E0']))
    b.metric('Stressed equity',money(result['E2']),delta=money(result['E2']-result['E0']))
    c.metric('Initial NPL',f"{100*result['d0']:.1f}%")
    e.metric('Stressed NPL',f"{100*result['d_final']:.1f}%")

    c1,c2=st.columns(2)
    with c1:
        f=go.Figure()
        x_lo=min(result['E2'],0,E)*1.25-1000
        x_hi=max(E,result['E2'],1000)*1.3+1000
        y_hi=max(H,result['C2'],1000)*1.3+1000
        y_lo=-max(1000,.12*y_hi)
        f.add_shape(type='rect',x0=0,x1=x_hi,y0=0,y1=y_hi,fillcolor='rgba(30,160,95,.14)',line_width=0)
        f.add_shape(type='rect',x0=x_lo,x1=0,y0=y_lo,y1=y_hi,fillcolor='rgba(220,70,90,.14)',line_width=0)
        f.add_shape(type='rect',x0=0,x1=x_hi,y0=y_lo,y1=0,fillcolor='rgba(238,168,41,.16)',line_width=0)
        f.add_hline(y=0,line_color='#555');f.add_vline(x=0,line_color='#555')
        f.add_trace(go.Scatter(x=[E,result['E2']],y=[H,result['C2']],mode='lines+markers+text',
            text=['Initial','Stressed'],textposition='top center',name='Illustrative path',
            marker=dict(size=10,color=['#198d65','#c73949']),line=dict(color='#2367d8',width=3)))
        f.update_xaxes(range=[x_lo,x_hi]);f.update_yaxes(range=[y_lo,y_hi])
        st.plotly_chart(fig_theme(f,'Equity and liquid-buffer trajectory','Equity (millions USD)','Liquid buffer (millions USD)'),use_container_width=True)
        explanation('The two markers compare initial and stressed accounting positions. Shaded quadrants identify signs of equity and liquid buffers; they are not legal insolvency boundaries or HJB first-passage contours. This is a two-stage stress comparison, not a continuous-time simulated path.')
    with c2:
        f=go.Figure()
        # Historic dashboard surrogate, retained for consistency with the default snapshot.
        xx=np.linspace(.5,4.,95); yy=np.linspace(0,.4,95)
        X,Y=np.meshgrid(xx,yy)
        Z=(.4*(X/1.5)**1.5 + .6*(np.maximum(Y,0)/.20)**1.2
           + .8*(X/1.5)**1.1*(np.maximum(Y,0)/.20)**1.1)
        # Categorical display of the historical *heuristic* score, not HJB PD.
        # A two-color heatmap makes the regions distinguishable without
        # falsely implying that the score is a calibrated probability.
        heuristic_high = (Z >= 1.0).astype(float)
        f.add_trace(go.Heatmap(x=xx,y=yy,z=heuristic_high,zmin=0,zmax=1,
            colorscale=[[0.0,'#e8f4fa'],[0.4999,'#e8f4fa'],
                        [0.5001,'#13537e'],[1.0,'#13537e']],
            showscale=False,hovertemplate='Proxy λ=%{x:.2f}<br>NPL d=%{y:.3f}<br>Heuristic zone=%{z:.0f}<extra></extra>'))
        f.add_trace(go.Contour(x=xx,y=yy,z=Z,
            contours=dict(start=1,end=1,size=1,coloring='none'),
            showscale=False,line=dict(color='#162a3f',width=3,dash='dash'),
            name='Heuristic score = 1'))
        f.add_annotation(x=.95,y=.035,text='Lower heuristic score',showarrow=False,
                         font=dict(color='#14446a',size=11),bgcolor='rgba(255,255,255,.75)')
        f.add_annotation(x=3.15,y=.34,text='Higher heuristic score',showarrow=False,
                         font=dict(color='white',size=11),bgcolor='rgba(5,30,60,.4)')
        f.add_trace(go.Scatter(x=[result['lambda_initial'],result['lambda_final']],
              y=[d0,result['d_final']],mode='lines+markers',marker=dict(size=12,color=['#2367d8','#c73949']),
              line=dict(color='#07182F',width=3),name='Scenario states'))
        f.update_xaxes(range=[.5,4]);f.update_yaxes(range=[0,.4])
        st.plotly_chart(fig_theme(f,'Legacy proxy state map (NOT the HJB boundary)','SCO / SCI scenario ratio','NPL ratio'),use_container_width=True)
        explanation('The dashed contour is a parameterized heuristic index retained to reproduce the original dashboard behavior. It is NOT a numerical HJB boundary. The horizontal ratio is scenario SCO/SCI, whereas the HJB liquidity coordinate uses 30-day NCOF/HQLA; the two must not be equated.')
        if result['lambda_final']>4 or result['d_final']>.4:
            st.warning('The stressed endpoint lies outside the visual axes; the chart does not support an in-range classification.')

    st.subheader('Stress transmission and response')
    f=go.Figure(go.Waterfall(orientation='v',measure=['absolute','relative','relative','total'],
          x=['Starting equity','Asset-value shock','Borrowing and sale costs','Stressed equity'],
          y=[E,result['shock_loss'],-result['cost'],0],
          connector=dict(line=dict(color='#a3a3a3'))))
    st.plotly_chart(fig_theme(f,'Illustrative equity loss decomposition','','Millions USD'),use_container_width=True)
    explanation(f"The asset shock changes the marketable-asset and loan-book values by {money(result['shock_loss'])} in aggregate. Modelled financing and sale costs subtract another {money(result['cost'])}; this arithmetic decomposition is not a loss-attribution estimate from observed banks.")
    c1,c2,c3=st.columns(3)
    c1.metric('Additional borrowing',money(result['borrow']))
    c2.metric('Executed asset sales',money(result['sale']))
    c3.metric('Unmet liquidity need',money(result['unmet_liquidity_gap']))
    if result['unmet_liquidity_gap']>1e-8:
        st.warning('The scenario includes an unresolved funding gap after available borrowing and sale capacity; this is not automatically a legal default event.')
    f=go.Figure()
    sh=np.linspace(-.5,0,26)
    rr=[stress(Scenario(**{**asdict(s),'shock':float(z)})) for z in sh]
    f.add_trace(go.Scatter(x=100*sh,y=[z['E2'] for z in rr],mode='lines+markers',name='Stressed equity'))
    f.add_hline(y=0,line_color='#c73949',line_dash='dash')
    st.plotly_chart(fig_theme(f,'Scenario sensitivity to asset shocks','Asset-value shock (%)','Ending equity (millions USD)'),use_container_width=True)
    explanation('Each plotted value recomputes the same two-stage balance-sheet stress calculation at a different shock, holding all other scenario inputs fixed. It shows scenario sensitivity, not an empirical default frequency or a stochastic HJB sensitivity.')

with tab_h:
    st.subheader('Finite-horizon numerical HJB — conditional reference calibration')
    st.caption('The arrays below come from research/model_outputs_v4/operational_survival_solution.npz (25×29×23 grid, 30 steps, one-year horizon). These are fixed-calibration surfaces, not bank-specific re-estimates.')
    in_bounds=in_domain(solution,w,lam,d)
    feas=minimum_feasible_controls(lam,calibration)
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Normalized equity w = E / Emin',f'{w:.3f}')
    c2.metric('Inverse LCR λ = NCOF / HQLA',f'{lam:.3f}')
    c3.metric('Credit state d',f'{d:.3f}')
    c4.metric('Model-implied 1-year boundary probability',f'{100*model_pd:.1f}%' if model_pd is not None else 'Outside domain')
    if w<=1 or not feas['feasible']:
        st.error('Absorbing state in the HJB model: minimum equity threshold or reference-model control infeasibility. This is not a legal insolvency determination.')
    if not in_bounds and w>1 and feas['feasible']:
        st.warning('One or more coordinates are outside the numerical HJB domain; no model probability is extrapolated. The risk contour remains a reference slice.')
    elif model_pd is not None and w>1 and feas['feasible']:
        text='model-implied high-risk region' if model_pd>=.5 else ('model warning zone' if model_pd>=.2 else 'lower modeled-risk region')
        st.caption(f'Illustrative classification: **{text}** (50% high-risk and 20% warning thresholds are author-selected, not regulatory).')
    c1,c2=st.columns(2)
    w_slice=float(np.clip(w,solution.wealth_grid[0],solution.wealth_grid[-1]))
    with c1:
        x,y,z=risk_slice(solution,w_slice)
        f=go.Figure(go.Contour(x=solution.liquidity_grid,y=solution.credit_grid,z=z,
            colorscale='RdYlGn_r',zmin=0,zmax=1,
            colorbar=dict(title='Model risk',tickformat='.0%'),
            contours=dict(showlabels=True)))
        f.add_trace(go.Scatter(x=[lam],y=[d],mode='markers',marker=dict(color='#1d2666',size=11,symbol='x'),name='Input state'))
        f.add_trace(go.Contour(x=solution.liquidity_grid,y=solution.credit_grid,z=z,
            contours=dict(start=alpha,end=alpha,size=1e-4,coloring='none'),line=dict(color='#07182F',width=3),
            showscale=False,name=f'{alpha:.0%} model contour'))
        st.plotly_chart(fig_theme(f,f'HJB PD slice at w={w_slice:.2f}','Inverse LCR λ','NPL ratio d'),use_container_width=True)
        explanation(f'This chart evaluates the numerical HJB solution at fixed normalized equity w={w_slice:.2f}. The contour marks a {alpha:.0%} model-implied threshold. Redder areas represent higher conditional first-passage probability. A marker outside the solved domain cannot be assigned a numerical probability.')
    with c2:
        boundary=boundary_surface(solution,alpha)
        f=go.Figure(go.Heatmap(x=solution.liquidity_grid,y=solution.credit_grid,z=boundary,
            colorscale='Viridis',colorbar=dict(title='Required w'),hoverongaps=False))
        f.add_trace(go.Scatter(x=[lam],y=[d],mode='markers',marker=dict(symbol='x',size=11,color='white',line=dict(width=2,color='#07182F')),name='Input liquidity/credit'))
        st.plotly_chart(fig_theme(f,f'Capital buffer for {alpha:.0%} model target','Inverse LCR λ','NPL ratio d'),use_container_width=True)
        explanation('Each colored location is the first discrete capital grid point whose HJB probability is no greater than the selected target. Blank cells mean that threshold is not attainable within the computed capital range; they are not zero-capital requirements.')
    with st.expander('Explore the three-dimensional model boundary', expanded=True):
        boundary_3d=boundary_surface(solution,alpha)
        f3=go.Figure(go.Surface(x=solution.liquidity_grid, y=solution.credit_grid,
                     z=boundary_3d,colorscale='Viridis',colorbar=dict(title='Capital / Emin'),
                     hovertemplate='Inverse LCR: %{x:.3f}<br>NPL: %{y:.3f}<br>Required w: %{z:.3f}<extra></extra>'))
        f3.update_layout(title=f'Model-implied three-dimensional {alpha:.0%} capital boundary',
                         scene=dict(xaxis_title='Inverse LCR λ',yaxis_title='Credit state d',
                                    zaxis_title='Required normalized equity w'),
                         template='plotly_white',height=480,margin=dict(l=0,r=0,t=42,b=0))
        st.plotly_chart(f3,use_container_width=True)
        explanation('The height is the smallest grid value of normalized capital that meets the selected conditional model-risk threshold for each liquidity–credit pair. Holes mean the target cannot be met within the solved capital grid. This is a surface of a numerical reference model, not a legal insolvency surface for observed banks.')
    c1,c2=st.columns(2)
    with c1:
        wp,pol,sale,borrow=policy_slice(solution,w_slice)
        f=go.Figure(go.Heatmap(x=solution.liquidity_grid,y=solution.credit_grid,z=pol,
            zmin=-.5,zmax=3.5,colorscale=[[0,'#2fa987'],[.249,'#2fa987'],[.25,'#5a8dee'],[.499,'#5a8dee'],[.5,'#efa95d'],[.749,'#efa95d'],[.75,'#7b8193'],[1,'#7b8193']],
            colorbar=dict(tickvals=[0,1,2,3],ticktext=['No action','Borrow','Sale','Infeasible'])))
        st.plotly_chart(fig_theme(f,f'HJB policy regimes (nearest w grid = {wp:.2f})','Inverse LCR λ','NPL ratio d'),use_container_width=True)
        explanation('Color indicates the optimizer’s regime in the reference constrained HJB: no additional action, borrowing, positive asset sales, or infeasible controls. Policies depend on fixed assumed sale costs, credit feedback and capacity bounds; they are not individualized financial advice.')
    with c2:
        wg,pg=risk_vs_capital(solution,lam,d)
        f=go.Figure(go.Scatter(x=wg,y=pg,mode='lines+markers',line=dict(color='#2367d8',width=3),name='Model PD'))
        f.add_hline(y=alpha,line_color='#c73949',line_dash='dash')
        if in_bounds:f.add_vline(x=w,line_color='#07182F',line_dash='dot')
        f.update_yaxes(tickformat='.0%',range=[0,1])
        st.plotly_chart(fig_theme(f,'Model risk versus normalized capital','Equity / minimum required equity','Conditional one-year risk'),use_container_width=True)
        explanation('Holding inverse LCR, NPL and the model calibration fixed, the curve isolates how the solved conditional boundary probability changes with additional equity buffer. It does not forecast actual defaults in a bank sample.')
    st.subheader('Why does the model assign this risk? — one-factor diagnostics')
    comparisons=local_counterfactuals(solution,w,lam,d, .10)
    valid=[row for row in comparisons if row['pd'] is not None]
    if valid and any(row['label']=='Current state' for row in valid):
        f=go.Figure(go.Bar(x=[row['label'] for row in valid],
               y=[100*row['pd'] for row in valid],marker_color=['#344966']+['#237a70']*(len(valid)-1),
               text=[f"{100*row['pd']:.1f}%" for row in valid],textposition='outside'))
        f.update_yaxes(range=[0,108]);f.update_layout(margin=dict(l=30,r=20,t=30,b=90),height=390)
        st.plotly_chart(fig_theme(f,'Model-implied one-factor stress comparison','One-at-a-time starting-state variation','Conditional one-year model risk (%)'),use_container_width=True)
        explanation('Each bar re-evaluates the SAME solved HJB function, holding its assumed dynamics, optimal-control objective and other state coordinates fixed. The three alternatives change initial normalized equity, inverse LCR or NPL by 10%, separately. This is explanatory scenario sensitivity, NOT observed default validation or causal attribution. A missing comparison means the new state leaves the numerical domain; changes in regulatory capacity require a new HJB solve.')
        st.dataframe(pd.DataFrame(comparisons),use_container_width=True,hide_index=True)
    else:
        st.info('The numerical HJB comparison is unavailable for these coordinates; no extrapolated probabilities are displayed.')

    st.subheader('Local feasibility and illustrative actions')
    lamgrid=solution.liquidity_grid
    min_controls=[minimum_feasible_controls(float(k),calibration) for k in lamgrid]
    sale_min=np.array([a['asset_sale'] if a['feasible'] else np.nan for a in min_controls])
    borrow_min=np.array([a['borrowing'] if a['feasible'] else np.nan for a in min_controls])
    f=go.Figure()
    f.add_trace(go.Scatter(x=lamgrid,y=sale_min,mode='lines+markers',name='Minimum sale / HQLA'))
    f.add_trace(go.Scatter(x=lamgrid,y=borrow_min,mode='lines+markers',name='Minimum borrowing / HQLA'))
    f.add_vline(x=lam,line_dash='dash',line_color='#444')
    st.plotly_chart(fig_theme(f,'LCR–NSFR feasible-control envelope','Inverse LCR λ','Minimum control intensity / HQLA'),use_container_width=True)
    explanation('For each liquidity state the plot uses the algebraic LCR/NSFR feasibility inequalities of the published reference model. Missing segments mean the institutional-capacity bounds make the joint constraints infeasible. This plot is not an official Basel compliance calculation.')
    if in_bounds and w>1 and feas['feasible']:
        h_int=f'Feasible: minimum sale intensity {feas["asset_sale"]:.3f}; minimum borrowing intensity {feas["borrowing"]:.3f}.'
        st.info(h_int + ' Multiply by the supplied eligible HQLA to express these example intensities in the selected monetary units; do not confuse minimum feasible controls with optimal HJB policy.')
    st.caption(f'Reference parameters: T={calibration.horizon_years:g} year; σw={calibration.wealth_volatility:.2f}, σλ={calibration.sigma_liquidity:.2f}, σd={calibration.sigma_credit:.2f}, ρ={calibration.liquidity_credit_correlation:.2f}, φ={calibration.fire_sale_credit_feedback:.2f}, s0={calibration.stable_funding_buffer:.2f}, ā={calibration.max_asset_sale:.2f}, b̄={calibration.max_borrowing:.2f}.')

with tab_c:
    st.subheader('Definitions, numerical verification and data scope')
    st.markdown('**Reference risk:** minimum model-admissible one-year probability of hitting normalized equity `w ≤ 1` or a reference control-infeasibility boundary, given the stated mathematical and computational assumptions. The 20% and 50% thresholds are visualization choices, not observed default rates.')
    st.markdown('**Two different representations:** the first tab uses a discrete balance-sheet shock and a legacy heuristic contour. The second tab reads the released numerical **three-state HJB grid**. Streamlit inputs cannot convert annual public CODAL accounting proxies into official LCR, NSFR or regulatory NPL without granular regulatory disclosures.')
    st.markdown('**Numerical limits:** the computation exhibits material time/space resolution sensitivity under some tested refinements. The plotted surface is therefore a conditional numerical approximation, not a certified convergence-accurate default probability or an externally calibrated predictive score.')
    with st.expander('Solver verification table'):
        path=ROOT/'research/model_outputs_v4/operational_convergence.csv'
        if path.exists():st.dataframe(pd.read_csv(path),use_container_width=True)
    with st.expander('Independent robustness experiments'):
        path=ROOT/'research/audit/numerical_controls_time.csv'
        if path.exists():st.dataframe(pd.read_csv(path),use_container_width=True)
    st.markdown('**Reproducibility:** all manuscript figures, numerical tables, source code, parameter files, processed public accounting proxies, and controlled regression checks are in the `research/` directory. The online dashboard is a presentation layer, not a substitute for this numerical archive.')
    st.download_button('Download current scenario and HJB coordinates (JSON)',
        data=json.dumps({'scenario':asdict(s),'scenario_result':result,
           'hjb_reference_coordinates':{'w':w,'lambda':lam,'d':d},
           'hjb_conditional_probability':model_pd,
           'hjb_parameters':asdict(calibration),
           'warning':'Illustrative scenario only; not external prediction or legal insolvency'},indent=2),
        file_name='bank_scenario_reproducibility.json',mime='application/json')
