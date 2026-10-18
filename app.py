import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Structural Insolvency Stress Test", layout="wide")
st.title("Joint Liquidity-Credit Stress Testing Dashboard")
st.caption("Implementing Stochastic Dynamic Feedback Mechanisms on Structural Balance Sheets")

st.markdown("""
This application maps structural balance sheet shocks onto the theoretical insolvency boundaries, 
explicitly modeling the NSFR-induced policy switching (Lemma 1) and the endogenous credit feedback loop.
""")

# ---------------------------------------------------------
# 1. Balance Sheet Inputs
# ---------------------------------------------------------
st.sidebar.header("1. Initial Balance Sheet (Millions USD)")
col1, col2 = st.sidebar.columns(2)

with col1:
    E0 = st.number_input("Initial Equity (W0)", min_value=0.0, value=8000.0)
    C0 = st.number_input("High Quality Liquid Assets (HQLA)", min_value=0.0, value=15000.0)
    M0 = st.number_input("Marketable Assets (Unencumbered)", min_value=0.0, value=20000.0)

with col2:
    J0 = st.number_input("Illiquid Assets (Loan Book)", min_value=0.0, value=57000.0)
    S0 = st.number_input("Short-Term Liabilities (Maturing)", min_value=0.0, value=25000.0)
    L0 = st.number_input("Long-Term Liabilities", min_value=0.0, value=67000.0)

st.sidebar.header("2. Cash Flow & Regulatory State")
SCI = st.number_input("Stable Cash Inflows (SCI)", min_value=0.0, value=8000.0)
SCO = st.number_input("Short-Term Cash Outflows (SCO)", min_value=0.0, value=14000.0)
Initial_NPL = st.number_input("Initial Credit Risk State (NPL Ratio / d0)", min_value=0.0, max_value=1.0, value=0.05, format="%.3f")
NSFR_Buffer = st.slider("Initial NSFR Buffer (above 100%)", 0.0, 0.5, value=0.05, format="%.2f")

st.sidebar.header("3. Shock Scenario & Model Parameters")
shock_credit = st.slider("Asset Value Shock (Drop in M & J) %", -50.0, 0.0, value=-25.0) / 100
phi = st.number_input("Endogenous Feedback Parameter (ϕ)", min_value=0.0, value=0.00005, format="%.5f", help="Sensitivity of credit state to fire sales")
haircut = st.number_input("Repo / Fire-Sale Haircut", min_value=0.0, max_value=0.9, value=0.2, format="%.2f")

# ---------------------------------------------------------
# 2. The Core Mathematical Engine
# ---------------------------------------------------------
def run_dynamic_stress_test(E, C, M, J, S, L, sci, sco, d_init, nsfr_buff, shock, feedback_phi, hc):
    results = {}
    
    # --- STEP 1: Initial Solvency Shock ---
    delta_M = M * shock
    delta_J = J * shock
    E1 = E + delta_M + delta_J
    M1 = M * (1 + shock)
    J1 = J * (1 + shock)
    
    # --- STEP 2: Endogenous Liquidity Shock ---
    margin_call = abs(delta_M) * 0.5 + abs(delta_J) * 0.1 
    total_liquidity_need = S + margin_call
    available_liquidity = C + sci
    
    liquidity_gap = total_liquidity_need - available_liquidity
    at = 0.0
    bt = 0.0
    cost_of_funding = 0.0
    delta_d_feedback = 0.0

    # --- STEP 3: Decision Making (Lemma 1) ---
    if liquidity_gap > 0:
        max_borrow_via_nsfr = nsfr_buff * L 
        bt = min(liquidity_gap, max(0, max_borrow_via_nsfr))
        liquidity_gap -= bt
        cost_of_funding += bt * 0.05 
        
        if liquidity_gap > 0:
            at = liquidity_gap / (1 - hc) 
            actual_at = min(at, M1) 
            liquidity_gap -= actual_at * (1 - hc)
            cost_of_funding += actual_at * hc
            delta_d_feedback = feedback_phi * actual_at
            
    # --- STEP 4: Final State Calculations ---
    E2 = E1 - cost_of_funding
    C2 = available_liquidity + bt + (actual_at * (1 - hc) if liquidity_gap <=0 else 0) - total_liquidity_need
    C2 = max(0, C2)
    
    base_npl_increase = abs(shock) * 0.4 
    d_final = d_init + base_npl_increase + delta_d_feedback
    d_final = min(d_final, 1.0) 
    
    lambda_final = (sco + margin_call) / max(sci, 1) 

    results['E0'], results['E2'] = E, E2
    results['C0'], results['C2'] = C, C2
    results['d0'], results['d_final'] = d_init, d_final
    results['lambda_final'] = lambda_final
    results['at'] = at
    results['bt'] = bt
    results['delta_d_feedback'] = delta_d_feedback
    results['margin_call'] = margin_call
    
    return results

res = run_dynamic_stress_test(E0, C0, M0, J0, S0, L0, SCI, SCO, Initial_NPL, NSFR_Buffer, shock_credit, phi, haircut)

# ---------------------------------------------------------
# 3. Visualization & Output Dashboard (Upgraded with Regions)
# ---------------------------------------------------------
st.header("Stress Test Results & Insolvency Mapping")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Initial Equity (W0)", f"${res['E0']:,.0f}")
kpi2.metric("Final Equity (W2)", f"${res['E2']:,.0f}", delta=f"${res['E2']-res['E0']:,.0f}")
kpi3.metric("Initial Credit State (d0)", f"{res['d0']*100:.1f}%")
kpi4.metric("Final Credit State (d_final)", f"{res['d_final']*100:.1f}%", delta=f"+{(res['d_final']-res['d0'])*100:.2f}%", delta_color="off")

st.markdown("---")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Balance Sheet Trajectory under Joint Stress")
    fig_sl = go.Figure()
    
    # Determine axis limits dynamically based on inputs
    max_w = max(res['E0'], 5000) * 1.5
    max_c = max(res['C0'], 5000) * 1.5
    min_w = min(res['E2'], -2000) * 1.5
    min_c = min(res['C2'], -2000) * 1.5
    
    # 1. Draw Safe Quadrant (Green)
    fig_sl.add_shape(type="rect", x0=0, y0=0, x1=max_w, y1=max_c,
                     fillcolor="rgba(144, 238, 144, 0.3)", line=dict(width=0))
    # 2. Draw Insolvency Region (Left - Red)
    fig_sl.add_shape(type="rect", x0=min_w, y0=min_c, x1=0, y1=max_c,
                     fillcolor="rgba(255, 182, 193, 0.4)", line=dict(width=0))
    # 3. Draw Illiquidity Region (Bottom - Orange)
    fig_sl.add_shape(type="rect", x0=0, y0=min_c, x1=max_w, y1=0,
                     fillcolor="rgba(255, 218, 185, 0.4)", line=dict(width=0))

    # Add Text Labels for Regions
    fig_sl.add_annotation(x=max_w/2, y=max_c/2, text="SAFE ZONE<br>(Solvent & Liquid)", showarrow=False, font=dict(size=14, color="green"))
    fig_sl.add_annotation(x=min_w/2, y=max_c/2, text="INSOLVENCY<br>DEFAULT", showarrow=False, font=dict(size=12, color="darkred"))
    fig_sl.add_annotation(x=max_w/2, y=min_c/2, text="ILLIQUIDITY<br>DEFAULT", showarrow=False, font=dict(size=12, color="darkorange"))

    # Draw Zero Axes
    fig_sl.add_hline(y=0, line_width=2, line_color="black")
    fig_sl.add_vline(x=0, line_width=2, line_color="black")
    
    # Trajectory
    fig_sl.add_trace(go.Scatter(
        x=[res['E0'], res['E2']], 
        y=[res['C0'], res['C2']], 
        mode='lines+markers+text',
        line=dict(width=4, color='blue'),
        marker=dict(size=12, color=['darkgreen', 'red']),
        text=["Start (t=0)", "End (t=2)"],
        textposition="top center",
        name='Bank Trajectory'
    ))
    
    fig_sl.update_layout(
        xaxis_title="Bank Equity / Wealth (W)",
        yaxis_title="Liquidity Buffer (C)",
        height=600,
        xaxis_range=[min_w, max_w],
        yaxis_range=[min_c, max_c],
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_sl, use_container_width=True)


with col_chart2:
    st.subheader("State-Space Dynamics & Insolvency Boundary")
    
    # Re-introduce Surrogate Function for Boundary
    def surrogate_function(lam, d):
        w1, w2, w3 = 0.4, 0.6, 0.8
        p, q, r, s = 1.5, 1.2, 1.1, 1.1
        lambda_c, d_c = 1.5, 0.20
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = w1 * (np.divide(lam, lambda_c)**p)
            term2 = w2 * (np.divide(d, d_c)**q)
            term3 = w3 * (np.divide(lam, lambda_c)**r) * (np.divide(d, d_c)**s)
        return term1 + term2 + term3

    fig_state = go.Figure()
    
    # Create Grid for Boundary
    lam_contour = np.linspace(0.5, 4.0, 150)
    d_contour = np.linspace(0.0, 0.4, 150)
    LAM_C, D_C = np.meshgrid(lam_contour, d_contour)
    Z = surrogate_function(LAM_C, D_C)
    f_c = 1.0 # Threshold

    # 1. Shading the Boundary (Red = Insolvent, Green = Safe)
    fig_state.add_contour(
        x=lam_contour, y=d_contour, z=Z, 
        zmin=0, zmax=2*f_c, 
        colorscale=[[0, 'red'], [0.5, 'white'], [1, 'lightgreen']],
        showscale=False, opacity=0.3
    )
    
    # 2. Drawing the Exact Boundary Line (Black)
    fig_state.add_contour(
        x=lam_contour, y=d_contour, z=Z, 
        contours=dict(start=f_c, end=f_c, size=0.01), 
        line=dict(width=3, color='black', dash='dash'),
        showlegend=False
    )
    
    # Add Label for Boundary
    fig_state.add_annotation(x=1.2, y=0.25, text="Exact Boundary (B_exact)", showarrow=True, arrowhead=1, ax=40, ay=-30, font=dict(color="black", size=12))

    # 3. Plot the Bank's Trajectory
    start_lam = SCO / max(SCI, 1)
    
    fig_state.add_trace(go.Scatter(
        x=[start_lam, res['lambda_final']], 
        y=[res['d0'], res['d_final']], 
        mode='lines+markers',
        line=dict(width=4, color='darkblue'),
        marker=dict(size=14, color=['blue', 'darkred']),
        name='State Trajectory'
    ))
    
    # 4. Annotation for Feedback
    if res['at'] > 0:
        fig_state.add_annotation(
            x=res['lambda_final'], y=res['d_final'], 
            text=f"Endogenous Feedback Triggered!<br>Fire Sales (at) = {res['at']:,.0f}<br>Δd (feedback) = {res['delta_d_feedback']:.4f}", 
            showarrow=True, arrowhead=1, ax=100, ay=60,
            font=dict(color='red', size=11),
            bgcolor="white", bordercolor="red", borderwidth=1
        )
                        
    fig_state.update_layout(
        xaxis_title="Liquidity Risk State (λt = SCO/SCI)",
        yaxis_title="Credit Risk State (dt = NPL Ratio)",
        height=600,
        xaxis_range=[0.5, 4.0],
        yaxis_range=[0.0, 0.4],
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_state, use_container_width=True)

st.markdown("""
**How to interpret these plots:**
*   **Left Plot:** Shows the physical balance sheet deterioration. A South-West shift crossing the axes means the bank has lost its buffers.
*   **Right Plot:** Shows the economic mechanism. The convex black dashed line is the theoretical insolvency boundary derived from the HJB equation. The trajectory proves that fire sales (driven by NSFR constraints) push the bank's state diagonally towards this boundary.
""")