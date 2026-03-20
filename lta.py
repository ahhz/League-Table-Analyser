import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="League Table Simulator", layout="wide")

st.title("🎓 League Table Simulator")

# 1. Data Loading & Setup
st.sidebar.header("1. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    all_cols = df.columns.tolist()
    
    st.sidebar.header("2. Column Mapping")
    name_col = st.sidebar.selectbox("University Name Column", all_cols)
    overall_col = st.sidebar.selectbox("Overall Score Column", all_cols)
    rank_col = st.sidebar.selectbox("Current Rank Column", all_cols)
    criteria_cols = st.sidebar.multiselect("Criteria Columns", 
                                         [c for c in all_cols if c not in [overall_col, rank_col, name_col]])

    if criteria_cols and overall_col:
        # Data Cleaning
        for col in criteria_cols + [overall_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_cleaned = df.replace(0, np.nan)
        df_reg = df_cleaned.dropna(subset=criteria_cols + [overall_col])
        
        # Regression
        X = df_reg[criteria_cols]
        X_with_const = sm.add_constant(X)
        y = df_reg[overall_col]
        model = sm.OLS(y, X_with_const).fit()
        
        # --- UNIVERSITY SELECTION ---
        selected_uni = st.selectbox("Select University", df[name_col].unique())
        
        if "last_selected_uni" not in st.session_state:
            st.session_state.last_selected_uni = selected_uni

        if st.session_state.last_selected_uni != selected_uni:
            for crit in criteria_cols:
                if f"slider_{crit}" in st.session_state:
                    del st.session_state[f"slider_{crit}"]
            st.session_state.last_selected_uni = selected_uni

        uni_data = df[df[name_col] == selected_uni].iloc[0]
        target_idx = df.index[df[name_col] == selected_uni][0]

        # --- PRE-CALCULATION FOR SLIDERS ---
        for crit in criteria_cols:
            if f"slider_{crit}" not in st.session_state:
                orig_val = uni_data[crit] if not (pd.isna(uni_data[crit]) or uni_data[crit] == 0) else df_cleaned[crit].min()
                if model.params[crit] < 0:
                    st.session_state[f"slider_{crit}"] = float(-orig_val)
                else:
                    st.session_state[f"slider_{crit}"] = float(orig_val)

        # --- CALCULATIONS ---
        scenario_values_current = {crit: abs(st.session_state[f"slider_{crit}"]) for crit in criteria_cols}
        orig_score = uni_data[overall_col] if not (pd.isna(uni_data[overall_col]) or uni_data[overall_col] == 0) else 0
        score_diff = sum(model.params[crit] * (scenario_values_current[crit] - (uni_data[crit] if not pd.isna(uni_data[crit]) else 0)) for crit in criteria_cols)
        new_score = orig_score + score_diff

        all_scores = df[overall_col].fillna(0).tolist()
        all_scores.append(new_score)
        rank_series = pd.Series(all_scores).rank(ascending=False, method='min')
        new_rank = int(rank_series.iloc[-1])
        orig_rank_val = int(uni_data[rank_col]) if not pd.isna(uni_data[rank_col]) else None

        # --- TOP DASHBOARD ---
        st.divider()
        st.subheader(f"Overall Rank Analysis: {selected_uni}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Rank", f"#{orig_rank_val}" if orig_rank_val else "N/A")
        c2.metric("Projected Rank", f"#{new_rank}", f"{orig_rank_val - new_rank:+}" if orig_rank_val else None)
        c3.metric("Projected Score", f"{new_score:.2f}", f"{score_diff:+.2f}")

        # --- SIMULATION SECTION ---
        st.header("Scenario Simulation")
        if st.button("🔄 Reset Sliders to Actual Scores"):
            for crit in criteria_cols:
                orig_val = uni_data[crit] if not (pd.isna(uni_data[crit]) or uni_data[crit] == 0) else df_cleaned[crit].min()
                if model.params[crit] < 0:
                    st.session_state[f"slider_{crit}"] = float(-orig_val)
                else:
                    st.session_state[f"slider_{crit}"] = float(orig_val)
            st.rerun()

        grid_cols = st.columns(3)
        for i, criterion in enumerate(criteria_cols):
            with grid_cols[i % 3]:
                coeff = model.params[criterion]
                low_better = coeff < 0
                abs_min = float(df_cleaned[criterion].min(skipna=True))
                abs_max = float(df_cleaned[criterion].max(skipna=True))
                abs_orig = uni_data[criterion] if not pd.isna(uni_data[criterion]) else abs_min

                st.markdown(f"**{criterion}**" + (" (Lower is Better)" if low_better else ""))
                
                if low_better:
                    val = st.slider(criterion, -abs_max, -abs_min, key=f"slider_{criterion}", label_visibility="collapsed")
                    display_val = abs(val)
                else:
                    val = st.slider(criterion, abs_min, abs_max, key=f"slider_{criterion}", label_visibility="collapsed")
                    display_val = val
                
                c_sector_base = df[criterion].fillna(df[criterion].median()).tolist()
                c_sector_sim = c_sector_base + [display_val]
                r_orig_c = int(pd.Series(c_sector_base).rank(ascending=low_better, method='min').iloc[target_idx])
                r_sim_c = int(pd.Series(c_sector_sim).rank(ascending=low_better, method='min').iloc[-1])
                
                m1, m2 = st.columns(2)
                m1.metric("Value", f"{display_val:.1f}", f"{display_val - abs_orig:+.1f}", delta_color="inverse" if low_better else "normal")
                m2.metric("Crit. Rank", f"#{r_sim_c}", f"{r_orig_c - r_sim_c:+} pos")
                st.write("---")

        # --- COLLAPSIBLE SECTIONS ---

        # 1. Ceteris Paribus Analysis
        with st.expander("🔍 Ceteris Paribus Analysis (Impact of Single Metric)", expanded=False):
            st.markdown("How does your rank change as you vary one metric, holding all other **actual** values constant?")
            
            cp_col1, cp_col2, cp_col3 = st.columns([1, 2, 1])
            with cp_col2:
                target_crit = st.selectbox("Select Criterion to Analyze", criteria_cols, key="cp_select")
            
            min_val = float(df_cleaned[target_crit].min())
            max_val = float(df_cleaned[target_crit].max())
            actual_crit_val = float(uni_data[target_crit]) if not pd.isna(uni_data[target_crit]) else min_val
            
            # Inject actual value into range to ensure marker sits exactly on the line
            step_range = np.linspace(min_val, max_val, 100).tolist()
            step_range.append(actual_crit_val)
            step_range = sorted(list(set(step_range)))
            
            base_score_actual = orig_score - (model.params[target_crit] * actual_crit_val)
            overall_ranks = []
            criteria_ranks = []
            
            low_better_cp = model.params[target_crit] < 0
            c_sector_base = df[target_crit].fillna(df[target_crit].median()).tolist()

            for val in step_range:
                sim_score_cp = base_score_actual + (model.params[target_crit] * val)
                temp_overall = df[overall_col].fillna(0).tolist()
                temp_overall.append(sim_score_cp)
                overall_ranks.append(pd.Series(temp_overall).rank(ascending=False, method='min').iloc[-1])
                
                temp_crit = c_sector_base + [val]
                criteria_ranks.append(pd.Series(temp_crit).rank(ascending=low_better_cp, method='min').iloc[-1])

            fig_cp = go.Figure()
            fig_cp.add_trace(go.Scatter(x=step_range, y=overall_ranks, mode='lines', name='Overall Rank', line=dict(color='#636EFA', width=3)))
            fig_cp.add_trace(go.Scatter(x=step_range, y=criteria_ranks, mode='lines', name=f'Rank in {target_crit}', line=dict(color='#00CC96', width=2, dash='dash')))
            fig_cp.add_trace(go.Scatter(x=[actual_crit_val], y=[orig_rank_val], mode='markers', marker=dict(color='#EF553B', size=14, symbol='diamond', line=dict(width=2, color='white')), name='Actual Position'))

            fig_cp.update_layout(
                xaxis_title=f"Value of {target_crit}", yaxis_title="Rank (1 is Top)",
                yaxis=dict(autorange="reversed", gridcolor='lightgray'),
                plot_bgcolor='white', hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                height=500, margin=dict(l=0, r=0, t=40, b=0)
            )
            
            with cp_col2:
                st.plotly_chart(fig_cp, use_container_width=True)

        # 2. Rank vs Score Plot
        with st.expander("📈 View Rank vs. Overall Score Plot", expanded=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[overall_col], y=df[rank_col], mode='markers', name='Sector', marker=dict(color='rgba(200, 200, 200, 0.5)', size=7), text=df[name_col], hoverinfo='text+x+y'))
            fig.add_trace(go.Scatter(x=[orig_score], y=[orig_rank_val], mode='markers', name='Current', marker=dict(color='#1f77b4', size=14, line=dict(width=2, color='white'))))
            fig.add_trace(go.Scatter(x=[new_score], y=[new_rank], mode='markers', name='Scenario', marker=dict(color='#d62728', size=18, symbol='star', line=dict(width=2, color='white'))))
            
            fig.add_annotation(x=orig_score, y=orig_rank_val if orig_rank_val else 0, text=f"<b>Current Status</b><br>Rank: #{orig_rank_val}", showarrow=True, arrowhead=2, ax=60, ay=-40, bgcolor="#1f77b4", font=dict(color="white"), bordercolor="white")
            fig.add_annotation(x=new_score, y=new_rank, text=f"<b>Scenario Status</b><br>Rank: #{new_rank}", showarrow=True, arrowhead=2, ax=-60, ay=40, bgcolor="#d62728", font=dict(color="white"), bordercolor="white")

            fig.update_layout(xaxis_title="Overall Score", yaxis_title="Overall Rank", xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"), plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        # 3. Criteria Rank Summary Table
        with st.expander("📊 Sector-Wide Criteria Rank Summary", expanded=False):
            rank_summary = []
            for c in criteria_cols:
                low_better = model.params[c] < 0
                c_sim_val = scenario_values_current[c]
                c_sector = df[c].fillna(df[c].median()).tolist()
                c_sector_sim = c_sector + [c_sim_val]
                r_orig = int(pd.Series(c_sector).rank(ascending=low_better, method='min').iloc[target_idx])
                r_sim = int(pd.Series(c_sector_sim).rank(ascending=low_better, method='min').iloc[-1])
                rank_summary.append({"Metric": c, "Current Rank": f"#{r_orig}", "Scenario Rank": f"#{r_sim}", "Place Change": r_orig - r_sim})
            st.table(pd.DataFrame(rank_summary))

        # 4. Reference Guide
        with st.expander("📚 A-Level to Entry Tariff Reference Guide", expanded=False):
            col_guide1, col_guide2 = st.columns([1, 2])
            with col_guide1:
                st.write("**UCAS Points per Grade**")
                tariff_data = {"A-Level Grade": ["A*", "A", "B", "C", "D", "E"], "UCAS Points": [56, 48, 40, 32, 24, 16]}
                st.table(pd.DataFrame(tariff_data))
            with col_guide2:
                st.write("**Common Grade Profile Conversions**")
                profiles = {"Grade Profile": ["A*A*A*", "A*AA", "AAA", "AAB", "ABB", "BBB", "BBC", "BCC", "CCC"], "Total Points": [168, 152, 144, 136, 128, 120, 112, 104, 96]}
                st.table(pd.DataFrame(profiles))

        # 5. Data & Stats
        with st.expander("📋 View Full League Table", expanded=False):
            st.dataframe(df)

        with st.expander("📊 View Regression Statistics", expanded=False):
            st.table(pd.DataFrame({"Coefficient": model.params, "P-Value": model.pvalues}))

else:
    st.info("Please upload an Excel file and map your columns to begin.")