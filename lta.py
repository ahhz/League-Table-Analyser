import streamlit as st
import pandas as pd
import statsmodels.api as sm
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="League Table Analyser", layout="wide")

st.title("🎓 League Table Analyser")

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
        for col in criteria_cols + [overall_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df_cleaned = df.replace(0, np.nan)
        df_reg = df_cleaned.dropna(subset=criteria_cols + [overall_col])
        
        # Regression Logic
        X = df_reg[criteria_cols]
        X_with_const = sm.add_constant(X)
        y = df_reg[overall_col]
        model = sm.OLS(y, X_with_const).fit()
        conf_interval = model.conf_int(alpha=0.05) 
        
        selected_uni = st.selectbox("Select University", df[name_col].unique())
        uni_data = df[df[name_col] == selected_uni].iloc[0]

        # --- SLIDER STATE MANAGEMENT ---
        if "last_selected_uni" not in st.session_state or st.session_state.last_selected_uni != selected_uni:
            for crit in criteria_cols:
                if f"slider_{crit}" in st.session_state:
                    del st.session_state[f"slider_{crit}"]
            st.session_state.last_selected_uni = selected_uni

        for crit in criteria_cols:
            if f"slider_{crit}" not in st.session_state:
                st.session_state[f"slider_{crit}"] = float(uni_data[crit]) if not pd.isna(uni_data[crit]) else float(df_cleaned[crit].min())

        # --- CALCULATIONS ---
        scenario_values = {crit: st.session_state[f"slider_{crit}"] for crit in criteria_cols}
        orig_score = uni_data[overall_col] if not pd.isna(uni_data[overall_col]) else 0
        score_diff = sum(model.params[crit] * (scenario_values[crit] - (uni_data[crit] if not pd.isna(uni_data[crit]) else 0)) for crit in criteria_cols)
        new_score = orig_score + score_diff

        all_scores = df[overall_col].fillna(0).tolist()
        all_scores.append(new_score)
        new_rank = int(pd.Series(all_scores).rank(ascending=False, method='min').iloc[-1])
        orig_rank_val = int(uni_data[rank_col]) if not pd.isna(uni_data[rank_col]) else None

        st.divider()

        # --- SECTION 1: SINGLE CRITERION SENSITIVITY ---
        with st.expander("🔍 Change Single Criterion", expanded=False):
            _, graph_col1, _ = st.columns([1, 2, 1])
            with graph_col1:
                target_crit = st.selectbox("Select Criterion", criteria_cols)
                coeff = model.params[target_crit]
                is_negative_beta = coeff < 0
                
                step_range = np.linspace(float(df_cleaned[target_crit].min()), float(df_cleaned[target_crit].max()), 100).tolist()
                actual_val = float(uni_data[target_crit]) if not pd.isna(uni_data[target_crit]) else step_range[0]
                
                ranks_mid, ranks_low, ranks_high, crit_ranks = [], [], [], []
                all_sector_scores = df[overall_col].fillna(0).tolist()
                c_sector_base = df[target_crit].fillna(df[target_crit].median()).tolist()

                for val in step_range:
                    diff = (val - actual_val)
                    s_mid = orig_score + (coeff * diff)
                    s_low = orig_score + (conf_interval.loc[target_crit, 0] * diff)
                    s_high = orig_score + (conf_interval.loc[target_crit, 1] * diff)
                    
                    ranks_mid.append(pd.Series(all_sector_scores + [s_mid]).rank(ascending=False, method='min').iloc[-1])
                    ranks_low.append(pd.Series(all_sector_scores + [s_low]).rank(ascending=False, method='min').iloc[-1])
                    ranks_high.append(pd.Series(all_sector_scores + [s_high]).rank(ascending=False, method='min').iloc[-1])
                    crit_ranks.append(pd.Series(c_sector_base + [val]).rank(ascending=is_negative_beta, method='min').iloc[-1])

                fig_cp = go.Figure()
                fig_cp.add_trace(go.Scatter(x=step_range + step_range[::-1], y=ranks_high + ranks_low[::-1], 
                                            fill='toself', fillcolor='rgba(99, 110, 250, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% CI'))
                fig_cp.add_trace(go.Scatter(x=step_range, y=crit_ranks, mode='lines', name=f'Rank in {target_crit}', line=dict(color='#00CC96', width=1.5)))
                fig_cp.add_trace(go.Scatter(x=step_range, y=ranks_mid, mode='lines', name='Projected Overall Rank', line=dict(color='#636EFA', width=3)))
                fig_cp.add_vline(x=actual_val, line_width=1.5, line_dash="dot", line_color="grey", annotation_text="Actual")

                fig_cp.update_layout(
                    xaxis_title=f"{target_crit} Score", yaxis_title="Rank (1 is Top)",
                    xaxis=dict(autorange="reversed" if not is_negative_beta else True, gridcolor='whitesmoke'),
                    yaxis=dict(autorange="reversed", gridcolor='whitesmoke'),
                    plot_bgcolor='white', hovermode="x unified", height=450
                )
                st.plotly_chart(fig_cp, use_container_width=True)
     
        # --- SECTION 2: CONSOLIDATED SIMULATOR HUB ---
        with st.expander("📈 Change Multiple Criteria", expanded=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Original Rank", f"#{orig_rank_val}" if orig_rank_val else "N/A")
            m2.metric("Projected Rank", f"#{new_rank}", f"{orig_rank_val - new_rank:+}" if orig_rank_val else None)
            m3.metric("Projected Score", f"{new_score:.2f}", f"{score_diff:+.2f}")
            
            st.write("---")
            
            if st.button("🔄 Reset to Actual Scores"):
                for crit in criteria_cols:
                    st.session_state[f"slider_{crit}"] = float(uni_data[crit]) if not pd.isna(uni_data[crit]) else float(df_cleaned[crit].min())
                st.rerun()

            grid_cols = st.columns(3)
            for i, criterion in enumerate(criteria_cols):
                with grid_cols[i % 3]:
                    abs_min, abs_max = float(df_cleaned[criterion].min()), float(df_cleaned[criterion].max())
                    st.markdown(f"**{criterion}**")
                    st.slider(criterion, abs_min, abs_max, key=f"slider_{criterion}", label_visibility="collapsed")
                    current_val = st.session_state[f"slider_{criterion}"]
                    orig_val = uni_data[criterion] if not pd.isna(uni_data[criterion]) else abs_min
                    st.caption(f"Cur: {current_val:.1f} | Δ: {current_val - orig_val:+.1f}")
            
            st.write("---")
            
            _, graph_col_main, _ = st.columns([1, 2, 1])
            with graph_col_main:
                fig_main = go.Figure()
                fig_main.add_trace(go.Scatter(x=df[overall_col], y=df[rank_col], mode='markers', name='Sector', marker=dict(color='rgba(200, 200, 200, 0.4)'), text=df[name_col]))
                fig_main.add_trace(go.Scatter(x=[orig_score], y=[orig_rank_val], mode='markers', name='Actual', 
                                              marker=dict(color='#636EFA', size=12, line=dict(color='white', width=2))))
                fig_main.add_trace(go.Scatter(x=[new_score], y=[new_rank], mode='markers', name='Scenario', 
                                              marker=dict(color='#d62728', size=16, symbol='star')))
                
                fig_main.update_layout(
                    xaxis_title="Overall Score", yaxis_title="Rank", 
                    xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"), 
                    plot_bgcolor='white', height=400, margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.5)")
                )
                st.plotly_chart(fig_main, use_container_width=True)

        # --- SECTION 3: REFERENCE GUIDES ---
        with st.expander("📚 Entry Tariff Reference Guide", expanded=False):
            col_guide1, col_guide2 = st.columns(2)
            with col_guide1:
                st.write("**UCAS Points per Grade**")
                st.table(pd.DataFrame({"Grade": ["A*", "A", "B", "C", "D", "E"], "A-Level": [56, 48, 40, 32, 24, 16], "EPQ": [28, 24, 20, 16, 12, 8]}))
                st.write("**AS Levels & Supplementals**")
                st.table(pd.DataFrame({"Grade": ["A", "B", "C", "D", "E"], "AS Level": [20, 16, 12, 10, 6], "Music (Gr. 8)": [30, 24, 18, "-", "-"]}))
            with col_guide2:
                st.write("**Common Grade Profile Conversions**")
                st.table(pd.DataFrame({"Grade Profile": ["A\*A\*A\*", "A\*AA", "AAA", "AAB", "ABB", "BBB", "BBC", "BCC", "CCC"], "Total Points": [168, 152, 144, 136, 128, 120, 112, 104, 96]}))

        # --- SECTION 4: FULL LEAGUE TABLE DATA ---
        with st.expander("📋 League Table Data", expanded=False):
            display_cols = [name_col, rank_col, overall_col] + criteria_cols
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # --- SECTION 5: STATISTICAL SUMMARY ---
        with st.expander("📊 Regression Statistics", expanded=False):
            st.write(f"**R-Squared:** {model.rsquared:.3f}")
            summary_df = pd.DataFrame({
                "Coefficient (Beta)": model.params,
                "P-Value": model.pvalues,
                "Lower 95% CI": conf_interval[0],
                "Upper 95% CI": conf_interval[1]
            })
            st.table(summary_df.drop("const", errors='ignore'))
            st.caption("A P-Value < 0.05 indicates the criterion is a statistically significant driver of the Overall Score.")

        # --- SECTION 6: METHODOLOGY ---
        with st.expander("🧠 Methodology & Analysis Logic", expanded=False):
            st.markdown("""
            ### How the Analysis Works
            
            **1. Ordinary Least Squares (OLS) Regression**
            The tool uses a statistical model to determine the 'weighting' of each criterion. By looking at the entire sector, it identifies how much a 1-point increase in a specific metric (e.g., Graduate Prospects) typically increases the Overall Score.
            
            **2. The 'Beta' Coefficient**
            In the statistics table above, the **Beta Coefficient** represents the sensitivity. If a criterion has a Beta of `0.2`, then increasing that score by 10 points will likely increase the Overall Score by 2 points.
            
            **3. Dynamic Ranking**
            When you adjust a slider, the tool:
            * Calculates your university's **New Overall Score**.
            * Compares this new score against the **Actual Scores** of every other university in the sector.
            * Assigns a **New Rank** based on where that score fits in the current distribution.
            
            **4. 95% Confidence Intervals (The Shaded Area)**
            Statistics involve uncertainty. The shaded blue area in the single-criterion graph shows the 'likely' rank outcome. There is a 95% probability that your actual rank change would fall within this band, accounting for the statistical noise in league table data.
            """)

    else:
        st.warning("Please complete the **Column Mapping** in the sidebar to visualize the data.")

else:
    # Instructions Landing Page
    st.info("👋 Welcome! Please upload an Excel file in the sidebar to get started.")
    
    col_inst1, col_inst2 = st.columns(2)
    with col_inst1:
        st.subheader("📁 1. Prepare your File")
        st.write("""
        Ensure your Excel file (`.xlsx`) contains:
        * **Institution Names**: A column with the names of the universities.
        * **Overall Scores**: The current total score (numerical).
        * **Rankings**: The current rank position (e.g., 1, 2, 3).
        * **Criteria**: Columns for each individual metric (e.g., Student Sat, Entry Standards).
        """)
        
    with col_inst2:
        st.subheader("⚙️ 2. Map & Analyse")
        st.write("""
        1.  Use the **Sidebar** to select the correct columns from your file.
        2.  Select **Criteria Columns** (these will become your interactive sliders).
        3.  The tool will automatically calculate a **Linear Regression** model to predict how changes in criteria impact the overall rank.
        """)
    
    st.divider()
    st.caption("Note: The model works best when all criteria and overall scores are provided for the majority of institutions.")
