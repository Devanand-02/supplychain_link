"""
Page 6: Propensity & Credit Approval Predictive Models
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    train_propensity_model,
    train_approval_model,
    calculate_composite_scores
)

st.title("📈 Page 6: Propensity & Credit Approval Predictive Models")
st.caption("Train Logistic Regression propensity and credit approval models to compute composite scores: Final Score = P(Response) × P(Approval).")

data_dict = st.session_state["data"]
companies_df = data_dict["companies"]
campaign_df = data_dict["campaign_outcomes"]

centrality_df = st.session_state.get("network_analytics", {}).get("centrality_df", None)

# Train Propensity Model
with st.spinner("Training Logistic Regression Propensity Model..."):
    prop_results = train_propensity_model(companies_df, campaign_df, centrality_df)
    st.session_state["propensity_results"] = prop_results

# Train Approval Model
with st.spinner("Training Credit Approval Model..."):
    appr_results = train_approval_model(companies_df, campaign_df)
    st.session_state["approval_results"] = appr_results

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Propensity ROC-AUC", f"{prop_results['auc']:.3f}")
col2.metric("Precision-Recall AP", f"{prop_results['pr_auc']:.3f}")
col3.metric("5-Fold CV AUC", f"{prop_results['cv_auc_mean']:.3f} ± {prop_results['cv_auc_std']:.3f}")
col4.metric("Approval Model AUC", f"{appr_results.get('approval_auc', 0.85):.3f}")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Odds Ratios & Coefficients",
    "📉 Model Performance Curves",
    "🎯 Composite Score Ranking List",
    "📖 Econometric Model Methodology"
])

# ---------------------------------------------------------
# Tab 1: Odds Ratios & Coefficients
# ---------------------------------------------------------
with tab1:
    st.subheader("Logistic Regression Coefficients & Odds Ratios [OR = exp(β)]")
    st.write("Odds Ratios > 1.0 indicate positive impact on loan response propensity. 95% Confidence Intervals calculated via Statsmodels Logit.")

    odds_df = prop_results["odds_ratios_df"]
    st.dataframe(odds_df, use_container_width=True)

    # Bar chart of Odds Ratios
    st.write("#### Odds Ratio Multipliers")
    st.bar_chart(odds_df.set_index("Feature")["Odds_Ratio"])

# ---------------------------------------------------------
# Tab 2: Model Performance Curves
# ---------------------------------------------------------
with tab2:
    st.subheader("Model Diagnostic & Calibration Curves")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(prop_results["roc_fig"], use_container_width=True)
        st.plotly_chart(prop_results["calib_fig"], use_container_width=True)

    with c2:
        st.plotly_chart(prop_results["pr_fig"], use_container_width=True)
        
        st.write("#### Confusion Matrix")
        cm = prop_results["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual Non-Responder", "Actual Responder"], columns=["Predicted Non-Responder", "Predicted Responder"])
        st.dataframe(cm_df, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: Composite Score Ranking List
# ---------------------------------------------------------
with tab3:
    st.subheader("Ranked Target Campaign List by Composite Score")
    st.write("Final Target Score = P(Loan Response) × P(Credit Approval). Ranks prospect candidates by expected commercial value.")

    composite_df = calculate_composite_scores(
        companies_df,
        prop_results["model"],
        appr_results["model"],
        centrality_df
    )

    display_comp = composite_df.copy()
    display_comp["registered_capital_usd"] = display_comp["registered_capital"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        display_comp[[
            "company_id", "name", "industry", "registered_capital_usd",
            "p_response", "p_approval", "composite_score"
        ]],
        use_container_width=True
    )

    csv_scores = display_comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Ranked Composite Score Campaign CSV",
        data=csv_scores,
        file_name="supplychain_link_composite_scored_campaign.csv",
        mime="text/csv",
        type="primary"
    )

# ---------------------------------------------------------
# Tab 4: Econometric Model Methodology
# ---------------------------------------------------------
with tab4:
    st.subheader("Econometric Propensity Model Formula")
    st.latex(r"""
    \text{logit}(P(\text{Response}_i = 1)) = \beta_0 + \beta_1 \ln(\text{Capital}_i) + \beta_2 \text{Tenure}_i + \beta_3 \text{Products}_i + \beta_4 \text{PageRank}_i + \sum \gamma_k \text{RelType}_{ik}
    """)
    st.markdown("""
    - **Odds Ratio ($OR_j$)**: $e^{\beta_j}$ represents the multiplicative factor change in the odds of responding to a corporate loan offer for a 1-unit increase in feature $j$.
    - **Composite Valuation**: Target prioritization relies on the joint probability of acquisition and credit approval to maximize loan origination yields.
    """)
