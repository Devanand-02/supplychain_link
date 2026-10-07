"""
Page 5: Association Rule Mining & Cross-Sell Engine
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.association import (
    mine_product_association_rules,
    mine_relationship_combination_rules,
    generate_rule_interpretations
)

st.title("💡 Page 5: Association Rule Mining (Apriori Cross-Sell Engine)")
st.caption("Mine frequent product holding baskets and relationship channel combinations to uncover high-lift cross-selling rules.")

data_dict = st.session_state["data"]
companies_df = data_dict["companies"]
campaign_df = data_dict["campaign_outcomes"]

# Sidebar Sliders
st.sidebar.header("Apriori Parameter Sliders")
min_supp = st.sidebar.slider("Min Support", min_value=0.005, max_value=0.20, value=0.03, step=0.005)
min_conf = st.sidebar.slider("Min Confidence", min_value=0.05, max_value=0.80, value=0.15, step=0.05)
min_lift = st.sidebar.slider("Min Lift Factor", min_value=1.0, max_value=5.0, value=1.1, step=0.1)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🛒 Product Holding Basket Rules",
    "🔗 Relationship Channel Combinations (Table 2)",
    "📝 Plain-English Cross-Sell Strategy"
])

# ---------------------------------------------------------
# Tab 1: Product Holding Basket Rules
# ---------------------------------------------------------
with tab1:
    st.subheader("Bank Product Basket Association Rules")
    
    rule_results = mine_product_association_rules(
        companies_df,
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lift
    )

    rules_df = rule_results["rules"]

    if not rules_df.empty:
        st.write(f"Identified **{len(rules_df)} association rules** meeting parameter thresholds.")
        
        st.dataframe(
            rules_df[["rule_str", "support", "confidence", "lift"]].round(4),
            use_container_width=True
        )

        st.plotly_chart(rule_results["scatter_fig"], use_container_width=True)
    else:
        st.warning("No product association rules found under the current sliders. Lower Support or Confidence to expand rules.")

# ---------------------------------------------------------
# Tab 2: Relationship Channel Combinations (Table 2)
# ---------------------------------------------------------
with tab2:
    st.subheader("Relationship Channel Combinations (Replicating Paper Table 2)")
    st.write("Mines rules connecting relationship sources (Affiliate, Upstream/Downstream, E-Wiring) to loan response and approval outcomes.")

    rel_rules_df = mine_relationship_combination_rules(
        campaign_df,
        min_support=min_supp,
        min_confidence=min_conf,
        min_lift=min_lift
    )

    if not rel_rules_df.empty:
        st.dataframe(
            rel_rules_df[["rule_str", "support", "confidence", "lift"]].round(4),
            use_container_width=True
        )
    else:
        st.info("No relationship combination rules meeting parameter thresholds.")

# ---------------------------------------------------------
# Tab 3: Plain-English Cross-Sell Strategy
# ---------------------------------------------------------
with tab3:
    st.subheader("Plain-English Cross-Sell Strategy Recommendations")
    
    if not rules_df.empty:
        interpretations = generate_rule_interpretations(rules_df, top_n=5)
        for interp in interpretations:
            st.markdown(interp)
            st.write("")
    else:
        st.info("Adjust sliders to generate plain-English business rules.")
