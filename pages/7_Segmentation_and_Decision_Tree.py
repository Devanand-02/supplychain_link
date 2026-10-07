"""
Page 7: Customer Segmentation & Decision Tree Segmentation
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    train_decision_tree_segmentation,
    perform_customer_segmentation
)

st.title("🌳 Page 7: Decision Tree & Customer K-Means Segmentation")
st.caption("Replicate Fig. 5 decision tree rules from the academic paper and profile customer segments using K-Means clustering.")

data_dict = st.session_state["data"]
companies_df = data_dict["companies"]
campaign_df = data_dict["campaign_outcomes"]
centrality_df = st.session_state.get("network_analytics", {}).get("centrality_df", None)

# Tabs
tab1, tab2 = st.tabs([
    "🌴 Decision Tree Segmentation (Fig. 5 Replication)",
    "🧩 K-Means Customer Clustering Profile"
])

# ---------------------------------------------------------
# Tab 1: Decision Tree Segmentation
# ---------------------------------------------------------
with tab1:
    st.subheader("Decision Tree Rules (Depth = 3, Min Leaf Size = 40)")
    st.write("Extracts explicit rule paths governing loan response rate differentials across customer tenure, e-wiring adoption, and product holdings.")

    dt_results = train_decision_tree_segmentation(companies_df, campaign_df)

    st.code(dt_results["tree_rules_text"], language="text")

    st.write("#### Leaf Segment Summary & Response Rates")
    st.dataframe(dt_results["leaf_summary"], use_container_width=True)

# ---------------------------------------------------------
# Tab 2: K-Means Customer Clustering
# ---------------------------------------------------------
with tab2:
    st.subheader("K-Means Customer Clustering Profile")
    st.write("Clusters commercial clients based on tenure, product holdings, e-wiring usage, registered capital, and network centrality.")

    max_k_slider = st.slider("Max Clusters (k)", min_value=3, max_value=8, value=6)

    with st.spinner("Executing K-Means clustering and calculating silhouette scores..."):
        cluster_results = perform_customer_segmentation(companies_df, centrality_df, max_k=max_k_slider)

    st.write(f"Optimal Cluster Count Selected: **k = {cluster_results['best_k']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(cluster_results["elbow_fig"], use_container_width=True)
    with col2:
        st.plotly_chart(cluster_results["sil_fig"], use_container_width=True)

    st.write("#### Cluster Profiles & Strategic Action Matrix")
    st.dataframe(cluster_results["profile_df"], use_container_width=True)
