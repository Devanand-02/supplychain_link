"""
Page 8: Fraud Screening & Anomaly Detection
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.fraud import (
    benford_law_analysis,
    isolation_forest_anomaly_detection
)

st.title("🛡️ Page 8: Fraud Screening & Transaction Anomaly Detection")
st.caption("Perform forensic auditing of e-wiring transactions using Benford's Law first-digit analysis and Isolation Forest unsupervised learning.")

data_dict = st.session_state["data"]
wiring_df = data_dict["wiring_transactions"]

# Tabs
tab1, tab2 = st.tabs([
    "📐 Benford's Law First-Digit Forensic Test",
    "🌲 Isolation Forest Transaction Anomaly Screening"
])

# ---------------------------------------------------------
# Tab 1: Benford's Law Analysis
# ---------------------------------------------------------
with tab1:
    st.subheader("Benford's Law First-Digit Audit")
    st.write("Extracts the leading digit of transaction amounts and compares against theoretical Benford probabilities [P(d) = log10(1 + 1/d)].")

    benford_res = benford_law_analysis(wiring_df)

    if "error" not in benford_res:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAD Statistic", f"{benford_res['mad']:.4f}")
        c2.metric("Conformity Rating", benford_res['conformity'])
        c3.metric("Chi-Square p-value", f"{benford_res['p_value']:.4f}")

        st.plotly_chart(benford_res["benford_fig"], use_container_width=True)

        st.write("#### First-Digit Frequency Comparison Table")
        st.dataframe(benford_res["results_df"], use_container_width=True)

        st.write("#### Flagged Suspicious Senders (High Round-Number Frequency)")
        st.dataframe(benford_res["suspicious_senders"].head(10), use_container_width=True)
    else:
        st.error(benford_res["error"])

# ---------------------------------------------------------
# Tab 2: Isolation Forest Anomaly Detection
# ---------------------------------------------------------
with tab2:
    st.subheader("Isolation Forest Anomaly Detection")
    st.write("Screen for transaction velocity spikes, industry mismatch payees, and uncharacteristically large transfer amounts.")

    contam = st.slider("Anomaly Contamination Rate", min_value=0.01, max_value=0.10, value=0.03, step=0.01)

    with st.spinner("Running Isolation Forest..."):
        df_iso, flagged_senders, iso_fig = isolation_forest_anomaly_detection(wiring_df, contamination=contam)

    st.plotly_chart(iso_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Top Anomalous Transactions")
        anom_only = df_iso[df_iso["anomaly_label"] == "Anomalous"]
        st.dataframe(
            anom_only[["transaction_id", "sender_id", "receiver_id", "amount", "date", "industry_match"]].head(10),
            use_container_width=True
        )

    with c2:
        st.write("#### Flagged Senders (High Anomaly Count)")
        st.dataframe(flagged_senders.head(10), use_container_width=True)
