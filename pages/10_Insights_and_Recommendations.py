"""
Page 10: Strategic Insights, Practical Banking Implications, & Limitations
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("🎯 Page 10: Executive Insights & Practical Banking Implications")
st.caption("Synthesizes overall empirical findings and details the paper's three practical commercial banking strategies.")

# Academic Paper Synthesis
st.markdown("""
<div style="background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); padding: 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 24px;">
    <h3>Executive Summary & Key Research Discoveries</h3>
    <p>By leveraging internal B2B transactional data (e-wiring transfers) combined with credit report relationship registries (affiliates and upstream/downstream partners), commercial banks can achieve a <strong>~5.4x expansion</strong> of their addressable corporate prospect pool.</p>
    <p>Targeted marketing campaigns utilizing supply chain network positioning achieve a <strong>2.17% corporate loan response rate</strong> (vs. 0.98% baseline) and an <strong>89.4% credit approval rate</strong> (vs. 48.1% baseline), drastically improving sales force efficiency and credit risk quality.</p>
</div>
""", unsafe_allow_html=True)

# 3 Practical Banking Implications
st.subheader("The Paper's 3 Practical Commercial Banking Implications")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 10px; border-top: 4px solid #38BDF8; height: 100%;">
        <h4>1. Promote E-Wiring & Account Opening</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            E-wiring transactional relationships represent the largest single relationship source (<strong>50.3% relationship mix share</strong>). 
            Commercial banks should subsidize digital transfer fees and incentivize corporate clients to mandate electronic payment channels, creating rich transactional visibility into upstream suppliers and downstream buyers.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 10px; border-top: 4px solid #2ECC71; height: 100%;">
        <h4>2. Deepen Product Holdings Cross-Selling</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Existing bank customers with tenure &lt; 4.5 years utilizing e-wiring exhibit the highest loan conversion rates (<strong>~20% response</strong>). 
            Account officers should deploy association rule bundles (e.g. <i>{Checking, E-Wiring} &rArr; Corporate Loan / Trade Finance</i>) to deepen multi-product holdings early in the customer lifecycle.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 10px; border-top: 4px solid #8E44AD; height: 100%;">
        <h4>3. Expand B2B Analytics to Risk & SCF</h4>
        <p style="color: #94A3B8; font-size: 0.95rem;">
            Supply chain network centralities (PageRank, Betweenness) identify key anchor hub enterprises (&gt;$50M capital). 
            Banks should structure reverse factoring and Supply Chain Finance (SCF) programs around anchor hubs, reducing credit underwriting friction for SME suppliers.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# Limitations Section
st.subheader("⚠️ Analytical Limitations & Future Scope")

with st.expander("1. Synthetic Data & Distributional Constraints"):
    st.markdown("""
    While synthetic data generation rigorously replicates parameter distributions (log-normal amounts, Pareto capital, network degree power laws), real-world B2B transaction graphs exhibit non-stationary seasonality and macroeconomic shocks.
    """)

with st.expander("2. Observational Correlation vs Causal Impact"):
    st.markdown("""
    High loan response rates among e-wiring users reflect both selection effects (more active businesses use digital channels) and operational affinity. Quasi-experimental causal inference (e.g. Propensity Score Matching or IV estimation) is recommended prior to full budget deployment.
    """)

with st.expander("3. Qualitative Account Officer Constraints"):
    st.markdown("""
    Algorithmic targeting provides prioritized prospect lists; however, final conversion depends on relationship manager soft skills, local branch coverage, and customized credit pricing negotiations.
    """)
