"""
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
Main Streamlit Application & Session State Manager
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_gen import generate_synthetic_supply_chain_data

# Page configuration
st.set_page_config(
    page_title="SupplyChain-Link | Corporate Banking Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling (Glassmorphism & Executive Theme)
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #F8FAFC;
    }
    
    /* Card Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Hero Header */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 24px;
    }
    
    /* Metric Badges */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 16px;
        border-radius: 8px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initializes shared session state dataset and variables."""
    if "data" not in st.session_state:
        st.session_state["n_companies"] = 2500 # Default size for fast initialization
        st.session_state["random_state"] = 42
        with st.spinner("Generating initial synthetic supply chain dataset..."):
            st.session_state["data"] = generate_synthetic_supply_chain_data(
                n_companies=st.session_state["n_companies"],
                random_state=st.session_state["random_state"]
            )
            st.session_state["data_source"] = "Synthetic (Seeded)"

init_session_state()

# Sidebar summary & control info
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/organization-chart.png", width=64)
    st.title("SupplyChain-Link")
    st.caption("Big Data Analytics for Corporate Banking")
    st.divider()

    st.subheader("Dataset State")
    data_dict = st.session_state["data"]
    comp_df = data_dict["companies"]
    wiring_df = data_dict["wiring_transactions"]

    st.write(f"**Source**: `{st.session_state.get('data_source', 'Synthetic')}`")
    st.write(f"**Total Companies**: `{len(comp_df):,}`")
    st.write(f"**Bank Customers**: `{comp_df['is_bank_customer'].sum():,}` ({comp_df['is_bank_customer'].mean()*100:.1f}%)")
    st.write(f"**Wiring Transactions**: `{len(wiring_df):,}`")
    st.write(f"**Total Transfer Volume**: `${wiring_df['amount'].sum()/1e6:,.1f}M`")
    
    st.divider()
    st.info("Navigate through the multi-page modules in the sidebar menu above to explore analytics.")

# Main Page Header
st.markdown('<div class="hero-title">SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Production-grade analytics platform expanding corporate loan acquisition and multi-product cross-selling through B2B credit and e-wiring networks.</div>', unsafe_allow_html=True)

# Key Benchmark Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-box">
        <div class="metric-label">Network Expansion Target</div>
        <div class="metric-value">~5.4x</div>
        <span style="color: #2ECC71; font-size: 0.85rem;">35k -> 225k Companies</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box" style="border-left-color: #2ECC71;">
        <div class="metric-label">Loan Response Rate</div>
        <div class="metric-value">2.17%</div>
        <span style="color: #2ECC71; font-size: 0.85rem;">vs 0.98% Baseline (2.2x)</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-box" style="border-left-color: #8E44AD;">
        <div class="metric-label">Credit Approval Rate</div>
        <div class="metric-value">89.4%</div>
        <span style="color: #2ECC71; font-size: 0.85rem;">vs 48.1% Baseline (1.9x)</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-box" style="border-left-color: #F39C12;">
        <div class="metric-label">Relationship Mix</div>
        <div class="metric-value">50.3%</div>
        <span style="color: #F8FAFC; font-size: 0.85rem;">Wiring Only Channel Share</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Overview of 10 Analytical Modules
st.subheader("Platform Architecture & Modules")

m1, m2 = st.columns(2)

with m1:
    st.markdown("""
    <div class="glass-card">
        <h4>1. Overview & Data Control Center</h4>
        <p style="color: #94A3B8;">Configurable synthetic data generator, real CSV uploader, column schema validator, and data quality checks.</p>
        
        <h4>2. Network Expansion Builder</h4>
        <p style="color: #94A3B8;">6-step iterative graph expansion algorithm pulling credit report affiliates, upstream/downstream partners, and common director ownership.</p>

        <h4>3. Campaign Filters & Target List</h4>
        <p style="color: #94A3B8;">Dynamic filtering controls (min wiring frequency, USD amount, core loan capital) and campaign list export.</p>

        <h4>4. Network Explorer & Analytics</h4>
        <p style="color: #94A3B8;">Interactive Pyvis/Plotly 2D graph, PageRank, degree centrality, modularity community detection, and anchor hub identification.</p>

        <h4>5. Association Rules Mining (Apriori)</h4>
        <p style="color: #94A3B8;">Apriori cross-sell itemset mining on product baskets and relationship channels with plain-English interpretations.</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="glass-card">
        <h4>6. Propensity & Credit Approval Models</h4>
        <p style="color: #94A3B8;">Logistic regression with Odds Ratios (OR = exp(&beta;)), 95% CIs, ROC-AUC, PR curve, calibration plot, and composite score ranking.</p>

        <h4>7. Segmentation & Decision Tree</h4>
        <p style="color: #94A3B8;">Decision tree segmentation (depth=3, min_leaf=40) replicating paper Fig. 5 plus K-Means customer profiling with elbow/silhouette charts.</p>

        <h4>8. Fraud & Anomaly Screening</h4>
        <p style="color: #94A3B8;">Benford's Law first-digit Chi-Square/MAD test and Isolation Forest transaction anomaly screening.</p>

        <h4>9. Campaign Results & Lift Analysis</h4>
        <p style="color: #94A3B8;">Statistical hypothesis testing (2-proportion z-test), decile analysis, cumulative gains, and lift charts.</p>

        <h4>10. Executive Insights & Recommendations</h4>
        <p style="color: #94A3B8;">Academic paper synthesis, 3 practical commercial banking implications, and analytical limitations.</p>
    </div>
    """, unsafe_allow_html=True)

st.success("Use the sidebar menu to navigate to Page 1 (Overview & Data) to configure your dataset or start exploring modules!")
