"""
Page 3: Filtering Engine & Target Campaign List
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import build_supply_chain_network, apply_false_relationship_filter, filter_network_graph

st.title("🎯 Page 3: Campaign Filtering Engine & Target List")
st.caption("Apply multi-criteria business rules and false-relationship filters to extract the target marketing list.")

data_dict = st.session_state["data"]

# Ensure network graph exists
if "network_results" not in st.session_state:
    st.session_state["network_results"] = build_supply_chain_network(
        data_dict["companies"],
        data_dict["credit_reports"],
        data_dict["wiring_transactions"],
        data_dict["directors"]
    )

G = st.session_state["network_results"]["graph"]

# Sidebar Filtering Controls
st.sidebar.header("Filter Criteria Controls")

min_tx = st.sidebar.slider("Min Annual Wiring Tx Count", min_value=1, max_value=30, value=8, help="Minimum required transfer count per annum")
min_amount = st.sidebar.number_input("Min Total Wiring Amount (USD)", min_value=1000, max_value=500000, value=13500, step=2500)
min_core_cap = st.sidebar.number_input("Core Company Capital Threshold ($)", min_value=10000000, max_value=100000000, value=50000000, step=10000000)

rel_types = st.sidebar.multiselect(
    "Allowed Relationship Channels",
    options=["Wiring Only", "Affiliate Only", "Upstream-Downstream Only", "Wiring + Affiliate", "Wiring + Upstream/Downstream", "All Sources"],
    default=["Wiring Only", "Affiliate Only", "Upstream-Downstream Only", "Wiring + Affiliate", "Wiring + Upstream/Downstream", "All Sources"]
)

exclude_missing_tax = st.sidebar.checkbox("Exclude Missing Tax IDs", value=False)
apply_hub_filter = st.sidebar.checkbox("Apply False-Relationship Filter (Hub Payees)", value=True)

# ---------------------------------------------------------
# Apply False Relationship Filter First
# ---------------------------------------------------------
if apply_hub_filter:
    filtered_wiring, false_rel_stats = apply_false_relationship_filter(
        data_dict["wiring_transactions"],
        data_dict["companies"],
        hub_payee_threshold=30
    )

    st.subheader("False-Relationship Screening (Hub Payee Utility/Shipping Filter)")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    f_col1.metric("Initial Transactions", f"{false_rel_stats['initial_tx_count']:,}")
    f_col2.metric("Filtered Transactions", f"{false_rel_stats['final_tx_count']:,}")
    f_col3.metric("Removed False Links", f"{false_rel_stats['removed_tx_count']:,}", delta_color="inverse")
    f_col4.metric("Hub Payees Identified", f"{false_rel_stats['identified_hub_payees_count']:,}")
    
    st.divider()

# ---------------------------------------------------------
# Apply Graph Filtering Rules
# ---------------------------------------------------------
subG, campaign_list_df = filter_network_graph(
    G=G,
    min_annual_tx=min_tx,
    min_total_amount=min_amount,
    min_core_capital=min_core_cap,
    allowed_rel_types=rel_types,
    exclude_missing_tax_id=exclude_missing_tax,
    companies_df=data_dict["companies"]
)

st.session_state["filtered_graph"] = subG
st.session_state["campaign_list_df"] = campaign_list_df

# Display Filtered Campaign Summary
st.subheader("Target Campaign Candidate List Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Qualified Companies", f"{len(campaign_list_df):,}")
c2.metric("Non-Bank Customers (List A)", f"{(~campaign_list_df['is_bank_customer']).sum():,}")
c3.metric("Existing Bank Customers", f"{campaign_list_df['is_bank_customer'].sum():,}")
c4.metric("Core Anchor Companies", f"{campaign_list_df['is_core_company'].sum():,}")

st.write("#### Qualified Campaign Target List Table")

# Format capital for display
display_df = campaign_list_df.copy()
display_df["registered_capital_usd"] = display_df["registered_capital"].apply(lambda x: f"${x:,.0f}")
display_df["status"] = np.where(display_df["is_bank_customer"], "Bank Customer", "Non-Customer (Prospect)")

st.dataframe(
    display_df[[
        "company_id", "name", "status", "industry", "registered_capital_usd",
        "is_core_company", "tax_id", "total_degree"
    ]],
    use_container_width=True
)

# Download CSV
csv_campaign = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Export Target Campaign List CSV",
    data=csv_campaign,
    file_name="supplychain_link_filtered_campaign_list.csv",
    mime="text/csv",
    type="primary"
)
