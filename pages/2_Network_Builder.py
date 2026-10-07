"""
Page 2: Network Builder & Iterative Graph Expansion
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import build_supply_chain_network

st.title("🌐 Page 2: Network Expansion Builder & Iterative Flow")
st.caption("Executes the 6-step iterative graph expansion flow described in the academic paper to discover non-customer supply chain prospects.")

data_dict = st.session_state["data"]

# Run Network Expansion Algorithm
with st.spinner("Executing 6-step iterative graph expansion algorithm..."):
    network_results = build_supply_chain_network(
        data_dict["companies"],
        data_dict["credit_reports"],
        data_dict["wiring_transactions"],
        data_dict["directors"]
    )
    st.session_state["network_results"] = network_results

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Initial Wiring Nodes (S0)", f"{network_results['initial_nodes_count']:,}")

with col2:
    st.metric("Expanded Total Nodes (Sk)", f"{network_results['final_nodes_count']:,}")

with col3:
    exp_factor = network_results["expansion_factor"]
    st.metric("Expansion Factor", f"{exp_factor:.2f}x", delta=f"Paper Benchmark: 5.4x")

with col4:
    list_a_len = len(network_results["list_a_non_customers"])
    st.metric("List A (Non-Customer Prospects)", f"{list_a_len:,}")

st.divider()

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Iterative Expansion Trace",
    "📋 List A (Non-Customer Prospect Pool)",
    "👔 Director Links & Ownership",
    "📖 Algorithm Academic Explanation"
])

# ---------------------------------------------------------
# Tab 1: Iterative Expansion Trace
# ---------------------------------------------------------
with tab1:
    st.subheader("Step-by-Step Expansion Trace Table")
    st.write("Tracks node discovery across credit report iteration cycles until network convergence.")

    step_stats = network_results["step_stats"]
    st.dataframe(step_stats, use_container_width=True)

    # Line chart of node accumulation
    st.write("#### Node Growth Trajectory")
    st.line_chart(step_stats.set_index("Iteration")[["Total_Nodes", "Bank_Customers", "Non_Customers"]])

# ---------------------------------------------------------
# Tab 2: List A (Non-Customer Prospect Pool)
# ---------------------------------------------------------
with tab2:
    st.subheader("List A: Discovered Non-Bank-Customer Companies")
    st.write("Companies identified through supply chain connections that do NOT currently hold accounts with the bank.")

    list_a_df = network_results["list_a_non_customers"]
    
    st.dataframe(
        list_a_df[["company_id", "name", "industry", "registered_capital", "is_core_company", "tax_id"]],
        use_container_width=True
    )

    csv_data = list_a_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download List A Prospects CSV",
        data=csv_data,
        file_name="supplychain_link_list_a_prospects.csv",
        mime="text/csv",
        type="primary"
    )

# ---------------------------------------------------------
# Tab 3: Director Links & Ownership
# ---------------------------------------------------------
with tab3:
    st.subheader("Director Common Ownership Network Links")
    st.write("Cross-company ownership connections identified via corporate board member registries (Step 6).")

    dir_df = network_results["director_links"]
    if not dir_df.empty:
        st.dataframe(dir_df, use_container_width=True)
    else:
        st.info("No director cross-ownership links identified in current sample dataset.")

# ---------------------------------------------------------
# Tab 4: Algorithm Academic Explanation
# ---------------------------------------------------------
with tab4:
    st.subheader("Academic Paper Algorithm Breakdown (Section 4.1)")
    
    with st.expander("Step 1: Initial Wiring Network Base"):
        st.markdown("""
        **Step 1**: Extract all unique senders and receivers from historical electronic bank wiring transaction logs ($S_0$).
        This forms the grounded empirical transaction foundation of active commercial relationships.
        """)

    with st.expander("Step 2: Bank Customer Verification"):
        st.markdown("""
        **Step 2**: Check internal Core Banking System (CBS) records to tag nodes as existing bank customers (`is_bank_customer = True`)
        or non-customers (`is_bank_customer = False`).
        """)

    with st.expander("Step 3 & 4: Credit Report Partner Expansion Loop"):
        st.markdown("""
        **Step 3 & 4**: For each node in the frontier $S_k$, query commercial credit reports to pull registered:
        - **Affiliated Companies** (Parents, Subsidiaries, Group Entities)
        - **Upstream / Downstream Partners** (Suppliers and Commercial Buyers)
        Append newly discovered entities into $S_{k+1}$.
        """)

    with st.expander("Step 5: Iterative Convergence"):
        st.markdown("""
        **Step 5**: Repeat the expansion query loop until no new company entities are discovered ($S_{k+1} = S_k$), signifying complete network saturation.
        """)

    with st.expander("Step 6: Director Cross-Ownership Lookup"):
        st.markdown("""
        **Step 6**: Cross-reference corporate board members against personal banking records and company ownership disclosures to link sister companies owned by common directors.
        """)
