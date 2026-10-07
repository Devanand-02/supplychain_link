"""
Page 4: Network Explorer & Graph Analytics
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import (
    build_supply_chain_network,
    compute_network_analytics,
    get_networkx_plotly_figure,
    generate_pyvis_html
)

st.title("🕸️ Page 4: Network Explorer & Graph Analytics")
st.caption("Explore network topology, node centralities (PageRank, Betweenness), modularity communities, and anchor hub companies.")

data_dict = st.session_state["data"]

# Obtain graph
if "filtered_graph" in st.session_state and len(st.session_state["filtered_graph"]) > 0:
    G = st.session_state["filtered_graph"]
    st.info("Displaying Sidebar-Filtered Network Graph.")
elif "network_results" in st.session_state:
    G = st.session_state["network_results"]["graph"]
else:
    net_res = build_supply_chain_network(
        data_dict["companies"],
        data_dict["credit_reports"],
        data_dict["wiring_transactions"],
        data_dict["directors"]
    )
    st.session_state["network_results"] = net_res
    G = net_res["graph"]

# Compute Analytics
analytics = compute_network_analytics(G, data_dict["companies"])
st.session_state["network_analytics"] = analytics

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Interactive Graph Visualizer",
    "📊 Centrality Metrics Table",
    "🧱 Modularity Communities",
    "⚓ Anchor Hub Companies (SCF Targets)"
])

# ---------------------------------------------------------
# Tab 1: Interactive Graph Visualizer
# ---------------------------------------------------------
with tab1:
    st.subheader("Network Topology Visualization")
    st.markdown("""
    **Legend**: 
    - <span style="color:#2ECC71; font-weight:bold;">● Green Nodes</span>: Existing Bank Customers
    - <span style="color:#8E44AD; font-weight:bold;">● Purple Nodes</span>: Non-Customers (List A Prospect Pool)
    """, unsafe_allow_html=True)

    viz_mode = st.radio("Visualization Engine", options=["Plotly Interactive 2D", "Pyvis Dynamic Physics HTML"], horizontal=True)

    if viz_mode == "Plotly Interactive 2D":
        fig = get_networkx_plotly_figure(G, max_nodes=250)
        st.plotly_chart(fig, use_container_width=True)
    else:
        with st.spinner("Rendering Pyvis physics graph..."):
            pyvis_html = generate_pyvis_html(G, max_nodes=150)
            components.html(pyvis_html, height=520, scrolling=False)

# ---------------------------------------------------------
# Tab 2: Centrality Metrics Table
# ---------------------------------------------------------
with tab2:
    st.subheader("Node Centrality & PageRank Table")
    st.write("Measures structural importance and transaction flow centrality across the supply chain network.")

    cent_df = analytics["centrality_df"]
    st.dataframe(
        cent_df.sort_values("pagerank", ascending=False),
        use_container_width=True
    )

# ---------------------------------------------------------
# Tab 3: Modularity Communities
# ---------------------------------------------------------
with tab3:
    st.subheader("Community Detection (Greedy Modularity)")
    st.write(f"Identified **{analytics['communities_count']} distinct supply chain communities/clusters**.")

    comm_dist = cent_df.groupby("community_id").size().reset_index(name="Member_Count")
    st.bar_chart(comm_dist.set_index("community_id"))

# ---------------------------------------------------------
# Tab 4: Anchor Hub Companies
# ---------------------------------------------------------
with tab4:
    st.subheader("Anchor Hub Companies (Supply Chain Finance Targets)")
    st.write("Hub companies with high counterparty degrees and registered capital >= $10M USD identified as prime candidates for Supply Chain Finance (SCF) anchor programs.")

    hubs_df = analytics["hub_companies"]
    if not hubs_df.empty:
        st.dataframe(hubs_df, use_container_width=True)
    else:
        st.info("No hub companies meeting threshold in current sub-graph.")
