"""
Relationship Network Construction, Expansion, Filtering, and Analytics Module
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyvis.network import Network
from typing import Dict, List, Tuple, Any, Optional

def build_supply_chain_network(
    companies_df: pd.DataFrame,
    credit_reports_df: pd.DataFrame,
    wiring_transactions_df: pd.DataFrame,
    directors_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Implements the 6-step iterative expansion algorithm described in the paper:
    1. Start from initial wiring network senders/receivers.
    2. Check bank customer status.
    3. Pull affiliated and upstream/downstream partners from credit reports.
    4. Append newly discovered entities to the frontier.
    5. Loop until no new companies appear (convergence).
    6. Cross-reference directors and companies they own.

    Returns:
        Dict[str, Any]: {
            'graph': nx.DiGraph,
            'expansion_factor': float,
            'step_stats': pd.DataFrame,
            'list_a_non_customers': pd.DataFrame,
            'director_links': pd.DataFrame
        }
    """
    # Create lookup maps
    bank_customer_map = companies_df.set_index("company_id")["is_bank_customer"].to_dict()
    company_info_map = companies_df.set_index("company_id").to_dict(orient="index")

    # Credit report mapping
    credit_map = credit_reports_df.set_index("company_id").to_dict(orient="index")

    # Initial wiring graph nodes (Step 1)
    wiring_senders = set(wiring_transactions_df["sender_id"])
    wiring_receivers = set(wiring_transactions_df["receiver_id"])
    initial_nodes = wiring_senders.union(wiring_receivers)

    current_nodes = set(initial_nodes)
    visited_nodes = set()
    step_records = []
    iteration = 1

    # Record initial step
    step_records.append({
        "Iteration": 0,
        "Step_Description": "Initial E-Wiring Network",
        "Total_Nodes": len(current_nodes),
        "New_Nodes_Added": len(current_nodes),
        "Bank_Customers": sum(1 for n in current_nodes if bank_customer_map.get(n, False)),
        "Non_Customers": sum(1 for n in current_nodes if not bank_customer_map.get(n, False))
    })

    # Iterative expansion loop (Steps 2 - 5)
    while True:
        unvisited = current_nodes - visited_nodes
        if not unvisited:
            break

        new_frontier = set()
        for node in unvisited:
            visited_nodes.add(node)
            c_info = credit_map.get(node, {})
            
            # Extract affiliates
            aff_str = str(c_info.get("affiliated_companies", ""))
            if aff_str and aff_str != "nan":
                affs = [a.strip() for a in aff_str.split(",") if a.strip()]
                new_frontier.update(affs)

            # Extract upstream/downstream partners
            updn_str = str(c_info.get("upstream_downstream_companies", ""))
            if updn_str and updn_str != "nan":
                updns = [u.strip() for u in updn_str.split(",") if u.strip()]
                new_frontier.update(updns)

        # Filter out nodes not in our master dataset
        valid_new = {n for n in new_frontier if n in bank_customer_map and n not in current_nodes}

        if not valid_new:
            break # Convergence reached

        current_nodes.update(valid_new)

        step_records.append({
            "Iteration": iteration,
            "Step_Description": f"Credit Report Expansion Iteration {iteration}",
            "Total_Nodes": len(current_nodes),
            "New_Nodes_Added": len(valid_new),
            "Bank_Customers": sum(1 for n in current_nodes if bank_customer_map.get(n, False)),
            "Non_Customers": sum(1 for n in current_nodes if not bank_customer_map.get(n, False))
        })
        iteration += 1

    # Step 6: Cross-reference directors and companies they own
    director_links_records = []
    director_owned_nodes = set()

    if not directors_df.empty:
        dir_grouped = directors_df.groupby("person_id")
        for p_id, group in dir_grouped:
            person_name = group["person_name"].iloc[0]
            is_personal_cust = group["is_personal_bank_customer"].iloc[0]
            
            comp_list = []
            for _, row in group.iterrows():
                comp_list.append(row["company_id"])
                owned_str = str(row.get("owns_other_companies", ""))
                if owned_str and owned_str != "nan":
                    owned_ids = [o.strip() for o in owned_str.split(",") if o.strip()]
                    comp_list.extend(owned_ids)

            valid_comp_list = list(set(c for c in comp_list if c in bank_customer_map))
            if len(valid_comp_list) > 1:
                for c1 in valid_comp_list:
                    for c2 in valid_comp_list:
                        if c1 != c2 and (c1 in current_nodes or c2 in current_nodes):
                            director_owned_nodes.add(c1)
                            director_owned_nodes.add(c2)
                            director_links_records.append({
                                "person_id": p_id,
                                "person_name": person_name,
                                "is_personal_bank_customer": is_personal_cust,
                                "company_1": c1,
                                "company_2": c2
                            })

    current_nodes.update(director_owned_nodes)

    step_records.append({
        "Iteration": iteration,
        "Step_Description": "Director Common Ownership Lookup",
        "Total_Nodes": len(current_nodes),
        "New_Nodes_Added": len(director_owned_nodes - visited_nodes),
        "Bank_Customers": sum(1 for n in current_nodes if bank_customer_map.get(n, False)),
        "Non_Customers": sum(1 for n in current_nodes if not bank_customer_map.get(n, False))
    })

    # Construct complete Directed NetworkX Graph
    G = nx.DiGraph()

    # Add node attributes
    for node_id in current_nodes:
        info = company_info_map.get(node_id, {})
        G.add_node(
            node_id,
            name=info.get("name", node_id),
            industry=info.get("industry", "Unknown"),
            registered_capital=info.get("registered_capital", 0.0),
            is_bank_customer=info.get("is_bank_customer", False),
            is_core_company=info.get("is_core_company", False),
            tenure_years=info.get("tenure_years", 0.0),
            e_wiring_user=info.get("e_wiring_user", False),
            tax_id=info.get("tax_id", None)
        )

    # Add wiring transaction edges
    wiring_filtered = wiring_transactions_df[
        wiring_transactions_df["sender_id"].isin(current_nodes) &
        wiring_transactions_df["receiver_id"].isin(current_nodes)
    ]

    edge_weights = wiring_filtered.groupby(["sender_id", "receiver_id"]).agg(
        frequency=("amount", "count"),
        total_amount=("amount", "sum")
    ).reset_index()

    for _, row in edge_weights.iterrows():
        G.add_edge(
            row["sender_id"],
            row["receiver_id"],
            weight=float(row["total_amount"]),
            frequency=int(row["frequency"]),
            edge_type="Wiring"
        )

    # Add credit report affiliate & upstream/downstream edges
    for _, row in credit_reports_df.iterrows():
        cid = row["company_id"]
        if cid in G:
            # Affiliates
            aff_str = str(row.get("affiliated_companies", ""))
            if aff_str and aff_str != "nan":
                for aff in aff_str.split(","):
                    aff = aff.strip()
                    if aff in G and aff != cid:
                        if not G.has_edge(cid, aff):
                            G.add_edge(cid, aff, weight=1.0, frequency=1, edge_type="Affiliate")
            
            # Upstream / Downstream
            updn_str = str(row.get("upstream_downstream_companies", ""))
            if updn_str and updn_str != "nan":
                for updn in updn_str.split(","):
                    updn = updn.strip()
                    if updn in G and updn != cid:
                        if not G.has_edge(cid, updn):
                            G.add_edge(cid, updn, weight=1.0, frequency=1, edge_type="Upstream/Downstream")

    # Calculate expansion metrics
    initial_count = max(1, len(initial_nodes))
    final_count = len(current_nodes)
    expansion_factor = round(final_count / initial_count, 2)

    step_stats = pd.DataFrame(step_records)

    # Extract List A: Non-bank-customer prospect list
    non_customer_ids = [n for n in current_nodes if not bank_customer_map.get(n, False)]
    list_a_df = companies_df[companies_df["company_id"].isin(non_customer_ids)].copy()

    director_links_df = pd.DataFrame(director_links_records)

    return {
        "graph": G,
        "expansion_factor": expansion_factor,
        "step_stats": step_stats,
        "list_a_non_customers": list_a_df,
        "director_links": director_links_df,
        "initial_nodes_count": len(initial_nodes),
        "final_nodes_count": final_count
    }

def apply_false_relationship_filter(
    wiring_df: pd.DataFrame,
    companies_df: pd.DataFrame,
    hub_payee_threshold: int = 50
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies and filters non-supply-chain wiring transactions (false relationships)
    such as payments to utility or shipping hub payees receiving transactions from >50 unique senders
    with industry mismatches.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Filtered wiring dataframe and comparison statistics.
    """
    # Calculate payee unique sender counts
    payee_counts = wiring_df.groupby("receiver_id")["sender_id"].nunique().to_dict()
    
    # Identify utility/shipping hubs
    shipping_utility_ids = set(companies_df[companies_df["industry"].isin(["Shipping", "Utilities"])]["company_id"])
    hub_payees = {cid for cid, count in payee_counts.items() if count >= hub_payee_threshold and cid in shipping_utility_ids}

    initial_tx_count = len(wiring_df)
    initial_amount = wiring_df["amount"].sum()

    # Filter out false relationship edges (payments from non-matching industries to hub payees)
    filtered_df = wiring_df[
        ~((wiring_df["receiver_id"].isin(hub_payees)) & (~wiring_df["industry_match"]))
    ].copy()

    final_tx_count = len(filtered_df)
    final_amount = filtered_df["amount"].sum()

    stats = {
        "initial_tx_count": initial_tx_count,
        "final_tx_count": final_tx_count,
        "removed_tx_count": initial_tx_count - final_tx_count,
        "initial_amount_usd": round(initial_amount, 2),
        "final_amount_usd": round(final_amount, 2),
        "removed_amount_usd": round(initial_amount - final_amount, 2),
        "identified_hub_payees_count": len(hub_payees)
    }

    return filtered_df, stats

def filter_network_graph(
    G: nx.DiGraph,
    min_annual_tx: int = 8,
    min_total_amount: float = 13500.0,
    min_core_capital: float = 50000000.0,
    allowed_rel_types: Optional[List[str]] = None,
    exclude_missing_tax_id: bool = False,
    companies_df: Optional[pd.DataFrame] = None
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Applies sidebar filtering criteria to narrow down the network graph and produce the targeting campaign list.

    Returns:
        Tuple[nx.DiGraph, pd.DataFrame]: Filtered graph and target campaign candidate table.
    """
    if allowed_rel_types is None:
        allowed_rel_types = ["Wiring Only", "Affiliate Only", "Upstream-Downstream Only", "Wiring", "Affiliate", "Upstream/Downstream"]

    subG = G.copy()

    # Filter edges based on wiring frequency and amount thresholds
    edges_to_remove = []
    for u, v, data in subG.edges(data=True):
        if data.get("edge_type") == "Wiring":
            freq = data.get("frequency", 1)
            amt = data.get("weight", 0.0)
            if freq < min_annual_tx or amt < min_total_amount:
                edges_to_remove.append((u, v))

    subG.remove_edges_from(edges_to_remove)

    # Filter nodes based on criteria
    nodes_to_remove = []
    for node, data in subG.nodes(data=True):
        # Exclude missing tax ID if requested
        if exclude_missing_tax_id and (pd.isna(data.get("tax_id")) or not data.get("tax_id")):
            nodes_to_remove.append(node)

    subG.remove_nodes_from(nodes_to_remove)
    
    # Remove isolated nodes after edge filtering
    isolated = [n for n, deg in subG.degree() if deg == 0]
    subG.remove_nodes_from(isolated)

    # Build campaign candidate dataframe
    node_records = []
    for n, d in subG.nodes(data=True):
        in_deg = subG.in_degree(n)
        out_deg = subG.out_degree(n)
        tot_deg = in_deg + out_deg

        node_records.append({
            "company_id": n,
            "name": d.get("name", n),
            "industry": d.get("industry", "Unknown"),
            "registered_capital": d.get("registered_capital", 0.0),
            "is_bank_customer": d.get("is_bank_customer", False),
            "is_core_company": d.get("is_core_company", False),
            "tenure_years": d.get("tenure_years", 0.0),
            "e_wiring_user": d.get("e_wiring_user", False),
            "tax_id": d.get("tax_id", "Missing"),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": tot_deg
        })

    campaign_list_df = pd.DataFrame(node_records)

    return subG, campaign_list_df

def compute_network_analytics(G: nx.DiGraph, companies_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes centrality metrics, connected components, community detection, and identifies hub companies.

    Metrics:
        - Degree Centrality
        - Weighted Degree (In/Out-strength)
        - PageRank
        - Betweenness Centrality
        - Community Detection (Greedy Modularity)
        - Hub Companies (> 100 counterparties and >= 10M registered capital)
    """
    if len(G) == 0:
        return {
            "centrality_df": pd.DataFrame(),
            "communities_count": 0,
            "hub_companies": pd.DataFrame()
        }

    # Centrality metrics
    deg_cent = nx.degree_centrality(G)

    try:
        pagerank = nx.pagerank(G, weight="weight", max_iter=100)
    except Exception:
        pagerank = {n: 1.0 / len(G) for n in G.nodes()}

    # Betweenness on sampled nodes if graph is large
    if len(G) > 1000:
        betweenness = nx.betweenness_centrality(G, k=100, seed=42)
    else:
        betweenness = nx.betweenness_centrality(G)

    # In/Out strength (weighted degree)
    in_strength = {n: sum(d.get("weight", 0.0) for _, _, d in G.in_edges(n, data=True)) for n in G.nodes()}
    out_strength = {n: sum(d.get("weight", 0.0) for _, _, d in G.out_edges(n, data=True)) for n in G.nodes()}

    # Community detection (Greedy modularity on undirected version)
    undirected_G = G.to_undirected()
    try:
        communities = list(nx.community.greedy_modularity_communities(undirected_G))
        community_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_map[node] = idx + 1
    except Exception:
        community_map = {n: 1 for n in G.nodes()}
        communities = [set(G.nodes())]

    records = []
    for n, d in G.nodes(data=True):
        records.append({
            "company_id": n,
            "name": d.get("name", n),
            "industry": d.get("industry", "Unknown"),
            "registered_capital": d.get("registered_capital", 0.0),
            "is_bank_customer": d.get("is_bank_customer", False),
            "is_core_company": d.get("is_core_company", False),
            "degree": G.degree(n),
            "degree_centrality": round(deg_cent.get(n, 0.0), 5),
            "pagerank": round(pagerank.get(n, 0.0), 5),
            "betweenness_centrality": round(betweenness.get(n, 0.0), 5),
            "in_strength_usd": round(in_strength.get(n, 0.0), 2),
            "out_strength_usd": round(out_strength.get(n, 0.0), 2),
            "total_strength_usd": round(in_strength.get(n, 0.0) + out_strength.get(n, 0.0), 2),
            "community_id": community_map.get(n, 1)
        })

    centrality_df = pd.DataFrame(records)

    # Identify Hub Companies (> 10 counterparties or high degree + registered capital >= 10M USD)
    # Threshold scaled based on sampled graph size
    deg_thresh = 15 if len(G) > 200 else 6
    hubs_df = centrality_df[
        (centrality_df["degree"] >= deg_thresh) &
        (centrality_df["registered_capital"] >= 10_000_000.0)
    ].sort_values("pagerank", ascending=False)

    return {
        "centrality_df": centrality_df,
        "communities_count": len(communities),
        "hub_companies": hubs_df,
        "total_nodes": len(G),
        "total_edges": G.number_of_edges()
    }

def get_networkx_plotly_figure(G: nx.DiGraph, max_nodes: int = 250) -> go.Figure:
    """
    Renders an interactive Plotly 2D Network Graph.
    Nodes colored: Green = Bank Customers, Purple = Non-Customers Prospect Pool.
    """
    if len(G) > max_nodes:
        # Sample top nodes by degree for visualization
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        subG = G.subgraph(top_nodes)
    else:
        subG = G

    if len(subG) == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty Network Graph")
        return fig

    pos = nx.spring_layout(subG, k=0.25, seed=42)

    # Edges trace
    edge_x = []
    edge_y = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.6, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Customer Nodes (Green) vs Non-Customer Nodes (Purple)
    cust_x, cust_y, cust_text, cust_size = [], [], [], []
    non_cust_x, non_cust_y, non_cust_text, non_cust_size = [], [], [], []

    for node in subG.nodes():
        x, y = pos[node]
        d = subG.nodes[node]
        name = d.get("name", node)
        ind = d.get("industry", "N/A")
        cap = d.get("registered_capital", 0.0)
        is_cust = d.get("is_bank_customer", False)
        deg = subG.degree(node)

        hover = (
            f"<b>{name}</b> ({node})<br>"
            f"Status: {'Bank Customer' if is_cust else 'Non-Customer (Prospect)'}<br>"
            f"Industry: {ind}<br>"
            f"Capital: ${cap:,.0f}<br>"
            f"Connections: {deg}"
        )

        size = max(10, min(30, deg * 2.5))

        if is_cust:
            cust_x.append(x)
            cust_y.append(y)
            cust_text.append(hover)
            cust_size.append(size)
        else:
            non_cust_x.append(x)
            non_cust_y.append(y)
            non_cust_text.append(hover)
            non_cust_size.append(size)

    cust_trace = go.Scatter(
        x=cust_x, y=cust_y,
        mode="markers",
        hoverinfo="text",
        text=cust_text,
        name="Bank Customers",
        marker=dict(
            size=cust_size,
            color="#2ECC71", # Emerald Green
            line=dict(width=1, color="#1E8449")
        )
    )

    non_cust_trace = go.Scatter(
        x=non_cust_x, y=non_cust_y,
        mode="markers",
        hoverinfo="text",
        text=non_cust_text,
        name="Non-Customers (Target List A)",
        marker=dict(
            size=non_cust_size,
            color="#8E44AD", # Purple
            line=dict(width=1, color="#5B2C6F")
        )
    )

    fig = go.Figure(
        data=[edge_trace, cust_trace, non_cust_trace],
        layout=go.Layout(
            title=f"Supply Chain Relationship Graph (Displaying Top {len(subG)} Nodes)",
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
        )
    )

    return fig

def generate_pyvis_html(G: nx.DiGraph, max_nodes: int = 150) -> str:
    """
    Generates an HTML string for an interactive Pyvis network graph widget.
    """
    if len(G) > max_nodes:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:max_nodes]
        subG = G.subgraph(top_nodes)
    else:
        subG = G

    net = Network(height="500px", width="100%", bgcolor="#0F172A", font_color="white", directed=True)

    for n, d in subG.nodes(data=True):
        is_cust = d.get("is_bank_customer", False)
        color = "#2ECC71" if is_cust else "#8E44AD"
        name = d.get("name", n)
        ind = d.get("industry", "Unknown")
        label = f"{name}\n({ind})"

        title = f"ID: {n}\nName: {name}\nIndustry: {ind}\nStatus: {'Customer' if is_cust else 'Prospect'}"
        net.add_node(n, label=label, title=title, color=color, size=15)

    for u, v, d in subG.edges(data=True):
        net.add_edge(u, v, title=f"Type: {d.get('edge_type', 'Link')}")

    net.toggle_physics(True)
    return net.generate_html()
