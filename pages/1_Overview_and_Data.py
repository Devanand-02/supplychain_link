"""
Page 1: Overview & Data Control Center
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen import (
    generate_synthetic_supply_chain_data,
    validate_uploaded_csv,
    get_schema_guide
)

st.title("📊 Page 1: Data Control Center & Schema Validation")
st.caption("Configure synthetic data generation parameters or upload real corporate banking CSV datasets.")

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Synthetic Data Generator",
    "📁 Upload CSV Datasets",
    "📖 Schema Guide",
    "🔍 Data Quality Audit"
])

# ---------------------------------------------------------
# Tab 1: Synthetic Data Generator
# ---------------------------------------------------------
with tab1:
    st.subheader("Synthetic Dataset Generation Settings")
    st.write("Generate benchmark supply chain datasets reflecting the academic paper's distributional parameters.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_comp = st.slider("Total Baseline Companies (n_companies)", min_value=500, max_value=20000, value=st.session_state.get("n_companies", 2500), step=500)
    with col2:
        rand_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
    with col3:
        st.write("<br>", unsafe_allow_html=True)
        btn_gen = st.button("🔄 Regenerate Synthetic Data", type="primary", use_container_width=True)

    if btn_gen:
        with st.spinner("Generating new synthetic supply chain network..."):
            st.session_state["n_companies"] = n_comp
            st.session_state["random_state"] = rand_seed
            st.session_state["data"] = generate_synthetic_supply_chain_data(n_companies=n_comp, random_state=rand_seed)
            st.session_state["data_source"] = f"Synthetic (n={n_comp:,}, seed={rand_seed})"
            st.success(f"Successfully generated data for {n_comp:,} companies!")
            st.rerun()

    st.divider()
    st.write("### Current Active Data Summary")
    data_dict = st.session_state["data"]
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Companies", f"{len(data_dict['companies']):,}")
    c2.metric("Credit Reports", f"{len(data_dict['credit_reports']):,}")
    c3.metric("Wiring Tx Count", f"{len(data_dict['wiring_transactions']):,}")
    c4.metric("Directors Recorded", f"{len(data_dict['directors']):,}")
    c5.metric("Campaign Records", f"{len(data_dict['campaign_outcomes']):,}")

# ---------------------------------------------------------
# Tab 2: Upload CSV Datasets
# ---------------------------------------------------------
with tab2:
    st.subheader("Upload Custom Corporate Banking CSVs")
    st.write("Upload custom CSV files matching the expected schema. The system validates required columns automatically.")

    uploaded_files = {}
    u_col1, u_col2 = st.columns(2)

    with u_col1:
        uploaded_files["companies"] = st.file_uploader("Upload Companies CSV", type=["csv"], key="u_comp")
        uploaded_files["credit_reports"] = st.file_uploader("Upload Credit Reports CSV", type=["csv"], key="u_cred")
        uploaded_files["wiring_transactions"] = st.file_uploader("Upload Wiring Transactions CSV", type=["csv"], key="u_wire")

    with u_col2:
        uploaded_files["directors"] = st.file_uploader("Upload Directors CSV", type=["csv"], key="u_dir")
        uploaded_files["campaign_outcomes"] = st.file_uploader("Upload Campaign Outcomes CSV", type=["csv"], key="u_camp")

    if st.button("Apply Uploaded CSV Datasets", type="primary"):
        new_data = {}
        all_valid = True

        for schema_name, file_obj in uploaded_files.items():
            if file_obj is not None:
                df_up = pd.read_csv(file_obj)
                is_valid, errors = validate_uploaded_csv(df_up, schema_name)
                if not is_valid:
                    st.error(f"Error in uploaded file for {schema_name}: {errors[0]}")
                    all_valid = False
                else:
                    new_data[schema_name] = df_up
            else:
                new_data[schema_name] = st.session_state["data"][schema_name]

        if all_valid and any(file_obj is not None for file_obj in uploaded_files.values()):
            st.session_state["data"] = new_data
            st.session_state["data_source"] = "Custom CSV Upload"
            st.success("Successfully validated and loaded custom CSV datasets!")
            st.rerun()

# ---------------------------------------------------------
# Tab 3: Schema Guide
# ---------------------------------------------------------
with tab3:
    st.subheader("Interactive Schema Guide & Field Requirements")
    guide = get_schema_guide()

    for table_name, field_list in guide.items():
        with st.expander(f"Table: {table_name.upper()} ({len(field_list)} fields)"):
            for field in field_list:
                st.markdown(f"- `{field}`")

# ---------------------------------------------------------
# Tab 4: Data Quality Audit
# ---------------------------------------------------------
with tab4:
    st.subheader("Automated Data Quality Audit")
    
    data_dict = st.session_state["data"]
    comp_df = data_dict["companies"]
    wiring_df = data_dict["wiring_transactions"]

    st.write("#### Companies Table Missing Value Audit")
    missing_df = comp_df.isnull().sum().reset_index()
    missing_df.columns = ["Column Name", "Missing Count"]
    missing_df["Missing Percentage"] = (missing_df["Missing Count"] / len(comp_df) * 100).round(2).astype(str) + "%"
    
    st.dataframe(missing_df, use_container_width=True)

    st.divider()
    st.write("#### Industry & Bank Customer Distribution")
    ind_dist = comp_df.groupby(["industry", "is_bank_customer"]).size().unstack(fill_value=0)
    ind_dist.columns = ["Non-Customer", "Bank Customer"]
    st.bar_chart(ind_dist)
