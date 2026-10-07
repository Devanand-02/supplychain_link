"""
Unit Tests for SupplyChain-Link Analytics & Machine Learning Pipeline
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen import generate_synthetic_supply_chain_data, validate_uploaded_csv
from src.network import build_supply_chain_network, apply_false_relationship_filter, filter_network_graph, compute_network_analytics
from src.association import mine_product_association_rules
from src.models import train_propensity_model, train_approval_model, perform_customer_segmentation
from src.fraud import benford_law_analysis
from src.evaluation import evaluate_campaign_performance, compute_lift_and_deciles

@pytest.fixture(scope="module")
def sample_data():
    """Generates a small synthetic dataset for testing."""
    return generate_synthetic_supply_chain_data(n_companies=200, random_state=42)

def test_data_generation(sample_data):
    assert "companies" in sample_data
    assert "credit_reports" in sample_data
    assert "wiring_transactions" in sample_data
    assert "directors" in sample_data
    assert "campaign_outcomes" in sample_data

    companies_df = sample_data["companies"]
    assert len(companies_df) == 200
    assert "company_id" in companies_df.columns
    assert "registered_capital" in companies_df.columns

def test_csv_validation(sample_data):
    valid, errors = validate_uploaded_csv(sample_data["companies"], "companies")
    assert valid
    assert len(errors) == 0

    invalid_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    valid_inv, errors_inv = validate_uploaded_csv(invalid_df, "companies")
    assert not valid_inv
    assert len(errors_inv) > 0

def test_network_expansion(sample_data):
    res = build_supply_chain_network(
        sample_data["companies"],
        sample_data["credit_reports"],
        sample_data["wiring_transactions"],
        sample_data["directors"]
    )
    assert "graph" in res
    assert "expansion_factor" in res
    assert res["expansion_factor"] >= 1.0
    assert not res["step_stats"].empty

def test_false_relationship_filter(sample_data):
    filtered_df, stats = apply_false_relationship_filter(
        sample_data["wiring_transactions"],
        sample_data["companies"],
        hub_payee_threshold=5
    )
    assert "initial_tx_count" in stats
    assert "final_tx_count" in stats
    assert len(filtered_df) <= len(sample_data["wiring_transactions"])

def test_association_rules(sample_data):
    res = mine_product_association_rules(sample_data["companies"], min_support=0.01, min_confidence=0.05)
    assert "rules" in res
    assert "basket_matrix" in res

def test_propensity_model(sample_data):
    net_res = build_supply_chain_network(
        sample_data["companies"],
        sample_data["credit_reports"],
        sample_data["wiring_transactions"],
        sample_data["directors"]
    )
    analytics = compute_network_analytics(net_res["graph"], sample_data["companies"])

    res = train_propensity_model(
        sample_data["companies"],
        sample_data["campaign_outcomes"],
        analytics["centrality_df"]
    )
    assert "auc" in res
    assert res["auc"] >= 0.5
    assert "odds_ratios_df" in res

def test_benford_law(sample_data):
    res = benford_law_analysis(sample_data["wiring_transactions"])
    assert "mad" in res
    assert "conformity" in res
    assert "chi2_stat" in res

def test_campaign_evaluation(sample_data):
    res = evaluate_campaign_performance(sample_data["campaign_outcomes"])
    assert "summary_df" in res
    assert "pct_resp_loan" in res
    assert not res["summary_df"].empty
