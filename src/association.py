"""
Association Rule Mining Module (Apriori & Cross-Selling Insights)
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from typing import Dict, Any, Tuple, List

def mine_product_association_rules(
    companies_df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.15,
    min_lift: float = 1.1
) -> Dict[str, Any]:
    """
    Mines association rules for bank product holdings baskets using Apriori.
    Identifies cross-sell rules like {checking, e-wiring} => {corporate loan}.

    Returns:
        Dict[str, Any]: {
            'rules': pd.DataFrame,
            'frequent_itemsets': pd.DataFrame,
            'scatter_fig': go.Figure,
            'basket_matrix': pd.DataFrame
        }
    """
    # Filter to existing bank customers with product holdings
    cust_df = companies_df[companies_df["is_bank_customer"] & (companies_df["products_held"].str.len() > 0)].copy()

    if len(cust_df) == 0:
        return {
            "rules": pd.DataFrame(),
            "frequent_itemsets": pd.DataFrame(),
            "scatter_fig": go.Figure(),
            "basket_matrix": pd.DataFrame()
        }

    # Extract all unique products
    all_products = set()
    for prods_str in cust_df["products_held"]:
        prods = [p.strip() for p in str(prods_str).split(",") if p.strip()]
        all_products.update(prods)

    product_list = sorted(list(all_products))

    # One-hot encoding matrix
    basket_records = []
    for prods_str in cust_df["products_held"]:
        prods = set([p.strip() for p in str(prods_str).split(",") if p.strip()])
        record = {prod: (prod in prods) for prod in product_list}
        basket_records.append(record)

    basket_df = pd.DataFrame(basket_records)

    # Run Apriori
    try:
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return {
                "rules": pd.DataFrame(),
                "frequent_itemsets": pd.DataFrame(),
                "scatter_fig": go.Figure(),
                "basket_matrix": basket_df
            }

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Filter by min_lift
        rules = rules[rules["lift"] >= min_lift].copy()

        # Format antecedents and consequents as readable strings
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules["rule_str"] = rules["antecedents_str"] + "  ==>  " + rules["consequents_str"]

        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    except Exception:
        return {
            "rules": pd.DataFrame(),
            "frequent_itemsets": pd.DataFrame(),
            "scatter_fig": go.Figure(),
            "basket_matrix": basket_df
        }

    # Build Interactive Scatter Plot (Support vs Confidence colored by Lift)
    if not rules.empty:
        scatter_fig = px.scatter(
            rules,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_name="rule_str",
            hover_data={"support": ":.3f", "confidence": ":.3f", "lift": ":.3f"},
            color_continuous_scale="Viridis",
            title="Association Rules: Support vs Confidence (Size & Color = Lift)"
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Support",
            yaxis_title="Confidence"
        )
    else:
        scatter_fig = go.Figure()

    return {
        "rules": rules,
        "frequent_itemsets": frequent_itemsets,
        "scatter_fig": scatter_fig,
        "basket_matrix": basket_df
    }

def mine_relationship_combination_rules(
    campaign_df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.10,
    min_lift: float = 1.0
) -> pd.DataFrame:
    """
    Mines association rules connecting relationship-type combinations to campaign outcomes
    (Replicating paper's Table 2 analysis).
    """
    records = []
    for _, row in campaign_df.iterrows():
        rel = str(row.get("relationship_type", ""))
        rec = {
            "Has_Affiliate": "Affiliate" in rel,
            "Has_Upstream_Downstream": ("Upstream" in rel) or ("Downstream" in rel),
            "Has_Wiring": "Wiring" in rel,
            "Responded_Loan": bool(row.get("responded_loan", False)),
            "Approved": bool(row.get("approved", False)),
            "Responded_Other": bool(row.get("response_other_services", False))
        }
        records.append(rec)

    df_matrix = pd.DataFrame(records)

    try:
        freq = apriori(df_matrix, min_support=min_support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        rules = rules[rules["lift"] >= min_lift].copy()

        # Keep rules targeting campaign outcomes
        outcome_rules = rules[
            rules["consequents"].apply(lambda c: any(item in c for item in ["Responded_Loan", "Approved", "Responded_Other"]))
        ].copy()

        outcome_rules["antecedents_str"] = outcome_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        outcome_rules["consequents_str"] = outcome_rules["consequents"].apply(lambda x: ", ".join(list(x)))
        outcome_rules["rule_str"] = outcome_rules["antecedents_str"] + " ==> " + outcome_rules["consequents_str"]

        return outcome_rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def generate_rule_interpretations(rules_df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Translates top association rules into plain-English cross-sell insights.
    """
    if rules_df.empty:
        return ["No association rules found under the current parameter thresholds."]

    interpretations = []
    for idx, row in rules_df.head(top_n).iterrows():
        ant = row.get("antecedents_str", "")
        cons = row.get("consequents_str", "")
        supp = row.get("support", 0.0)
        conf = row.get("confidence", 0.0)
        lift = row.get("lift", 0.0)

        text = (
            f"**Rule #{idx+1}**: Customers holding **{{{ant}}}** are **{conf*100:.1f}% likely** "
            f"to adopt **{{{cons}}}** (Lift: **{lift:.2f}x**, Support: **{supp*100:.1f}%**). "
            f"--> *Action*: Target customers using {{{ant}}} with direct marketing for {{{cons}}}."
        )
        interpretations.append(text)

    return interpretations
