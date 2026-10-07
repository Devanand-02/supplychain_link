"""
Fraud Screening & Transaction Anomaly Detection Module (Benford's Law & Isolation Forest)
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px

def benford_law_analysis(wiring_df: pd.DataFrame, amount_col: str = "amount") -> Dict[str, Any]:
    """
    Performs Benford's Law First-Digit Analysis on e-wiring transaction amounts.
    Calculates Chi-Square statistic, p-value, and Mean Absolute Deviation (MAD).

    Benford's Law Probability: P(d) = log10(1 + 1/d) for d in [1..9].
    """
    if wiring_df.empty or amount_col not in wiring_df.columns:
        return {"error": "Invalid or empty transaction dataframe."}

    # Filter non-zero amounts
    amounts = wiring_df[wiring_df[amount_col] > 0][amount_col]
    
    # Extract first digit
    first_digits = amounts.astype(str).str.lstrip("0").str.replace(".", "").str[0]
    valid_digits = first_digits[first_digits.isin([str(i) for i in range(1, 10)])].astype(int)

    total_count = len(valid_digits)
    if total_count == 0:
        return {"error": "No valid positive amounts for Benford analysis."}

    # Actual counts & frequencies
    actual_counts = valid_digits.value_counts().reindex(range(1, 10), fill_value=0)
    actual_freq = actual_counts / total_count

    # Theoretical Benford frequencies
    digits = np.arange(1, 10)
    benford_freq = np.log10(1.0 + 1.0 / digits)
    expected_counts = benford_freq * total_count

    # Chi-Square test
    chi2_stat, p_value = chisquare(f_obs=actual_counts, f_exp=expected_counts)

    # Mean Absolute Deviation (MAD)
    mad = np.mean(np.abs(actual_freq.values - benford_freq))

    # MAD Conformity Evaluation
    if mad <= 0.006:
        conformity = "Close Conformity"
    elif mad <= 0.012:
        conformity = "Acceptable Conformity"
    elif mad <= 0.015:
        conformity = "Marginally Acceptable"
    else:
        conformity = "Non-Conformity (Suspicious Distribution)"

    # Identify senders with highest deviation from Benford (e.g. repeated round amounts)
    sender_round_counts = wiring_df.groupby("sender_id").apply(
        lambda g: (g[amount_col] % 100 == 0).mean()
    ).reset_index(name="round_amount_pct")

    suspicious_senders = sender_round_counts[sender_round_counts["round_amount_pct"] > 0.40].sort_values(
        "round_amount_pct", ascending=False
    )

    # Plot comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(d) for d in digits],
        y=actual_freq.values * 100,
        name="Actual Transaction Distribution",
        marker_color="#3498DB"
    ))
    fig.add_trace(go.Scatter(
        x=[str(d) for d in digits],
        y=benford_freq * 100,
        mode="lines+markers",
        name="Benford's Law (Theoretical)",
        line=dict(color="#E74C3C", width=3)
    ))
    fig.update_layout(
        title=f"Benford's Law First-Digit Analysis (MAD = {mad:.4f} - {conformity})",
        xaxis_title="First Digit",
        yaxis_title="Frequency (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    results_df = pd.DataFrame({
        "Digit": digits,
        "Actual_Count": actual_counts.values,
        "Actual_Pct": (actual_freq.values * 100).round(2),
        "Benford_Pct": (benford_freq * 100).round(2),
        "Difference_Pct": ((actual_freq.values - benford_freq) * 100).round(2)
    })

    return {
        "results_df": results_df,
        "chi2_stat": round(chi2_stat, 2),
        "p_value": round(p_value, 5),
        "mad": round(mad, 5),
        "conformity": conformity,
        "benford_fig": fig,
        "suspicious_senders": suspicious_senders
    }

def isolation_forest_anomaly_detection(
    wiring_df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, go.Figure]:
    """
    Applies Isolation Forest anomaly detection to identify suspicious wiring transactions.

    Features:
        - Transaction Amount (Log USD)
        - Sender Transaction Velocity
        - Industry Mismatch Flag
    """
    if wiring_df.empty:
        return wiring_df, pd.DataFrame(), go.Figure()

    df = wiring_df.copy()

    # Calculate transaction velocity (count per sender)
    velocity = df.groupby("sender_id")["transaction_id"].transform("count")
    df["sender_velocity"] = velocity

    X = pd.DataFrame({
        "log_amount": np.log1p(np.maximum(0, df["amount"])),
        "velocity": df["sender_velocity"],
        "ind_mismatch": (~df["industry_match"]).astype(int)
    })

    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df["anomaly_score"] = iso.fit_predict(X)
    df["anomaly_label"] = np.where(df["anomaly_score"] == -1, "Anomalous", "Normal")

    anomalous_tx = df[df["anomaly_label"] == "Anomalous"].sort_values("amount", ascending=False)

    # Flagged Senders
    flagged_senders = anomalous_tx.groupby("sender_id").agg(
        anomalous_count=("transaction_id", "count"),
        total_anomalous_amount=("amount", "sum")
    ).reset_index().sort_values("anomalous_count", ascending=False)

    # Scatter plot
    fig = px.scatter(
        df,
        x="sender_velocity",
        y="amount",
        color="anomaly_label",
        color_discrete_map={"Normal": "#2ECC71", "Anomalous": "#E74C3C"},
        hover_data=["transaction_id", "sender_id", "receiver_id", "amount"],
        log_y=True,
        title="Isolation Forest Anomaly Detection on Wiring Transactions"
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return df, flagged_senders, fig
