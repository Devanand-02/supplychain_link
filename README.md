# SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking

Production-grade Streamlit Web Application implementing end-to-end B2B supply chain network expansion, predictive propensity modeling, association rule mining, fraud screening, and campaign evaluation for commercial corporate banking.

---

## 📌 Executive Summary & Academic Paper Mapping

Commercial banks face significant customer acquisition costs and credit underwriting risks in B2B corporate lending. **SupplyChain-Link** leverages internal electronic bank wiring transactions (e-wiring) and credit report relationship registries (affiliated companies and upstream/downstream commercial partners) to:
1. Expand the addressable corporate prospect pool by **~5.4x** (from initial e-wiring clients to non-customer supply chain partners).
2. Prioritize corporate loan prospecting using **Logistic Regression Propensity & Credit Approval Models** ($P(\text{Response}) \times P(\text{Approval})$), achieving a **2.17% loan response rate** (vs 0.98% baseline) and an **89.4% credit approval rate** (vs 48.1% baseline).
3. Mine cross-selling product bundles using **Apriori Association Rule Mining** (e.g., $\{ \text{Checking, E-Wiring} \} \Rightarrow \text{Corporate Loan}$).

---

## 🛠️ Project Structure & Modular Folder Layout

```text
supplychain_link/
├── app.py                      # Main Streamlit application & session state manager
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & paper mapping
├── data/                       # Directory for CSV dataset persistence
├── src/                        # Core analytics & machine learning modules
│   ├── __init__.py
│   ├── data_gen.py             # Configurable synthetic data generator & CSV validator
│   ├── network.py              # 6-Step iterative network expansion & centrality analytics
│   ├── association.py          # Apriori & association rule mining (mlxtend)
│   ├── models.py               # Econometric propensity model (Odds Ratios) & K-Means
│   ├── fraud.py                # Benford's Law Chi-Square/MAD & Isolation Forest screening
│   └── evaluation.py           # Two-proportion z-tests, deciles, & cumulative gains
├── pages/                      # 10 Streamlit multi-page interface screens
│   ├── 1_Overview_and_Data.py
│   ├── 2_Network_Builder.py
│   ├── 3_Filters_and_Campaign_List.py
│   ├── 4_Network_Explorer.py
│   ├── 5_Association_Rules.py
│   ├── 6_Propensity_and_Approval_Models.py
│   ├── 7_Segmentation_and_Decision_Tree.py
│   ├── 8_Fraud_Screening.py
│   ├── 9_Campaign_Results_and_Lift.py
│   └── 10_Insights_and_Recommendations.py
└── tests/                      # Automated Pytest suite
    ├── __init__.py
    └── test_analytics.py
```

---

## 💻 Tech Stack

- **Python**: 3.10+
- **Frontend / Framework**: Streamlit (Multi-page Architecture)
- **Data Wrangling**: Pandas, NumPy
- **Graph & Network Analytics**: NetworkX, Pyvis, Plotly
- **Machine Learning & Econometrics**: Scikit-Learn, Statsmodels, Scipy, Joblib
- **Association Rule Mining**: Mlxtend (`apriori`, `association_rules`)
- **Testing**: Pytest

---

## 🚀 How to Run the Application

### 1. Installation
Clone or navigate to the project folder and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Platform
```bash
streamlit run app.py
```

### 3. Run Automated Unit Tests
```bash
pytest tests/
```

---

## 📖 Academic Paper Section Mapping

| Paper Section | Application Page / Module | Key Functions / Analytics |
|---|---|---|
| **Sec 1 & 2: Background** | `app.py` & `Page 1: Overview & Data` | Synthetic Data Generator (`data_gen.py`), CSV schema validation |
| **Sec 3 & 4.1: Expansion Algorithm** | `Page 2: Network Builder` | 6-step iterative graph expansion (`build_supply_chain_network`) |
| **Sec 4.2: False-Relationship Filter** | `Page 3: Filters & Campaign List` | Hub payee screening (`apply_false_relationship_filter`) |
| **Sec 4.3 & 4.4: Network Analytics** | `Page 4: Network Explorer` | PageRank, Betweenness, Modularity Communities, Hub Companies |
| **Table 2 & Sec 5: Association Rules** | `Page 5: Association Rules` | Apriori itemset mining (`mine_product_association_rules`) |
| **Sec 6: Econometric Propensity** | `Page 6: Propensity & Approval Models` | Logistic regression Odds Ratios $OR = e^\beta$, ROC-AUC, Calibration |
| **Fig. 5: Segmentation Tree** | `Page 7: Segmentation & Decision Tree` | Decision Tree (depth=3, min_leaf=40) & K-Means Clustering |
| **Forensic Audit** | `Page 8: Fraud Screening` | Benford's Law Chi-Square / MAD test & Isolation Forest |
| **Sec 7: Campaign Evaluation** | `Page 9: Campaign Results & Lift` | Two-proportion Z-test, decile table, cumulative gains |
| **Sec 8: Practical Implications** | `Page 10: Insights & Recommendations` | 3 commercial banking strategies & analytical limitations |

---

## 🛡️ Analytical Limitations

1. **Synthetic Data Constraints**: Synthetic generators simulate log-normal transfer amounts and Pareto capital distributions; real-world B2B graphs experience non-stationary macroeconomic shocks.
2. **Correlation vs Causation**: Higher loan response among e-wiring users reflects operational digital affinity alongside selection effects.
3. **Field Account Officer Constraints**: Final conversion relies on relationship manager soft skills, local branch coverage, and credit negotiations.
