"""
Synthetic Data Generator & Schema Validation Module
SupplyChain-Link: Big Data Analytics for Supply Chain Relationships in Corporate Banking
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import random

# Core Product List
ALL_PRODUCTS = [
    "checking",
    "savings",
    "FX",
    "trade finance",
    "payroll",
    "e-wiring",
    "corporate loan",
    "cash management"
]

INDUSTRIES = [
    "Technology",
    "Manufacturing",
    "Retail",
    "Shipping",
    "Utilities",
    "Energy",
    "Healthcare",
    "Construction",
    "Financial Services"
]

BRANCHES = [f"BRANCH_{i:02d}" for i in range(1, 16)]

COMPANY_PREFIXES = ["Apex", "Global", "Vanguard", "Nexus", "Titan", "Summit", "Horizon", "Pinnacle", "Crest", "Omni", "Quantum", "Synergy", "Pacific", "Atlas", "Sterling"]
COMPANY_SUFFIXES = ["Corp", "Inc", "Holdings", "Logistics", "Tech", "Industries", "Group", "Solutions", "Energy", "Trading", "Mfg", "Services"]
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]

def generate_synthetic_supply_chain_data(
    n_companies: int = 5000,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generates realistic synthetic supply chain data for corporate banking analytics.

    Parameters:
        n_companies (int): Total number of companies to generate (default 5,000 for fast execution, adjustable up to 50,000+).
        random_state (int): Seed for reproducibility.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing dataframes for 'companies',
        'credit_reports', 'wiring_transactions', 'directors', and 'campaign_outcomes'.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    company_ids = [f"COMP_{i:05d}" for i in range(1, n_companies + 1)]

    # 1. Companies Table Generation
    # Tax ID with ~3% missing
    tax_ids = [
        f"TAX-{np.random.randint(10000000, 99999999)}" if np.random.rand() > 0.03 else None
        for _ in range(n_companies)
    ]

    names = [
        f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
        for _ in range(n_companies)
    ]

    industries = np.random.choice(INDUSTRIES, size=n_companies, p=[0.15, 0.22, 0.15, 0.08, 0.06, 0.09, 0.08, 0.10, 0.07])
    
    # Registered capital: Log-normal distribution (ranging $100k to $500M)
    log_cap = np.random.normal(loc=15.2, scale=1.5, size=n_companies)
    registered_capital = np.clip(np.exp(log_cap), 100_000, 500_000_000).round(2)

    # Bank customer status (~35% existing bank customers)
    is_bank_customer = np.random.rand(n_companies) < 0.35
    
    # Core company flag (Loan >= 50M USD or Capital >= 50M USD & exchange listed)
    is_core_company = (registered_capital >= 50_000_000) & (np.random.rand(n_companies) < 0.40)

    # Customer tenure (years)
    tenure_years = np.where(
        is_bank_customer,
        np.round(np.random.uniform(0.5, 18.0, size=n_companies), 1),
        0.0
    )

    # E-wiring adoption
    e_wiring_user = np.where(
        is_bank_customer,
        np.random.rand(n_companies) < 0.65,
        False
    )

    # Product holdings generation
    products_held_list = []
    num_product_holdings = []

    for i in range(n_companies):
        if not is_bank_customer[i]:
            products_held_list.append("")
            num_product_holdings.append(0)
        else:
            # Bank customer products
            possible = ALL_PRODUCTS.copy()
            if e_wiring_user[i]:
                held = ["checking", "e-wiring"]
            else:
                held = ["checking"] if np.random.rand() > 0.1 else []

            # Add additional random products based on tenure and core status
            p_add = 0.45 if is_core_company[i] else 0.25
            for prod in possible:
                if prod not in held and np.random.rand() < p_add:
                    held.append(prod)

            products_held_list.append(", ".join(held))
            num_product_holdings.append(len(held))

    # Board members
    board_members_list = []
    for _ in range(n_companies):
        n_members = random.randint(2, 5)
        members = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n_members)]
        board_members_list.append(", ".join(members))

    companies_df = pd.DataFrame({
        "company_id": company_ids,
        "tax_id": tax_ids,
        "name": names,
        "industry": industries,
        "registered_capital": registered_capital,
        "is_bank_customer": is_bank_customer,
        "is_core_company": is_core_company,
        "tenure_years": tenure_years,
        "e_wiring_user": e_wiring_user,
        "num_product_holdings": num_product_holdings,
        "products_held": products_held_list,
        "board_members": board_members_list
    })

    # 2. Credit Reports Table Generation (Affiliates & Upstream/Downstream)
    credit_reports_records = []
    
    # Identify core company indices
    core_company_ids = companies_df[companies_df["is_core_company"]]["company_id"].tolist()
    if not core_company_ids:
        core_company_ids = company_ids[:10]

    for cid in company_ids:
        # Generate 0-3 affiliates
        n_aff = random.choices([0, 1, 2, 3], weights=[0.60, 0.25, 0.10, 0.05])[0]
        affiliates = random.sample(company_ids, n_aff) if n_aff > 0 else []
        if cid in affiliates:
            affiliates.remove(cid)

        # Generate 0-5 upstream/downstream partners
        n_updn = random.choices([0, 1, 2, 3, 4, 5], weights=[0.40, 0.30, 0.15, 0.10, 0.03, 0.02])[0]
        up_downstream = random.sample(company_ids, n_updn) if n_updn > 0 else []
        if cid in up_downstream:
            up_downstream.remove(cid)

        # Higher likelihood of linking to a core company
        if random.random() < 0.25 and core_company_ids:
            linked_core = random.choice(core_company_ids)
            if linked_core != cid and linked_core not in affiliates:
                affiliates.append(linked_core)

        credit_reports_records.append({
            "company_id": cid,
            "affiliated_companies": ", ".join(affiliates),
            "upstream_downstream_companies": ", ".join(up_downstream)
        })

    credit_reports_df = pd.DataFrame(credit_reports_records)

    # 3. E-Wiring Transactions Table Generation
    # Include legitimate B2B supply chain transactions + "false supply chain" hub payees (Shipping / Utilities)
    wiring_records = []
    n_transactions = int(n_companies * 2.5)

    # Pick hub payee companies (utilities / shipping receiving payments from many unrelated firms)
    utility_shipping_ids = companies_df[companies_df["industry"].isin(["Shipping", "Utilities"])]["company_id"].tolist()
    if not utility_shipping_ids:
        utility_shipping_ids = company_ids[:5]
    
    hub_payees = set(random.sample(utility_shipping_ids, min(len(utility_shipping_ids), 10)))

    # Dates spanning 2016-01-01 to 2017-12-31
    start_date = pd.Timestamp("2016-01-01")
    end_date = pd.Timestamp("2017-12-31")
    date_range_days = (end_date - start_date).days

    company_industry_map = companies_df.set_index("company_id")["industry"].to_dict()

    for tx_id in range(1, n_transactions + 1):
        sender = random.choice(company_ids)
        
        # 20% chance to pay a hub payee (false supply chain relationship)
        if random.random() < 0.20 and hub_payees:
            receiver = random.choice(list(hub_payees))
        else:
            receiver = random.choice(company_ids)

        if sender == receiver:
            continue

        # Amount: log-normal around $15,000 USD
        amount = round(float(np.random.lognormal(mean=9.5, sigma=1.2)), 2)
        amount = max(100.0, min(amount, 5_000_000.0))

        tx_date = start_date + pd.Timedelta(days=random.randint(0, date_range_days))
        
        sender_ind = company_industry_map.get(sender, "Other")
        receiver_ind = company_industry_map.get(receiver, "Other")
        industry_match = (sender_ind == receiver_ind) or (receiver in hub_payees)

        wiring_records.append({
            "transaction_id": f"TX_{tx_id:06d}",
            "sender_id": sender,
            "receiver_id": receiver,
            "amount": amount,
            "date": tx_date.strftime("%Y-%m-%d"),
            "industry_match": industry_match
        })

    wiring_transactions_df = pd.DataFrame(wiring_records)

    # 4. Directors Table Generation
    directors_records = []
    p_id = 1
    for cid in company_ids[:int(n_companies * 0.4)]: # 40% of companies have director records
        n_dirs = random.randint(1, 3)
        for _ in range(n_dirs):
            person_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            is_personal_cust = random.random() < 0.45
            
            # Director owns 0-2 other companies
            owned = random.sample(company_ids, random.choice([0, 1, 2])) if random.random() < 0.3 else []
            if cid in owned:
                owned.remove(cid)

            directors_records.append({
                "person_id": f"DIR_{p_id:05d}",
                "company_id": cid,
                "person_name": person_name,
                "is_personal_bank_customer": is_personal_cust,
                "owns_other_companies": ", ".join(owned)
            })
            p_id += 1

    directors_df = pd.DataFrame(directors_records)

    # 5. Campaign Outcomes Table Generation
    # Constructing realistic campaign target statistics from the paper:
    # Baseline corporate loan response ~0.98%, loan approval ~48.1%
    # Targeted campaign response ~2.17%, loan approval ~89.4%
    # Relationship mix: 34.4% affiliate only, 12.7% upstream/downstream only, 50.3% wiring only
    
    # We select a target campaign subset (~24% of companies)
    campaign_size = int(n_companies * 0.24)
    campaign_companies = set(random.sample(company_ids, campaign_size))

    # Pre-calculate wiring & credit relationships for relationship_type mapping
    aff_map = set(credit_reports_df[credit_reports_df["affiliated_companies"] != ""]["company_id"])
    updn_map = set(credit_reports_df[credit_reports_df["upstream_downstream_companies"] != ""]["company_id"])
    wiring_senders = set(wiring_transactions_df["sender_id"]).union(set(wiring_transactions_df["receiver_id"]))

    comp_lookup = companies_df.set_index("company_id").to_dict(orient="index")

    campaign_records = []
    for cid in company_ids:
        has_aff = cid in aff_map
        has_updn = cid in updn_map
        has_wire = cid in wiring_senders

        # Classify relationship type according to paper's Table 2 proportions
        rel_types = []
        if has_aff:
            rel_types.append("Affiliate")
        if has_updn:
            rel_types.append("Upstream/Downstream")
        if has_wire:
            rel_types.append("Wiring")

        if not rel_types:
            # Default assign based on paper weights
            rel_type = np.random.choice(
                ["Wiring Only", "Affiliate Only", "Upstream-Downstream Only"],
                p=[0.503, 0.344, 0.153]
            )
        elif "Wiring" in rel_types and len(rel_types) == 1:
            rel_type = "Wiring Only"
        elif "Affiliate" in rel_types and len(rel_types) == 1:
            rel_type = "Affiliate Only"
        elif "Upstream/Downstream" in rel_types and len(rel_types) == 1:
            rel_type = "Upstream-Downstream Only"
        else:
            rel_type = " + ".join(rel_types)

        contacted = cid in campaign_companies
        c_info = comp_lookup[cid]
        tenure = c_info["tenure_years"]
        e_wire = c_info["e_wiring_user"]
        holdings = c_info["num_product_holdings"]
        is_bank_cust = c_info["is_bank_customer"]
        is_core = c_info["is_core_company"]

        # Paper segmentation logic for response probability:
        # 1) Newer customers (< 4.5 yrs) with e-wiring respond ~20%
        # 2) Older customers (>= 4.5 yrs) with >= 3.5 holdings respond ~9.5%
        # 3) Zero holdings respond ~0%
        # 4) Non-customers respond ~1.25%
        if not is_bank_cust:
            p_response = 0.0125
        elif holdings == 0:
            p_response = 0.001
        elif tenure < 4.5 and e_wire:
            p_response = 0.20
        elif tenure >= 4.5 and holdings >= 3.5:
            p_response = 0.095
        else:
            p_response = 0.045

        # Relationship boost: Wiring + Affiliate connection boosts response
        if "Wiring" in rel_type and ("Affiliate" in rel_type or "Upstream" in rel_type):
            p_response *= 1.35
        if is_core:
            p_response *= 1.25

        p_response = min(0.40, p_response)

        responded_loan = contacted and (np.random.rand() < p_response)
        applied = responded_loan # loan respondents apply

        # Approval logic: Targeted campaign ~89.4% approval vs baseline ~48.1%
        if applied:
            # High approval if core, high capital, or strong tenure
            p_approval = 0.894 if (is_core or c_info["registered_capital"] > 5_000_000 or holdings >= 2) else 0.65
            approved = np.random.rand() < p_approval
        else:
            approved = False

        # Response to other bank services (FX, cash management, trade finance)
        response_other = contacted and (np.random.rand() < (p_response * 1.8))

        campaign_records.append({
            "company_id": cid,
            "relationship_type": rel_type,
            "contacted": contacted,
            "responded_loan": responded_loan,
            "applied": applied,
            "approved": approved,
            "response_other_services": response_other,
            "branch_id": random.choice(BRANCHES)
        })

    campaign_outcomes_df = pd.DataFrame(campaign_records)

    return {
        "companies": companies_df,
        "credit_reports": credit_reports_df,
        "wiring_transactions": wiring_transactions_df,
        "directors": directors_df,
        "campaign_outcomes": campaign_outcomes_df
    }

def validate_uploaded_csv(df: pd.DataFrame, schema_name: str) -> Tuple[bool, List[str]]:
    """
    Validates an uploaded DataFrame against expected schema requirements.

    Parameters:
        df (pd.DataFrame): Uploaded dataframe.
        schema_name (str): Name of schema ('companies', 'credit_reports', 'wiring_transactions', 'directors', 'campaign_outcomes').

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_messages).
    """
    schemas = {
        "companies": ["company_id", "name", "industry", "registered_capital", "is_bank_customer"],
        "credit_reports": ["company_id", "affiliated_companies", "upstream_downstream_companies"],
        "wiring_transactions": ["sender_id", "receiver_id", "amount", "date"],
        "directors": ["person_id", "company_id", "person_name"],
        "campaign_outcomes": ["company_id", "relationship_type", "contacted", "responded_loan", "approved"]
    }

    if schema_name not in schemas:
        return False, [f"Unknown schema name: {schema_name}"]

    required = schemas[schema_name]
    missing = [col for col in required if col not in df.columns]

    if missing:
        return False, [f"Missing required columns for '{schema_name}': {', '.join(missing)}"]

    return True, []

def get_schema_guide() -> Dict[str, List[str]]:
    """
    Returns column requirements and sample data types for all tables.
    """
    return {
        "companies": [
            "company_id (str): Unique ID e.g. COMP_00001",
            "tax_id (str, optional): Tax ID e.g. TAX-12345678",
            "name (str): Enterprise Name",
            "industry (str): Industry Sector",
            "registered_capital (float): Capital in USD",
            "is_bank_customer (bool): Bank Customer Status",
            "is_core_company (bool): Core Anchor Enterprise Flag",
            "tenure_years (float): Banking Relationship Length",
            "e_wiring_user (bool): E-Wiring User Status",
            "num_product_holdings (int): Count of Held Banking Products",
            "products_held (str): Comma-separated Products Held",
            "board_members (str): Comma-separated Director Names"
        ],
        "credit_reports": [
            "company_id (str): Company ID",
            "affiliated_companies (str): Comma-separated Parent/Subsidiary IDs",
            "upstream_downstream_companies (str): Comma-separated Supplier/Customer IDs"
        ],
        "wiring_transactions": [
            "transaction_id (str): Transaction ID",
            "sender_id (str): Payer Company ID",
            "receiver_id (str): Payee Company ID",
            "amount (float): Transaction Amount in USD",
            "date (str): Date (YYYY-MM-DD)",
            "industry_match (bool): Industry Consistency Flag"
        ],
        "directors": [
            "person_id (str): Director Person ID",
            "company_id (str): Enterprise Company ID",
            "person_name (str): Full Name",
            "is_personal_bank_customer (bool): Personal Banking Client Flag",
            "owns_other_companies (str): Comma-separated Owned Company IDs"
        ],
        "campaign_outcomes": [
            "company_id (str): Company ID",
            "relationship_type (str): Network Link Source",
            "contacted (bool): Target Campaign Contacted Flag",
            "responded_loan (bool): Responded to Corporate Loan Offer",
            "applied (bool): Applied for Loan",
            "approved (bool): Credit Approval Granted",
            "response_other_services (bool): Responded to Other Banking Services",
            "branch_id (str): Managing Bank Branch ID"
        ]
    }
