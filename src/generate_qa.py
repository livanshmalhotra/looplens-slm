import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are LoopLens AI, a procurement analytics assistant. You answer questions "
    "about spend, suppliers, contracts, sourcing events, and procurement KPIs using "
    "data from the LoopLens platform. Always respond in clear, non-technical language. "
    "Format currency values with commas. Include relevant context like percentages and comparisons when helpful."
)

def fmt_curr(val: float, symbol: str = "$") -> str:
    """Formats numeric value as standard currency string."""
    try:
        return f"{symbol}{val:,.2f}"
    except Exception:
        return f"{symbol}0.00"

def fmt_num(val: float) -> str:
    """Formats integer or float with commas."""
    try:
        return f"{val:,.0f}" if isinstance(val, (int, np.integer)) or (isinstance(val, float) and val.is_integer()) else f"{val:,.2f}"
    except Exception:
        return str(val)

# ==========================================
# DOMAIN 1: SPEND ANALYTICS (Invoices)
# ==========================================
def generate_spend_qa(df_inv: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_inv.empty:
        return qa_pairs

    total_spend = float(df_inv['Total Amount (Local)'].sum())
    total_inv = len(df_inv)
    total_vendors = int(df_inv['Vendor Name'].nunique())
    total_cats = int(df_inv['Spend Category'].nunique())

    # 1. Macro Spend Questions
    macro_q = [
        "What is our total procurement spend?",
        "How much did we spend overall across all invoices?",
        "Can you summarize total spend and invoice volume?",
        "What is the total expenditure recorded in LoopLens?"
    ]
    macro_a = (
        f"Your total procurement spend is {fmt_curr(total_spend)} across {fmt_num(total_inv)} invoice line items "
        f"from {fmt_num(total_vendors)} unique vendors across {fmt_num(total_cats)} spend categories."
    )
    for q in macro_q:
        qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": macro_a})

    # 2. Vendor Specific Spend
    vendor_grp = df_inv.groupby('Vendor Name')['Total Amount (Local)'].agg(['sum', 'count']).reset_index()
    for _, row in vendor_grp.iterrows():
        vendor = str(row['Vendor Name'])
        v_spend = float(row['sum'])
        v_cnt = int(row['count'])
        pct = (v_spend / total_spend * 100) if total_spend > 0 else 0

        qs = [
            f"How much did we spend with {vendor}?",
            f"What is the total expenditure for supplier {vendor}?",
            f"Show me spending summary for {vendor}.",
            f"What's our spend history with vendor {vendor}?",
            f"How many invoices do we have for {vendor} and total amount?"
        ]
        ans = (
            f"Total spend with {vendor} is {fmt_curr(v_spend)} across {fmt_num(v_cnt)} invoice line item(s). "
            f"This accounts for {pct:.2f}% of our overall procurement expenditure."
        )
        for q in qs:
            qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": ans})

    # 3. Category Specific Spend
    cat_grp = df_inv.groupby('Spend Category')['Total Amount (Local)'].agg(['sum', 'count']).reset_index()
    for _, row in cat_grp.iterrows():
        cat = str(row['Spend Category'])
        c_spend = float(row['sum'])
        c_cnt = int(row['count'])
        pct = (c_spend / total_spend * 100) if total_spend > 0 else 0

        qs = [
            f"How much was spent on {cat}?",
            f"What is the total spend in category {cat}?",
            f"Show category breakdown for {cat}.",
            f"What is the expenditure for {cat}?"
        ]
        ans = (
            f"Expenditure for '{cat}' totals {fmt_curr(c_spend)} across {fmt_num(c_cnt)} invoice line(s), "
            f"representing {pct:.2f}% of total spend."
        )
        for q in qs:
            qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": ans})

    # 4. Top Vendors / Categories
    top_v = vendor_grp.sort_values(by='sum', ascending=False).head(5)
    top_v_str = "\n".join([f"{idx+1}. {r['Vendor Name']} — {fmt_curr(r['sum'])} ({r['sum']/total_spend*100:.1f}%)" for idx, (_, r) in enumerate(top_v.iterrows())])
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "Who are our top 5 vendors by spend?",
        "output": f"The top 5 suppliers by spend are:\n{top_v_str}"
    })
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "Show me top 5 suppliers",
        "output": f"Here are the top 5 vendors ranked by spend:\n{top_v_str}"
    })

    top_c = cat_grp.sort_values(by='sum', ascending=False).head(5)
    top_c_str = "\n".join([f"{idx+1}. {r['Spend Category']} — {fmt_curr(r['sum'])} ({r['sum']/total_spend*100:.1f}%)" for idx, (_, r) in enumerate(top_c.iterrows())])
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "What are our top 5 spend categories?",
        "output": f"The top 5 spend categories are:\n{top_c_str}"
    })

    # 5. Contract Compliance / Rogue Spend
    if 'Contract Status' in df_inv.columns:
        contract_summary = df_inv.groupby('Contract Status')['Total Amount (Local)'].agg(['sum', 'count']).reset_index()
        contract_str = ", ".join([f"{r['Contract Status']}: {fmt_curr(r['sum'])} ({r['count']} invoices)" for _, r in contract_summary.iterrows()])
        qa_pairs.append({
            "instruction": SYSTEM_PROMPT,
            "input": "What is our contract compliance status on invoices?",
            "output": f"Invoice breakdown by contract status is as follows: {contract_str}."
        })

    return qa_pairs


# ==========================================
# DOMAIN 2: SUPPLIER ANALYTICS
# ==========================================
def generate_supplier_qa(df_sup: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_sup.empty:
        return qa_pairs

    tot_sup = len(df_sup)
    status_counts = df_sup['Status'].value_counts().to_dict() if 'Status' in df_sup.columns else {}
    status_str = ", ".join([f"{k}: {v}" for k, v in status_counts.items()])

    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "How many suppliers are registered in LoopLens?",
        "output": f"There are {fmt_num(tot_sup)} suppliers registered in the system. Breakdown by status: {status_str}."
    })
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "What is the breakdown of supplier status?",
        "output": f"Out of {fmt_num(tot_sup)} suppliers: {status_str}."
    })

    # Sample individual supplier lookups safely
    sample_size = min(300, len(df_sup))
    sample_sups = df_sup.sample(sample_size, random_state=42)
    for _, row in sample_sups.iterrows():
        s_name = str(row.get('SupplierName', row.get('Supplier Code', 'Supplier')))
        s_status = str(row.get('Status', 'Active'))
        s_country = str(row.get('CountryCode', 'Global'))
        s_spend = float(row.get('TotalSpend_YTD', 0.0) or 0.0)
        s_risk = str(row.get('RiskScore', 'N/A'))

        qs = [
            f"What is the status and profile for supplier {s_name}?",
            f"Tell me about supplier {s_name}.",
            f"Is supplier {s_name} active and what is its location?"
        ]
        ans = (
            f"Supplier '{s_name}' is currently {s_status}. Country of operation: {s_country}. "
            f"YTD spend: {fmt_curr(s_spend)}. Risk rating score: {s_risk}."
        )
        for q in qs:
            qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": ans})

    return qa_pairs


# ==========================================
# DOMAIN 3: CONTRACT ANALYTICS
# ==========================================
def generate_contract_qa(df_con: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_con.empty:
        return qa_pairs

    tot_con = len(df_con)
    status_counts = df_con['Status'].value_counts().to_dict() if 'Status' in df_con.columns else {}
    status_str = ", ".join([f"{k}: {v}" for k, v in status_counts.items()])
    tot_val = float(df_con['ContractValue'].sum()) if 'ContractValue' in df_con.columns else 0.0

    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "How many active and total contracts do we have?",
        "output": f"We have {fmt_num(tot_con)} total contracts with a combined value of {fmt_curr(tot_val)}. Breakdown by status: {status_str}."
    })
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "What is our total contract portfolio value?",
        "output": f"The total value of all contracts in ContractLens is {fmt_curr(tot_val)} across {fmt_num(tot_con)} contracts ({status_str})."
    })

    # Sample contract lookups safely
    valid_cons = df_con.dropna(subset=['CONTRACT_ID'])
    sample_size = min(300, len(valid_cons))
    sample_cons = valid_cons.sample(sample_size, random_state=42)
    for _, row in sample_cons.iterrows():
        c_id = str(row.get('CONTRACT_ID', row.get('ContractNumber', 'CON-000')))
        c_title = str(row.get('ContractTitle', 'Contract'))
        c_vendor = str(row.get('SUPPLIER_COMPANY_NAME', 'Vendor'))
        c_val = float(row.get('ContractValue', 0.0) or 0.0)
        c_status = str(row.get('Status', 'Active'))
        c_owner = str(row.get('ContractOwner', 'Unassigned'))

        qs = [
            f"What are the details for contract {c_id}?",
            f"Who owns contract {c_id} and what is its status?",
            f"Show details for contract {c_title} ({c_id})."
        ]
        ans = (
            f"Contract {c_id} ('{c_title}') with vendor '{c_vendor}' has a value of {fmt_curr(c_val)}. "
            f"Current status is {c_status}, managed by {c_owner}."
        )
        for q in qs:
            qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": ans})

    return qa_pairs


# ==========================================
# DOMAIN 4: PO ANALYTICS
# ==========================================
def generate_po_qa(df_po: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_po.empty:
        return qa_pairs

    tot_po = len(df_po)
    tot_val = float(df_po['AMOUNT'].sum()) if 'AMOUNT' in df_po.columns else 0.0
    buyers = int(df_po['BUYER_NAME'].nunique()) if 'BUYER_NAME' in df_po.columns else 0

    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "What is our total Purchase Order (PO) volume and spend?",
        "output": f"Total PO spend is {fmt_curr(tot_val)} across {fmt_num(tot_po)} purchase order lines, managed by {fmt_num(buyers)} buyers."
    })

    if 'BUYER_NAME' in df_po.columns and 'AMOUNT' in df_po.columns:
        buyer_grp = df_po.groupby('BUYER_NAME')['AMOUNT'].agg(['sum', 'count']).reset_index().head(20)
        for _, row in buyer_grp.iterrows():
            b_name = str(row['BUYER_NAME'])
            b_spend = float(row['sum'])
            b_cnt = int(row['count'])
            qa_pairs.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"How much PO value was issued by buyer {b_name}?",
                "output": f"Buyer {b_name} has issued {fmt_num(b_cnt)} PO lines totaling {fmt_curr(b_spend)}."
            })

    return qa_pairs


# ==========================================
# DOMAIN 5: SOURCING / RFQ ANALYTICS
# ==========================================
def generate_rfq_qa(df_rfq: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_rfq.empty:
        return qa_pairs

    tot_rfq_events = int(df_rfq['RFQ_CODE'].nunique()) if 'RFQ_CODE' in df_rfq.columns else len(df_rfq)
    tot_responses = len(df_rfq)
    status_counts = df_rfq['RFQ Status'].value_counts().to_dict() if 'RFQ Status' in df_rfq.columns else {}
    status_str = ", ".join([f"{k}: {v}" for k, v in list(status_counts.items())[:5]])

    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "How many RFQ sourcing events have been conducted?",
        "output": f"LoopLens SourcingLens records {fmt_num(tot_rfq_events)} unique RFQ events with {fmt_num(tot_responses)} supplier participation responses. Primary statuses: {status_str}."
    })
    qa_pairs.append({
        "instruction": SYSTEM_PROMPT,
        "input": "What is the status of our RFQ sourcing pipeline?",
        "output": f"There are {fmt_num(tot_rfq_events)} sourcing events recorded. Breakdown by top status: {status_str}."
    })

    if 'DIVISION_NAME' in df_rfq.columns:
        div_grp = df_rfq.groupby('DIVISION_NAME')['RFQ_CODE'].nunique().reset_index()
        for _, row in div_grp.iterrows():
            div = str(row['DIVISION_NAME'])
            cnt = int(row['RFQ_CODE'])
            qa_pairs.append({
                "instruction": SYSTEM_PROMPT,
                "input": f"How many RFQs were published for division {div}?",
                "output": f"Division '{div}' has initiated {fmt_num(cnt)} RFQ sourcing events."
            })

    return qa_pairs


# ==========================================
# DOMAIN 6: SPEND CLASSIFICATION
# ==========================================
def generate_classification_qa(df_inv: pd.DataFrame) -> List[Dict[str, str]]:
    qa_pairs = []
    if df_inv.empty:
        return qa_pairs

    filtered_invs = df_inv.dropna(subset=['Line Item Description', 'Spend Category'])
    sample_size = min(2000, len(filtered_invs))
    sample_invs = filtered_invs.sample(sample_size, random_state=42)
    for _, row in sample_invs.iterrows():
        desc = str(row['Line Item Description'])
        cat = str(row['Spend Category'])
        vendor = str(row.get('Vendor Name', 'Unknown Vendor'))
        amt = float(row.get('Total Amount (Local)', 0.0) or 0.0)
        curr = str(row.get('Currency', 'USD'))

        qs = [
            f"Classify this invoice line: '{desc}' from vendor '{vendor}' worth {amt} {curr}.",
            f"What spend category should be assigned to: '{desc}'?",
            f"Categorize the following line item: '{desc}'"
        ]
        ans = (
            f"Based on the description and vendor context, this item is classified as '{cat}'. "
            f"Recommendation: Record under {cat} for spend category reporting."
        )
        for q in qs:
            qa_pairs.append({"instruction": SYSTEM_PROMPT, "input": q, "output": ans})

    return qa_pairs


# ==========================================
# DOMAIN 7: GLOSSARY & EXPLANATIONS
# ==========================================
def generate_glossary_qa() -> List[Dict[str, str]]:
    glossary = [
        ("What is contract compliance rate?", "Contract compliance rate measures the percentage of total spend covered by active, formal contracts versus non-contracted (rogue) spend. High compliance reduces supply chain risk and captures negotiated savings."),
        ("What is rogue spend?", "Rogue spend (or maverick spend) refers to unapproved or off-contract purchases made outside agreed procurement frameworks, often resulting in higher prices and unvetted supplier risk."),
        ("What is Pareto analysis in spend analytics?", "Pareto analysis (the 80/20 rule) identifies top spend drivers, typically showing that 80% of organization spend comes from 20% of suppliers or categories."),
        ("What is an RFQ cycle time?", "RFQ cycle time measures the number of calendar days elapsed from Purchase Requisition (PR) approval to RFQ publishing, technical evaluation, commercial evaluation, and final award."),
        ("What is SpendLens?", "SpendLens is the LoopLens dashboard dedicated to invoice spend aggregation, category taxonomies, vendor volume analysis, payment terms, and PO compliance."),
        ("What is SourcingLens?", "SourcingLens tracks strategic sourcing events, RFQ performance, supplier response rates, evaluation SLAs, and buyer productivity."),
        ("What is SupplierLens?", "SupplierLens consolidates supplier identity profiles, risk scores, ESG ratings, location distribution, and total engagement across POs and RFQs."),
        ("What is ContractLens?", "ContractLens monitors contract header lifecycles, renewal alerts, expiry calendar buckets (30/60/90 days), and contract owner allocations.")
    ]
    qa = []
    for q, a in glossary:
        qa.append({"instruction": SYSTEM_PROMPT, "input": q, "output": a})
    return qa


def prepare_qa_dataset(raw_dir: str, output_dir: str, val_ratio: float = 0.1):
    print(f"[*] Starting LoopLens SLM Q&A Dataset Generator from raw extracts in: {raw_dir}")
    all_qa = []

    # 1. Invoices
    inv_path = os.path.join(raw_dir, "invoice_spend_extract_final.csv")
    if os.path.exists(inv_path):
        print(f"[*] Reading Invoices extract: {inv_path}")
        df_inv = pd.read_csv(inv_path)
        all_qa.extend(generate_spend_qa(df_inv))
        all_qa.extend(generate_classification_qa(df_inv))

    # 2. Suppliers
    sup_path = os.path.join(raw_dir, "Supplier Extract.xlsx")
    if os.path.exists(sup_path):
        print(f"[*] Reading Supplier extract: {sup_path}")
        df_sup = pd.read_excel(sup_path)
        all_qa.extend(generate_supplier_qa(df_sup))

    # 3. Contracts
    con_path = os.path.join(raw_dir, "DIM_CONTRACT.xlsx")
    if os.path.exists(con_path):
        print(f"[*] Reading Contract extract: {con_path}")
        df_con = pd.read_excel(con_path)
        all_qa.extend(generate_contract_qa(df_con))

    # 4. POs
    po_path = os.path.join(raw_dir, "PO_Details.xlsx")
    if os.path.exists(po_path):
        print(f"[*] Reading PO extract: {po_path}")
        df_po = pd.read_excel(po_path)
        all_qa.extend(generate_po_qa(df_po))

    # 5. RFQs
    rfq_path = os.path.join(raw_dir, "RFQ Extract_DEMO_SYNTHETIC.csv")
    if os.path.exists(rfq_path):
        print(f"[*] Reading RFQ extract: {rfq_path}")
        df_rfq = pd.read_csv(rfq_path, usecols=['RFQ_CODE', 'RFQ Status', 'DIVISION_NAME', 'SUPPLIER_COMPANY_NAME'])
        all_qa.extend(generate_rfq_qa(df_rfq))

    # 6. Glossary
    all_qa.extend(generate_glossary_qa())

    print(f"[*] Total Q&A Pairs generated: {len(all_qa)}")

    # Shuffle dataset
    random.seed(42)
    random.shuffle(all_qa)

    # Split train/val
    os.makedirs(output_dir, exist_ok=True)
    val_count = max(1, int(len(all_qa) * val_ratio))
    train_count = len(all_qa) - val_count

    train_data = all_qa[:train_count]
    val_data = all_qa[train_count:]

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"[OK] Saved {len(train_data)} train samples to {train_path}")
    print(f"[OK] Saved {len(val_data)} validation samples to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoopLens Natural Language Q&A Dataset Generator")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output path for JSONL dataset")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation dataset split ratio")
    args = parser.parse_args()

    prepare_qa_dataset(args.raw_dir, args.output_dir, args.val_ratio)
