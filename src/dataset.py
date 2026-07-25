import os
import json
import argparse
import pandas as pd
from typing import Dict, Any, List, Optional

SYSTEM_PROMPT = """You are a specialized Procurement and Spend Analytics AI Assistant. Predict standard 4-level spend taxonomy, UNSPSC code, normalized vendor name, procurement risk, and sourcing insights for the following invoice item."""

def get_col_val(row: Dict[str, Any], candidates: List[str], default: Any = "") -> Any:
    """Helper to retrieve column value matching any candidate key name."""
    for key in candidates:
        if key in row and pd.notna(row[key]):
            return row[key]
        # Also try lowercase / underscore variants
        for row_key in row.keys():
            if row_key.lower().replace(" ", "_").replace("(", "").replace(")", "") == key.lower().replace(" ", "_").replace("(", "").replace(")", ""):
                if pd.notna(row[row_key]):
                    return row[row_key]
    return default

def parse_taxonomy_levels(spend_category: str) -> Dict[str, str]:
    """Splits raw hyphen-separated spend category into hierarchy levels."""
    if not spend_category or not isinstance(spend_category, str):
        return {
            "level_1": "General Spend",
            "level_2": "Unclassified",
            "level_3": "Unclassified",
            "category_name": "General"
        }
    
    parts = [p.strip() for p in spend_category.split("-") if p.strip()]
    l1 = parts[0] if len(parts) > 0 else "General Spend"
    l2 = parts[1] if len(parts) > 1 else "Unclassified"
    l3 = parts[2] if len(parts) > 2 else "Unclassified"
    cat_name = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else l1)
    
    return {
        "level_1": l1,
        "level_2": l2,
        "level_3": l3,
        "category_name": cat_name
    }

def format_input_prompt(row: Dict[str, Any]) -> str:
    """Formats raw invoice attributes into structured LLM input prompt."""
    desc = get_col_val(row, ["Line Item Description", "line_item_description", "po_line_desc"], "N/A")
    vendor = get_col_val(row, ["Vendor Name", "vendor_name", "Vendor ID"], "N/A")
    amount = get_col_val(row, ["Total Amount (Local)", "total_amount_local", "Line Amount (Local)", "total_amount"], 0.0)
    currency = get_col_val(row, ["Currency", "currency"], "USD")
    raw_cat = get_col_val(row, ["Spend Category", "spend_category"], "N/A")
    gl_acc = get_col_val(row, ["GL Account", "gl_account"], "N/A")
    
    return f"Line Description: {desc}\nVendor: {vendor}\nAmount: {amount} {currency}\nRaw Spend Category: {raw_cat}\nGL Account: {gl_acc}"

def format_output_json(row: Dict[str, Any], idx: int) -> str:
    """Formats target procurement intelligence payload into strict JSON response."""
    desc = str(get_col_val(row, ["Line Item Description", "line_item_description"], "Procurement Item"))
    vendor = str(get_col_val(row, ["Vendor Name", "vendor_name"], "UNKNOWN"))
    spend_cat = str(get_col_val(row, ["Spend Category", "spend_category"], ""))
    contract_status = str(get_col_val(row, ["Contract Status", "contract_status"], "No Contract"))
    tot_amount = float(get_col_val(row, ["Total Amount (Local)", "total_amount_local"], 0.0) or 0.0)
    inv_id = get_col_val(row, ["Invoice ID", "invoice_id", "line_item_id"], idx)

    tax_levels = parse_taxonomy_levels(spend_cat)
    is_rogue = (contract_status.lower() == "no contract") or (tot_amount > 100000.0)
    
    output = {
        "line_item_id": inv_id,
        "taxonomy_classification": {
            "level_1": tax_levels["level_1"],
            "level_2": tax_levels["level_2"],
            "level_3": tax_levels["level_3"],
            "category_name": tax_levels["category_name"],
            "unspsc_code": "86132201" if "software" in desc.lower() or "elearning" in desc.lower() else "80101500",
            "unspsc_title": "Educational software" if "software" in desc.lower() or "elearning" in desc.lower() else "Business management consultancy services",
            "confidence_score": 0.95
        },
        "entity_resolution": {
            "normalized_vendor_name": vendor.upper().strip(),
            "matched_master_supplier_id": 100000 + (hash(vendor) % 800000),
            "match_confidence": 0.98
        },
        "risk_and_compliance": {
            "is_rogue_spend": is_rogue,
            "risk_level": "HIGH" if (tot_amount > 100000 and is_rogue) else ("MEDIUM" if is_rogue else "LOW"),
            "anomaly_reasons": ["Non-contract spend threshold exceeded"] if is_rogue else [],
            "contract_compliance_flag": not is_rogue
        },
        "sourcing_insights": {
            "item_standardized_title": desc[:60],
            "recommended_sourcing_strategy": "Consolidate volume with preferred vendor"
        }
    }
    return json.dumps(output, indent=2)

def prepare_dataset_from_csv(raw_dir: str, output_dir: str, val_ratio: float = 0.1):
    """Reads raw CSV exports from LoopLens and outputs train.jsonl and val.jsonl."""
    candidate_files = [
        os.path.join(raw_dir, "invoice_spend_extract_final.csv"),
        os.path.join(raw_dir, "fact_invoice.csv")
    ]
    
    invoice_csv = None
    for cf in candidate_files:
        if os.path.exists(cf):
            invoice_csv = cf
            break
            
    if not invoice_csv and os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.endswith(".csv") and ("invoice" in f.lower() or "spend" in f.lower()):
                invoice_csv = os.path.join(raw_dir, f)
                break

    if not invoice_csv or not os.path.exists(invoice_csv):
        print(f"[!] Warning: No invoice CSV file found in {raw_dir}. Skipping CSV extraction.")
        return

    print(f"[*] Processing raw extracts from {invoice_csv}...")
    df = pd.read_csv(invoice_csv)
    
    formatted_data = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        formatted_data.append({
            "line_item_id": get_col_val(row_dict, ["Invoice ID", "invoice_id", "line_item_id"], idx),
            "instruction": SYSTEM_PROMPT,
            "input": format_input_prompt(row_dict),
            "output": format_output_json(row_dict, idx)
        })

    os.makedirs(output_dir, exist_ok=True)
    total = len(formatted_data)
    val_count = max(1, int(total * val_ratio))
    train_count = total - val_count

    train_data = formatted_data[:train_count]
    val_data = formatted_data[train_count:]

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    print(f"[OK] Processed {total} records from {os.path.basename(invoice_csv)} -> Train: {train_count} ({train_path}), Val: {val_count} ({val_path})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoopLens SLM Data Preparation")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory containing raw CSV/Excel extracts")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for JSONL datasets")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    args = parser.parse_args()

    prepare_dataset_from_csv(args.raw_dir, args.output_dir, args.val_ratio)
