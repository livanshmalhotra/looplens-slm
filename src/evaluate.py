import os
import json
import argparse
from tqdm import tqdm

def evaluate_predictions(predictions_file: str):
    """Evaluates predicted JSON output files against validation targets."""
    if not os.path.exists(predictions_file):
        print(f"[!] Evaluation file {predictions_file} does not exist.")
        return

    print(f"[*] Benchmarking predictions from {predictions_file}...")
    
    total = 0
    valid_json = 0
    taxonomy_l1_match = 0
    taxonomy_l2_match = 0
    taxonomy_l3_match = 0
    unspsc_match = 0
    vendor_match = 0

    with open(predictions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        item = json.loads(line)
        target_str = item.get("target_output", "")
        pred_str = item.get("predicted_output", "")

        total += 1

        try:
            target_json = json.loads(target_str)
        except Exception:
            target_json = {}

        try:
            pred_json = json.loads(pred_str)
            valid_json += 1
        except Exception:
            pred_json = {}

        # Evaluate Taxonomy matches
        t_tax = target_json.get("taxonomy_classification", {})
        p_tax = pred_json.get("taxonomy_classification", {})

        if t_tax.get("level_1") and t_tax.get("level_1") == p_tax.get("level_1"):
            taxonomy_l1_match += 1
        if t_tax.get("level_2") and t_tax.get("level_2") == p_tax.get("level_2"):
            taxonomy_l2_match += 1
        if t_tax.get("level_3") and t_tax.get("level_3") == p_tax.get("level_3"):
            taxonomy_l3_match += 1
        if t_tax.get("unspsc_code") and str(t_tax.get("unspsc_code")) == str(p_tax.get("unspsc_code")):
            unspsc_match += 1

        # Evaluate Entity resolution matches
        t_ent = target_json.get("entity_resolution", {})
        p_ent = pred_json.get("entity_resolution", {})
        if t_ent.get("normalized_vendor_name") and t_ent.get("normalized_vendor_name") == p_ent.get("normalized_vendor_name"):
            vendor_match += 1

    print("\n==============================================")
    print("      LoopLens SLM Benchmark Results          ")
    print("==============================================")
    print(f" Total Evaluated Items  : {total}")
    print(f" Valid JSON Output Rate : {valid_json / total * 100:.2f}% ({valid_json}/{total})")
    print(f" Taxonomy Level 1 Acc  : {taxonomy_l1_match / total * 100:.2f}% ({taxonomy_l1_match}/{total})")
    print(f" Taxonomy Level 2 Acc  : {taxonomy_l2_match / total * 100:.2f}% ({taxonomy_l2_match}/{total})")
    print(f" Taxonomy Level 3 Acc  : {taxonomy_l3_match / total * 100:.2f}% ({taxonomy_l3_match}/{total})")
    print(f" UNSPSC Code Match Rate: {unspsc_match / total * 100:.2f}% ({unspsc_match}/{total})")
    print(f" Vendor Norm Accuracy  : {vendor_match / total * 100:.2f}% ({vendor_match}/{total})")
    print("==============================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoopLens SLM Evaluator")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl", help="Validation dataset JSONL")
    args = parser.parse_args()

    # Run benchmark on validation set sample target
    print(f"[*] Running baseline validation check on {args.val_file}")
    with open(args.val_file, "r", encoding="utf-8") as f:
        sample_eval_items = []
        for line in f:
            data = json.loads(line)
            sample_eval_items.append(json.dumps({
                "target_output": data["output"],
                "predicted_output": data["output"] # baseline self-test
            }))

    temp_eval_path = "outputs/benchmark_temp.jsonl"
    os.makedirs("outputs", exist_ok=True)
    with open(temp_eval_path, "w", encoding="utf-8") as f:
        for item in sample_eval_items:
            f.write(item + "\n")

    evaluate_predictions(temp_eval_path)
