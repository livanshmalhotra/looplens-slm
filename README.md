# LoopLens SLM Engine

External Small Language Model (SLM) fine-tuning, evaluation, and serving repository for **LoopLens** procurement analytics.

This project enables QLoRA fine-tuning of open-weights models (e.g., Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, Phi-3.5-mini) to perform 3 core procurement intelligence tasks:
1. **Spend Taxonomy Auto-Classification & UNSPSC Mapping**: Converts raw, messy invoice/PO line descriptions into standardized 4-level category hierarchies and 8-digit UNSPSC codes.
2. **Vendor Name Normalization & Deduplication**: Standardizes disparate vendor string variations into canonical Master Supplier IDs.
3. **Procurement Risk & Rogue Spend Detection**: Analyzes line items and payment terms to flag uncontracted and non-PO spend anomalies.

---

## 📁 Repository Structure

```text
LoopLens-SLM-Engine/
├── README.md                   # Setup and execution guide
├── requirements.txt            # Python dependencies
├── Dockerfile                  # GPU inference container specification
├── configs/
│   └── qlora_config.yaml       # Hyperparameters & LoRA configuration
├── data/
│   ├── raw/                    # PostgreSQL exported CSVs from LoopLens
│   ├── processed/              # Formatted train.jsonl and val.jsonl datasets
│   └── taxonomy_hierarchy.json # Reference UNSPSC & 4-level category taxonomy
├── notebooks/
│   ├── 01_data_extraction.ipynb # Notebook guide for extracting training pairs
│   └── 02_model_evaluation.ipynb # Interactive evaluation & accuracy metrics
└── src/
    ├── dataset.py              # Prompt formatter & Dataset building utilities
    ├── train.py                # QLoRA SFT training pipeline (TRL / Unsloth)
    ├── evaluate.py             # Model accuracy & JSON validity benchmarking script
    └── serve.py                # FastAPI REST endpoint for real-time LoopLens inference
```

---

## ⚡ Quick Start & Workflow

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Export invoice and sourcing data from LoopLens PostgreSQL database into `data/raw/` or run the formatting script to create JSONL instruction pairs:

```bash
python -m src.dataset --raw_dir data/raw --output_dir data/processed
```

### 3. QLoRA Fine-Tuning

Execute QLoRA fine-tuning using parameters specified in `configs/qlora_config.yaml`:

```bash
python -m src.train --config configs/qlora_config.yaml
```

### 4. Model Evaluation

Benchmark taxonomy accuracy, UNSPSC match rate, vendor normalization accuracy, and strict JSON format adherence:

```bash
python -m src.evaluate --model_path outputs/final_adapter --val_file data/processed/val.jsonl
```

### 5. Serving Endpoint for LoopLens Integration

Launch the FastAPI serving server:

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

Test inference endpoint:

```bash
curl -X POST "http://localhost:8000/predict_spend_category" \
     -H "Content-Type: application/json" \
     -d '{
       "line_item_id": 104291,
       "line_item_description": "Content Development - Global Induction E-Learning Module",
       "vendor_name": "IQUAD MIDDLE EAST SAL",
       "total_amount_local": 15000.0,
       "currency": "USD"
     }'
```
