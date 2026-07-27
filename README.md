# 🤖 LoopLens SLM — Natural Language Procurement Assistant

**LoopLens SLM** transforms enterprise procurement data into a natural-language Q&A engine powered by fine-tuned Small Language Models (**Qwen2.5-0.5B-Instruct**). Non-technical procurement managers can ask plain-English questions about spend, suppliers, contracts, and RFQs, and receive instant, plain-English answers.

---

## 🌟 Features

- **Programmatic Q&A Dataset Generator**: Automatically extracts structured facts from 13 raw database extracts (CSV/XLSX) to produce 15,000+ instruction-tuned Q&A pairs.
- **Resource-Efficient QLoRA Fine-Tuning**: Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` using 4-bit quantization, requiring minimal VRAM (~1.5GB) and supporting CPU execution.
- **Interactive Browser UI**: Glassmorphic, dark-mode web chat interface served via FastAPI.
- **Terminal CLI Chat**: Interactive color-coded CLI for command-line users.
- **Comprehensive Procurement Coverage**:
  1. **Spend Analytics**: Total spend, vendor totals, category breakdowns, PO compliance.
  2. **Supplier Analytics**: Status, country distribution, risk scores, ESG ratings.
  3. **Contract Analytics**: Total portfolio value, active/expired statuses, contract details.
  4. **PO Analytics**: Buyer totals, line item descriptions, order amounts.
  5. **Sourcing Analytics**: RFQ counts, participation rates, division metrics.
  6. **Spend Classification**: Direct invoice line-item category predictions.

---

## 🚀 Quickstart Commands

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Training Dataset from Raw Data
Extracts facts from all 13 raw CSV/XLSX files in `data/raw/` and generates `train.jsonl` and `val.jsonl`:
```bash
python -m src.generate_qa --raw_dir data/raw --output_dir data/processed
```

### Step 3: Train the SLM (QLoRA Fine-Tuning)
Fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` and saves the adapter to `outputs/qlora_adapter/final_adapter/`:
```bash
python -m src.train --config configs/qlora_config.yaml
```

### Step 4: Run the Web Server (Browser Chat UI)
Starts the FastAPI server with the embedded chat application at `http://localhost:8000`:
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```
- **Browser UI**: Open `http://localhost:8000` in your web browser.
- **OpenAPI / Swagger Docs**: Open `http://localhost:8000/docs`.

### Step 5: Run the Terminal CLI
In a separate terminal window, launch the interactive command-line chat:
```bash
python -m src.cli
```

---

## 📁 Repository Structure

```
c:/looplens-slm/
├── configs/
│   └── qlora_config.yaml     # QLoRA hyperparameters & model config
├── data/
│   ├── raw/                  # 13 raw CSV & XLSX data files
│   └── processed/            # Generated train.jsonl & val.jsonl datasets
├── src/
│   ├── generate_qa.py        # Programmatic dataset generator script
│   ├── train.py              # QLoRA fine-tuning script
│   ├── serve.py              # FastAPI server & embedded web UI
│   └── cli.py                # Terminal interactive chat CLI
├── Dockerfile                # Docker deployment file
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```
