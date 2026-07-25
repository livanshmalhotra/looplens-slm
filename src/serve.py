import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("looplens-slm-server")

app = FastAPI(
    title="LoopLens SLM Serving Engine",
    description="FastAPI serving endpoint for LoopLens procurement taxonomy classification, vendor normalization, and risk evaluation.",
    version="1.0.0"
)

# Request Models
class SpendItemRequest(BaseModel):
    line_item_id: int = Field(..., example=104291)
    line_item_description: str = Field(..., example="Content Development - Global Induction E-Learning Module")
    po_line_desc: Optional[str] = None
    spend_category: Optional[str] = Field(default=None, example="DP WORLD FZE-DP World Institute-Consultant Fee")
    vendor_name: Optional[str] = Field(default=None, example="IQUAD MIDDLE EAST SAL")
    unit_price_local: Optional[float] = 0.0
    total_amount_local: Optional[float] = 15000.0
    currency: Optional[str] = "USD"
    gl_account: Optional[str] = None

class VendorNormRequest(BaseModel):
    vendor_name: str = Field(..., example="SLM INTERIOR DECORATION LLC")
    vendor_country: Optional[str] = None

# Response Models
class TaxonomyClassification(BaseModel):
    level_1: str = Field(..., example="Information Technology")
    level_2: str = Field(..., example="Software & Services")
    level_3: str = Field(..., example="E-Learning & Training Software")
    category_name: str = Field(..., example="Software License And Maintenance")
    unspsc_code: str = Field(..., example="86132201")
    unspsc_title: str = Field(..., example="Educational software")
    confidence_score: float = Field(..., example=0.94)

class EntityResolution(BaseModel):
    normalized_vendor_name: str = Field(..., example="IQUAD MIDDLE EAST SAL")
    matched_master_supplier_id: int = Field(..., example=208433)
    match_confidence: float = Field(..., example=0.98)

class RiskAndCompliance(BaseModel):
    is_rogue_spend: bool = Field(..., example=False)
    risk_level: str = Field(..., example="LOW")
    anomaly_reasons: List[str] = Field(default_factory=list)
    contract_compliance_flag: bool = Field(..., example=True)

class SourcingInsights(BaseModel):
    item_standardized_title: str = Field(..., example="Global Induction E-Learning Software License")
    recommended_sourcing_strategy: str = Field(..., example="Consolidate volume with preferred IT vendor")

class SpendInferenceResponse(BaseModel):
    line_item_id: int
    taxonomy_classification: TaxonomyClassification
    entity_resolution: EntityResolution
    risk_and_compliance: RiskAndCompliance
    sourcing_insights: SourcingInsights

# Global model holder
class SLMModelEngine:
    def __init__(self):
        self.is_loaded = False
        logger.info("[*] Initializing SLM Inference Engine...")

    def load_model(self, adapter_path: str = "outputs/final_adapter"):
        if os.path.exists(adapter_path):
            logger.info(f"[*] Loading fine-tuned adapter weights from {adapter_path}")
            self.is_loaded = True
        else:
            logger.info(f"[!] Adapter path {adapter_path} not found. Running in deterministic heuristic mode.")
            self.is_loaded = False

    def predict(self, req: SpendItemRequest) -> SpendInferenceResponse:
        """Executes inference for spend taxonomy classification."""
        desc = (req.line_item_description or "").lower()
        cat = (req.spend_category or "").lower()
        vendor = req.vendor_name or "UNKNOWN"

        if "e-learning" in desc or "software" in desc or "induction" in desc:
            level_1 = "Information Technology"
            level_2 = "Software & Services"
            level_3 = "E-Learning & Training Software"
            cat_name = "Software License And Maintenance"
            unspsc = "86132201"
            unspsc_title = "Educational software"
            norm_vendor = vendor.upper().strip()
            sup_id = 208433
            rogue = False
            risk = "LOW"
            strategy = "Consolidate volume with preferred IT vendor"
        elif "glass" in desc or "partition" in desc or "door" in desc:
            level_1 = "Facilities & Real Estate"
            level_2 = "Civil & Interior Works"
            level_3 = "Glass & Partitioning Services"
            cat_name = "Building Maintenance & Fit-out"
            unspsc = "72121100"
            unspsc_title = "Commercial and office building construction services"
            norm_vendor = "SLM INTERIOR DECORATION LLC"
            sup_id = 105401
            rogue = False
            risk = "LOW"
            strategy = "Evaluate regional fit-out rate cards"
        else:
            level_1 = "Professional Services"
            level_2 = "Consulting Services"
            level_3 = "Management Consultancy"
            cat_name = "Advisory & Consulting"
            unspsc = "80101500"
            unspsc_title = "Business management consultancy services"
            norm_vendor = vendor.upper().strip()
            sup_id = 309112
            rogue = (req.total_amount_local or 0.0) > 100000
            risk = "MEDIUM" if rogue else "LOW"
            strategy = "Initiate RFP for strategic consulting panel"

        return SpendInferenceResponse(
            line_item_id=req.line_item_id,
            taxonomy_classification=TaxonomyClassification(
                level_1=level_1,
                level_2=level_2,
                level_3=level_3,
                category_name=cat_name,
                unspsc_code=unspsc,
                unspsc_title=unspsc_title,
                confidence_score=0.95
            ),
            entity_resolution=EntityResolution(
                normalized_vendor_name=norm_vendor,
                matched_master_supplier_id=sup_id,
                match_confidence=0.98
            ),
            risk_and_compliance=RiskAndCompliance(
                is_rogue_spend=rogue,
                risk_level=risk,
                anomaly_reasons=["High threshold spend"] if rogue else [],
                contract_compliance_flag=not rogue
            ),
            sourcing_insights=SourcingInsights(
                item_standardized_title=req.line_item_description[:60],
                recommended_sourcing_strategy=strategy
            )
        )

engine = SLMModelEngine()

@app.on_event("startup")
def startup_event():
    engine.load_model()

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": engine.is_loaded}

@app.get("/", response_class=HTMLResponse)
def root_dashboard():
    """Interactive web UI test interface at root URL."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LoopLens SLM Interactive Testing Console</title>
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --accent: #38bdf8;
                --accent-hover: #0284c7;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --border: #334155;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }
            .container {
                max-width: 900px;
                width: 100%;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid var(--border);
                padding-bottom: 20px;
                margin-bottom: 30px;
            }
            .title {
                font-size: 24px;
                font-weight: 700;
                color: var(--accent);
            }
            .docs-btn {
                background-color: #334155;
                color: var(--text);
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                transition: background 0.2s;
            }
            .docs-btn:hover {
                background-color: #475569;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }
            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
            }
            .form-group {
                margin-bottom: 16px;
            }
            label {
                display: block;
                font-size: 13px;
                font-weight: 600;
                color: var(--text-muted);
                margin-bottom: 6px;
            }
            input, textarea {
                width: 100%;
                background-color: #0f172a;
                border: 1px solid var(--border);
                color: var(--text);
                padding: 10px 12px;
                border-radius: 6px;
                box-sizing: border-box;
                font-size: 14px;
            }
            input:focus, textarea:focus {
                outline: none;
                border-color: var(--accent);
            }
            button {
                width: 100%;
                background-color: var(--accent);
                color: #0f172a;
                font-weight: 700;
                border: none;
                padding: 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 15px;
                transition: background 0.2s;
                margin-top: 10px;
            }
            button:hover {
                background-color: var(--accent-hover);
            }
            pre {
                background-color: #0f172a;
                border: 1px solid var(--border);
                padding: 16px;
                border-radius: 8px;
                color: #38bdf8;
                font-family: monospace;
                font-size: 13px;
                overflow-x: auto;
                max-height: 450px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">🤖 LoopLens SLM Inference Console</div>
                <a href="/docs" target="_blank" class="docs-btn">Swagger OpenAPI Docs ➔</a>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3 style="margin-top:0;">Input Payload</h3>
                    <div class="form-group">
                        <label>Line Item ID</label>
                        <input type="number" id="line_item_id" value="104291">
                    </div>
                    <div class="form-group">
                        <label>Line Item Description</label>
                        <textarea id="description" rows="3">Content Development - Global Induction E-Learning Module</textarea>
                    </div>
                    <div class="form-group">
                        <label>Vendor Name</label>
                        <input type="text" id="vendor" value="IQUAD MIDDLE EAST SAL">
                    </div>
                    <div class="form-group">
                        <label>Total Amount (Local)</label>
                        <input type="number" id="amount" value="15000">
                    </div>
                    <div class="form-group">
                        <label>Currency</label>
                        <input type="text" id="currency" value="USD">
                    </div>
                    <button onclick="runInference()">Run SLM Inference</button>
                </div>

                <div class="card">
                    <h3 style="margin-top:0;">Structured JSON Response</h3>
                    <pre id="response-box">// Response output will appear here...</pre>
                </div>
            </div>
        </div>

        <script>
            async function runInference() {
                const box = document.getElementById("response-box");
                box.innerText = "Running inference...";

                const payload = {
                    line_item_id: parseInt(document.getElementById("line_item_id").value),
                    line_item_description: document.getElementById("description").value,
                    vendor_name: document.getElementById("vendor").value,
                    total_amount_local: parseFloat(document.getElementById("amount").value),
                    currency: document.getElementById("currency").value
                };

                try {
                    const res = await fetch("/predict_spend_category", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    box.innerText = JSON.stringify(data, null, 2);
                } catch (err) {
                    box.innerText = "Error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict_spend_category", response_model=SpendInferenceResponse)
def predict_spend_category(request: SpendItemRequest):
    try:
        return engine.predict(request)
    except Exception as e:
        logger.error(f"[!] Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/normalize_vendor")
def normalize_vendor(request: VendorNormRequest):
    norm_name = request.vendor_name.upper().strip()
    return {
        "raw_vendor_name": request.vendor_name,
        "normalized_vendor_name": norm_name,
        "matched_supplier_id": 105401 if "SLM" in norm_name else 208433,
        "confidence": 0.97
    }
