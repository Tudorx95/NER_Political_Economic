"""
NER Demo Backend — FastAPI
Descarca modelele GLiNER si spaCy de pe HuggingFace la pornire,
apoi serveste inferenta pe ambele modele via REST API.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
GLINER_REPO = os.getenv("GLINER_REPO", "Tudorx95/NER_Economic_Political")
SPACY_REPO  = os.getenv("SPACY_REPO",  "Tudorx95/NER_Economic_Political_Spacy")
MODELS_DIR  = Path("/app/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NER_LABELS = [
    "POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
    "CURRENCY", "TRADE_AGREEMENT", "GPE",
]

LABEL_COLORS = {
    "POLITICIAN":         "#FF6B6B",
    "POLITICAL_PARTY":    "#4ECDC4",
    "POLITICAL_ORG":      "#45B7D1",
    "FINANCIAL_ORG":      "#96CEB4",
    "ECONOMIC_INDICATOR": "#FFEAA7",
    "POLICY":             "#DDA0DD",
    "LEGISLATION":        "#98D8C8",
    "MARKET_EVENT":       "#F7DC6F",
    "CURRENCY":           "#BB8FCE",
    "TRADE_AGREEMENT":    "#85C1E9",
    "GPE":                "#F0B27A",
}

# ── Global model holders ──────────────────────────────────────
gliner_model = None
spacy_model  = None

# ── Pydantic schemas ──────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: str  # "gliner" | "spacy"
    threshold: float = 0.5
    extra_labels: Optional[List[str]] = None  # zero-shot labels pt GLiNER

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: Optional[float] = None

class PredictResponse(BaseModel):
    model: str
    entities: List[Entity]
    labels_used: List[str]

class HealthResponse(BaseModel):
    status: str
    gliner_loaded: bool
    spacy_loaded: bool

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="NER Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────
def load_gliner():
    global gliner_model
    logger.info(f"Descarc GLiNER: {GLINER_REPO}...")
    torch.backends.cudnn.enabled = False
    from gliner import GLiNER
    gliner_model = GLiNER.from_pretrained(GLINER_REPO)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gliner_model.to(device)
    logger.info(f"GLiNER incarcat pe {device}")

def load_spacy():
    global spacy_model
    logger.info(f"Descarc spaCy: {SPACY_REPO}...")
    from huggingface_hub import snapshot_download
    import spacy
    local_path = snapshot_download(
        repo_id=SPACY_REPO,
        local_dir=str(MODELS_DIR / "spacy"),
    )
    spacy_model = spacy.load(local_path)
    logger.info(f"spaCy incarcat din {local_path}")

@app.on_event("startup")
async def startup():
    """Descarca si incarca ambele modele la pornirea containerului."""
    try:
        load_gliner()
    except Exception as e:
        logger.error(f"Eroare GLiNER: {e}")
    try:
        load_spacy()
    except Exception as e:
        logger.error(f"Eroare spaCy: {e}")

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        gliner_loaded=gliner_model is not None,
        spacy_loaded=spacy_model is not None,
    )

@app.get("/labels")
def get_labels():
    return {
        "labels": NER_LABELS,
        "colors": LABEL_COLORS,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text gol")

    if req.model == "gliner":
        if gliner_model is None:
            raise HTTPException(503, "GLiNER nu este incarcat")
        labels = list(NER_LABELS)
        if req.extra_labels:
            labels.extend(req.extra_labels)
        raw = gliner_model.predict_entities(
            req.text, labels, threshold=req.threshold
        )
        entities = [
            Entity(
                text=e["text"], label=e["label"],
                start=e["start"], end=e["end"],
                score=round(e["score"], 4),
            ) for e in raw
        ]
        return PredictResponse(model="gliner", entities=entities, labels_used=labels)

    elif req.model == "spacy":
        if spacy_model is None:
            raise HTTPException(503, "spaCy nu este incarcat")
        doc = spacy_model(req.text)
        entities = [
            Entity(
                text=ent.text, label=ent.label_,
                start=ent.start_char, end=ent.end_char,
                score=None,
            ) for ent in doc.ents
        ]
        return PredictResponse(
            model="spacy", entities=entities, labels_used=NER_LABELS
        )
    else:
        raise HTTPException(400, f"Model necunoscut: {req.model}")

@app.post("/predict_both")
def predict_both(req: PredictRequest):
    """Ruleaza inferenta pe AMBELE modele simultan."""
    results = {}
    for model_name in ["gliner", "spacy"]:
        try:
            sub_req = PredictRequest(
                text=req.text, model=model_name,
                threshold=req.threshold,
                extra_labels=req.extra_labels,
            )
            results[model_name] = predict(sub_req).dict()
        except HTTPException as e:
            results[model_name] = {"error": str(e.detail)}
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
