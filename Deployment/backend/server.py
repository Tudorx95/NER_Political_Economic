from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import os
import uuid
from datetime import datetime, timezone
import spacy
from gliner import GLiNER
import torch

# ------------------- Config -------------------
load_dotenv()
mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
db_name = os.getenv("DB_NAME", "test_database")

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

app = FastAPI(title="NER Compare API - Real Models")
api_router = APIRouter(prefix="/api")

COUNTRY_CONTEXTS = {
    "Romania": "Romania's economy is highly integrated with the European Union, experiencing growth in the IT and manufacturing sectors. The central bank recently discussed monetary policy adjustments.",
}

DEFAULT_CONTEXT = "The political and economic context for {country} involves ongoing legislative reforms and shifts in market indicators. Regional trade agreements and central bank policies play a crucial role in shaping its financial landscape."

# ------------------- Model Loading at Startup -------------------
print("Loading models... (this may take a while on first run)")

# spaCy
print("Loading spaCy model...")
spacy_nlp = spacy.load("/app/models/spacy")

# GLiNER
print("Loading GLiNER model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
gliner_model = GLiNER.from_pretrained("Tudorx95/NER_Economic_Political")
gliner_model.to(device)

NER_LABELS = [
    "POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
    "CURRENCY", "TRADE_AGREEMENT", "GPE",
]

# ------------------- Pydantic Models -------------------
class NERRequest(BaseModel):
    text: str = Field(min_length=1)
    model_type: str = Field(default="compare", pattern="^(spacy|gliner|compare)$")
    country: Optional[str] = None

class NEREntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class NERResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    country: Optional[str] = None
    spacy_entities: Optional[List[NEREntity]] = None
    gliner_entities: Optional[List[NEREntity]] = None
    model_type: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ------------------- Real Inference Functions -------------------
def run_spacy(text: str) -> List[Dict]:
    doc = spacy_nlp(text)
    entities = []
    for ent in doc.ents:
        # Map to your label set if needed (spaCy may use different labels)
        label = ent.label_  # or map if necessary
        entities.append({
            "text": ent.text,
            "label": label,
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": 0.95,   # spaCy doesn't give confidence by default
        })
    return entities

def run_gliner(text: str) -> List[Dict]:
    entities = gliner_model.predict_entities(
        text,
        NER_LABELS,
        threshold=0.5
    )
    return [
        {
            "text": e["text"],
            "label": e["label"],
            "start": e["start"],
            "end": e["end"],
            "confidence": round(float(e["score"]), 3),
        }
        for e in entities
    ]

# ------------------- Endpoints -------------------
@api_router.post("/ner/analyze", response_model=NERResponse)
async def analyze_ner(request: NERRequest):
    spacy_entities = None
    gliner_entities = None

    if request.model_type in ("spacy", "compare"):
        spacy_entities = run_spacy(request.text)
    if request.model_type in ("gliner", "compare"):
        gliner_entities = run_gliner(request.text)

    result = NERResponse(
        text=request.text,
        country=request.country,
        spacy_entities=spacy_entities,
        gliner_entities=gliner_entities,
        model_type=request.model_type,
    )

    await db.ner_analyses.insert_one(result.model_dump())

    return result

@api_router.get("/countries")
async def get_countries():
    countries_list = ["Libya", "Chad", "Romania", "USA", "Germany", "France", "China"]
    return {"countries": countries_list}

@api_router.get("/countries/{country}/context")
async def get_country_context(country: str):
    context = COUNTRY_CONTEXTS.get(country, DEFAULT_CONTEXT.format(country=country))
    return {"text": context}

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # or configure properly in production
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)