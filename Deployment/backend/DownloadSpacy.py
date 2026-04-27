from huggingface_hub import snapshot_download
import spacy
from pathlib import Path

HF_MODEL_REPO = "Tudorx95/NER_Economic_Political_Spacy"
MODEL_DIR = Path("/app/models/spacy")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading spaCy model {HF_MODEL_REPO} ...")
local_path = snapshot_download(repo_id=HF_MODEL_REPO, local_dir=str(MODEL_DIR))

nlp = spacy.load(local_path)
print("spaCy model loaded successfully.")
print(f"Model path: {local_path}")