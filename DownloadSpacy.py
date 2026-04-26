from huggingface_hub import snapshot_download
import os
from pathlib import Path
import spacy

HF_DATASET_REPO = "Tudorx95/NER_Political_Economic"
HF_MODEL_REPO   = "Tudorx95/NER_Economic_Political_Spacy"
BASE_DIR = Path("/mnt/ssd/tudor.lepadatu/AI_CD")
MODEL_OUT_DIR = BASE_DIR / "spacy_finetuned"

print(f"Se descarca {HF_MODEL_REPO} de pe HuggingFace...")
local_model_path = snapshot_download(
    repo_id=HF_MODEL_REPO,
    local_dir=str(MODEL_OUT_DIR / "hf_download"),
)

print(f"Model descarcat in: {local_model_path}")
hf_model = spacy.load(local_model_path)

text = "The IMF warned that rising US interest rates could trigger a global recession."
doc = hf_model(text)

print(f"\nText: {text}\n")
print("Entitati detectate:")
for ent in doc.ents:
    print(f"  {ent.text:<30} -> {ent.label_:<20}")

print("\n=== TOTUL OK ===")
print(f"Dataset:  https://huggingface.co/datasets/{HF_DATASET_REPO}")
print(f"Model:    https://huggingface.co/{HF_MODEL_REPO}")
