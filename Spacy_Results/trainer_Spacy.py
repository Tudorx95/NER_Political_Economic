import os
from pathlib import Path

# Mount Google Drive (Colab)

BASE_DIR = Path("/mnt/ssd/tudor.lepadatu/AI_CD")

SPLITS_DIR    = BASE_DIR / "splits"
MODEL_OUT_DIR = BASE_DIR / "spacy_finetuned"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Hiperparametri ──────────────────────────────────────────
NUM_EPOCHS       = 15
BATCH_SIZE       = 8
LEARNING_RATE    = 2e-5                     # learning rate pt fine-tuning NER head
DROP_RATE        = 0.35                     # dropout pt NER layer
EVAL_THRESHOLD   = 0.5

# HuggingFace repos
HF_DATASET_REPO = "Tudorx95/NER_Political_Economic"
HF_MODEL_REPO   = "Tudorx95/NER_Economic_Political_Spacy"

# Schema NER — aceleasi 11 labels ca la GLiNER
NER_LABELS = [
    "POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
    "CURRENCY", "TRADE_AGREEMENT", "GPE",
]

print(f"BASE_DIR        = {BASE_DIR}")
print(f"MODEL_OUT_DIR   = {MODEL_OUT_DIR}")
print(f"HF model repo   = {HF_MODEL_REPO}")
print(f"Labels:           {len(NER_LABELS)} custom NER labels")


import torch
import spacy


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:          {torch.cuda.get_device_name(0)}")
    print(f"Memory total:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = torch.device("cuda:0")
else:
    print("ATENTIE: rulezi pe CPU. Antrenarea va fi mai lenta.")
    DEVICE = torch.device("cpu")

print(f"\nspaCy version:   {spacy.__version__}")
print(f"GPU spaCy:       {spacy.prefer_gpu()}")


# Load Dataset and convert to spacy format

import json
import random

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# Incarcam splits-urile
train_raw = load_jsonl(SPLITS_DIR / "train.jsonl")
dev_raw   = load_jsonl(SPLITS_DIR / "dev.jsonl")
test_raw  = load_jsonl(SPLITS_DIR / "test.jsonl")

print(f"Raw counts: train={len(train_raw)}, dev={len(dev_raw)}, test={len(test_raw)}")


# ─── Conversie JSONL -> format spaCy training ────────────────────────────────
# Format: [(text, {"entities": [(start, end, label), ...]}), ...]

def convert_to_spacy_format(examples):
    """Converteste lista de exemple JSONL in format spaCy."""
    data = []
    skipped_docs = 0
    total_ents = 0
    kept_ents = 0

    for ex in examples:
        text = ex["text"]
        entities = []

        for ent in ex.get("entities", []):
            total_ents += 1
            start, end, label = ent["start"], ent["end"], ent["label"]

            # Validare: span-ul trebuie sa fie valid
            if start < 0 or end > len(text) or start >= end:
                continue

            # Verificam suprapuneri cu entitati deja adaugate
            overlap = False
            for (s2, e2, _) in entities:
                if start < e2 and end > s2:
                    overlap = True
                    break

            if not overlap:
                entities.append((start, end, label))
                kept_ents += 1

        if entities:
            data.append((text, {"entities": entities}))
        else:
            skipped_docs += 1

    print(f"  Docs convertite: {len(data)}, sarite (fara entitati): {skipped_docs}")
    print(f"  Entitati: {kept_ents}/{total_ents} pastrate")
    return data

print("Conversie train...")
train_data = convert_to_spacy_format(train_raw)
print("Conversie dev...")
dev_data = convert_to_spacy_format(dev_raw)
print("Conversie test...")
test_data = convert_to_spacy_format(test_raw)


# Verificam primele 3 exemple
for i, (text, annot) in enumerate(train_data[:3]):
    print(f"\nExemplu {i}: \"{text[:80]}...\"")
    for (s, e, lbl) in annot["entities"]:
        print(f"  {text[s:e]:<35} -> {lbl:<20} [{s}:{e}]")


import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import warnings
warnings.filterwarnings("ignore")


# Incarcam modelul transformer pretrained
print("Se incarca en_core_web_trf...")
nlp = spacy.load("en_core_web_trf")

# Eliminam NER-ul pretrained (cu labels standard: PERSON, ORG, etc.)
if "ner" in nlp.pipe_names:
    nlp.remove_pipe("ner")
    print("NER pretrained eliminat.")

# Adaugam NER head nou cu labels custom
ner = nlp.add_pipe("ner", last=True)
for label in NER_LABELS:
    ner.add_label(label)

print(f"\nPipeline: {nlp.pipe_names}")
print(f"NER labels: {ner.labels}")
print(f"Total labels: {len(ner.labels)}")


## Training

# ─── Pregatire date de antrenare ──────────────────────────────────────────────
# Cream Example objects din datele noastre

print("Se creeaza exemple de antrenare...")
train_examples = []
for text, annots in train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annots)
    train_examples.append(example)

dev_examples = []
for text, annots in dev_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annots)
    dev_examples.append(example)

print(f"Train examples: {len(train_examples)}")
print(f"Dev examples:   {len(dev_examples)}")



# Initializam DOAR NER-ul cu exemple (ca sa invete tranzitiile B/I/L/U/O)
ner = nlp.get_pipe("ner")
ner.initialize(lambda: train_examples, nlp=nlp)



# ─── Training loop ───────────────────────────────────────────────────────────
from tqdm.auto import tqdm
import numpy as np

# Componente pe care NU le antrenam (freezam transformer-ul pt viteza)
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in ("ner", "transformer")]

best_model_path = MODEL_OUT_DIR / "model-best"  # valoare default

best_f1 = 0.0
patience_counter = 0
PATIENCE = 5        # opreste daca F1 nu se imbunatateste 3 epoci consecutiv

print(f"\nIncepe antrenarea ({NUM_EPOCHS} epoci)...")
print(f"  Pipes frozen: {other_pipes}")
print(f"  NER labels:   {ner.labels}")
print(f"  Patience:     {PATIENCE}")
print("=" * 60)

# Initializam optimizatorul
optimizer = nlp.resume_training()
optimizer.learn_rate = 1e-3       # LR mare pentru NER head (neinițializat)
optimizer.eps = 1e-8


with nlp.disable_pipes(*other_pipes):
    for epoch in range(NUM_EPOCHS):
        random.shuffle(train_examples)
        losses = {}

        # Mini-batch training
        batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
        batch_count = 0

        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=DROP_RATE, losses=losses)
            batch_count += 1

        # Evaluare pe dev set
        scores = nlp.evaluate(dev_examples)
        dev_f1 = scores.get("ents_f", 0.0)
        dev_p  = scores.get("ents_p", 0.0)
        dev_r  = scores.get("ents_r", 0.0)

        print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
              f"Loss: {losses.get('ner', 0):.4f} | "
              f"Dev P: {dev_p:.4f} R: {dev_r:.4f} F1: {dev_f1:.4f}", end="")

        # Salvare best model
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            best_model_path = MODEL_OUT_DIR / "model-best"
            nlp.to_disk(str(best_model_path))
            print(f" ✓ BEST (salvat)")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{PATIENCE})")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping la epoca {epoch+1} (F1 nu s-a imbunatatit {PATIENCE} epoci)")
            break

# Salvam si modelul final (ultima epoca)
last_model_path = MODEL_OUT_DIR / "model-last"
nlp.to_disk(str(last_model_path))

print(f"\nAntrenare completa!")
print(f"  Best F1 (dev): {best_f1:.4f}")
print(f"  Model best:    {best_model_path}")
print(f"  Model last:    {last_model_path}")



# Incarcam best model pentru evaluare curata
eval_nlp = spacy.load(str(best_model_path))
print(f"Model incarcat: {best_model_path}")
print(f"Pipeline: {eval_nlp.pipe_names}")
print(f"NER labels: {eval_nlp.get_pipe('ner').labels}")


# ─── Evaluare nativa spaCy ────────────────────────────────────────────────────
test_examples = []
for text, annots in test_data:
    doc = eval_nlp.make_doc(text)
    example = Example.from_dict(doc, annots)
    test_examples.append(example)

spacy_scores = eval_nlp.evaluate(test_examples)

print("Rezultate spaCy evaluate (test set):")
print(f"  Precision: {spacy_scores.get('ents_p', 0):.4f}")
print(f"  Recall:    {spacy_scores.get('ents_r', 0):.4f}")
print(f"  F1:        {spacy_scores.get('ents_f', 0):.4f}")

if 'ents_per_type' in spacy_scores:
    print(f"\n  {'Label':<22} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"  {'-'*50}")
    for label, metrics in sorted(spacy_scores['ents_per_type'].items()):
        print(f"  {label:<22} {metrics['p']:>8.2f} {metrics['r']:>8.2f} {metrics['f']:>8.2f}")



# ─── Evaluare cu nervaluate (comparabil cu GLiNER) ────────────────────────────
from tqdm.auto import tqdm

print(f"\nSe evalueaza pe {len(test_raw)} exemple test (nervaluate)...")
gold_entities_per_doc = []
pred_entities_per_doc = []

for ex in tqdm(test_raw, desc="Predictie test"):
    text = ex["text"]

    # Gold entities — direct din JSONL
    gold = [
        {"start": e["start"], "end": e["end"], "label": e["label"]}
        for e in ex.get("entities", [])
    ]

    # Predictii spaCy
    doc = eval_nlp(text)
    pred = [
        {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        for ent in doc.ents
    ]

    gold_entities_per_doc.append(gold)
    pred_entities_per_doc.append(pred)

print(f"\nDocumente evaluate: {len(gold_entities_per_doc)}")
print(f"Total entitati gold: {sum(len(g) for g in gold_entities_per_doc)}")
print(f"Total entitati pred: {sum(len(p) for p in pred_entities_per_doc)}")



from nervaluate import Evaluator

evaluator = Evaluator(
    gold_entities_per_doc,
    pred_entities_per_doc,
    tags=NER_LABELS,
)
raw_result = evaluator.evaluate()

# nervaluate poate returna dict sau tuple
if isinstance(raw_result, tuple):
    results         = raw_result[0]
    results_per_tag = raw_result[1]
else:
    results         = raw_result.get("overall", raw_result)
    results_per_tag = raw_result.get("entities", {})

def get_metric(obj, key):
    if isinstance(obj, dict):
        return obj.get(key, 0.0)
    return getattr(obj, key, 0.0)

# ─── Metrici globale ──────────────────────────────────────────
print("=" * 70)
print("METRICI GLOBALE (nervaluate — comparabil cu GLiNER)")
print("=" * 70)
for mode in ["ent_type", "partial", "strict", "exact"]:
    if isinstance(results, dict) and mode in results:
        r = results[mode]
        print(f"\nMod '{mode}':")
        print(f"  Precision: {get_metric(r, 'precision'):.4f}")
        print(f"  Recall:    {get_metric(r, 'recall'):.4f}")
        print(f"  F1:        {get_metric(r, 'f1'):.4f}")

# ─── Metrici per label ────────────────────────────────────────
print("\n" + "=" * 70)
print("METRICI PER LABEL")
print("=" * 70)
print(f"{'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 70)

for label in NER_LABELS:
    support = sum(1 for doc in gold_entities_per_doc for e in doc if e["label"] == label)
    if isinstance(results_per_tag, dict) and label in results_per_tag:
        tag_data = results_per_tag[label]
        if isinstance(tag_data, dict) and "ent_type" in tag_data:
            m = tag_data["ent_type"]
        else:
            m = tag_data
        p = get_metric(m, "precision")
        r = get_metric(m, "recall")
        f = get_metric(m, "f1")
        print(f"{label:<22} {p:>10.4f} {r:>10.4f} {f:>10.4f} {support:>10}")
    else:
        print(f"{label:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {support:>10}")



from datetime import datetime

metrics_summary = {
    "timestamp": datetime.now().isoformat(),
    "framework": "spaCy",
    "backbone": "en_core_web_trf (roberta-base)",
    "num_train_examples": len(train_data),
    "num_dev_examples": len(dev_data),
    "num_test_examples": len(test_data),
    "training_config": {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROP_RATE,
    },
    "best_dev_f1": best_f1,
    "spacy_test_scores": {
        "precision": spacy_scores.get("ents_p", 0),
        "recall": spacy_scores.get("ents_r", 0),
        "f1": spacy_scores.get("ents_f", 0),
    },
    "global_metrics": {},
    "per_label_metrics": {},
}

for mode in ["ent_type", "partial", "strict", "exact"]:
    if isinstance(results, dict) and mode in results:
        r = results[mode]
        metrics_summary["global_metrics"][mode] = {
            "precision": float(get_metric(r, "precision")),
            "recall":    float(get_metric(r, "recall")),
            "f1":        float(get_metric(r, "f1")),
        }

for label in NER_LABELS:
    if isinstance(results_per_tag, dict) and label in results_per_tag:
        tag_data = results_per_tag[label]
        label_entry = {}
        for mode in ["ent_type", "strict"]:
            if isinstance(tag_data, dict) and mode in tag_data:
                m = tag_data[mode]
            else:
                m = tag_data
            label_entry[mode] = {
                "precision": float(get_metric(m, "precision")),
                "recall":    float(get_metric(m, "recall")),
                "f1":        float(get_metric(m, "f1")),
            }
        metrics_summary["per_label_metrics"][label] = label_entry

metrics_path = MODEL_OUT_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_summary, f, indent=2)
print(f"Metrici salvate: {metrics_path}")

if "ent_type" in metrics_summary["global_metrics"]:
    print(f"\nF1 global (ent_type): {metrics_summary['global_metrics']['ent_type']['f1']:.4f}")
if "strict" in metrics_summary["global_metrics"]:
    print(f"F1 global (strict):   {metrics_summary['global_metrics']['strict']['f1']:.4f}")



test_sentences = [
    "The Federal Reserve raised interest rates by 50 basis points, citing persistent inflation pressures.",
    "President Biden signed the Inflation Reduction Act into law after months of congressional negotiations.",
    "The European Central Bank announced a new round of quantitative easing to support the eurozone economy.",
    "Christine Lagarde said the ECB remains committed to its 2% inflation target despite rising energy costs.",
    "NATO members agreed to increase defense spending following the latest G20 summit in Tokyo.",
    "Goldman Sachs analysts predict the dollar will weaken against the yen in the coming quarter.",
    "The Republican Party secured a narrow majority in the Senate after the midterm elections.",
]

print("PREDICTII MODEL FINE-TUNAT (spaCy)")
print("=" * 70)
for sent in test_sentences:
    print(f"\nText: {sent}")
    doc = eval_nlp(sent)
    for ent in doc.ents:
        print(f"  {ent.text:<35} -> {ent.label_:<20}")
    if not doc.ents:
        print("  (nicio entitate detectata)")


from huggingface_hub import login, HfApi

login()

# Verificare ca login-ul a reusit
api = HfApi()
user_info = api.whoami()
print(f"\nAutentificat ca: {user_info['name']}")

from huggingface_hub import HfApi, create_repo

api = HfApi()

# Cream repo daca nu exista
try:
    create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True)
    print(f"Repo model OK: https://huggingface.co/{HF_MODEL_REPO}")
except Exception as e:
    print(f"Eroare la create_repo: {e}")

# ─── Generam README pentru model ──────────────────────────────────────────────
with open(MODEL_OUT_DIR / "metrics.json") as f:
    m = json.load(f)

metrics_table_lines = [
    "| Label | Precision | Recall | F1 |",
    "|-------|-----------|--------|----| "
]
for label in NER_LABELS:
    if label in m.get("per_label_metrics", {}):
        ent = m["per_label_metrics"][label].get("ent_type", {})
        metrics_table_lines.append(
            f"| {label} | {ent.get('precision',0):.3f} | {ent.get('recall',0):.3f} | {ent.get('f1',0):.3f} |"
        )
metrics_table = "\n".join(metrics_table_lines)

g = m.get("global_metrics", {}).get("ent_type", {"precision": 0, "recall": 0, "f1": 0})
s = m.get("spacy_test_scores", {"precision": 0, "recall": 0, "f1": 0})

model_readme = f"""---
license: mit
language:
- en
library_name: spacy
pipeline_tag: token-classification
tags:
- ner
- spacy
- spacy-transformers
- politics
- economics
- roberta
base_model: roberta-base
datasets:
- {HF_DATASET_REPO}
---

# spaCy NER Fine-tuned for Political & Economic Entities

Fine-tuned spaCy `en_core_web_trf` (RoBERTa-base backbone) on a custom
politico-economic NER dataset. Trained to recognize 11 entity types.

## Entity types

`POLITICIAN`, `POLITICAL_PARTY`, `POLITICAL_ORG`, `FINANCIAL_ORG`,
`ECONOMIC_INDICATOR`, `POLICY`, `LEGISLATION`, `MARKET_EVENT`,
`CURRENCY`, `TRADE_AGREEMENT`, `GPE`

## Performance (Test Set)

**spaCy native evaluation:**
- Precision: **{s.get('precision',0):.4f}**
- Recall:    **{s.get('recall',0):.4f}**
- F1:        **{s.get('f1',0):.4f}**

**nervaluate ent_type (comparabil cu GLiNER):**
- Precision: **{g.get('precision',0):.4f}**
- Recall:    **{g.get('recall',0):.4f}**
- F1:        **{g.get('f1',0):.4f}**

**Per label (ent_type):**

{metrics_table}

## Usage

```python
import spacy

nlp = spacy.load("path/to/model")
# sau dupa descarcare de pe HuggingFace:
# from huggingface_hub import snapshot_download
# nlp = spacy.load(snapshot_download("{HF_MODEL_REPO}"))

doc = nlp("The Federal Reserve raised rates after President Biden signed the new bill.")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)
```

## Training Details

- **Framework:** spaCy {spacy.__version__}
- **Backbone:** en_core_web_trf (RoBERTa-base)
- **Strategy:** Frozen transformer + NER head fine-tuning
- **Train examples:** {m.get('num_train_examples', '?')}
- **Dev examples:** {m.get('num_dev_examples', '?')}
- **Test examples:** {m.get('num_test_examples', '?')}
- **Best dev F1:** {m.get('best_dev_f1', 0):.4f}
"""

model_readme_path = MODEL_OUT_DIR / "README.md"
with open(model_readme_path, "w") as f:
    f.write(model_readme)
print(f"README generat: {model_readme_path}")


# ─── Upload model pe HuggingFace ──────────────────────────────────────────────
print(f"Continut {best_model_path}:")
for p in sorted(best_model_path.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(best_model_path)}  ({p.stat().st_size / 1e6:.1f} MB)")

print(f"\nIncarcam modelul in repo {HF_MODEL_REPO}...")
api.upload_folder(
    folder_path=str(best_model_path),
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Upload fine-tuned spaCy NER model (en_core_web_trf)",
)

# Upload README si metrics
api.upload_file(
    path_or_fileobj=str(model_readme_path),
    path_in_repo="README.md",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj=str(MODEL_OUT_DIR / "metrics.json"),
    path_in_repo="metrics.json",
    repo_id=HF_MODEL_REPO,
    repo_type="model",
)

print(f"\nModel uploaded: https://huggingface.co/{HF_MODEL_REPO}")



