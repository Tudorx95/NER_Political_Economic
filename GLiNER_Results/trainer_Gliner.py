import os
from pathlib import Path

BASE_DIR = Path("/mnt/ssd/tudor.lepadatu/AI_CD")

SPLITS_DIR    = BASE_DIR / "splits"                 # dir for train/val/test splits
MODEL_OUT_DIR = BASE_DIR / "gliner_finetuned"       # dir for saving the fine-tuned model and tokenizer

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)    

PRETRAINED_MODEL = "urchade/gliner_small-v2.1"  # alternative: gliner-community/gliner_medium-v2.5
NUM_EPOCHS       = 10
BATCH_SIZE       = 8
LEARNING_RATE    = 3e-6

WARMUP_RATIO     = 0.15     # procentul initial din pasii de antrenare pt care lr creste treptat pana la valoarea specificata
                            # previne gradienti instabili la inceputul antrenamentului
WEIGHT_DECAY     = 0.05     # termen de regularizare ce penalizeaza ponderile mari (previne overfitting)
MAX_GRAD_NORM    = 1.0      # pt gradient clipping (if norm gradient > MAX_GRAD_NORM, se scaleaza gradientii astfel incat norm = MAX_GRAD_NORM)
EVAL_THRESHOLD   = 0.5      # prag folosit la inferenta pt a decide daca predictia e acceptata sau nu (if confidence > EVAL_THRESHOLD, accepta predictia else o respinge)


# HuggingFace repos 
HF_DATASET_REPO = "Tudorx95/NER_Political_Economic"
HF_MODEL_REPO   = "Tudorx95/NER_Economic_Political"

# Schema NER
NER_LABELS = [
    "POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
    "CURRENCY", "TRADE_AGREEMENT", "GPE",  # Geopolitical Entity
]

import torch
torch.backends.cudnn.enabled = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:          {torch.cuda.get_device_name(0)}")
    print(f"Memory total:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = torch.device("cuda:0")
else:
    print("ATENTIE: rulezi pe CPU. Antrenarea va fi LENTA. Activeaza GPU in Colab: Runtime > Change runtime type > GPU.")
    DEVICE = torch.device("cpu")


# Load Dataset and Convert it to Gliner format

import json
import re
from typing import List, Dict, Tuple

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# Tokenizer simplu compatibil cu GLiNER
TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z]\.){2,}"          # abrevieri: U.S., U.K., e.g., St.
    r"|\w+(?:[-']\w+)*"            # cuvinte cu cratima/apostrof: Kwazulu-Natal, don't
    r"|[^\w\s]"                    # punctuatie ramasa
)

# Returneaza lista de tokeni impreuna cu offset-urile lor in text: [(token, char_start, char_end), ...]
# Prin acest Tokenizer se extrag obiectele din lista "entities"

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Returneaza [(token, char_start, char_end), ...]."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]
    # aici efectiv group = token/cuvant, start = pozitia de inceput din text, end = pozitia de sfarsit din text (exclusive)

def convert_to_gliner_format(example: Dict) -> Dict:
    """Converteste un exemplu char-span in format GLiNER token-span."""
    # Pt fiecare obiect din fisierul de intrare, se extrage textul si obiectele din "entities".
    text = example["text"]                                                              # text-ul original
    tokens_with_offsets = tokenize_with_offsets(text)                                   # sparge textul in tokeni si retine offset-urile lor in text
    tokens = [t for t, _, _ in tokens_with_offsets]                                     # extrage doar tokenii (fara offset-uri)

    ner = []
    # se cauta pentru fiecare entitate care sunt tokenii care corespund span-ului de caractere al entitatii.
    for ent in example.get("entities", []):
        # extragem pozitiile de start si end din entitate.
        ent_start_char = ent["start"]
        ent_end_char   = ent["end"]
        # Cautam token care incepe la ent_start_char (sau cel mai apropiat dupa)
        start_tok = None
        end_tok = None
        # se verifica in textul original daca exista tokeni care incep sau se termina exact la pozitiile de start si end ale entitatii. Daca da, se retin indexii acelor tokeni.
        for i, (_, cs, ce) in enumerate(tokens_with_offsets):
            if start_tok is None and cs >= ent_start_char:
                start_tok = i       # indexul cuvantului in text
            if ce <= ent_end_char:
                end_tok = i

        # Fallback: in caz ca nu exista tokenul in text, atunci cautam tokeni care se suprapun cu span-ul de caractere al entitatii (cs < ent_end_char si ce > ent_start_char).
        # Daca gasim astfel de tokeni, retinem indexul primului gasit ca start_tok si indexul ultimului gasit ca end_tok.
        if start_tok is None or end_tok is None or start_tok > end_tok:
            for i, (_, cs, ce) in enumerate(tokens_with_offsets):
                if cs < ent_end_char and ce > ent_start_char:
                    if start_tok is None:
                        start_tok = i
                    end_tok = i
        # daca am gasit un token care apare in text, il adaugam in lista (pozitia cuvantului in text!!!).
        if start_tok is not None and end_tok is not None and start_tok <= end_tok:
            ner.append([start_tok, end_tok, ent["label"]])

    return {"tokenized_text": tokens, "ner": ner}

# Se returneaza un dict cu "tokenized_text" (lista de tokeni) si "ner" (lista de entitati in format [start_token_idx, end_token_idx, label]).

# Incarcam dataset-urile
train_raw = load_jsonl(SPLITS_DIR / "train.jsonl")
dev_raw   = load_jsonl(SPLITS_DIR / "dev.jsonl")
test_raw  = load_jsonl(SPLITS_DIR / "test.jsonl")

print(f"Raw counts: train={len(train_raw)}, dev={len(dev_raw)}, test={len(test_raw)}")

# Conversie
train_data = [convert_to_gliner_format(ex) for ex in train_raw]
dev_data   = [convert_to_gliner_format(ex) for ex in dev_raw]
test_data  = [convert_to_gliner_format(ex) for ex in test_raw]

# Filtram exemplele care au pierdut toate entitatile la conversie
train_data = [ex for ex in train_data if ex["ner"]]
dev_data   = [ex for ex in dev_data   if ex["ner"]]
test_data  = [ex for ex in test_data  if ex["ner"]]

print(f"Dupa conversie + filtrare:")
print(f"  train: {len(train_data)}")
print(f"  dev:   {len(dev_data)}")
print(f"  test:  {len(test_data)}")

# Sanity check pe primul exemplu
print("\nExemplu convertit:")
print(json.dumps(train_data[0], indent=2)[:400])


# Identifies discrepancies

real_discrepancies = 0

for i in range(len(train_data)):
    train_r = train_raw[i]['entities']
    train_d = train_data[i]['ner']
    tokens  = train_data[i]['tokenized_text']

    # Reconstruim textul fiecarei entitati din GLiNER format
    gliner_texts = {" ".join(tokens[s:e+1]) for s, e, lbl in train_d}
    raw_texts    = {ent['text'].strip() for ent in train_r}

    # Comparam textele, nu pozitiile
    lost_words = raw_texts - gliner_texts      # in raw dar nu in gliner
    added_words = gliner_texts - raw_texts      # in gliner dar nu in raw

    if lost_words or added_words:
        real_discrepancies += 1
        if real_discrepancies <= 5:  # afisam doar primele 5
            print(f"\n🔴 Exemplul {i}")
            print(f"  Pierdute la conversie: {lost_words}")
            print(f"  Adaugate eronat:       {added_words}")
            print(f"  Text: {train_raw[i]['text'][:100]}")

print(f"\nTotal discrepante reale: {real_discrepancies} / {len(train_data)}")



# Incarcare Model pretrained Gliner

from gliner import GLiNER

print(f"Se incarca {PRETRAINED_MODEL}...")
model = GLiNER.from_pretrained(PRETRAINED_MODEL)
model.to(DEVICE)

# Test inferenta zero-shot inainte de fine-tuning
test_text = "The Federal Reserve raised interest rates by 25 basis points after the FOMC meeting, citing inflation concerns. President Biden welcomed the decision."
zero_shot_entities = model.predict_entities(test_text, NER_LABELS, threshold=EVAL_THRESHOLD)

print(f"\nPredictii zero-shot (inainte de fine-tuning):")
for ent in zero_shot_entities:
    print(f"  {ent['text']:<35} -> {ent['label']:<20} (score: {ent['score']:.3f})")


from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import SpanDataCollator

# ─── Data collator — versiunea corecta pentru GLiNER >= 0.2.14 ───────────────
# DataCollator a fost eliminat. Modelele span-based (gliner_small, gliner_medium)
# folosesc SpanDataCollator. Modelele token-based folosesc TokenDataCollator.
data_collator = SpanDataCollator(
    config=model.config,                    # contine configuratia modelului (backbone: Bert, RoBerta, etc. ; hidden size, nb layers, etc.)
    data_processor=model.data_processor,    # contine logica de preprocesare a datelor (converteste text+entitati in formatul necesar modelului)
    prepare_labels=True,
)

# Calcul total steps pentru scheduler
num_steps = (len(train_data) // BATCH_SIZE) * NUM_EPOCHS           # numarul total de update-uri pe care le va face modelul in timpul trainingului
num_warmup = int(num_steps * WARMUP_RATIO)                         # numarul de update-uri pentru care lr creste treptat la inceputul antrenamentului (previne gradienti instabili)
print(f"Total training steps: {num_steps}, warmup: {num_warmup}")

training_args = TrainingArguments(
    output_dir=str(MODEL_OUT_DIR / "checkpoints"),  # unde se salveaza checkpoint-urile in timpul trainingului
    learning_rate=LEARNING_RATE,                
    weight_decay=WEIGHT_DECAY,                      # termen de regularizare ce penalizeaza ponderile mari (previne overfitting)
    others_lr=1e-5,                                 # learning rate mic pentru restul parametrilor (ex: backbone) pentru a nu strica cunostintele deja invatate in pretraining
    others_weight_decay=0.01,                       # termen de regularizare pentru restul parametrilor
    lr_scheduler_type="linear",                     # scheduler liniar care scade lr de la valoarea initiala la 0 pe parcursul trainingului
    warmup_steps=WARMUP_RATIO,                      # procentul initial din pasii de antrenare pt care lr creste treptat pana la valoarea specificata
    per_device_train_batch_size=BATCH_SIZE,         # numarul de exemple procesate simultan pe fiecare dispozitiv (GPU/CPU) in timpul trainingului
    per_device_eval_batch_size=BATCH_SIZE,          # numarul de exemple procesate simultan pe fiecare dispozitiv (GPU/CPU) in timpul evaluarii
    num_train_epochs=NUM_EPOCHS,
    eval_strategy="epoch",                          # evaluation_strategy in versiunile vechi
    save_strategy="epoch",
    dataloader_num_workers=0,                       # numarul de subprocesses folosite pentru a incarca datele (0 = se incarca in procesul principal, >0 = se folosesc subprocesses pentru a incarca datele in paralel)
    use_cpu=(DEVICE.type == "cpu"),
    report_to="none",
    logging_steps=50,
    max_grad_norm=MAX_GRAD_NORM,                    # pt gradient clipping (if norm gradient > MAX_GRAD_NORM, se scaleaza gradientii astfel incat norm = MAX_GRAD_NORM)
    remove_unused_columns=False,                    # CRITIC: fara asta Trainer sterge coloanele custom (ex: "tokenized_text", "ner") din dataset, ceea ce strica logica de preprocesare a datelor din data_processor. Seteaza False pentru a pastra toate coloanele din dataset.
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,                        # pentru ca vrem sa minimizam pierderea (loss) la evaluare
    save_total_limit=3,        # pastreaza top 3 checkpoints
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    processing_class=model.data_processor.transformer_tokenizer,  # 'tokenizer' -> 'processing_class'
    data_collator=data_collator,
)

print("Trainer configurat cu succes!")
print(f"  Collator:  {type(data_collator).__name__}")
print(f"  Processor: {type(model.data_processor).__name__}")


# Training 

print("Incepe antrenarea...\n")
trainer.train()
print("\nAntrenare completa.")

# Salvam modelul final
final_model_path = MODEL_OUT_DIR / "final"
model.save_pretrained(str(final_model_path))
print(f"\nModel salvat in: {final_model_path}")


# Evaluate Model

# Reincarcam modelul salvat pentru evaluare curata
from gliner import GLiNER

eval_model = GLiNER.from_pretrained(str(final_model_path), local_files_only=True) # incarcam modelul din calea locala (fara a incerca sa-l descarce de pe HuggingFace Hub)
eval_model.to(DEVICE)
eval_model.eval()           # seteaza modelul in modul de evaluare (dezactiveaza dropout, etc.) pentru a obtine predictii consistente la inferenta

# Convertim test set inapoi in format text + entitati char-span pentru evaluare
def gliner_to_char_spans(ex):
    """Reconstruieste textul cu spatii intre tokens si calculeaza char spans."""
    tokens = ex["tokenized_text"]
    text_parts = []
    char_pos = 0
    token_starts, token_ends = [], []
    for i, t in enumerate(tokens):
        if i > 0:                   # daca nu e primul cuvant
            text_parts.append(" ")
            char_pos += 1
        # recalculeaza pozitiile si adauga in liste
        token_starts.append(char_pos)
        text_parts.append(t)
        char_pos += len(t)
        token_ends.append(char_pos)
    # reconstruieste textul
    text = "".join(text_parts)
    entities = []
    for s_tok, e_tok, lbl in ex["ner"]:
        entities.append({
            "text": text[token_starts[s_tok]:token_ends[e_tok]],
            "label": lbl,
            "start": token_starts[s_tok],
            "end": token_ends[e_tok],
        })
    return text, entities

# Generam predictii pe test set
print(f"Se evalueaza pe {len(test_data)} exemple test...")
gold_entities_per_doc = []
pred_entities_per_doc = []

from tqdm.auto import tqdm
for ex in tqdm(test_data, desc="Predictie test"):   # parcurgem fiecare exemplu din test_data
    text, gold = gliner_to_char_spans(ex)           # reconstruieste textul original si entitatile gold in format char-span
    pred_raw = eval_model.predict_entities(text, NER_LABELS, threshold=EVAL_THRESHOLD)      # obtine predictiile modelului in format char-span (lista de dicturi cu "text", "label", "start", "end")
    # Format pred pentru nervaluate
    pred = [{"start": p["start"], "end": p["end"], "label": p["label"]} for p in pred_raw]
    gold_n = [{"start": g["start"], "end": g["end"], "label": g["label"]} for g in gold]
    gold_entities_per_doc.append(gold_n)
    pred_entities_per_doc.append(pred)

print(f"\nDocumente evaluate: {len(gold_entities_per_doc)}")
print(f"Total entitati gold: {sum(len(g) for g in gold_entities_per_doc)}")
print(f"Total entitati pred: {sum(len(p) for p in pred_entities_per_doc)}")


print(f"Se evalueaza pe {len(test_raw)} exemple test...")
gold_entities_per_doc = []
pred_entities_per_doc = []

for ex in tqdm(test_raw, desc="Predictie test"):
    # Folosim textul ORIGINAL, nu reconstruit din tokeni
    text = ex["text"]

    # Gold direct din test_raw — char spans originale, fara conversie
    gold_n = [
        {"start": e["start"], "end": e["end"], "label": e["label"]}
        for e in ex.get("entities", [])
    ]

    # Predictii pe textul original
    pred_raw = eval_model.predict_entities(text, NER_LABELS, threshold=EVAL_THRESHOLD)
    pred = [
        {"start": p["start"], "end": p["end"], "label": p["label"]}
        for p in pred_raw
    ]

    gold_entities_per_doc.append(gold_n)
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

# nervaluate poate returna fie un dict {"overall":..., "entities":...}
# fie un tuple (results_dict, results_per_tag_dict) in functie de versiune
if isinstance(raw_result, tuple):
    results         = raw_result[0]   # dict cu chei "ent_type", "partial", "strict", "exact"
    results_per_tag = raw_result[1]   # dict cu chei per label
else:
    results         = raw_result.get("overall", raw_result)
    results_per_tag = raw_result.get("entities", {})

# nervaluate returneaza obiecte cu atribute SAU dicturi — normalizam accesul
def get_metric(obj, key):
    """Acceseaza un camp fie ca atribut, fie ca cheie de dict."""
    if isinstance(obj, dict):
        return obj.get(key, 0.0)
    return getattr(obj, key, 0.0)

# ─── Metrici globale ─────────────────────────────────────────────────────────
print("=" * 70)
print("METRICI GLOBALE (micro-averaged pe toate labels)")
print("=" * 70)
for mode in ["ent_type", "partial", "strict", "exact"]:
    if isinstance(results, dict) and mode in results:
        r = results[mode]
    else:
        r = results  # unele versiuni returneaza direct rezultatul fara mod
        mode = "overall"
    print(f"\nMod '{mode}':")
    print(f"  Precision: {get_metric(r, 'precision'):.4f}")
    print(f"  Recall:    {get_metric(r, 'recall'):.4f}")
    print(f"  F1:        {get_metric(r, 'f1'):.4f}")
    if mode == "overall":
        break

# ─── Metrici per label ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("METRICI PER LABEL")
print("=" * 70)
print(f"{'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print("-" * 70)

for label in NER_LABELS:
    support = sum(1 for doc in gold_entities_per_doc for e in doc if e["label"] == label)
    if isinstance(results_per_tag, dict) and label in results_per_tag:
        tag_data = results_per_tag[label]
        # poate fi {"ent_type": obj, ...} sau direct obiect cu atribute
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


# ─── Salvare metrici in fisier ──────────────────────────────────────────────
import json
from datetime import datetime

metrics_summary = {
    "timestamp": datetime.now().isoformat(),
    "pretrained_model": PRETRAINED_MODEL,
    "num_train_examples": len(train_data),
    "num_dev_examples":   len(dev_data),
    "num_test_examples":  len(test_data),
    "training_config": {
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
    },
    "eval_threshold": EVAL_THRESHOLD,
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
if 'ent_type' in metrics_summary['global_metrics']:
    print(f"F1 global (ent_type): {metrics_summary['global_metrics']['ent_type']['f1']:.4f}")
if 'strict' in metrics_summary['global_metrics']:
    print(f"F1 global (strict):   {metrics_summary['global_metrics']['strict']['f1']:.4f}")


# Test the model


# Texte de test reprezentative pentru domeniul nostru
test_sentences = [
    "The Federal Reserve raised interest rates by 50 basis points, citing persistent inflation pressures.",
    "President Biden signed the Inflation Reduction Act into law after months of congressional negotiations.",
    "The European Central Bank announced a new round of quantitative easing to support the eurozone economy.",
    "Christine Lagarde said the ECB remains committed to its 2% inflation target despite rising energy costs.",
    "NATO members agreed to increase defense spending following the latest G20 summit in Tokyo.",
    "Goldman Sachs analysts predict the dollar will weaken against the yen in the coming quarter.",
    "The Republican Party secured a narrow majority in the Senate after the midterm elections.",
]

print("PREDICTII MODEL FINE-TUNAT")
print("=" * 70)
for sent in test_sentences:
    print(f"\nText: {sent}")
    entities = eval_model.predict_entities(sent, NER_LABELS, threshold=EVAL_THRESHOLD)
    for ent in entities:
        print(f"  {ent['text']:<35} -> {ent['label']:<20} (score: {ent['score']:.3f})")


from huggingface_hub import HfApi

pass  # login via: hf login din terminal
# Verificare ca login-ul a reusit
api = HfApi()
user_info = api.whoami()
print(f"\nAutentificat ca: {user_info['name']}")


from huggingface_hub import HfApi, create_repo

api = HfApi()

# Cream repo daca nu exista (idempotent — exist_ok=True nu suprascrie)
try:
    create_repo(HF_DATASET_REPO, repo_type="dataset", exist_ok=True)
    print(f"Repo dataset OK: https://huggingface.co/datasets/{HF_DATASET_REPO}")
except Exception as e:
    print(f"Eroare la create_repo (probabil exista deja): {e}")


# Cream repo model (idempotent)
try:
    create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True)
    print(f"Repo model OK: https://huggingface.co/{HF_MODEL_REPO}")
except Exception as e:
    print(f"Eroare la create_repo (probabil exista deja): {e}")


# ─── Generam README pentru model ─────────────────────────────────────────────
# Citim metricile salvate ca sa le includem in README

with open(MODEL_OUT_DIR / "metrics.json") as f:
    m = json.load(f)

metrics_table_lines = [
    "| Label | Precision | Recall | F1 |",
    "|-------|-----------|--------|----|"
]
for label in NER_LABELS:
    if label in m["per_label_metrics"]:
        ent = m["per_label_metrics"][label]["ent_type"]
        metrics_table_lines.append(
            f"| {label} | {ent['precision']:.3f} | {ent['recall']:.3f} | {ent['f1']:.3f} |"
        )
metrics_table = "\n".join(metrics_table_lines)

g = m["global_metrics"]["ent_type"]

model_readme = f"""---
license: mit
language:
- en
library_name: gliner
pipeline_tag: token-classification
tags:
- ner
- gliner
- politics
- economics
base_model: {PRETRAINED_MODEL}
datasets:
- {HF_DATASET_REPO}
---

# GLiNER Fine-tuned for Political & Economic NER

Fine-tuned version of [`{PRETRAINED_MODEL}`]({PRETRAINED_MODEL}) on a custom
politico-economic NER dataset. Trained to recognize 11 entity types.

## Entity types

`POLITICIAN`, `POLITICAL_PARTY`, `POLITICAL_ORG`, `FINANCIAL_ORG`,
`ECONOMIC_INDICATOR`, `POLICY`, `LEGISLATION`, `MARKET_EVENT`,
`CURRENCY`, `TRADE_AGREEMENT`, `GPE`

## Performance

Test set: {m['num_test_examples']} examples.
Evaluation mode: `ent_type` (label match, ignoring exact boundaries).

**Global (micro-averaged):**
- Precision: **{g['precision']:.4f}**
- Recall:    **{g['recall']:.4f}**
- F1:        **{g['f1']:.4f}**

**Per label:**

{metrics_table}

## Usage

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("{HF_MODEL_REPO}")
labels = ["POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
          "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
          "CURRENCY", "TRADE_AGREEMENT", "GPE"]

text = "The Federal Reserve raised rates after President Biden signed the new bill."
entities = model.predict_entities(text, labels, threshold=0.5)
for e in entities:
    print(e["text"], "->", e["label"])
```

## Training details

- Base model: `{PRETRAINED_MODEL}`
- Training examples: {m['num_train_examples']}
- Validation examples: {m['num_dev_examples']}
- Epochs: {m['training_config']['num_epochs']}
- Batch size: {m['training_config']['batch_size']}
- Learning rate: {m['training_config']['learning_rate']}
"""

model_readme_path = MODEL_OUT_DIR / "README.md"
with open(model_readme_path, "w") as f:
    f.write(model_readme)
print("README model generat.")


# ─── Upload model files ──────────────────────────────────────────────────────
# GLiNER salveaza modelul ca un folder cu config + greutati. Le incarcam pe toate.

print(f"Continut {final_model_path}:")
for p in sorted(final_model_path.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(final_model_path)}  ({p.stat().st_size / 1e6:.1f} MB)")

# Upload intregul folder model
print(f"\nIncarcam folderul model in repo {HF_MODEL_REPO}...")
api.upload_folder(
    folder_path=str(final_model_path),
    repo_id=HF_MODEL_REPO,
    repo_type="model",
    commit_message="Upload fine-tuned GLiNER model",
)

# Upload README si metrics.json separat
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

print(f"\nModel uploaded la: https://huggingface.co/{HF_MODEL_REPO}")
