<div align="center">

# NER Political & Economic Dataset

**Building a specialized NER dataset for the political and economic domain**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-NER__Political__Economic-yellow)](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](../LICENSE)

</div>

---

## Overview

This directory contains the code and data used to build the specialized NER dataset for the **political and economic** domain, publicly available on HuggingFace:

> **[Tudorx95/NER_Political_Economic](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)**

The dataset covers **11 entity types** and contains approximately **9,100 annotated examples**, obtained through a multi-stage pipeline described in detail below.

---

## NER Schema — 11 Entity Types

| Label | Description | Examples |
|-------|-------------|---------|
| `POLITICIAN` | Persons holding political office | *Joe Biden*, *Christine Lagarde* |
| `POLITICAL_PARTY` | Political parties | *Republican Party*, *CDU* |
| `POLITICAL_ORG` | Intergovernmental political organizations | *NATO*, *European Union*, *G7* |
| `FINANCIAL_ORG` | Financial institutions and organizations | *Federal Reserve*, *IMF*, *Goldman Sachs* |
| `ECONOMIC_INDICATOR` | Macroeconomic indicators | *GDP*, *CPI*, *unemployment rate* |
| `POLICY` | Monetary or fiscal policies | *quantitative easing*, *rate hike* |
| `LEGISLATION` | Legislative acts | *Dodd-Frank Act*, *CHIPS Act* |
| `MARKET_EVENT` | Major market events | *2008 financial crisis*, *Great Recession* |
| `CURRENCY` | Currencies and cryptocurrencies | *USD*, *euro*, *Bitcoin* |
| `TRADE_AGREEMENT` | International trade agreements | *NAFTA*, *USMCA*, *TPP* |
| `GPE` | Geopolitical entities (countries, regions) | *United States*, *China*, *Germany* |

---

## Files

| File | Description |
|------|-------------|
| `DatasetCreation.py` | Main script — full pipeline from data collection to final splits |
| `DatasetCreation.ipynb` | Notebook version of the pipeline (interactive exploration) |
| `DatasetCreation_Results.ipynb` | Notebook with statistics and visualizations of the final dataset |
| `synthetic_augmented.jsonl` | Synthetic examples generated for underrepresented classes |

---

## Dataset Construction Pipeline

### Stage 1 — Data Collection from 5 Sources

Raw texts come from complementary sources, covering different linguistic styles and registers:

| Source | Collection Method | Role in Dataset |
|--------|-------------------|-----------------|
| **CC-News** | Filtered on 30 politico-economic keywords (`GDP`, `Federal Reserve`, `NATO` etc.); streamed via `datasets` | Large volume, linguistic diversity, recent news |
| **Wikipedia** | `wikipediaapi` for 60+ articles on politicians, organizations and economic concepts | High quality, verifiable facts, encyclopedic language |
| **SEC EDGAR** | Automatic download of 10-K filings (`sec_edgar_downloader`) for major financial tickers | Technical and formal financial language |
| **CoNLL-2003** | Loaded via HuggingFace `datasets`; BIO labels remapped to the project schema | Professionally annotated data, gold-standard for test |
| **WNUT-2017** | Similar to CoNLL; includes social media texts | Stylistic diversity, informal language |

**Collection parameters:**
- CC-News articles: max 5,000, min 200 characters per article
- Wikipedia articles: 60 predefined topics
- SEC filings: max 200 `.txt` files

---

### Stage 2 — Sentence Segmentation with spaCy

All raw texts are segmented into individual sentences using the `en_core_web_sm` model (with `ner` and `lemmatizer` components disabled for efficiency):

- Minimum sentence length: **40 characters**
- Maximum sentence length: **400 characters**
- Maximum **20 sentences** extracted per document
- Strict deduplication on exact text

---

### Stage 3 — Gazetteer Construction

Manual gazetteers were built for 9 of the 11 entity types, used both by Labeling Functions and for span extraction:

| Gazetteer | Initial Size | Extension Technique |
|-----------|-------------|---------------------|
| `POLITICIANS_GAZETTEER` | 23 politicians | Extended with **~3,000 politicians** from Wikidata via SPARQL |
| `FINANCIAL_ORGS_GAZETTEER` | 23 organizations | Manual |
| `POLITICAL_ORGS_GAZETTEER` | 21 organizations | Manual |
| `POLITICAL_PARTIES_GAZETTEER` | 17 parties | Manual |
| `ECONOMIC_INDICATORS_GAZETTEER` | 17 indicators | Manual |
| `LEGISLATION_GAZETTEER` | 13 legislative acts | Manual |
| `CURRENCIES_GAZETTEER` | 16 currencies | Manual |
| `TRADE_AGREEMENTS_GAZETTEER` | 9 agreements | Manual |
| `MARKET_EVENTS_GAZETTEER` | 10 events | Manual |

**Wikidata extension:** SPARQL query retrieving all persons with occupation `politician` (`Q82955`) and at least 5 inter-wiki links, returning ~3,000 real politicians.

---

### Stage 4 — External Dataset Remapping (CoNLL-2003 & WNUT-2017)

Labels from standard BIO schemes were remapped to the project schema via gazetteer lookups:

**CoNLL-2003 (BIO schema: PER, ORG, LOC, MISC):**

| Original Label | Mapping to Project Schema |
|----------------|--------------------------|
| `B/I-PER` | → `POLITICIAN` (if found in gazetteer), otherwise discarded |
| `B/I-ORG` | → `FINANCIAL_ORG` or `POLITICAL_ORG` (by gazetteer) |
| `B/I-LOC` | → `GPE` (all locations become GPE) |
| `B/I-MISC` | → `POLITICAL_PARTY`, `LEGISLATION`, `TRADE_AGREEMENT`, `MARKET_EVENT`, `CURRENCY` (by gazetteer) |

**WNUT-2017 (schema: corporation, person, location, group, etc.):**

| Original Label | Mapping to Project Schema |
|----------------|--------------------------|
| `corporation` | → `FINANCIAL_ORG` or `POLITICAL_ORG` |
| `person` | → `POLITICIAN` (if found in gazetteer) |
| `location` | → `GPE` |
| `group` | → `POLITICAL_PARTY` or `POLITICAL_ORG` |
| `creative-work`, `product` | discarded (do not fit the domain) |

CoNLL test set examples were kept separately as the **gold standard** for the final test split.

---

### Stage 5 — Weak Supervision with Snorkel

Instead of costly manual annotation, the project uses **Snorkel** to generate sentence-level pseudo-labels through 14 Labeling Functions (LFs):

| LF | Strategy | Target Entity |
|----|----------|---------------|
| `lf_politician_gazetteer` | Gazetteer lookup (case-insensitive) | `POLITICIAN` |
| `lf_politician_title` | Regex: political title + proper name (*"President Biden"*) | `POLITICIAN` |
| `lf_political_party` | Gazetteer lookup | `POLITICAL_PARTY` |
| `lf_political_org` | Gazetteer lookup | `POLITICAL_ORG` |
| `lf_financial_org` | Gazetteer lookup | `FINANCIAL_ORG` |
| `lf_economic_indicator` | Gazetteer lookup | `ECONOMIC_INDICATOR` |
| `lf_legislation` | Gazetteer lookup | `LEGISLATION` |
| `lf_currency` | Gazetteer lookup | `CURRENCY` |
| `lf_trade_agreement` | Gazetteer lookup | `TRADE_AGREEMENT` |
| `lf_market_event` | Gazetteer lookup | `MARKET_EVENT` |
| `lf_policy_pattern` | Regex: monetary/fiscal policy terms | `POLICY` |
| `lf_currency_symbol` | Regex: currency symbol + digits (`$`, `€`, `£`, `¥`) | `CURRENCY` |
| `lf_gpe_country` | Fixed list of 17 countries | `GPE` |
| `lf_external_dataset_match` | Match against entities extracted from remapped CoNLL/WNUT | all types |

**Label Model:**
- Cardinality: 11 classes
- Trained for **500 epochs**, lr=0.001
- **Confidence threshold: ≥ 0.7** — examples below threshold are discarded

**Span extraction:** after filtering, exact entity positions are identified via regex + gazetteer matching, with overlap resolution (longest match wins).

---

### Stage 6 — Synthetic Augmentation for Underrepresented Classes

For classes with few examples (`MARKET_EVENT`, `TRADE_AGREEMENT`, `ECONOMIC_INDICATOR`, `POLICY`, `LEGISLATION`), synthetic examples were generated using varied templates:

**Template structure (5 templates × N entities per class):**

```
# MARKET_EVENT (9 events × 5 templates = 45 examples)
"Analysts compared the recent downturn to the {EVENT}, noting similar warning signs."

# TRADE_AGREEMENT (5 agreements × 5 templates = 25 examples)
"Negotiations over {TRADE} dragged on for several years before a final deal was reached."

# ECONOMIC_INDICATOR (5 indicators × 5 templates = 25 examples)
"The latest {IND} reading came in below market expectations, prompting concern."

# POLICY (7 policies × 5 templates = 35 examples)
"The committee announced a new round of {POLICY} to address market volatility."

# LEGISLATION (5 laws × 5 templates = 25 examples)
"Provisions of the {LAW} have been the subject of intense judicial review."
```

Entity spans are automatically located with regex in the generated text.

---

### Stage 7 — Validation and Deduplication

Each example passes through a validator that checks:
- Minimum text length ≥ 40 characters
- At least one entity present
- Valid `start`/`end` offsets matching the text
- Label belongs to the 11 defined classes

Deduplication is performed on exact text (hash-set).

---

### Stage 8 — Train / Dev / Test Split

The final split follows **data quality stratification**:

1. CoNLL gold test examples are reserved entirely for the **test set**
2. Remaining data is split with `train_test_split` (sklearn, `random_state=42`):
   - 15% → additional test (combined with gold)
   - 85% → pool for train + dev
3. From the train/dev pool: 17.6% → **dev**, 82.4% → **train** (≈ 15% of total)

| Split | Examples | Proportion |
|-------|----------|------------|
| **train** | ~5,747 | ~63% |
| **dev** | ~1,228 | ~13% |
| **test** | ~2,122 | ~23% |
| **TOTAL** | **~9,097** | — |

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | ~9,100 |
| Entity types | 11 |
| Data sources | 5 (CC-News, Wikipedia, SEC EDGAR, CoNLL-2003, WNUT-2017) |
| Snorkel Labeling Functions | 14 |
| Snorkel confidence threshold | 0.70 |
| Politicians in gazetteer (after Wikidata) | ~3,000+ |

---

## HuggingFace

The dataset is publicly available at:

**[https://huggingface.co/datasets/Tudorx95/NER_Political_Economic](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)**

```python
from datasets import load_dataset

dataset = load_dataset("Tudorx95/NER_Political_Economic")
```

---

## Main Dependencies

```
datasets
snorkel
spacy (en_core_web_sm)
wikipediaapi
sec-edgar-downloader
scikit-learn
pandas
numpy
requests
tqdm
```

---

## Author

**Sd. Sg. Maj. Lepădatu Tudor**  
Military Technical Academy "Ferdinand I", 2026
