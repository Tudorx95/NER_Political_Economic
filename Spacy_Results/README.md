---
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
- Tudorx95/NER_Political_Economic
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
- Precision: **0.8507**
- Recall:    **0.7768**
- F1:        **0.8121**

**nervaluate ent_type (comparabil cu GLiNER):**
- Precision: **0.8633**
- Recall:    **0.7891**
- F1:        **0.8245**

**Per label (ent_type):**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----| 
| POLITICIAN | 0.813 | 0.685 | 0.743 |
| POLITICAL_PARTY | 0.914 | 0.934 | 0.924 |
| POLITICAL_ORG | 0.779 | 0.567 | 0.656 |
| FINANCIAL_ORG | 0.886 | 0.750 | 0.813 |
| ECONOMIC_INDICATOR | 0.571 | 0.800 | 0.667 |
| POLICY | 0.750 | 0.750 | 0.750 |
| LEGISLATION | 1.000 | 0.800 | 0.889 |
| MARKET_EVENT | 0.968 | 0.968 | 0.968 |
| CURRENCY | 0.630 | 0.567 | 0.596 |
| TRADE_AGREEMENT | 0.684 | 0.929 | 0.788 |
| GPE | 0.879 | 0.824 | 0.851 |

## Usage

```python
import spacy

nlp = spacy.load("path/to/model")
# sau dupa descarcare de pe HuggingFace:
# from huggingface_hub import snapshot_download
# nlp = spacy.load(snapshot_download("Tudorx95/NER_Economic_Political_Spacy"))

doc = nlp("The Federal Reserve raised rates after President Biden signed the new bill.")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)
```

## Training Details

- **Framework:** spaCy 3.8.14
- **Backbone:** en_core_web_trf (RoBERTa-base)
- **Strategy:** Frozen transformer + NER head fine-tuning
- **Train examples:** 5747
- **Dev examples:** 1228
- **Test examples:** 2124
- **Best dev F1:** 0.8374
