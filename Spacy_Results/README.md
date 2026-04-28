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

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.8507 |
| Recall    | 0.7768 |
| F1        | 0.8121 |

**nervaluate evaluation (comparable with GLiNER):**

| Mode      | Precision | Recall | F1     |
| --------- | --------- | ------ | ------ |
| ent_type  | 0.8633    | 0.7891 | 0.8245 |
| partial   | 0.8770    | 0.8016 | 0.8376 |
| exact     | 0.8670    | 0.7925 | 0.8281 |
| strict    | 0.8490    | 0.7761 | 0.8109 |

**Per label (ent_type, test set):**

| Label              | Precision | Recall | F1    | Support |
| ------------------ | --------- | ------ | ----- | ------- |
| POLITICIAN         | 0.813     | 0.685  | 0.743 | 501     |
| POLITICAL_PARTY    | 0.914     | 0.934  | 0.924 | 137     |
| POLITICAL_ORG      | 0.779     | 0.567  | 0.656 | 187     |
| FINANCIAL_ORG      | 0.886     | 0.750  | 0.813 | 104     |
| ECONOMIC_INDICATOR | 0.571     | 0.800  | 0.667 | 5       |
| POLICY             | 0.750     | 0.750  | 0.750 | 4       |
| LEGISLATION        | 1.000     | 0.800  | 0.889 | 5       |
| MARKET_EVENT       | 0.968     | 0.968  | 0.968 | 31      |
| CURRENCY           | 0.630     | 0.567  | 0.596 | 30      |
| TRADE_AGREEMENT    | 0.684     | 0.929  | 0.788 | 14      |
| GPE                | 0.879     | 0.824  | 0.851 | 2206    |

## Training curve

| Epoch | Loss      | Dev P  | Dev R  | Dev F1 |
| ----- | --------- | ------ | ------ | ------ |
| 1     | 10187.12  | 0.6390 | 0.7430 | 0.6871 |
| 2     | 6728.36   | 0.7856 | 0.7649 | 0.7751 |
| 4     | 5261.25   | 0.8325 | 0.7783 | 0.8045 |
| 6     | 4689.88   | 0.8360 | 0.8070 | 0.8212 |
| 10    | 3716.44   | 0.8382 | 0.8143 | 0.8261 |
| 11    | 3443.75   | 0.8346 | 0.8182 | 0.8263 |
| 12    | 3384.84   | 0.8220 | 0.8373 | 0.8296 |
| 15    | 3035.63   | 0.8563 | 0.8193 | **0.8374** |

## Usage

```python
import spacy

nlp = spacy.load("path/to/model")
# or after downloading from HuggingFace:
# from huggingface_hub import snapshot_download
# nlp = spacy.load(snapshot_download("Tudorx95/NER_Economic_Political_Spacy"))

doc = nlp("The Federal Reserve raised rates after President Biden signed the new bill.")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)
```

## Training Details

| Parameter         | Value                          |
| ----------------- | ------------------------------ |
| Framework         | spaCy 3.8.14                   |
| Backbone          | en_core_web_trf (RoBERTa-base) |
| PyTorch           | 2.5.1+cu121                    |
| Strategy          | Frozen transformer, NER head fine-tuned |
| Epochs            | 15 (patience 5)                |
| Best dev F1       | 0.8374 (epoch 15)              |
| Train examples    | 5747                           |
| Dev examples      | 1228                           |
| Test examples     | 2124                           |
| Hardware          | NVIDIA A40 (47.6 GB VRAM)      |
| Dataset           | [Tudorx95/NER_Political_Economic](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic) |
| Model on HF Hub   | [Tudorx95/NER_Economic_Political_Spacy](https://huggingface.co/Tudorx95/NER_Economic_Political_Spacy) |
