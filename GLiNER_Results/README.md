---
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
base_model: urchade/gliner_small-v2.1
datasets:
  - Tudorx95/NER_Political_Economic
---

# GLiNER Fine-tuned for Political & Economic NER

Fine-tuned version of [`urchade/gliner_small-v2.1`](urchade/gliner_small-v2.1) on a custom
politico-economic NER dataset. Trained to recognize 11 entity types.

## Entity types

`POLITICIAN`, `POLITICAL_PARTY`, `POLITICAL_ORG`, `FINANCIAL_ORG`,
`ECONOMIC_INDICATOR`, `POLICY`, `LEGISLATION`, `MARKET_EVENT`,
`CURRENCY`, `TRADE_AGREEMENT`, `GPE`

## Performance (Test Set)

**nervaluate evaluation:**

| Mode     | Precision | Recall | F1     |
| -------- | --------- | ------ | ------ |
| ent_type | 0.6811    | 0.9094 | 0.7789 |
| partial  | 0.7127    | 0.9516 | 0.8150 |
| exact    | 0.7059    | 0.9426 | 0.8073 |
| strict   | 0.6720    | 0.8973 | 0.7685 |

**Per label (ent_type, test set — 2122 examples):**

| Label              | Precision | Recall | F1    |
| ------------------ | --------- | ------ | ----- |
| POLITICIAN         | 0.603     | 0.932  | 0.732 |
| POLITICAL_PARTY    | 0.750     | 0.964  | 0.843 |
| POLITICAL_ORG      | 0.324     | 0.497  | 0.392 |
| FINANCIAL_ORG      | 0.257     | 0.471  | 0.332 |
| ECONOMIC_INDICATOR | 0.294     | 1.000  | 0.455 |
| POLICY             | 0.111     | 0.250  | 0.154 |
| LEGISLATION        | 0.238     | 1.000  | 0.385 |
| MARKET_EVENT       | 0.188     | 0.710  | 0.297 |
| CURRENCY           | 0.094     | 0.400  | 0.153 |
| TRADE_AGREEMENT    | 0.122     | 0.357  | 0.182 |
| GPE                | 0.842     | 0.971  | 0.901 |

## Usage

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("Tudorx95/NER_Economic_Political")
labels = ["POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
          "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
          "CURRENCY", "TRADE_AGREEMENT", "GPE"]

text = "The Federal Reserve raised rates after President Biden signed the new bill."
entities = model.predict_entities(text, labels, threshold=0.5)
for e in entities:
    print(e["text"], "->", e["label"])
```

## Training Details

| Parameter              | Value                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| Base model             | `urchade/gliner_small-v2.1`                                                                        |
| Library                | GLiNER                                                                                             |
| Epochs                 | 10                                                                                                 |
| Batch size             | 8                                                                                                  |
| Learning rate (head)   | 3e-6                                                                                               |
| Learning rate (others) | 1e-5                                                                                               |
| Weight decay (head)    | 0.05                                                                                               |
| Weight decay (others)  | 0.01                                                                                               |
| LR scheduler           | linear                                                                                             |
| Warmup ratio           | 0.15                                                                                               |
| Max grad norm          | 1.0                                                                                                |
| Eval threshold         | 0.5                                                                                                |
| Best model metric      | eval_loss                                                                                          |
| Train examples         | 5747                                                                                               |
| Dev examples           | 1228                                                                                               |
| Test examples          | 2122                                                                                               |
| Hardware               | NVIDIA A40 (47.6 GB VRAM)                                                                          |
| Dataset                | [Tudorx95/NER_Political_Economic](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic) |
| Model on HF Hub        | [Tudorx95/NER_Economic_Political](https://huggingface.co/Tudorx95/NER_Economic_Political)          |
