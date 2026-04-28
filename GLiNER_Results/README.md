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

## Performance

Test set: 2122 examples.
Evaluation mode: `ent_type` (label match, ignoring exact boundaries).

**Global (micro-averaged):**
- Precision: **0.6811**
- Recall:    **0.9094**
- F1:        **0.7789**

**Per label:**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|----|
| POLITICIAN | 0.603 | 0.932 | 0.732 |
| POLITICAL_PARTY | 0.750 | 0.964 | 0.843 |
| POLITICAL_ORG | 0.324 | 0.497 | 0.392 |
| FINANCIAL_ORG | 0.257 | 0.471 | 0.332 |
| ECONOMIC_INDICATOR | 0.294 | 1.000 | 0.455 |
| POLICY | 0.111 | 0.250 | 0.154 |
| LEGISLATION | 0.238 | 1.000 | 0.385 |
| MARKET_EVENT | 0.188 | 0.710 | 0.297 |
| CURRENCY | 0.094 | 0.400 | 0.153 |
| TRADE_AGREEMENT | 0.122 | 0.357 | 0.182 |
| GPE | 0.842 | 0.971 | 0.901 |

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

## Training details

- Base model: `urchade/gliner_small-v2.1`
- Training examples: 5747
- Validation examples: 1228
- Epochs: 10
- Batch size: 8
- Learning rate: 3e-06
