<div align="center">

# 🌐 NER Political & Economic — End-to-End ML Pipeline

**Automatic extraction of political and economic entities from text using NLP**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-NER__Political__Economic-yellow)](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)
[![HuggingFace GLiNER](https://img.shields.io/badge/🤗_Model-GLiNER-blue)](https://huggingface.co/Tudorx95/NER_Economic_Political)
[![HuggingFace spaCy](https://img.shields.io/badge/🤗_Model-spaCy-green)](https://huggingface.co/Tudorx95/NER_Economic_Political_Spacy)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](./docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

</div>

---

## 📑 Table of Contents

- [Description](#-description)
- [Project Architecture](#-project-architecture)
- [External Resources (HuggingFace)](#-external-resources-huggingface)
- [NER Schema — 11 Entity Types](#-ner-schema--11-entity-types)
- [ML Pipeline](#-ml-pipeline)
- [Results](#-results)
- [⚡ Quick Deployment — Pre-built Docker Images](#-quick-deployment--pre-built-docker-images)
- [Quick Start — Local Build](#-quick-start--local-build)
- [Repository Structure](#-repository-structure)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 📖 Description

This project implements a **complete Machine Learning pipeline** for Named Entity Recognition (NER) in the **political and economic** domain.

The project covers all stages:

1. **Data collection** from multiple sources (CC-News, Wikipedia, SEC EDGAR, CoNLL-2003, WNUT-17)
2. **Weak Supervision with Snorkel** — programmatic labeling using 14 Labeling Functions
3. **Synthetic augmentation** for underrepresented classes
4. **Fine-tuning** on two different architectures: **GLiNER** (zero-shot capable) and **spaCy** (transformer-based)
5. **Deployment** in a fully containerized full-stack application with Docker (React + FastAPI + Nginx + MongoDB)

---

## 🏗 Project Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        📦 GitHub Repository                         │
│                   (Source Code, Notebooks, Application)              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────┐  │
│  │  📓 ML Pipeline │   │  ⚙️  Backend     │   │  🖥  Frontend     │  │
│  │                 │   │                 │   │                   │  │
│  │ DatasetCreation │   │ FastAPI Server  │   │ React + Nginx     │  │
│  │ GLiNER Training │   │ Model Inference │   │ Interactive UI    │  │
│  │ spaCy Training  │   │ REST API        │   │ Entity Highlight  │  │
│  └────────┬────────┘   └────────┬────────┘   └────────┬──────────┘  │
│           │                     │                      │             │
└───────────┼─────────────────────┼──────────────────────┼─────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       🤗 HuggingFace Hub                             │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  🗄️ Dataset       │  │  🧠 Model GLiNER │  │  🧠 Model spaCy   │  │
│  │  9.1k examples   │  │  F1: 0.7789      │  │  F1: 0.8245       │  │
│  │  11 labels       │  │  Zero-Shot NER   │  │  RoBERTa-base     │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🤗 External Resources (HuggingFace)

| Resource            | Link                                                                                                    | Description                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 🗄️ **Dataset**      | [`Tudorx95/NER_Political_Economic`](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)    | ~9,100 examples, 11 entity types, built with Snorkel + multiple sources |
| 🧠 **GLiNER Model** | [`Tudorx95/NER_Economic_Political`](https://huggingface.co/Tudorx95/NER_Economic_Political)             | Fine-tuned `gliner_small-v2.1`, supports zero-shot NER                  |
| 🧠 **spaCy Model**  | [`Tudorx95/NER_Economic_Political_Spacy`](https://huggingface.co/Tudorx95/NER_Economic_Political_Spacy) | Fine-tuned `en_core_web_trf` (RoBERTa-base backbone)                    |

> **Note:** Models are downloaded automatically from HuggingFace Hub on first application startup.

---

## 🏷 NER Schema — 11 Entity Types

| Entity               | Description                      | Example                                    |
| -------------------- | -------------------------------- | ------------------------------------------ |
| `POLITICIAN`         | Persons holding political office | _Joe Biden_, _Christine Lagarde_           |
| `POLITICAL_PARTY`    | Political parties                | _Republican Party_, _CDU_                  |
| `POLITICAL_ORG`      | Political organizations          | _NATO_, _European Union_, _G7_             |
| `FINANCIAL_ORG`      | Financial organizations          | _Federal Reserve_, _IMF_, _Goldman Sachs_  |
| `ECONOMIC_INDICATOR` | Economic indicators              | _GDP_, _CPI_, _unemployment rate_          |
| `POLICY`             | Government policies              | _quantitative easing_, _rate hike_         |
| `LEGISLATION`        | Legislative acts                 | _Dodd-Frank Act_, _CHIPS Act_              |
| `MARKET_EVENT`       | Market events                    | _2008 financial crisis_, _Great Recession_ |
| `CURRENCY`           | Currencies                       | _USD_, _euro_, _Bitcoin_                   |
| `TRADE_AGREEMENT`    | Trade agreements                 | _NAFTA_, _USMCA_, _TPP_                    |
| `GPE`                | Geopolitical entities            | _United States_, _China_, _Germany_        |

---

## 🔬 ML Pipeline

### 1. Data Collection (`dataset/`)

> **Compute environment:** The dataset was built in **Google Colab** using a **Tesla T4 GPU**.

Data comes from **5 complementary sources**:

| Source         | Method                                                                                     | Role                                    |
| -------------- | ------------------------------------------------------------------------------------------ | --------------------------------------- |
| **CC-News**    | Filtered on politico-economic keywords, processed with spaCy                               | Large volume, linguistic diversity      |
| **Wikipedia**  | `wikipediaapi` API, articles on politicians, organizations, economic concepts              | High quality, verifiable facts          |
| **SEC EDGAR**  | Automatically downloaded 10-K filings                                                      | Real financial language                 |
| **CoNLL-2003** | Remapped from BIO schema (`PER`→`POLITICIAN`, `ORG`→`FINANCIAL_ORG`/`POLITICAL_ORG`, etc.) | Professionally annotated data           |
| **WNUT-2017**  | Similar remapping to CoNLL                                                                 | Social media texts, stylistic diversity |

### 2. Weak Supervision with Snorkel

Instead of costly manual annotation, **Snorkel** was used to generate pseudo-labels:

- **14 Labeling Functions (LF)** based on:
  - 📚 Gazetteers (lists of politicians, organizations, indicators)
  - 🔍 Regex pattern-matching (political titles, currency symbols)
  - 🗂️ Match against entities from remapped external datasets (CoNLL/WNUT)
  - 🌐 SPARQL query on Wikidata (~3,000 politicians)
- **Label Model** trained for 500 epochs, with confidence threshold ≥ 0.7
- **Precise span extraction** using regex + gazetteer matching

### 3. Synthetic Augmentation

For underrepresented classes (`MARKET_EVENT`, `TRADE_AGREEMENT`, `POLICY`, `LEGISLATION`, `ECONOMIC_INDICATOR`), **synthetic examples** were generated using varied templates:

```
"Analysts compared the recent downturn to the {EVENT}, noting similar warning signs."
"Negotiations over {TRADE} dragged on for several years before a final deal was reached."
```

### 4. Fine-tuning

> **Compute environment:** Both models were trained on a dedicated server equipped with an **NVIDIA A40 GPU**.

Two architectures trained on the **same dataset** for **fair comparison**:

|                        | GLiNER                                  | spaCy                             |
| ---------------------- | --------------------------------------- | --------------------------------- |
| **Base**               | `urchade/gliner_small-v2.1`             | `en_core_web_trf` (RoBERTa-base)  |
| **Strategy**           | Full fine-tuning                        | Transformer frozen + NER head     |
| **Epochs**             | 10                                      | 15 (+ early stopping, patience=5) |
| **Batch size**         | 8                                       | 8 (compounding 4→32)              |
| **Learning rate**      | 3e-6                                    | 2e-5 (NER head)                   |
| **Zero-shot**          | ✅ Yes, can add new labels at inference | ❌ No                             |
| **Train / Dev / Test** | 5747 / 1228 / 2122                      | 5747 / 1228 / 2124                |

---

## 📊 Results

### Global Metrics (micro-averaged, `ent_type` mode)

| Model      | Precision | Recall | F1         |
| ---------- | --------- | ------ | ---------- |
| **GLiNER** | 0.6811    | 0.9094 | **0.7789** |
| **spaCy**  | 0.8633    | 0.7891 | **0.8245** |

> **Interpretation:** GLiNER has higher recall (finds more entities), while spaCy has higher precision (fewer false positives). spaCy wins overall on F1.

### GLiNER Unique Advantage: Zero-Shot Capability

GLiNER allows adding **new labels at inference** without retraining:

```
SANCTION, ELECTION, SUMMIT, CENTRAL_BANK_DECISION
```

This is possible due to its architecture that operates in the semantic space of labels rather than a fixed label set.

---

## ⚡ Quick Deployment — Pre-built Docker Images

> **The fastest way to run the application** — no build step required. Pre-built images are available on Docker Hub.

### Requirements

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed

### Steps

**1. Pull the pre-built images:**

```bash
docker pull tudorx95/ner-political-economic:frontend
docker pull tudorx95/ner-political-economic:backend
```

**2. Clone this repository:**

```bash
git clone https://github.com/Tudorx95/NER_Political_Economic.git
cd NER_Political_Economic
```

**3. Start all services using the root-level Compose file:**

```bash
docker-compose up -d
```

Open **http://localhost:8080** in your browser.

> ⏱ **First startup** takes a few minutes — models are downloaded from HuggingFace (~500MB) and cached in a Docker volume. Subsequent starts are instant.

---

### Option B — Build from Source

If you prefer to build the images yourself instead of pulling them, use the Compose file inside the `Deployment/` directory:

```bash
git clone https://github.com/Tudorx95/NER_Political_Economic.git
cd NER_Political_Economic/Deployment

docker-compose up --build -d
```

> This will build both the backend and frontend images locally from the Dockerfiles in `Deployment/backend/` and `Deployment/frontend/`.

---

### ⚠️ MongoDB AVX Compatibility Warning

Starting with **MongoDB 5.0**, the database requires a CPU that supports the **AVX (Advanced Vector Extensions)** instruction set. If your machine has an older CPU without AVX support (e.g. Intel Xeon Gold 6240R), the MongoDB container will crash in a restart loop and cause a **502 Bad Gateway** error.

```
WARNING: MongoDB 5.0+ requires a CPU with AVX support, and your current system
does not appear to have that!
```

---

## 🚀 Quick Start — Local Build

### With Docker Compose (build from source)

```bash
# Clone the repository
git clone https://github.com/Tudorx95/NER_Political_Economic.git
cd NER_Political_Economic/Deployment

# Build and start all services (backend + frontend)
docker-compose up --build
```

Open **http://localhost:8080** in your browser.

> ⏱ **First startup** takes a few minutes — models are downloaded from HuggingFace (~500MB).  
> Subsequent starts are instant (models are cached in a Docker volume).

### GPU Support (optional)

If you have an NVIDIA GPU with `nvidia-docker` installed, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Without Docker (development)

**Backend:**

```bash
cd Deployment/backend
pip install -r requirements.txt
python server.py
# Server running at http://localhost:8000
```

**Frontend:**

```bash
cd Deployment/frontend
npm install --legacy-peer-deps
npm start
# App running at http://localhost:3000
```

---

## 📁 Repository Structure

```
NER_Political_Economic/
│
├── 📁 dataset/                               # Dataset creation pipeline + synthetic data
│   ├── DatasetCreation.py                    # Main script: collection → Snorkel → split
│   ├── DatasetCreation.ipynb                 # Iterative pipeline notebook
│   ├── DatasetCreation_Results.ipynb         # Final dataset statistics and visualizations
│   ├── synthetic_augmented.jsonl             # Synthetic augmented data
│   └── README.md                             # Dataset technical documentation
│
├── 🧠 GLiNER_Results/                        # GLiNER fine-tuning and results
│   ├── trainer_Gliner.py                     # GLiNER training (evaluation + HF upload)
│   ├── GLiNER_FineTuning.ipynb               # Interactive GLiNER notebook
│   ├── DownloadGliner.py                     # Script: download and test GLiNER model
│   └── metrics.json                          # GLiNER evaluation metrics
│
├── 🧠 Spacy_Results/                         # spaCy fine-tuning and results
│   ├── trainer_Spacy.py                      # spaCy training (evaluation + HF upload)
│   ├── spaCy_FineTuning.ipynb                # Interactive spaCy notebook
│   ├── DownloadSpacy.py                      # Script: download and test spaCy model
│   └── metrics.json                          # spaCy evaluation metrics
│
├── 🖥  Deployment/                            # Containerized web application
│   ├── docker-compose.yaml                   # Container orchestration (backend + frontend + nginx)
│   ├── backend/
│   │   ├── server.py                         # FastAPI — GLiNER + spaCy model inference
│   │   ├── DownloadModel.py                  # Download GLiNER model from HF
│   │   ├── DownloadSpacy.py                  # Download spaCy model from HF
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/
│   │   ├── src/App.js                        # React — animated globe UI + entity highlighting
│   │   ├── src/components/                   # GlobeView, NERResults, CountryPanel, NERTag
│   │   ├── Dockerfile                        # Multi-stage build (React → Nginx)
│   │   └── nginx/nginx.conf                  # Frontend Nginx config
│   └── nginx/
│       └── nginx.conf                        # Global reverse proxy
│
├── docker-compose.yml                        # ← Quick deployment (pre-built images)
└── README.md                                 # ← This file
```

### What does each component do?

| Component                          | Function                                                                                                                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset/DatasetCreation.py`       | Full dataset creation pipeline: collection from 5 sources → spaCy segmentation → Snorkel LabelModel with 14 LFs → synthetic augmentation → validation + deduplication → train/dev/test split |
| `GLiNER_Results/trainer_Gliner.py` | Converts JSONL→GLiNER token-span format, trains with `gliner.training.Trainer`, evaluates with `nervaluate`, uploads to HuggingFace                                                          |
| `Spacy_Results/trainer_Spacy.py`   | Converts JSONL→spaCy format `(text, {"entities": [(s,e,label)]})`, trains with frozen transformer + NER head, evaluates, uploads to HuggingFace                                              |
| `Deployment/backend/server.py`     | FastAPI server: downloads both models from HF at startup, exposes `/predict` (per model) and `/predict_both` (compare both)                                                                  |
| `Deployment/frontend/src/App.js`   | React interface with animated CSS globe, country selector, editable text input, side-by-side GLiNER vs spaCy comparison, zero-shot label support                                             |

---

## 🛠 Technologies Used

### ML Pipeline

| Technology                     | Usage                                                                 |
| ------------------------------ | --------------------------------------------------------------------- |
| **Snorkel**                    | Weak Supervision — Label Model with 14 Labeling Functions             |
| **GLiNER**                     | Zero-shot capable NER architecture, fine-tuned on `gliner_small-v2.1` |
| **spaCy + spacy-transformers** | NER head fine-tuned on `en_core_web_trf` (RoBERTa-base)               |
| **PyTorch**                    | Deep learning backend                                                 |
| **nervaluate**                 | Standardized NER evaluation (ent_type, strict, partial, exact)        |
| **HuggingFace Hub**            | Model and dataset hosting                                             |
| **scikit-learn**               | Stratified train/dev/test split                                       |
| **Wikidata SPARQL**            | Gazetteer extension with ~3,000 politicians                           |
| **Wikipedia API**              | High-quality text collection                                          |

### Web Application

| Technology         | Usage                                  |
| ------------------ | -------------------------------------- |
| **FastAPI**        | REST API with inference on both models |
| **React**          | Frontend with interactive interface    |
| **Docker Compose** | Multi-container orchestration          |
| **Nginx**          | Reverse proxy, static asset serving    |
| **MongoDB**        | NER result persistence                 |

---

## 👤 Author

**Sd. Sg. Maj. Lepădatu Tudor**  
Military Technical Academy "Ferdinand I", 2026

---

<div align="center">

_Built using Snorkel, GLiNER, spaCy, FastAPI & React_

</div>
