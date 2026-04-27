<div align="center">

# 🌐 NER Political & Economic — End-to-End ML Pipeline

**Extragerea automată a entităților politice și economice din text folosind NLP**

[![HuggingFace Dataset](https://img.shields.io/badge/🤗_Dataset-NER__Political__Economic-yellow)](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic)
[![HuggingFace GLiNER](https://img.shields.io/badge/🤗_Model-GLiNER-blue)](https://huggingface.co/Tudorx95/NER_Economic_Political)
[![HuggingFace spaCy](https://img.shields.io/badge/🤗_Model-spaCy-green)](https://huggingface.co/Tudorx95/NER_Economic_Political_Spacy)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](./ner-demo/docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)


</div>

---

## 📑 Cuprins

- [Descriere](#-descriere)
- [Arhitectura Proiectului](#-arhitectura-proiectului)
- [Resurse Externe (HuggingFace)](#-resurse-externe-huggingface)
- [Schema NER — 11 Tipuri de Entități](#-schema-ner--11-tipuri-de-entități)
- [Pipeline-ul ML](#-pipeline-ul-ml)
- [Rezultate](#-rezultate)
- [Quick Start — Rulare Locală](#-quick-start--rulare-locală)
- [Structura Repository-ului](#-structura-repository-ului)
- [Tehnologii Utilizate](#-tehnologii-utilizate)
- [Autor](#-autor)

---

## 📖 Descriere

Acest proiect implementează un **pipeline complet de Machine Learning** pentru recunoașterea entităților denumite (Named Entity Recognition — NER) din domeniul **politic și economic**. 

Proiectul acoperă toate etapele:

1. **Colectarea datelor** din surse multiple (CC-News, Wikipedia, SEC EDGAR, CoNLL-2003, WNUT-17)
2. **Weak Supervision cu Snorkel** — etichetare programatică folosind 14 Labeling Functions
3. **Augmentare sintetică** pentru clasele sub-reprezentate
4. **Fine-tuning** pe două arhitecturi diferite: **GLiNER** (zero-shot capable) și **spaCy** (transformer-based)
5. **Deployment** într-o aplicație full-stack containerizată cu Docker (React + FastAPI + Nginx)

---

## 🏗 Arhitectura Proiectului

```
┌──────────────────────────────────────────────────────────────────────┐
│                        📦 GitHub Repository                         │
│                   (Cod sursă, Notebooks, Aplicație)                  │
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
│  │  9.1k exemple    │  │  F1: 0.7789      │  │  F1: 0.8245       │  │
│  │  11 labels       │  │  Zero-Shot NER   │  │  RoBERTa-base     │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🤗 Resurse Externe (HuggingFace)

| Resursă | Link | Descriere |
|---------|------|-----------|
| 🗄️ **Dataset** | [`Tudorx95/NER_Political_Economic`](https://huggingface.co/datasets/Tudorx95/NER_Political_Economic) | ~9.100 exemple, 11 tipuri de entități, creat prin Snorkel + surse multiple |
| 🧠 **Model GLiNER** | [`Tudorx95/NER_Economic_Political`](https://huggingface.co/Tudorx95/NER_Economic_Political) | Fine-tuned `gliner_small-v2.1`, suportă zero-shot NER |
| 🧠 **Model spaCy** | [`Tudorx95/NER_Economic_Political_Spacy`](https://huggingface.co/Tudorx95/NER_Economic_Political_Spacy) | Fine-tuned `en_core_web_trf` (RoBERTa-base backbone) |

> **Notă:** Modelele sunt descărcate automat de pe HuggingFace Hub la prima pornire a aplicației.

---

## 🏷 Schema NER — 11 Tipuri de Entități

| Entitate | Descriere | Exemplu |
|----------|-----------|---------|
| `POLITICIAN` | Persoane politice | *Joe Biden*, *Christine Lagarde* |
| `POLITICAL_PARTY` | Partide politice | *Republican Party*, *CDU* |
| `POLITICAL_ORG` | Organizații politice | *NATO*, *European Union*, *G7* |
| `FINANCIAL_ORG` | Organizații financiare | *Federal Reserve*, *IMF*, *Goldman Sachs* |
| `ECONOMIC_INDICATOR` | Indicatori economici | *GDP*, *CPI*, *unemployment rate* |
| `POLICY` | Politici guvernamentale | *quantitative easing*, *rate hike* |
| `LEGISLATION` | Acte legislative | *Dodd-Frank Act*, *CHIPS Act* |
| `MARKET_EVENT` | Evenimente de piață | *2008 financial crisis*, *Great Recession* |
| `CURRENCY` | Monede | *USD*, *euro*, *Bitcoin* |
| `TRADE_AGREEMENT` | Acorduri comerciale | *NAFTA*, *USMCA*, *TPP* |
| `GPE` | Entități geopolitice | *United States*, *China*, *Germany* |

---

## 🔬 Pipeline-ul ML

### 1. Colectarea Datelor (`dataset/`)

Datele provin din **5 surse complementare**:

| Sursă | Metoda | Rol |
|-------|--------|-----|
| **CC-News** | Filtrare pe keywords politico-economice, procesat cu spaCy | Volum mare, diversitate lingvistică |
| **Wikipedia** | API `wikipediaapi`, articole despre politicieni, organizații, concepte economice | Calitate ridicată, fapte verificabile |
| **SEC EDGAR** | Rapoarte 10-K descărcate automat | Limbaj financiar real |
| **CoNLL-2003** | Remapare din schema BIO (`PER`→`POLITICIAN`, `ORG`→`FINANCIAL_ORG`/`POLITICAL_ORG`, etc.) | Date etichetate profesional |
| **WNUT-2017** | Remapare similară cu CoNLL | Texte din social media, diversitate stilistică |

### 2. Weak Supervision cu Snorkel

În loc de etichetare manuală costisitoare, am folosit **Snorkel** pentru a genera pseudo-etichete:

- **14 Labeling Functions (LF)** bazate pe:
  - 📚 Gazetteers (liste de politicieni, organizații, indicatori)
  - 🔍 Pattern-matching regex (titluri politice, simboluri valutare)
  - 🗂️ Match cu entități din dataseturile externe remapate (CoNLL/WNUT)
  - 🌐 Interogare SPARQL pe Wikidata (~3000 politicieni)
- **Label Model** antrenat pe 500 de epoci, cu prag de confidență ≥ 0.7
- **Extracție de span-uri** precise folosind regex + gazetteer matching

### 3. Augmentare Sintetică

Pentru clasele sub-reprezentate (`MARKET_EVENT`, `TRADE_AGREEMENT`, `POLICY`, `LEGISLATION`, `ECONOMIC_INDICATOR`), am generat **exemple sintetice** cu template-uri variate:

```
"Analysts compared the recent downturn to the {EVENT}, noting similar warning signs."
"Negotiations over {TRADE} dragged on for several years before a final deal was reached."
```

### 4. Fine-tuning

Două arhitecturi antrenate pe **același dataset** pentru **comparație echitabilă**:

| | GLiNER | spaCy |
|---|--------|-------|
| **Bază** | `urchade/gliner_small-v2.1` | `en_core_web_trf` (RoBERTa-base) |
| **Strategie** | Fine-tuning complet | Transformer frozen + NER head |
| **Epoci** | 10 | 1 (+ early stopping, patience=5) |
| **Batch size** | 8 | 8 (compounding 4→32) |
| **Learning rate** | 3e-6 | 1e-3 (NER head) |
| **Zero-shot** | ✅ Da, poate adăuga etichete noi la inferență | ❌ Nu |
| **Train / Dev / Test** | 5747 / 1228 / 2122 | 5747 / 1228 / 2124 |

---

## 📊 Rezultate

### Metrici Globale (micro-averaged, modul `ent_type`)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| **GLiNER** | 0.6811 | 0.9094 | **0.7789** |
| **spaCy** | 0.8633 | 0.7891 | **0.8245** |

> **Interpretare:** GLiNER are recall mai mare (găsește mai multe entități), iar spaCy are precizie mai mare (mai puține false positive). spaCy câștigă per total la F1.

### Avantaj unic GLiNER: Zero-Shot Capability

GLiNER permite adăugarea de **etichete noi la inferență** fără re-antrenare:
```
SANCTION, ELECTION, SUMMIT, CENTRAL_BANK_DECISION
```

Acest lucru este posibil datorită arhitecturii sale care operează pe spațiul semantic al etichetelor, nu pe un set fix.

---

## 🚀 Quick Start — Rulare Locală

### Cu Docker Compose (recomandat)

```bash
# Clonează repository-ul
git clone https://github.com/Tudorx95/NER_Political_Economic.git
cd NER_Political_Economic/ner-demo

# Pornește toate serviciile (backend + frontend)
docker-compose up --build
```

Deschide **http://localhost:3000** în browser.

> ⏱ **Prima pornire** durează câteva minute — se descarcă modelele de pe HuggingFace (~500MB).  
> Pornirile ulterioare sunt instantanee (modelele sunt cache-uite într-un Docker volume).

### GPU Support (opțional)

Dacă ai un GPU NVIDIA cu `nvidia-docker` instalat, decomentează secțiunea `deploy` din `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Fără Docker (dezvoltare)

**Backend:**
```bash
cd ner-demo/backend
pip install -r requirements.txt
python app.py
# Server pornit pe http://localhost:8000
```

**Frontend:**
```bash
cd ner-demo/frontend
npm install --legacy-peer-deps
npm start
# Aplicație pornită pe http://localhost:3000
```

---

## 📁 Structura Repository-ului

```
NER_Political_Economic/
│
├── 📓 ML Pipeline (Research & Training)
│   ├── dataset/
│   │   ├── DatasetCreation.py            # Pipeline complet: colectare → Snorkel → split
│   │   ├── DatasetCreationV2.ipynb       # Notebook iterația 2
│   │   ├── DatasetCreation_v3.ipynb      # Notebook iterația 3 (finală)
│   │   └── synthetic_augmented.jsonl     # Date sintetice augmentate
│   │
│   ├── trainer_Gliner.py                 # Training GLiNER complet (cu evaluare + upload HF)
│   ├── trainer_Spacy.py                  # Training spaCy complet (cu evaluare + upload HF)
│   ├── GLiNER_Training.ipynb             # Notebook interactiv GLiNER
│   ├── spaCy_FineTuning.ipynb            # Notebook interactiv spaCy
│   ├── DownloadModel.py                  # Script test: descarcă și testează modelul GLiNER
│   └── DownloadSpacy.py                  # Script test: descarcă și testează modelul spaCy
│
├── 🖥  Web Application (Deployment)
│   ├── ner-demo/                         # ← Aplicația principală
│   │   ├── docker-compose.yml            # Orchestrare containere
│   │   ├── backend/
│   │   │   ├── app.py                    # FastAPI — descarcă modele de pe HF + servește inferență
│   │   │   ├── Dockerfile
│   │   │   └── requirements.txt
│   │   ├── frontend/
│   │   │   ├── src/App.js                # React — UI cu glob animat + entity highlighting
│   │   │   ├── Dockerfile                # Multi-stage build (React → Nginx)
│   │   │   └── nginx.conf               # Reverse proxy → backend
│   │   └── README.md
│   │
│   └── Deployment/                       # Versiune alternativă cu MongoDB
│       ├── docker-compose.yaml           # Include MongoDB pentru persistență
│       ├── backend/
│       │   ├── server.py                 # FastAPI + Motor (MongoDB async)
│       │   └── Dockerfile
│       ├── frontend/
│       └── nginx/
│
└── README.md                             # ← Acest fișier
```

### Ce face fiecare componentă?

| Componentă | Funcție |
|-----------|---------|
| `dataset/DatasetCreation.py` | Pipeline complet de creare a datasetului: colectare din 5 surse → segmentare cu spaCy → Snorkel LabelModel cu 14 LFs → augmentare sintetică → validare + deduplicare → split train/dev/test |
| `trainer_Gliner.py` | Convertește JSONL→format GLiNER token-span, antrenează cu `gliner.training.Trainer`, evaluează cu `nervaluate`, uploadează pe HuggingFace |
| `trainer_Spacy.py` | Convertește JSONL→format spaCy `(text, {"entities": [(s,e,label)]})`, antrenează cu transformer frozen + NER head, evaluează, uploadează pe HuggingFace |
| `ner-demo/backend/app.py` | Server FastAPI: descarcă ambele modele de pe HF la startup, expune `/predict` (per model) și `/predict_both` (compara ambele) |
| `ner-demo/frontend/src/App.js` | Interfață React cu glob CSS animat, selector de țări, input text editabil, comparare side-by-side GLiNER vs spaCy, suport zero-shot labels |

---

## 🛠 Tehnologii Utilizate

### ML Pipeline
| Tehnologie | Utilizare |
|-----------|-----------|
| **Snorkel** | Weak Supervision — Label Model cu 14 Labeling Functions |
| **GLiNER** | Arhitectură NER zero-shot capable, fine-tuned pe `gliner_small-v2.1` |
| **spaCy + spacy-transformers** | NER head fine-tuned pe `en_core_web_trf` (RoBERTa-base) |
| **PyTorch** | Backend deep learning |
| **nervaluate** | Evaluare NER standardizată (ent_type, strict, partial, exact) |
| **HuggingFace Hub** | Hosting modele și dataset |
| **scikit-learn** | Stratified train/dev/test split |
| **Wikidata SPARQL** | Extindere gazetteer cu ~3000 politicieni |
| **Wikipedia API** | Colectare texte de calitate |

### Web Application
| Tehnologie | Utilizare |
|-----------|-----------|
| **FastAPI** | REST API cu inferență pe ambele modele |
| **React** | Frontend cu interfață interactivă |
| **Docker Compose** | Orchestrare multi-container |
| **Nginx** | Reverse proxy, servire static assets |
| **MongoDB** *(opțional)* | Persistență rezultate NER (varianta `Deployment/`) |

---

## 👤 Autor

**Sd. Sg. Maj. Lepădatu Tudor**  
Academia Tehnică Militară „Ferdinand I", 2026

---

<div align="center">

*Built using Snorkel, GLiNER, spaCy, FastAPI & React*

</div>
