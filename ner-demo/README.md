# NER Entity Compare — Political & Economic

Aplicatie demo pentru compararea a doua modele NER (GLiNER vs spaCy)
antrenate pe entitati politice si economice.

## Arhitectura

```
┌─────────────────┐     ┌──────────────────┐
│   Frontend      │────▶│   Backend        │
│   React + Nginx │     │   FastAPI        │
│   Port 3000     │     │   Port 8000      │
│                 │     │                  │
│   3D Globe      │     │   GLiNER model   │
│   Text Input    │     │   spaCy model    │
│   Entity Viz    │     │   HuggingFace ↓  │
└─────────────────┘     └──────────────────┘
```

## Quick Start

### 1. Cu Docker Compose (recomandat)

```bash
cd ner-demo
docker-compose up --build
```

Deschide http://localhost:3000

### 2. Fara Docker (dezvoltare)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

## Configurare GPU

Decomentaza sectiunea `deploy` din `docker-compose.yml` daca ai GPU NVIDIA cu docker-nvidia instalat.

## Modele

- **GLiNER**: `Tudorx95/NER_Economic_Political` — fine-tuned pe dataset custom
- **spaCy**: `Tudorx95/NER_Economic_Political_Spacy` — en_core_web_trf fine-tuned

Modelele se descarca automat de pe HuggingFace la prima pornire.

## Zero-Shot Learning (GLiNER)

GLiNER suporta zero-shot NER — poti adauga etichete noi fara re-antrenare.
In interfata, scrie etichete separate prin virgula in campul "Zero-shot extra labels":

Exemple:
- `SANCTION` — detecteaza mentionarea sanctiunilor economice
- `ELECTION` — detecteaza referinte la alegeri
- `SUMMIT` — detecteaza summit-uri internationale
- `CENTRAL_BANK_DECISION` — decizii ale bancilor centrale

## Structura fisiere

```
ner-demo/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py              # FastAPI server
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── index.js
│       └── App.js           # React app
└── README.md
```

## Autor

Sd. Sg. Maj. Lepadatu Tudor  
Academia Tehnica Militara "Ferdinand I", 2026
