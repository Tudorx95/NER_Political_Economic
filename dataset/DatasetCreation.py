import os
from pathlib import Path
import re
import json
import time
import random
import requests
import warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter, defaultdict
from datasets import load_dataset
import wikipediaapi
import spacy


BASE_DIR = Path("/home/tudor.lepadatu/AI_CD")

RAW_DIR        = BASE_DIR / "raw"
ANNOTATED_DIR  = BASE_DIR / "annotated"
SPLITS_DIR     = BASE_DIR / "splits"
EXTERNAL_DIR   = BASE_DIR / "external"

for d in [RAW_DIR, ANNOTATED_DIR, SPLITS_DIR, EXTERNAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)


NER_LABELS = [
    "POLITICIAN",
    "POLITICAL_PARTY",
    "POLITICAL_ORG",
    "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR",
    "POLICY",
    "LEGISLATION",
    "MARKET_EVENT",
    "CURRENCY",
    "TRADE_AGREEMENT",
    "GPE",
]

MAX_CC_NEWS_ARTICLES   = 5000
MAX_WIKI_ARTICLES      = 60
MAX_SEC_FILINGS        = 200
MAX_SENTENCES_PER_DOC  = 20
MIN_SENTENCE_LEN       = 40
MAX_SENTENCE_LEN       = 400

SNORKEL_CONFIDENCE_THRESHOLD = 0.7

# Source 1: CC-News

KEYWORDS = [
    # Economic
    "GDP", "inflation", "interest rate", "Federal Reserve", "IMF",
    "World Bank", "stock market", "bond yield", "fiscal policy",
    "monetary policy", "recession", "unemployment", "CPI", "trade deficit",
    # Political
    "president", "senator", "prime minister", "parliament", "election",
    "congress", "legislation", "sanctions", "NATO", "treaty", "minister",
    "European Commission", "White House", "G7", "G20",
]

cc_news_dataset = load_dataset("cc_news", split="train", streaming=True, trust_remote_code=True)

cc_news_texts = []
pbar = tqdm(total=MAX_CC_NEWS_ARTICLES, desc="CC-News")

for article in cc_news_dataset:
    text = article.get("text", "").strip()
    if not text or len(text) < 200:
        continue
    text_lower = text.lower()
    if any(kw.lower() in text_lower for kw in KEYWORDS):
        cc_news_texts.append(text[:5000])  # limitam lungimea per articol
        pbar.update(1)
        if len(cc_news_texts) >= MAX_CC_NEWS_ARTICLES:
            break

pbar.close()

with open(RAW_DIR / "cc_news.json", "w") as f:
    json.dump(cc_news_texts, f)
print(f"Salvat: {RAW_DIR / 'cc_news.json'}")


# Source 2: Wikipedia
WIKI_TOPICS = [
    # Politicieni
    "Joe Biden", "Donald Trump", "Jerome Powell", "Christine Lagarde",
    "Janet Yellen", "Emmanuel Macron", "Olaf Scholz", "Rishi Sunak",
    "Angela Merkel", "Barack Obama", "Xi Jinping", "Vladimir Putin",
    # Organizatii politice
    "NATO", "European Union", "United Nations", "European Commission",
    "Republican Party (United States)", "Democratic Party (United States)",
    "G7", "G20", "BRICS",
    # Organizatii economice
    "International Monetary Fund", "World Bank", "Federal Reserve",
    "European Central Bank", "Bank of England", "Goldman Sachs",
    "JPMorgan Chase", "BlackRock",
    # Concepte economice
    "Quantitative easing", "Inflation", "Gross domestic product",
    "Monetary policy", "Fiscal policy", "Interest rate",
    "Stock market crash", "2008 financial crisis", "Stagflation",
    # Legislatie / Tratate
    "Dodd–Frank Wall Street Reform and Consumer Protection Act",
    "Paris Agreement", "North American Free Trade Agreement",
    "Maastricht Treaty", "Basel III",
]

wiki = wikipediaapi.Wikipedia(language="en", user_agent="NER-Dataset-Builder/2.0")
wiki_texts = []

for title in tqdm(WIKI_TOPICS[:MAX_WIKI_ARTICLES], desc="Wikipedia"):
    page = wiki.page(title)
    if page.exists():
        wiki_texts.append(page.text[:8000])

with open(RAW_DIR / "wikipedia.json", "w") as f:
    json.dump(wiki_texts, f)
print(f"Salvat: {RAW_DIR / 'wikipedia.json'}")


# Source 3: EDGAR

try:
    from sec_edgar_downloader import Downloader

    TICKERS = ["AAPL", "JPM", "GS", "BAC", "MS"]

    sec_raw_dir = RAW_DIR / "sec_raw"
    dl = Downloader("NER Research", "ner@research.com", str(sec_raw_dir))
    sec_texts = []

    for ticker in tqdm(TICKERS, desc="SEC EDGAR"):
        try:
            dl.get("10-K", ticker, limit=1)
        except Exception as e:
            print(f"Skip {ticker}: {e}")

    if sec_raw_dir.exists():
        for txt_file in list(sec_raw_dir.rglob("*.txt"))[:MAX_SEC_FILINGS]:
            try:
                text = txt_file.read_text(errors="ignore")[:3000]
                if len(text) > 200:
                    sec_texts.append(text)
            except Exception:
                pass

    print(f"Colectate {len(sec_texts)} fragmente SEC EDGAR.")
    with open(RAW_DIR / "sec_edgar.json", "w") as f:
        json.dump(sec_texts, f)
    print(f"Salvat: {RAW_DIR / 'sec_edgar.json'}")

except Exception as e:
    print(f"SEC EDGAR esuat (continuam fara): {e}")
    sec_texts = []
    with open(RAW_DIR / "sec_edgar.json", "w") as f:
        json.dump([], f)

# Divide into phrases using spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def extract_sentences(texts, source_name, max_per_doc=MAX_SENTENCES_PER_DOC):
    sentences = []
    for text in tqdm(texts, desc=f"Segmentare {source_name}", leave=False):
        if isinstance(text, dict):
            text = text.get("text", "")
        try:
            doc = nlp(text[:10000])
            sents = [
                sent.text.strip()
                for sent in doc.sents
                if MIN_SENTENCE_LEN <= len(sent.text.strip()) <= MAX_SENTENCE_LEN
            ]
            sentences.extend(sents[:max_per_doc])
        except Exception:
            pass
    return sentences

# Reincarcam datele raw (in caz de restart runtime)
with open(RAW_DIR / "cc_news.json") as f:
    cc_news_texts = json.load(f)
with open(RAW_DIR / "wikipedia.json") as f:
    wiki_texts = json.load(f)
try:
    with open(RAW_DIR / "sec_edgar.json") as f:
        sec_texts = json.load(f)
except FileNotFoundError:
    sec_texts = []

sents_cc   = extract_sentences(cc_news_texts, "CC-News")
sents_wiki = extract_sentences(wiki_texts, "Wikipedia")
sents_sec  = extract_sentences(sec_texts, "SEC EDGAR")

# Combinam si deduplicam
all_sentences = list(set(sents_cc + sents_wiki + sents_sec))
random.shuffle(all_sentences)

# Marcare sursa pentru fiecare propozitie
def mark_source(sents, source):
    return [{"text": s, "source": source} for s in sents]

cc_marked   = mark_source(sents_cc, "cc_news")
wiki_marked = mark_source(sents_wiki, "wikipedia")
sec_marked  = mark_source(sents_sec, "sec_edgar")

# Deduplicare pastrand sursa
seen = set()
all_marked = []
for s in cc_marked + wiki_marked + sec_marked:
    if s["text"] not in seen:
        seen.add(s["text"])
        all_marked.append(s)
random.shuffle(all_marked)

df_sentences = pd.DataFrame(all_marked)
print(f"Total propozitii unice: {len(df_sentences)}")
print(df_sentences['source'].value_counts())

df_sentences.to_json(RAW_DIR / "sentences.jsonl", orient="records", lines=True)


# Gazetters 


POLITICIANS_GAZETTEER = [
    "Joe Biden", "Donald Trump", "Barack Obama", "George W. Bush",
    "Bill Clinton", "Jerome Powell", "Janet Yellen", "Christine Lagarde",
    "Emmanuel Macron", "Olaf Scholz", "Rishi Sunak", "Angela Merkel",
    "Boris Johnson", "Xi Jinping", "Vladimir Putin", "Narendra Modi",
    "Ursula von der Leyen", "Mario Draghi", "Ben Bernanke", "Alan Greenspan",
    "Larry Summers", "Paul Volcker", "Mark Carney",
    "Lagarde", "Yellen", "Powell", "Bernanke",
]

FINANCIAL_ORGS_GAZETTEER = [
    "Federal Reserve", "Fed", "IMF", "International Monetary Fund",
    "World Bank", "European Central Bank", "ECB", "Bank of England",
    "Goldman Sachs", "JPMorgan", "JPMorgan Chase", "Morgan Stanley",
    "BlackRock", "Citigroup", "Bank of America", "Wells Fargo",
    "Deutsche Bank", "Barclays", "HSBC", "Credit Suisse",
    "Bank for International Settlements", "BIS", "FDIC",
    "Securities and Exchange Commission", "SEC", "Bank of Japan",
    "People's Bank of China", "Reserve Bank of India",
]

POLITICAL_ORGS_GAZETTEER = [
    "NATO", "United Nations", "UN", "European Union", "EU",
    "European Commission", "European Parliament", "G7", "G20", "BRICS",
    "OECD", "WTO", "World Trade Organization",
    "Congress", "Senate", "House of Representatives", "White House",
    "Pentagon", "State Department", "Treasury Department",
    "European Council", "African Union", "ASEAN",
]

POLITICAL_PARTIES_GAZETTEER = [
    "Republican Party", "Democratic Party", "Labour Party",
    "Conservative Party", "Liberal Democrats", "CDU", "SPD", "Greens",
    "En Marche", "Five Star Movement", "Lega", "Vox", "Podemos",
    "Communist Party of China", "Bharatiya Janata Party", "BJP",
    "Republicans", "Democrats",
]

ECONOMIC_INDICATORS_GAZETTEER = [
    "GDP", "gross domestic product", "CPI", "consumer price index",
    "PPI", "producer price index", "unemployment rate", "inflation rate",
    "interest rate", "federal funds rate", "yield curve",
    "trade deficit", "trade surplus", "PMI", "purchasing managers index",
    "nonfarm payrolls", "retail sales", "industrial production",
]

LEGISLATION_GAZETTEER = [
    "Dodd-Frank Act", "Dodd-Frank", "Inflation Reduction Act",
    "CHIPS Act", "Sarbanes-Oxley Act", "Glass-Steagall Act",
    "Affordable Care Act", "Patriot Act", "Volcker Rule",
    "Sherman Antitrust Act", "Basel III", "MiFID II",
    "GDPR", "General Data Protection Regulation",
]

CURRENCIES_GAZETTEER = [
    "USD", "EUR", "GBP", "JPY", "CNY", "CHF", "CAD", "AUD",
    "dollar", "euro", "pound sterling", "yen", "yuan", "renminbi",
    "Bitcoin", "BTC", "Ethereum", "ETH",
]

TRADE_AGREEMENTS_GAZETTEER = [
    "NAFTA", "USMCA", "TPP", "Trans-Pacific Partnership",
    "RCEP", "CPTPP", "EU-Mercosur", "WTO",
    "North American Free Trade Agreement",
    "Comprehensive and Progressive Agreement for Trans-Pacific Partnership",
]

MARKET_EVENTS_GAZETTEER = [
    "2008 financial crisis", "Great Recession", "Black Monday",
    "dot-com bubble", "Asian financial crisis", "European debt crisis",
    "COVID-19 recession", "subprime mortgage crisis",
    "Lehman Brothers collapse", "flash crash",
]

print(f"Gazetteers incarcate. Politicieni manuali: {len(POLITICIANS_GAZETTEER)}")


# ─── Extindere politicieni cu Wikidata (optional) ────────────────────────────
def get_wikidata_politicians(limit=3000):
    query = f"""
    SELECT DISTINCT ?personLabel WHERE {{
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P106 wd:Q82955 .
      ?person wikibase:sitelinks ?links .
      FILTER(?links > 5)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }} LIMIT {limit}
    """
    headers = {
        "User-Agent": "NER-Dataset-Builder/2.0 (research project)",
        "Accept": "application/json"
    }
    try:
        r = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=30
        )
        if r.status_code == 200:
            return [
                item["personLabel"]["value"]
                for item in r.json()["results"]["bindings"]
                if len(item["personLabel"]["value"].split()) >= 2
            ]
    except Exception as e:
        print(f"Wikidata query esuat: {e}")
    return []

print("Se interogheaza Wikidata...")
wikidata_politicians = get_wikidata_politicians(limit=3000)
ALL_POLITICIANS = list(set(POLITICIANS_GAZETTEER + wikidata_politicians))
print(f"Total politicieni in gazetteer: {len(ALL_POLITICIANS)}")


# Remapare CoNLL-2003 + WNUT-2017

# ─── Helper: cautare in gazetteer (case-insensitive, normalizat) ─────────────
def _normalize(s):
    return re.sub(r'\s+', ' ', s.lower().strip())

POLITICIANS_NORM = {_normalize(p) for p in ALL_POLITICIANS}
FIN_ORGS_NORM    = {_normalize(o) for o in FINANCIAL_ORGS_GAZETTEER}
POL_ORGS_NORM    = {_normalize(o) for o in POLITICAL_ORGS_GAZETTEER}
PARTIES_NORM     = {_normalize(p) for p in POLITICAL_PARTIES_GAZETTEER}
LEGIS_NORM       = {_normalize(l) for l in LEGISLATION_GAZETTEER}
TRADE_NORM       = {_normalize(t) for t in TRADE_AGREEMENTS_GAZETTEER}
EVENTS_NORM      = {_normalize(e) for e in MARKET_EVENTS_GAZETTEER}
CURR_NORM        = {_normalize(c) for c in CURRENCIES_GAZETTEER}
INDIC_NORM       = {_normalize(i) for i in ECONOMIC_INDICATORS_GAZETTEER}

def _gazetteer_lookup(text, gazetteer_norm):
    t = _normalize(text)
    if t in gazetteer_norm:
        return True
    # match partial: orice cuvant din gazetteer e in text
    for g in gazetteer_norm:
        if g in t or t in g:
            return True
    return False

def map_per(entity_text):
    return "POLITICIAN" if _gazetteer_lookup(entity_text, POLITICIANS_NORM) else None

def map_org(entity_text):
    if _gazetteer_lookup(entity_text, FIN_ORGS_NORM):
        return "FINANCIAL_ORG"
    if _gazetteer_lookup(entity_text, POL_ORGS_NORM):
        return "POLITICAL_ORG"
    return None

def map_loc(entity_text):
    return "GPE"  # toate locatiile devin GPE

def map_misc(entity_text):
    if _gazetteer_lookup(entity_text, PARTIES_NORM):
        return "POLITICAL_PARTY"
    if _gazetteer_lookup(entity_text, LEGIS_NORM):
        return "LEGISLATION"
    if _gazetteer_lookup(entity_text, TRADE_NORM):
        return "TRADE_AGREEMENT"
    if _gazetteer_lookup(entity_text, EVENTS_NORM):
        return "MARKET_EVENT"
    if _gazetteer_lookup(entity_text, CURR_NORM):
        return "CURRENCY"
    return None

def map_group(entity_text):
    if _gazetteer_lookup(entity_text, PARTIES_NORM):
        return "POLITICAL_PARTY"
    if _gazetteer_lookup(entity_text, POL_ORGS_NORM):
        return "POLITICAL_ORG"
    return None

def map_corporation(entity_text):
    if _gazetteer_lookup(entity_text, FIN_ORGS_NORM):
        return "FINANCIAL_ORG"
    if _gazetteer_lookup(entity_text, POL_ORGS_NORM):
        return "POLITICAL_ORG"
    return None

print("Functii de mapare definite.")



print("Se incarca CoNLL-2003...")
try:
    conll = load_dataset("conll2003", trust_remote_code=True)
except Exception as e:
    # fallback la versiunea community-ported daca cea oficiala da erori
    print(f"Variant 1 esuat ({e}); incerc tner/conll2003...")
    conll = load_dataset("tner/conll2003", trust_remote_code=True)

# Tag mapping CoNLL (BIO scheme)
# 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC, 7=B-MISC, 8=I-MISC
CONLL_TAGS = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-MISC", 8: "I-MISC",
}

def extract_entities_bio(tokens, ner_tags, tag_map):
    """Convert BIO tags into list of {text, label, start, end} on the joined text."""
    entities = []
    text_parts = []
    char_pos = 0
    token_starts = []

    # Reconstruim textul si pozitiile char per token
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            char_pos += 1
        token_starts.append(char_pos)
        text_parts.append(tok)
        char_pos += len(tok)

    text = "".join(text_parts)

    # Parsam BIO
    i = 0
    while i < len(ner_tags):
        tag = tag_map.get(ner_tags[i], "O")
        if tag.startswith("B-"):
            label_raw = tag[2:]
            start_tok = i
            j = i + 1
            while j < len(ner_tags) and tag_map.get(ner_tags[j], "O") == f"I-{label_raw}":
                j += 1
            end_tok = j - 1
            ent_text = " ".join(tokens[start_tok:end_tok+1])
            ent_start = token_starts[start_tok]
            ent_end = ent_start + len(ent_text)
            entities.append({
                "text": ent_text,
                "raw_label": label_raw,
                "start": ent_start,
                "end": ent_end
            })
            i = j
        else:
            i += 1
    return text, entities

# Mapper pentru CoNLL raw labels -> schema noastra
CONLL_MAPPERS = {
    "PER": map_per,
    "ORG": map_org,
    "LOC": map_loc,
    "MISC": map_misc,
}

def remap_conll_split(split_data, split_name):
    remapped = []
    skipped_no_entity = 0
    for example in tqdm(split_data, desc=f"CoNLL {split_name}"):
        text, raw_ents = extract_entities_bio(
            example["tokens"], example["ner_tags"], CONLL_TAGS
        )
        mapped_ents = []
        for e in raw_ents:
            mapper = CONLL_MAPPERS.get(e["raw_label"])
            if mapper is None:
                continue
            new_label = mapper(e["text"])
            if new_label is not None:
                mapped_ents.append({
                    "text": e["text"],
                    "label": new_label,
                    "start": e["start"],
                    "end": e["end"]
                })
        if mapped_ents:
            remapped.append({
                "text": text,
                "entities": mapped_ents,
                "source": f"conll2003_{split_name}"
            })
        else:
            skipped_no_entity += 1
    return remapped, skipped_no_entity

conll_train, sk1 = remap_conll_split(conll["train"], "train")
conll_val,   sk2 = remap_conll_split(conll["validation"], "val")
conll_test,  sk3 = remap_conll_split(conll["test"], "test")

print(f"\nCoNLL-2003 remapat:")
print(f"  train: {len(conll_train)} (skipped {sk1})")
print(f"  val:   {len(conll_val)}   (skipped {sk2})")
print(f"  test:  {len(conll_test)}  (skipped {sk3})")
print(f"  TOTAL utile: {len(conll_train) + len(conll_val) + len(conll_test)}")

# Salvam: train+val intra in pool-ul comun, test ramane gold standard
conll_pool = conll_train + conll_val
conll_gold = conll_test

with open(EXTERNAL_DIR / "conll_pool.jsonl", "w") as f:
    for ex in conll_pool:
        f.write(json.dumps(ex) + "\n")
with open(EXTERNAL_DIR / "conll_gold.jsonl", "w") as f:
    for ex in conll_gold:
        f.write(json.dumps(ex) + "\n")
print(f"\nSalvat: {EXTERNAL_DIR / 'conll_pool.jsonl'} si conll_gold.jsonl")


print("Se incarca WNUT-2017...")
try:
    wnut = load_dataset("wnut_17", trust_remote_code=True)
except Exception as e:
    print(f"Variant 1 esuat ({e}); incerc tner/wnut2017...")
    wnut = load_dataset("tner/wnut2017", trust_remote_code=True)

# WNUT-17 BIO tags:
# 0=O, 1=B-corporation, 2=I-corporation, 3=B-creative-work, 4=I-creative-work,
# 5=B-group, 6=I-group, 7=B-location, 8=I-location, 9=B-person, 10=I-person,
# 11=B-product, 12=I-product
WNUT_TAGS = {
    0: "O",
    1: "B-corporation",  2: "I-corporation",
    3: "B-creative-work", 4: "I-creative-work",
    5: "B-group",        6: "I-group",
    7: "B-location",     8: "I-location",
    9: "B-person",       10: "I-person",
    11: "B-product",     12: "I-product",
}

WNUT_MAPPERS = {
    "corporation": map_corporation,
    "person": map_per,
    "location": map_loc,
    "group": map_group,
    # creative-work si product le ignoram (nu fit-eaza schema politico-economica)
}

def remap_wnut_split(split_data, split_name):
    remapped = []
    for example in tqdm(split_data, desc=f"WNUT {split_name}"):
        text, raw_ents = extract_entities_bio(
            example["tokens"], example["ner_tags"], WNUT_TAGS
        )
        mapped_ents = []
        for e in raw_ents:
            mapper = WNUT_MAPPERS.get(e["raw_label"])
            if mapper is None:
                continue
            new_label = mapper(e["text"])
            if new_label is not None:
                mapped_ents.append({
                    "text": e["text"],
                    "label": new_label,
                    "start": e["start"],
                    "end": e["end"]
                })
        if mapped_ents:
            remapped.append({
                "text": text,
                "entities": mapped_ents,
                "source": f"wnut17_{split_name}"
            })
    return remapped

wnut_train = remap_wnut_split(wnut["train"], "train")
wnut_val   = remap_wnut_split(wnut["validation"], "val")
wnut_test  = remap_wnut_split(wnut["test"], "test")

print(f"\nWNUT-17 remapat:")
print(f"  train: {len(wnut_train)}")
print(f"  val:   {len(wnut_val)}")
print(f"  test:  {len(wnut_test)}")

wnut_pool = wnut_train + wnut_val + wnut_test  # tot pool, e mic

with open(EXTERNAL_DIR / "wnut_pool.jsonl", "w") as f:
    for ex in wnut_pool:
        f.write(json.dumps(ex) + "\n")
print(f"Salvat: {EXTERNAL_DIR / 'wnut_pool.jsonl'}")


# Statistics

def label_distribution(examples):
    return Counter(ent["label"] for ex in examples for ent in ex["entities"])

print("CoNLL pool (train+val):")
for lbl, n in sorted(label_distribution(conll_pool).items(), key=lambda x: -x[1]):
    print(f"  {lbl:<22} {n}")

print("\nCoNLL gold (test):")
for lbl, n in sorted(label_distribution(conll_gold).items(), key=lambda x: -x[1]):
    print(f"  {lbl:<22} {n}")

print("\nWNUT pool:")
for lbl, n in sorted(label_distribution(wnut_pool).items(), key=lambda x: -x[1]):
    print(f"  {lbl:<22} {n}")


# Snorkel


from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

ABSTAIN          = -1
L_POLITICIAN     = 0
L_POLITICAL_PARTY= 1
L_POLITICAL_ORG  = 2
L_FINANCIAL_ORG  = 3
L_ECONOMIC_IND   = 4
L_POLICY         = 5
L_LEGISLATION    = 6
L_MARKET_EVENT   = 7
L_CURRENCY       = 8
L_TRADE_AGREEMENT= 9
L_GPE            = 10

CARDINALITY = 11

LABEL_NAMES = {
    L_POLITICIAN:      "POLITICIAN",
    L_POLITICAL_PARTY: "POLITICAL_PARTY",
    L_POLITICAL_ORG:   "POLITICAL_ORG",
    L_FINANCIAL_ORG:   "FINANCIAL_ORG",
    L_ECONOMIC_IND:    "ECONOMIC_INDICATOR",
    L_POLICY:          "POLICY",
    L_LEGISLATION:     "LEGISLATION",
    L_MARKET_EVENT:    "MARKET_EVENT",
    L_CURRENCY:        "CURRENCY",
    L_TRADE_AGREEMENT: "TRADE_AGREEMENT",
    L_GPE:             "GPE",
}

# Set de entitati cunoscute din pool-urile externe (folosit de LF-ul nou)
EXTERNAL_ENTITIES = defaultdict(set)
for ex in conll_pool + wnut_pool:
    for ent in ex["entities"]:
        EXTERNAL_ENTITIES[ent["label"]].add(_normalize(ent["text"]))

print(f"Entitati cunoscute din dataseturi externe:")
for lbl, ents in EXTERNAL_ENTITIES.items():
    print(f"  {lbl:<22} {len(ents)} unice")

# Label Functions (LF)


def _contains_any(text, items):
    text_low = text.lower()
    return any(item.lower() in text_low for item in items)

@labeling_function()
def lf_politician_gazetteer(x):
    return L_POLITICIAN if _contains_any(x.text, ALL_POLITICIANS) else ABSTAIN

@labeling_function()
def lf_politician_title(x):
    pattern = (r"\b(President|Vice President|Senator|Prime Minister|Chancellor|"
               r"Secretary of State|Foreign Minister|Finance Minister|Governor|"
               r"Congressman|Congresswoman|Representative)\s+[A-Z][a-z]+")
    return L_POLITICIAN if re.search(pattern, x.text) else ABSTAIN

@labeling_function()
def lf_political_party(x):
    return L_POLITICAL_PARTY if _contains_any(x.text, POLITICAL_PARTIES_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_political_org(x):
    return L_POLITICAL_ORG if _contains_any(x.text, POLITICAL_ORGS_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_financial_org(x):
    return L_FINANCIAL_ORG if _contains_any(x.text, FINANCIAL_ORGS_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_economic_indicator(x):
    return L_ECONOMIC_IND if _contains_any(x.text, ECONOMIC_INDICATORS_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_legislation(x):
    return L_LEGISLATION if _contains_any(x.text, LEGISLATION_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_currency(x):
    return L_CURRENCY if _contains_any(x.text, CURRENCIES_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_trade_agreement(x):
    return L_TRADE_AGREEMENT if _contains_any(x.text, TRADE_AGREEMENTS_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_market_event(x):
    return L_MARKET_EVENT if _contains_any(x.text, MARKET_EVENTS_GAZETTEER) else ABSTAIN

@labeling_function()
def lf_policy_pattern(x):
    pattern = (r"\b(quantitative easing|quantitative tightening|rate hike|rate cut|"
               r"fiscal stimulus|austerity measures|tax reform|monetary easing|"
               r"interest rate decision)\b")
    return L_POLICY if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN

@labeling_function()
def lf_currency_symbol(x):
    if re.search(r"[\$€£¥]\s?\d", x.text):
        return L_CURRENCY
    return ABSTAIN

@labeling_function()
def lf_gpe_country(x):
    countries = ["United States", "USA", "China", "Russia", "Germany", "France",
                 "Japan", "India", "Brazil", "United Kingdom", "UK", "Italy",
                 "Spain", "Canada", "Australia", "South Korea", "Mexico"]
    return L_GPE if _contains_any(x.text, countries) else ABSTAIN

# ─── LF NOU: match cu entitati din CoNLL/WNUT externe ────────────────────────
LABEL_TO_ID = {v: k for k, v in LABEL_NAMES.items()}

@labeling_function()
def lf_external_dataset_match(x):
    """Voteaza daca textul contine o entitate cunoscuta din CoNLL/WNUT remapate."""
    text_low = x.text.lower()
    # Verificam fiecare label, in ordinea prioritatii (entitati specifice intai)
    priority_order = ["LEGISLATION", "TRADE_AGREEMENT", "MARKET_EVENT",
                      "POLITICAL_PARTY", "FINANCIAL_ORG", "POLITICAL_ORG",
                      "POLITICIAN", "GPE"]
    for label in priority_order:
        for entity_norm in EXTERNAL_ENTITIES.get(label, set()):
            if len(entity_norm) < 4:  # evitam match-uri pe entitati prea scurte
                continue
            if entity_norm in text_low:
                return LABEL_TO_ID[label]
    return ABSTAIN

ALL_LFS = [
    lf_politician_gazetteer,
    lf_politician_title,
    lf_political_party,
    lf_political_org,
    lf_financial_org,
    lf_economic_indicator,
    lf_legislation,
    lf_currency,
    lf_trade_agreement,
    lf_market_event,
    lf_policy_pattern,
    lf_currency_symbol,
    lf_gpe_country,
    lf_external_dataset_match,
]

print(f"Total LF-uri: {len(ALL_LFS)}")

print("Se aplica Labeling Functions...")
applier = PandasLFApplier(lfs=ALL_LFS)
L_matrix = applier.apply(df=df_sentences)

print(f"Shape L_matrix: {L_matrix.shape}")

lf_analysis = LFAnalysis(L=L_matrix, lfs=ALL_LFS).lf_summary()
print("\nAnaliza LF-uri:")
print(lf_analysis.to_string())

print("Se antreneaza Label Model...")
label_model = LabelModel(cardinality=CARDINALITY, verbose=True)
label_model.fit(
    L_train=L_matrix,
    n_epochs=500,
    lr=0.001,
    log_freq=100,
    seed=42
)

pseudo_labels = label_model.predict(L=L_matrix)
pseudo_probs  = label_model.predict_proba(L=L_matrix)

df_sentences["snorkel_label"]      = pseudo_labels
df_sentences["snorkel_confidence"] = pseudo_probs.max(axis=1)
df_sentences["snorkel_label_name"] = df_sentences["snorkel_label"].map(LABEL_NAMES)

# Filtram doar exemplele cu confidence ridicat
df_weak = df_sentences[
    (df_sentences["snorkel_label"] != ABSTAIN) &
    (df_sentences["snorkel_confidence"] >= SNORKEL_CONFIDENCE_THRESHOLD)
].copy()

print(f"\nExemple cu confidence >= {SNORKEL_CONFIDENCE_THRESHOLD}: {len(df_weak)}")
print("\nDistributie pe label:")
print(df_weak["snorkel_label_name"].value_counts())



# Extract the exact spans 

LABEL_TO_GAZETTEER = {
    "POLITICIAN":         ALL_POLITICIANS,
    "POLITICAL_ORG":      POLITICAL_ORGS_GAZETTEER,
    "POLITICAL_PARTY":    POLITICAL_PARTIES_GAZETTEER,
    "FINANCIAL_ORG":      FINANCIAL_ORGS_GAZETTEER,
    "ECONOMIC_INDICATOR": ECONOMIC_INDICATORS_GAZETTEER,
    "LEGISLATION":        LEGISLATION_GAZETTEER,
    "CURRENCY":           CURRENCIES_GAZETTEER,
    "TRADE_AGREEMENT":    TRADE_AGREEMENTS_GAZETTEER,
    "MARKET_EVENT":       MARKET_EVENTS_GAZETTEER,
}

def find_entity_spans(text, label):
    spans = []
    gazetteer = LABEL_TO_GAZETTEER.get(label, [])
    for entity in gazetteer:
        pattern = r"\b" + re.escape(entity) + r"\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append({
                "text":  match.group(),
                "label": label,
                "start": match.start(),
                "end":   match.end()
            })

    if label == "POLICY":
        policy_patterns = [
            r"\b(quantitative easing|quantitative tightening|rate hike|rate cut|"
            r"fiscal stimulus|austerity measures|tax reform|monetary easing|"
            r"interest rate decision)\b"
        ]
        for p in policy_patterns:
            for m in re.finditer(p, text, re.IGNORECASE):
                spans.append({
                    "text": m.group(),
                    "label": "POLICY",
                    "start": m.start(),
                    "end": m.end()
                })

    if label == "GPE":
        countries = ["United States", "USA", "China", "Russia", "Germany", "France",
                     "Japan", "India", "Brazil", "United Kingdom", "UK", "Italy",
                     "Spain", "Canada", "Australia", "South Korea", "Mexico"]
        for c in countries:
            for m in re.finditer(r"\b" + re.escape(c) + r"\b", text):
                spans.append({
                    "text": m.group(),
                    "label": "GPE",
                    "start": m.start(),
                    "end": m.end()
                })

    # Eliminam suprapuneri (pastram primul match)
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    cleaned = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            cleaned.append(s)
            last_end = s["end"]
    return cleaned

# Construim exemplele weak supervision finale
weak_examples = []
for _, row in tqdm(df_weak.iterrows(), total=len(df_weak), desc="Extract spans"):
    label = row["snorkel_label_name"]
    spans = find_entity_spans(row["text"], label)
    if spans:
        weak_examples.append({
            "text": row["text"],
            "entities": spans,
            "source": f"weak_supervision_{row['source']}",
            "snorkel_confidence": float(row["snorkel_confidence"])
        })

print(f"Exemple weak supervision cu span-uri: {len(weak_examples)}")

with open(ANNOTATED_DIR / "weak_supervision.jsonl", "w") as f:
    for ex in weak_examples:
        f.write(json.dumps(ex) + "\n")
print(f"Salvat: {ANNOTATED_DIR / 'weak_supervision.jsonl'}")

# Generate Syntetic data for low classes examples

def find_spans_in_template(text, entities_to_find):
    """entities_to_find = list of (entity_text, label)."""
    spans = []
    for ent_text, label in entities_to_find:
        pattern = r"\b" + re.escape(ent_text) + r"\b"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            spans.append({
                "text": m.group(),
                "label": label,
                "start": m.start(),
                "end": m.end()
            })
    return spans

synthetic_data = []

# ─── MARKET_EVENT ────────────────────────────────────────────────────────────
events = ["2008 financial crisis", "Great Recession", "Black Monday",
          "dot-com bubble", "Asian financial crisis", "European debt crisis",
          "COVID-19 recession", "subprime mortgage crisis", "Lehman Brothers collapse"]
event_templates = [
    "Analysts compared the recent downturn to the {EVENT}, noting similar warning signs.",
    "The {EVENT} reshaped global financial regulation for a decade.",
    "Many economists trace current policy frameworks to the lessons of the {EVENT}.",
    "The {EVENT} caused unprecedented losses across major stock exchanges.",
    "Memories of the {EVENT} still influence central bank decisions today.",
]
for ev in events:
    for tpl in event_templates:
        text = tpl.replace("{EVENT}", ev)
        synthetic_data.append({
            "text": text,
            "entities": find_spans_in_template(text, [(ev, "MARKET_EVENT")]),
            "source": "synthetic_market_event"
        })

# ─── TRADE_AGREEMENT ─────────────────────────────────────────────────────────
trade = ["NAFTA", "USMCA", "TPP", "RCEP", "CPTPP"]
trade_templates = [
    "Negotiations over {TRADE} dragged on for several years before a final deal was reached.",
    "{TRADE} significantly altered trade flows between the participating countries.",
    "Critics argued that {TRADE} disproportionately benefited multinational corporations.",
    "The administration announced plans to renegotiate the terms of {TRADE}.",
    "{TRADE} replaced an earlier framework that many viewed as outdated.",
]
for t in trade:
    for tpl in trade_templates:
        text = tpl.replace("{TRADE}", t)
        synthetic_data.append({
            "text": text,
            "entities": find_spans_in_template(text, [(t, "TRADE_AGREEMENT")]),
            "source": "synthetic_trade"
        })

# ─── ECONOMIC_INDICATOR ──────────────────────────────────────────────────────
indicators = ["GDP", "CPI", "unemployment rate", "inflation rate", "PMI"]
indicator_templates = [
    "The latest {IND} reading came in below market expectations, prompting concern.",
    "Economists revised their forecasts for {IND} after the central bank meeting.",
    "{IND} growth slowed in the second quarter, according to the new report.",
    "Markets reacted sharply to the surprise {IND} figure released this morning.",
    "Analysts expect {IND} to stabilize over the next two quarters.",
]
for ind in indicators:
    for tpl in indicator_templates:
        text = tpl.replace("{IND}", ind)
        synthetic_data.append({
            "text": text,
            "entities": find_spans_in_template(text, [(ind, "ECONOMIC_INDICATOR")]),
            "source": "synthetic_indicator"
        })

# ─── POLICY ──────────────────────────────────────────────────────────────────
policies = ["quantitative easing", "quantitative tightening", "rate hike",
            "rate cut", "fiscal stimulus", "austerity measures", "tax reform"]
policy_templates = [
    "The committee announced a new round of {POLICY} to address market volatility.",
    "Critics warned that {POLICY} could lead to long-term inflationary pressures.",
    "{POLICY} was implemented despite vocal opposition from several lawmakers.",
    "The minutes revealed deep divisions over the appropriate timing of {POLICY}.",
    "Markets had largely priced in the expected {POLICY} before the official announcement.",
]
for p in policies:
    for tpl in policy_templates:
        text = tpl.replace("{POLICY}", p)
        synthetic_data.append({
            "text": text,
            "entities": find_spans_in_template(text, [(p, "POLICY")]),
            "source": "synthetic_policy"
        })

# ─── LEGISLATION ─────────────────────────────────────────────────────────────
laws = ["Dodd-Frank Act", "Inflation Reduction Act", "CHIPS Act",
        "Sarbanes-Oxley Act", "Affordable Care Act"]
law_templates = [
    "Provisions of the {LAW} have been the subject of intense judicial review.",
    "Lawmakers proposed amendments to the {LAW} to close loopholes.",
    "The {LAW} fundamentally restructured oversight in the affected sector.",
    "Industry groups lobbied heavily against several sections of the {LAW}.",
    "Implementation of the {LAW} required new agency rules and guidance.",
]
for law in laws:
    for tpl in law_templates:
        text = tpl.replace("{LAW}", law)
        synthetic_data.append({
            "text": text,
            "entities": find_spans_in_template(text, [(law, "LEGISLATION")]),
            "source": "synthetic_legislation"
        })

# Filtram exemplele care nu au gasit span-uri
synthetic_data = [ex for ex in synthetic_data if ex["entities"]]
random.shuffle(synthetic_data)

print(f"Generate {len(synthetic_data)} exemple sintetice.")
print("\nDistributie:")
print(Counter(ex["source"] for ex in synthetic_data))

with open(ANNOTATED_DIR / "synthetic.jsonl", "w") as f:
    for ex in synthetic_data:
        f.write(json.dumps(ex) + "\n")
print(f"\nSalvat: {ANNOTATED_DIR / 'synthetic.jsonl'}")

# Aggregation, Validation, Deduplication

def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

# Reincarcam pentru claritate
weak_examples      = load_jsonl(ANNOTATED_DIR / "weak_supervision.jsonl")
synthetic_examples = load_jsonl(ANNOTATED_DIR / "synthetic.jsonl")
conll_pool_loaded  = load_jsonl(EXTERNAL_DIR / "conll_pool.jsonl")
conll_gold_loaded  = load_jsonl(EXTERNAL_DIR / "conll_gold.jsonl")
wnut_pool_loaded   = load_jsonl(EXTERNAL_DIR / "wnut_pool.jsonl")

print("=== Surse incarcate ===")
print(f"  Weak supervision (Snorkel) : {len(weak_examples):>6}")
print(f"  Synthetic                  : {len(synthetic_examples):>6}")
print(f"  CoNLL pool                 : {len(conll_pool_loaded):>6}")
print(f"  CoNLL gold (test)          : {len(conll_gold_loaded):>6}")
print(f"  WNUT pool                  : {len(wnut_pool_loaded):>6}")
total = (len(weak_examples) + len(synthetic_examples) + len(conll_pool_loaded)
         + len(conll_gold_loaded) + len(wnut_pool_loaded))
print(f"  TOTAL                      : {total:>6}")

def validate_example(ex):
    text = ex.get("text", "")
    if not text or len(text) < MIN_SENTENCE_LEN:
        return False
    entities = ex.get("entities", [])
    if not entities:
        return False
    for ent in entities:
        start, end = ent.get("start", 0), ent.get("end", 0)
        if start < 0 or end > len(text) or start >= end:
            return False
        if ent.get("text") and ent["text"] != text[start:end]:
            return False
        if ent.get("label") and ent["label"] not in NER_LABELS:
            return False
    return True

def deduplicate(examples):
    seen, unique = set(), []
    for ex in examples:
        if ex["text"] not in seen:
            seen.add(ex["text"])
            unique.append(ex)
    return unique

# Combinam in ordinea: gold > extern pool > sintetic > weak
all_examples = (conll_gold_loaded
                + conll_pool_loaded
                + wnut_pool_loaded
                + synthetic_examples
                + weak_examples)

print(f"Total inainte validare: {len(all_examples)}")

valid_examples = [ex for ex in all_examples if validate_example(ex)]
print(f"Dupa validare: {len(valid_examples)}")

unique_examples = deduplicate(valid_examples)
print(f"Dupa deduplicare: {len(unique_examples)}")

# Distributie pe source
print("\nDistributie pe sursa:")
src_counts = Counter(ex.get("source", "unknown") for ex in unique_examples)
for src, n in sorted(src_counts.items(), key=lambda x: -x[1]):
    print(f"  {src:<35} {n}")


# Split Train / Dev / Test


from sklearn.model_selection import train_test_split

# Identificam exemplele gold (din CoNLL test)
gold_texts = set(ex["text"] for ex in conll_gold_loaded)
test_gold = [ex for ex in unique_examples if ex["text"] in gold_texts]
rest      = [ex for ex in unique_examples if ex["text"] not in gold_texts]

print(f"Gold standard (CoNLL test): {len(test_gold)}")
print(f"Rest pentru split:          {len(rest)}")

# Split rest in train_full + test_extra
train_full, test_extra = train_test_split(rest, test_size=0.15, random_state=42)

# Test set = gold + test_extra
test_set = test_gold + test_extra
random.shuffle(test_set)

# Split train_full in train + dev
train_set, dev_set = train_test_split(train_full, test_size=0.176, random_state=42)
# 0.176 din 85% ≈ 15% din total

def save_jsonl(examples, path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Salvat: {path} ({len(examples)} exemple)")

save_jsonl(train_set, SPLITS_DIR / "train.jsonl")
save_jsonl(dev_set,   SPLITS_DIR / "dev.jsonl")
save_jsonl(test_set,  SPLITS_DIR / "test.jsonl")

total = len(train_set) + len(dev_set) + len(test_set)
print(f"\n=== SPLIT FINAL ===")
print(f"  train : {len(train_set):>5} ({100*len(train_set)/total:.1f}%)")
print(f"  dev   : {len(dev_set):>5} ({100*len(dev_set)/total:.1f}%)")
print(f"  test  : {len(test_set):>5} ({100*len(test_set)/total:.1f}%)  (din care gold: {len(test_gold)})")
print(f"  TOTAL : {total}")


# Final statistics

train_texts = set(ex["text"] for ex in train_set)
dev_texts   = set(ex["text"] for ex in dev_set)
test_texts  = set(ex["text"] for ex in test_set)

print("=== Verificare overlap (trebuie sa fie 0) ===")
print(f"  train ∩ dev  : {len(train_texts & dev_texts)}")
print(f"  train ∩ test : {len(train_texts & test_texts)}")
print(f"  dev   ∩ test : {len(dev_texts & test_texts)}")

print("\n=== Distributie entitati per split ===")
for split_name, split_data in [("train", train_set), ("dev", dev_set), ("test", test_set)]:
    counts = Counter(ent["label"] for ex in split_data for ent in ex["entities"])
    total_ents = sum(counts.values())
    print(f"\n  [{split_name}] {total_ents} entitati totale, {len(split_data)} propozitii:")
    for label in NER_LABELS:
        n = counts.get(label, 0)
        marker = "OK" if n >= 50 else ("LIMITAT" if n >= 10 else "INSUFICIENT")
        print(f"    {label:<22} {n:>5}  [{marker}]")

# ─── Salvare statistici ──────────────────────────────────────────────────────
splits = {
    "train": train_set,
    "dev":   dev_set,
    "test":  test_set,
}

stats = {}
for split_name, data in splits.items():
    label_counts  = Counter(ent["label"] for ex in data for ent in ex["entities"])
    source_counts = Counter(ex.get("source", "unknown") for ex in data)
    stats[split_name] = {
        "n_examples":          len(data),
        "n_entities":          sum(len(ex["entities"]) for ex in data),
        "label_distribution":  dict(label_counts),
        "source_distribution": dict(source_counts),
    }

stats["meta"] = {
    "labels": NER_LABELS,
    "snorkel_confidence_threshold": SNORKEL_CONFIDENCE_THRESHOLD,
    "n_total_examples": sum(stats[s]["n_examples"] for s in ["train", "dev", "test"]),
}

with open(BASE_DIR / "dataset_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
print(f"\nStatistici salvate in: {BASE_DIR / 'dataset_stats.json'}")





