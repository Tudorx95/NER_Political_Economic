from gliner import GLiNER
import torch

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device:          {torch.cuda.get_device_name(0)}")
    print(f"Memory total:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = torch.device("cuda:0")
else:
    print("ATENTIE: rulezi pe CPU.")
    DEVICE = torch.device("cpu")

NER_LABELS = [
    "POLITICIAN", "POLITICAL_PARTY", "POLITICAL_ORG", "FINANCIAL_ORG",
    "ECONOMIC_INDICATOR", "POLICY", "LEGISLATION", "MARKET_EVENT",
    "CURRENCY", "TRADE_AGREEMENT", "GPE",
]
EVAL_THRESHOLD = 0.5
HF_DATASET_REPO = "Tudorx95/NER_Political_Economic"
HF_MODEL_REPO = "Tudorx95/NER_Economic_Political"

print(f"Se descarca {HF_MODEL_REPO} de pe HuggingFace...")
hf_model = GLiNER.from_pretrained(HF_MODEL_REPO)
hf_model.to(DEVICE)

text = "The IMF warned that rising US interest rates could trigger a global recession."
entities = hf_model.predict_entities(text, NER_LABELS, threshold=EVAL_THRESHOLD)

print(f"\nText: {text}\n")
print("Entitati detectate:")
for e in entities:
    print(f"  {e['text']:<30} -> {e['label']:<20} (score: {e['score']:.3f})")

print("\n=== TOTUL OK ===")
print(f"Dataset:  https://huggingface.co/datasets/{HF_DATASET_REPO}")
print(f"Model:    https://huggingface.co/{HF_MODEL_REPO}")
