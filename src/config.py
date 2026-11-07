from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
DATA_PATH = ROOT / "data" / "train.csv"
TARGET = "default_payment_next_month"
SENSITIVE = ["sex", "education", "marriage", "age"]
