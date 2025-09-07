from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_TICKER = os.getenv("TICKER", "PETR4.SA")
YEARS = int(os.getenv("YEARS", "5"))
