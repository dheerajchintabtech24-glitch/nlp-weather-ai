from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
RESOURCES = PACKAGE_DIR / "resources"

DATA_DIR = ROOT / "data"
RAW_MASSIVE = DATA_DIR / "raw" / "1.1" / "data"
PROCESSED = DATA_DIR / "processed"
MASSIVE_DIR = PROCESSED / "massive"
DIALECT_DIR = PROCESSED / "dialect"
COLLECTED_DIALECT_DIR = DATA_DIR / "dialect" / "collected"
FOLLOWUP_DIR = PROCESSED / "followup"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache"

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"

LOCALES = {"en": "en-US", "hi": "hi-IN", "ta": "ta-IN"}
LANGUAGES = list(LOCALES)

DIALECTS = {
    "en": ["standard"],
    "hi": ["standard", "bhojpuri", "haryanvi", "bambaiya"],
    "ta": ["standard", "chennai", "kongu", "madurai", "nellai"],
}

WEATHER_INTENT = "weather_query"
WEATHER_SLOTS = ("place_name", "date", "time", "timeofday", "weather_descriptor")

DEFAULT_LOCATION = "Bengaluru"
FORECAST_DAYS = 16


def dialect_label(lang: str, dialect: str) -> str:
    return f"{lang}:{dialect}"


def all_dialect_labels():
    return [dialect_label(l, d) for l in LANGUAGES for d in DIALECTS[l]]
