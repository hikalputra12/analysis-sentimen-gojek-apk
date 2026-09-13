"""Configuration module for Gojek Sentiment Analysis pipeline.

Defines default hyperparameters, paths, random seeds, and slang/stopword dictionaries.
"""

from pathlib import Path
from typing import Dict, Set

# Base Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"

# Default Data File Paths
DEFAULT_RAW_DATA: Path = BASE_DIR / "ulasan_aplikasi.csv"
CLEANED_DATA_PATH: Path = BASE_DIR / "data_cleaned.csv"

# Reproducibility
RANDOM_SEED: int = 42

# Scraping Configuration
APP_PACKAGE_ID: str = "com.gojek.app"
DEFAULT_SCRAPE_COUNT: int = 15000
SCRAPE_LANG: str = "id"
SCRAPE_COUNTRY: str = "id"

# Label Mapping
LABEL_TO_INT: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

INT_TO_LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Negation words to preserve during stopword removal (crucial for sentiment analysis)
NEGATION_WORDS: Set[str] = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang", 
    "tak", "ndak", "ga", "gak", "nggak", "gk", "gda", "kagak"
}

# Expanded Indonesian Slang & Colloquialisms Dictionary
SLANG_DICTIONARY: Dict[str, str] = {
    # Pronouns & Conjunctions
    "sy": "saya", "gw": "saya", "gue": "saya", "aku": "saya", "ogut": "saya",
    "lu": "kamu", "lo": "kamu", "ente": "kamu", "sampeyan": "kamu", "kmu": "kamu",
    "jg": "juga", "jga": "juga",
    "dgn": "dengan", "dg": "dengan",
    "udh": "sudah", "sdh": "sudah", "udah": "sudah",
    "blm": "belum", "blom": "belum",
    "tp": "tapi", "tpi": "tapi",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau", "klu": "kalau",
    "utk": "untuk", "untk": "untuk", "buat": "untuk",
    "krn": "karena", "karna": "karena", "coz": "karena", "sbab": "karena",
    "dr": "dari", "dri": "dari",
    "yg": "yang", "yng": "yang",
    "aja": "saja", "ae": "saja",
    "bgt": "banget", "bngt": "banget", "bener": "benar", "bnget": "banget",
    "bgtu": "begitu", "gitu": "begitu", "gini": "begini",
    "bisa": "bisa", "bs": "bisa", "bsa": "bisa",
    "jd": "jadi", "jdi": "jadi",
    "trs": "terus", "trus": "terus",
    "sampe": "sampai", "ampe": "sampai",

    # Negation standardizations
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "g": "tidak", "tdk": "tidak", "tak": "tidak", "kagak": "tidak", "ndak": "tidak",

    # App & Service Specific Slang
    "app": "aplikasi", "apk": "aplikasi", "apps": "aplikasi",
    "lemot": "lambat", "lola": "lambat", "lelet": "lambat", "ngelag": "macet", "hang": "macet",
    "bapuk": "jelek", "ancur": "rusak", "rusak": "rusak", "cacat": "rusak", "zonk": "kecewa",
    "driver": "pengemudi", "ojol": "pengemudi", "abang": "pengemudi",
    "ongkir": "ongkos kirim", "ongkirnya": "ongkos kirim",
    "diskon": "potongan", "promo": "promosi", "voucher": "kupon",
    "topup": "isi saldo", "saldo": "saldo", "paylater": "bayar nanti",
    "error": "kesalahan", "eror": "kesalahan", "bug": "kesalahan", "crash": "rusak",
    "mantap": "bagus", "mantep": "bagus", "top": "bagus", "keren": "bagus",
    "oke": "baik", "ok": "baik", "okey": "baik", "sip": "baik", "siap": "baik",
    "makasih": "terima kasih", "tks": "terima kasih", "thx": "terima kasih", "ty": "terima kasih",
    "kecewa": "kecewa", "parah": "buruk", "parahhh": "buruk", "nyesel": "menyesal",
    "mahal": "mahal", "kemahalan": "mahal", "boros": "boros"
}
