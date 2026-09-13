"""Rule-based and Lexicon Sentiment Annotator with Contextual Negation Handling.

Provides robust sentiment labeling for Indonesian mobile app reviews.
"""

from typing import Tuple, Set, Dict

# Positive sentiment seed words
POSITIVE_WORDS: Dict[str, int] = {
    "bagus": 2, "membantu": 2, "cepat": 2, "puas": 2, "mantap": 2, "keren": 2,
    "murah": 2, "mudah": 2, "ramah": 2, "terbaik": 3, "bintang": 1, "hemat": 2,
    "top": 2, "suka": 2, "senang": 2, "hebat": 2, "nyaman": 2, "rapi": 1,
    "aman": 2, "bermanfaat": 2, "jelas": 1, "responsif": 2, "mantul": 2,
    "memuaskan": 3, "terima kasih": 2, "makasih": 2, "terjangkau": 2, "luar biasa": 3,
    "praktis": 2, "lancar": 2, "rekomended": 2, "recommended": 2, "oke": 1, "baik": 2
}

# Negative sentiment seed words
NEGATIVE_WORDS: Dict[str, int] = {
    "jelek": -2, "lambat": -2, "lelet": -2, "kecewa": -3, "lemot": -2, "error": -2,
    "eror": -2, "bug": -2, "gagal": -2, "macet": -2, "rugi": -3, "buruk": -2,
    "ribet": -2, "payah": -2, "mahal": -2, "penipu": -3, "bohong": -3, "cancel": -2,
    "batal": -1, "hilang": -2, "hang": -2, "crash": -2, "terpotong": -2, "parah": -3,
    "nyesel": -3, "menyesal": -3, "susah": -2, "sulit": -2, "berat": -1, "buang": -2,
    "kapok": -3, "bobrok": -3, "hancur": -3, "emosi": -2, "salah": -1, "terjelek": -3,
    "males": -2, "kurang": -1, "rusak": -2, "potong": -1, "kemahalan": -2, "boros": -2
}

NEGATION_SET: Set[str] = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang",
    "tak", "ndak", "ga", "gak", "nggak", "gk", "gda", "kagak"
}


def label_sentiment_indonesian(text: str) -> Tuple[int, str]:
    """Calculate sentiment score and polarity with negation context window.

    Args:
        text: Normalized review string.

    Returns:
        Tuple of (score: int, polarity: 'positive' | 'negative' | 'neutral').
    """
    tokens = text.lower().split()
    total_score = 0
    negation_active = False

    for i, token in enumerate(tokens):
        if token in NEGATION_SET:
            negation_active = True
            continue

        token_score = 0
        if token in POSITIVE_WORDS:
            token_score = POSITIVE_WORDS[token]
            if negation_active:
                token_score = -token_score  # "tidak bagus" -> negative
        elif token in NEGATIVE_WORDS:
            token_score = NEGATIVE_WORDS[token]
            if negation_active:
                token_score = abs(token_score)  # "tidak lemot" -> positive

        total_score += token_score
        negation_active = False  # Reset negation after applying to the next word

    if total_score > 0:
        return total_score, "positive"
    elif total_score < 0:
        return total_score, "negative"
    else:
        return 0, "neutral"
