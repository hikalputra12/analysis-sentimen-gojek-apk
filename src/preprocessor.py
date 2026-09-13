"""Text Preprocessing Module for Indonesian Sentiment Analysis.

Handles cleaning, slang normalization, negation-aware stopword removal,
and tokenization following PEP 8 and production NLP best practices.
"""

import logging
import re
import string
from typing import List, Optional, Set, Dict

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None
    stopwords = None
    word_tokenize = None

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError:
    StemmerFactory = None

from src.config import SLANG_DICTIONARY, NEGATION_WORDS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fallback default Indonesian stopwords
DEFAULT_ID_STOPWORDS = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia", "dua",
    "ia", "seperti", "jika", "jikalau", "sehingga", "kembali", "dan", "ini", "karena",
    "oleh", "saat", "olehnya", "tentang", "dengan", "adalah", "suatu", "juga", "bisa",
    "ada", "mereka", "sudah", "saya", "kamu", "kita", "kami", "saja", "dari", "dalam",
    "akan", "bisa", "itu", "atau", "hanya", "setelah", "bagi", "sampai", "tersebut"
}


def ensure_nltk_resources() -> None:
    """Download required NLTK resource packages if nltk is installed."""
    if nltk is None:
        return
    required_packages = ["punkt", "punkt_tab", "stopwords"]
    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}" if "punkt" in package else f"corpora/{package}")
        except (LookupError, IndexError, Exception):
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass


class IndonesianTextPreprocessor:
    """Production-grade text preprocessor tailored for Indonesian mobile app reviews."""

    def __init__(
        self,
        slang_dict: Optional[Dict[str, str]] = None,
        preserve_negations: bool = True,
        use_stemming: bool = False
    ) -> None:
        """Initialize preprocessor with configurable components.

        Args:
            slang_dict: Custom dictionary for informal/slang words mapping.
            preserve_negations: Whether to protect negation words from stopword pruning.
            use_stemming: Whether to apply Sastrawi stemming (expensive, default False).
        """
        ensure_nltk_resources()
        self.slang_dict = slang_dict if slang_dict is not None else SLANG_DICTIONARY
        self.preserve_negations = preserve_negations
        self.use_stemming = use_stemming

        # Build Stopword List
        self.stop_words: Set[str] = set()
        if stopwords is not None:
            try:
                base_stopwords = set(stopwords.words("indonesian"))
                english_stopwords = set(stopwords.words("english"))
                self.stop_words = base_stopwords.union(english_stopwords)
            except Exception:
                self.stop_words = set(DEFAULT_ID_STOPWORDS)
        else:
            self.stop_words = set(DEFAULT_ID_STOPWORDS)

        # Add informal noise words
        noise_words = {
            "iya", "yaa", "nya", "na", "sih", "ku", "di", "ya", "gaa",
            "loh", "kah", "woi", "woii", "woy", "deh", "dong", "nih", "tuh"
        }
        self.stop_words.update(noise_words)

        # Remove negation words so sentiment context is preserved
        if self.preserve_negations:
            self.stop_words = self.stop_words.difference(NEGATION_WORDS)

        # Lazy load Sastrawi Stemmer
        self._stemmer = None
        if self.use_stemming and StemmerFactory is not None:
            self._stemmer = StemmerFactory().create_stemmer()

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip URLs, mentions, hashtags, digits, emojis, and punctuation from text.

        Args:
            text: Raw input string.

        Returns:
            Sanitized text string.
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove mentions, hashtags, RT tags, and URLs
        text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
        text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
        text = re.sub(r"\bRT\b", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove HTML entities & tags
        text = re.sub(r"<.*?>", " ", text)

        # Remove numbers and special characters/emojis
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Normalize repeating characters (e.g. 'baguuuus' -> 'bagus')
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_slang(self, text: str) -> str:
        """Replace slang and informal words with formal/standard equivalents.

        Args:
            text: Cleaned text string.

        Returns:
            Normalized text string.
        """
        if not text:
            return ""

        tokens = text.split()
        normalized = [self.slang_dict.get(token, token) for token in tokens]
        return " ".join(normalized)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of word tokens.

        Args:
            text: Input string.

        Returns:
            List of string tokens.
        """
        if not text:
            return []
        if word_tokenize is not None:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        # Fallback whitespace / word regex tokenization
        return re.findall(r"\b\w+\b", text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stopwords while retaining sentiment-critical negations.

        Args:
            tokens: List of word tokens.

        Returns:
            Filtered list of word tokens.
        """
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply morphological stemming using Sastrawi.

        Args:
            tokens: List of tokens.

        Returns:
            List of stemmed tokens.
        """
        if self._stemmer is None:
            return tokens
        return [self._stemmer.stem(token) for token in tokens]

    def preprocess_pipeline(self, raw_text: str) -> str:
        """Full end-to-end preprocessing pipeline for a single review string.

        Args:
            raw_text: Raw review text from Google Play Store.

        Returns:
            Cleaned and normalized sentence string ready for feature extraction.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            return ""

        # Step 1: Clean special characters & lowercase
        cleaned = self.clean_text(raw_text)

        # Step 2: Normalize Slang / Typo
        normalized = self.normalize_slang(cleaned)

        # Step 3: Tokenize
        tokens = self.tokenize(normalized)

        # Step 4: Stopword removal (preserving negation)
        filtered_tokens = self.remove_stopwords(tokens)

        # Step 5: Optional Stemming
        if self.use_stemming:
            filtered_tokens = self.stem_tokens(filtered_tokens)

        return " ".join(filtered_tokens)
