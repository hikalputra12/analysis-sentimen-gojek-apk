"""Feature Engineering Module for Text Vectorization.

Supports TF-IDF with unigram/bigram features and Word2Vec dense sentence embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Manages TF-IDF and Word2Vec vectorization pipelines with serialization."""

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 2500,
        ngram_range: Tuple[int, int] = (1, 2),
        w2v_vector_size: int = 100,
        w2v_window: int = 5,
        w2v_min_count: int = 1
    ) -> None:
        """Initialize Feature Extractor.

        Args:
            method: 'tfidf' or 'word2vec'.
            max_features: Maximum vocabulary size for TF-IDF.
            ngram_range: N-gram range (default unigram + bigram (1, 2)).
            w2v_vector_size: Embedding dimensionality for Word2Vec.
            w2v_window: Context window for Word2Vec.
            w2v_min_count: Minimum token frequency for Word2Vec.
        """
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.w2v_vector_size = w2v_vector_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count

        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.w2v_model: Optional[Word2Vec] = None

        if self.method == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True
            )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit feature extractor on training texts and transform into numeric matrix.

        Args:
            texts: List of preprocessed text documents.

        Returns:
            2D numpy array or sparse matrix of features.
        """
        if self.method == "tfidf":
            logger.info("Fitting TF-IDF vectorizer (max_features=%d, ngrams=%s)...",
                        self.max_features, self.ngram_range)
            matrix = self.tfidf_vectorizer.fit_transform(texts)
            return matrix
        elif self.method == "word2vec":
            logger.info("Training Word2Vec model on %d sentences...", len(texts))
            tokenized_sentences = [text.split() for text in texts]
            self.w2v_model = Word2Vec(
                sentences=tokenized_sentences,
                vector_size=self.w2v_vector_size,
                window=self.w2v_window,
                min_count=self.w2v_min_count,
                workers=4
            )
            return self._transform_w2v(texts)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform unseen text data using fitted vectorizer.

        Args:
            texts: List of preprocessed text documents.

        Returns:
            Feature matrix for inference.
        """
        if self.method == "tfidf":
            if self.tfidf_vectorizer is None:
                raise RuntimeError("TF-IDF Vectorizer has not been fitted.")
            return self.tfidf_vectorizer.transform(texts)
        elif self.method == "word2vec":
            if self.w2v_model is None:
                raise RuntimeError("Word2Vec model has not been trained.")
            return self._transform_w2v(texts)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")

    def _sentence_to_w2v_vector(self, sentence: str) -> np.ndarray:
        """Compute average vector representation for a sentence."""
        words = sentence.split()
        vectors = [self.w2v_model.wv[w] for w in words if w in self.w2v_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.w2v_vector_size)

    def _transform_w2v(self, texts: List[str]) -> np.ndarray:
        """Transform batch of texts into Word2Vec sentence embeddings."""
        return np.array([self._sentence_to_w2v_vector(text) for text in texts])

    def save(self, filepath: Path) -> None:
        """Persist fitted vectorizer/model to disk.

        Args:
            filepath: Destination path (.pkl file).
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved FeatureExtractor to %s", filepath)

    @classmethod
    def load(cls, filepath: Path) -> "FeatureExtractor":
        """Load persisted FeatureExtractor from disk."""
        with open(filepath, "rb") as f:
            extractor = pickle.load(f)
        logger.info("Loaded FeatureExtractor from %s", filepath)
        return extractor
