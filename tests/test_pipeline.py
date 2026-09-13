"""Automated Unit & Integration Test Suite for Gojek Sentiment Analysis.

Tests all modules: preprocessor, labeler, feature extractor, model trainer, and inference.
"""

import unittest
from pathlib import Path
import numpy as np

from src.config import SLANG_DICTIONARY, NEGATION_WORDS, LABEL_TO_INT
from src.preprocessor import IndonesianTextPreprocessor
from src.labeler import label_sentiment_indonesian
from src.feature_engineering import FeatureExtractor
from src.model import SentimentModelTrainer
from src.inference import SentimentPredictor


class TestNLPPreprocessor(unittest.TestCase):
    """Unit tests for Indonesian text preprocessor."""

    def setUp(self):
        self.preprocessor = IndonesianTextPreprocessor(preserve_negations=True)

    def test_clean_text_removes_urls_mentions_numbers(self):
        raw = "Wah @gojekindonesia bagus banget http://gojek.com promo 1000 #promo"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("1000", cleaned)
        self.assertNotIn("#", cleaned)

    def test_slang_normalization(self):
        raw = "apk lemot bgt ga bsa topup saldo zonk"
        normalized = self.preprocessor.normalize_slang(raw)
        self.assertIn("aplikasi", normalized)
        self.assertIn("lambat", normalized)
        self.assertIn("banget", normalized)
        self.assertIn("tidak", normalized)

    def test_negation_preservation_in_stopwords(self):
        raw = "aplikasi ini tidak membantu dan sangat lambat"
        processed = self.preprocessor.preprocess_pipeline(raw)
        self.assertIn("tidak", processed)
        self.assertIn("membantu", processed)
        self.assertIn("lambat", processed)


class TestSentimentLabeler(unittest.TestCase):
    """Unit tests for sentiment annotation and negation handling."""

    def test_positive_sentiment(self):
        score, label = label_sentiment_indonesian("aplikasi sangat bagus cepat dan membantu")
        self.assertEqual(label, "positive")
        self.assertGreater(score, 0)

    def test_negative_sentiment(self):
        score, label = label_sentiment_indonesian("aplikasi jelek sering error dan mengecewakan")
        self.assertEqual(label, "negative")
        self.assertLess(score, 0)

    def test_negated_positive_becomes_negative(self):
        score, label = label_sentiment_indonesian("aplikasi ini tidak bagus sama sekali")
        self.assertEqual(label, "negative")


class TestFeatureEngineering(unittest.TestCase):
    """Unit tests for feature extraction."""

    def setUp(self):
        self.corpus = [
            "aplikasi gojek sangat bagus dan cepat",
            "aplikasi jelek sering error dan mengecewakan",
            "fitur pembayaran gopay mudah digunakan",
            "driver ramah dan pengantaran tepat waktu"
        ]

    def test_tfidf_fit_transform(self):
        extractor = FeatureExtractor(method="tfidf", max_features=100)
        X = extractor.fit_transform(self.corpus)
        self.assertEqual(X.shape[0], len(self.corpus))
        self.assertGreater(X.shape[1], 0)

        # Test transformation of new sample
        X_new = extractor.transform(["aplikasi bagus"])
        self.assertEqual(X_new.shape[0], 1)
        self.assertEqual(X_new.shape[1], X.shape[1])


class TestEndToEndInference(unittest.TestCase):
    """Integration test for full preprocessing -> extraction -> inference pipeline."""

    def test_pipeline_integration(self):
        corpus = [
            "aplikasi gojek sangat bagus dan cepat",
            "fitur pembayaran gopay mudah digunakan",
            "driver ramah dan pengantaran tepat waktu",
            "biasa saja tidak ada yang istimewa",
            "fitur standar tidak buruk dan tidak luar biasa",
            "cukup oke untuk penggunaan sehari hari",
            "aplikasi jelek sering error dan mengecewakan",
            "pelayanan buruk lambat dan tidak ramah",
            "sistem sering gangguan saldo terpotong"
        ]
        labels = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0])

        preprocessor = IndonesianTextPreprocessor(preserve_negations=True)
        cleaned_corpus = [preprocessor.preprocess_pipeline(text) for text in corpus]

        extractor = FeatureExtractor(method="tfidf", max_features=50)
        X = extractor.fit_transform(cleaned_corpus)

        trainer = SentimentModelTrainer(model_type="svm", use_smote=False)
        trainer.train(X, labels)

        predictor = SentimentPredictor(preprocessor, extractor, trainer)
        results = predictor.predict(["aplikasi sangat bagus", "aplikasi jelek sekali"])

        self.assertEqual(len(results), 2)
        self.assertIn(results[0]["sentiment"], ["positive", "neutral", "negative"])
        self.assertIn("confidence", results[0])


if __name__ == "__main__":
    unittest.main()
