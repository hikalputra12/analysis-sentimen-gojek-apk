"""Inference Module for Sentiment Prediction on Raw Indonesian Text."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np

from src.config import INT_TO_LABEL
from src.feature_engineering import FeatureExtractor
from src.model import SentimentModelTrainer
from src.preprocessor import IndonesianTextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """Production inference engine combining preprocessor, vectorizer, and trained model."""

    def __init__(
        self,
        preprocessor: IndonesianTextPreprocessor,
        feature_extractor: FeatureExtractor,
        model_trainer: SentimentModelTrainer
    ) -> None:
        """Initialize the predictor with pre-trained pipeline components."""
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predict sentiment polarity and confidence for raw Indonesian text(s).

        Args:
            texts: Single string or list of review texts.

        Returns:
            List of dictionaries containing raw text, cleaned text, predicted label,
            and confidence probability score.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # 1. Preprocessing
        cleaned_texts = [self.preprocessor.preprocess_pipeline(t) for t in texts]

        # 2. Feature Extraction
        features = self.feature_extractor.transform(cleaned_texts)

        # 3. Model Prediction & Probabilities
        predictions = self.model_trainer.model.predict(features)
        
        has_proba = hasattr(self.model_trainer.model, "predict_proba")
        probabilities = self.model_trainer.model.predict_proba(features) if has_proba else None

        results = []
        for idx, (raw, clean, pred) in enumerate(zip(texts, cleaned_texts, predictions)):
            label = INT_TO_LABEL.get(pred, "unknown")
            conf = float(np.max(probabilities[idx])) if probabilities is not None else 1.0
            results.append({
                "text": raw,
                "cleaned_text": clean,
                "sentiment": label,
                "confidence": round(conf, 4)
            })

        return results
