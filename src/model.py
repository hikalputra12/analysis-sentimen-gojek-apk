"""Model training and evaluation module for sentiment classification."""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.svm import SVC

from src.config import RANDOM_SEED, INT_TO_LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SentimentModelTrainer:
    """Trains, optimizes, evaluates, and persists sentiment classification models."""

    def __init__(
        self,
        model_type: str = "svm",
        use_smote: bool = True,
        random_state: int = RANDOM_SEED
    ) -> None:
        """Initialize trainer.

        Args:
            model_type: 'svm' or 'random_forest'.
            use_smote: Whether to balance classes using SMOTE.
            random_state: Random seed for reproducibility.
        """
        self.model_type = model_type.lower()
        self.use_smote = use_smote
        self.random_state = random_state
        self.model = None

    def _init_model(self) -> Any:
        """Instantiate classifier based on model type."""
        if self.model_type == "svm":
            base_svc = SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=self.random_state
            )
            # Use CalibratedClassifierCV to support probabilities cleanly without future warnings
            return CalibratedClassifierCV(estimator=base_svc, cv=3)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Any:
        """Train classifier with optional SMOTE resampling.

        Args:
            X_train: Feature matrix.
            y_train: Target label array.

        Returns:
            Trained model instance.
        """
        self.model = self._init_model()

        if self.use_smote:
            logger.info("Applying SMOTE oversampling on training set...")
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logger.info("Resampled training distribution: %s", np.bincount(y_resampled))
        else:
            X_resampled, y_resampled = X_train, y_train

        logger.info("Training %s model...", self.model_type.upper())
        self.model.fit(X_resampled, y_resampled)
        logger.info("Model training completed successfully.")
        return self.model

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate model performance on test dataset.

        Args:
            X_test: Test features.
            y_test: True test labels.

        Returns:
            Dictionary containing accuracy, f1_score, confusion matrix, and report.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        target_names = [INT_TO_LABEL[i] for i in sorted(INT_TO_LABEL.keys())]

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)

        logger.info("\n--- EVALUATION REPORT (%s) ---\n%s", self.model_type.upper(), report)
        logger.info("Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f", acc, f1_macro, f1_weighted)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report
        }

    def save(self, filepath: Path) -> None:
        """Save trained model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", filepath)

    @classmethod
    def load(cls, filepath: Path, model_type: str = "svm") -> "SentimentModelTrainer":
        """Load trained model from disk."""
        instance = cls(model_type=model_type)
        with open(filepath, "rb") as f:
            instance.model = pickle.load(f)
        logger.info("Loaded model from %s", filepath)
        return instance
