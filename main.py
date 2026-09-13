"""Main pipeline runner CLI for Gojek Sentiment Analysis.

Allows end-to-end execution of scraping, preprocessing, training, and inference.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEFAULT_RAW_DATA,
    RANDOM_SEED,
    LABEL_TO_INT,
    MODELS_DIR,
    APP_PACKAGE_ID
)
from src.feature_engineering import FeatureExtractor
from src.inference import SentimentPredictor
from src.labeler import label_sentiment_indonesian
from src.model import SentimentModelTrainer
from src.preprocessor import IndonesianTextPreprocessor
from src.scraper import scrape_playstore_reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_pipeline(
    model_type: str = "svm",
    feature_method: str = "tfidf",
    test_size: float = 0.2,
    use_smote: bool = True,
    sample_size: int = 5000
) -> None:
    """Execute the full data loading, preprocessing, training, and evaluation pipeline."""
    logger.info("=== Starting Gojek Sentiment Analysis Pipeline ===")

    if not DEFAULT_RAW_DATA.exists():
        logger.warning("Raw data not found at %s. Scraping initial dataset...", DEFAULT_RAW_DATA)
        scrape_playstore_reviews(count=sample_size)

    # 1. Load Data
    logger.info("Loading dataset from %s...", DEFAULT_RAW_DATA)
    df = pd.read_csv(DEFAULT_RAW_DATA)
    df = df.dropna(subset=["Review"]).drop_duplicates(subset=["Review"])

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info("Using %d reviews for training/evaluation.", len(df))

    # 2. Preprocessing
    preprocessor = IndonesianTextPreprocessor(preserve_negations=True, use_stemming=False)
    logger.info("Preprocessing reviews with negation-aware filter & slang normalizer...")
    df["clean_text"] = df["Review"].apply(preprocessor.preprocess_pipeline)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    # 3. Sentiment Labeling (if no pre-existing polarity column)
    if "polarity" not in df.columns or df["polarity"].isna().all():
        logger.info("Annotating sentiment labels using contextual negation-aware lexicon engine...")
        label_results = df["clean_text"].apply(label_sentiment_indonesian)
        df["polarity"] = [res[1] for res in label_results]

    logger.info("Sentiment class distribution:\n%s", df["polarity"].value_counts())
    df["label"] = df["polarity"].map(LABEL_TO_INT).fillna(1).astype(int)

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["label"].values,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"].values if len(df["label"].unique()) > 1 else None
    )

    # 5. Feature Extraction
    extractor = FeatureExtractor(method=feature_method)
    X_train_vec = extractor.fit_transform(X_train)
    X_test_vec = extractor.transform(X_test)

    # 6. Model Training & Evaluation
    trainer = SentimentModelTrainer(model_type=model_type, use_smote=use_smote)
    trainer.train(X_train_vec, y_train)
    metrics = trainer.evaluate(X_test_vec, y_test)

    # 7. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    extractor.save(MODELS_DIR / f"{feature_method}_extractor.pkl")
    trainer.save(MODELS_DIR / f"{model_type}_model.pkl")

    # 8. Demo Inferencing
    demo_texts = [
        "Aplikasi Gojek sangat membantu saya memesan makanan dengan cepat dan praktis!",
        "Biasa saja, tidak ada yang istimewa dari update terbaru.",
        "Kecewa banget, aplikasinya sering error, topup saldo gagal dan driver tidak datang.",
        "Aplikasi ini sangat jelek, maps tidak akurat sama sekali.",
        "Pelayanan driver ramah dan pengantaran tepat waktu, terima kasih Gojek!"
    ]

    predictor = SentimentPredictor(preprocessor, extractor, trainer)
    demo_results = predictor.predict(demo_texts)

    print("\n" + "=" * 65)
    print("DEMO INFERENCE RESULTS")
    print("=" * 65)
    for res in demo_results:
        print(f"Review     : {res['text']}")
        print(f"Cleaned    : {res['cleaned_text']}")
        print(f"Sentiment  : {res['sentiment'].upper()} (Confidence: {res['confidence'] * 100:.1f}%)")
        print("-" * 65)


def main() -> None:
    """CLI Argument parser entry point."""
    parser = argparse.ArgumentParser(
        description="Gojek App Sentiment Analysis Pipeline (Apple Developer Academy Showcase)"
    )
    parser.add_argument("--scrape", action="store_true", help="Scrape latest reviews from Play Store")
    parser.add_argument("--count", type=int, default=15000, help="Number of reviews to scrape")
    parser.add_argument("--train", action="store_true", help="Run model training and evaluation")
    parser.add_argument("--model", type=str, default="svm", choices=["svm", "random_forest"], help="Model type")
    parser.add_argument("--features", type=str, default="tfidf", choices=["tfidf", "word2vec"], help="Feature extractor")
    parser.add_argument("--smote", action="store_true", default=True, help="Use SMOTE balancing")
    parser.add_argument("--sample", type=int, default=2000, help="Sample size for training")

    args = parser.parse_args()

    if args.scrape:
        scrape_playstore_reviews(count=args.count)
    else:
        run_pipeline(
            model_type=args.model,
            feature_method=args.features,
            use_smote=args.smote,
            sample_size=args.sample
        )


if __name__ == "__main__":
    main()
