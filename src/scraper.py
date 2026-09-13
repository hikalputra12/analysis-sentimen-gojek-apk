"""Scraper module for downloading Google Play Store reviews for Gojek."""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from google_play_scraper import reviews, Sort

from src.config import APP_PACKAGE_ID, DEFAULT_SCRAPE_COUNT, SCRAPE_LANG, SCRAPE_COUNTRY, DEFAULT_RAW_DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def scrape_playstore_reviews(
    app_id: str = APP_PACKAGE_ID,
    count: int = DEFAULT_SCRAPE_COUNT,
    lang: str = SCRAPE_LANG,
    country: str = SCRAPE_COUNTRY,
    output_filepath: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Scrape reviews from Google Play Store with pagination and save to CSV.

    Args:
        app_id: Google Play App Package identifier (e.g. 'com.gojek.app').
        count: Desired number of reviews to extract.
        lang: Review language code.
        country: Country locale code.
        output_filepath: Destination CSV file path.

    Returns:
        List of review dictionaries.

    Raises:
        RuntimeError: If scraping fails due to network or scraper issues.
    """
    target_path = output_filepath or DEFAULT_RAW_DATA
    logger.info("Starting scrape for %s (Target count: %d)...", app_id, count)

    try:
        scraped_data, _ = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.MOST_RELEVANT,
            count=count
        )
        logger.info("Successfully scraped %d raw reviews.", len(scraped_data))
    except Exception as exc:
        logger.error("Failed to scrape reviews for %s: %s", app_id, exc)
        raise RuntimeError(f"Scraping failed: {exc}") from exc

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    try:
        with open(target_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Review", "Score", "ThumbsUp", "CreatedAt"])
            for item in scraped_data:
                writer.writerow([
                    item.get("content", ""),
                    item.get("score", ""),
                    item.get("thumbsUpCount", 0),
                    item.get("at", "")
                ])
        logger.info("Saved %d reviews to %s", len(scraped_data), target_path)
    except IOError as exc:
        logger.error("Failed to write scraped data to CSV: %s", exc)
        raise

    return scraped_data


if __name__ == "__main__":
    scrape_playstore_reviews(count=100)
