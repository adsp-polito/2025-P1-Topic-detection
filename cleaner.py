from typing import List, Optional

import pandas as pd
from langdetect import DetectorFactory, detect
from tqdm import tqdm

DetectorFactory.seed = 0  # Ensure reproducibility for language detection


class ReviewCleaner:
    """
    Handles data loading, exploratory analysis, and pre-processing
    specifically for the HYPE topic detection task.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None

        tqdm.pandas()

    def load_data(self) -> pd.DataFrame:
        """Loads data from Excel file into a DataFrame."""
        try:
            if self.file_path.endswith((".xls", ".xlsx")):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError(
                    "Unsupported file format. Please provide an Excel file."
                )

            print(f"--> [Loader] Data loaded. Total rows: {len(self.df)}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def filter_by_sentiment(
        self, target_sentiments: List[str] = ["negative", "neutral"]
    ) -> None:
        """
        Filters the dataset to keep only specific sentiments.
        Goal: Detect 'negative topics (issues)'.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_count = len(self.df)

        # Normalize to lowercase for comparison
        mask = (
            self.df["sentiment"]
            .str.lower()
            .isin([s.lower() for s in target_sentiments])
        )
        self.df = self.df[mask].copy()

        print(
            f"--> [Filter] Kept sentiments {target_sentiments}. Dropped {initial_count - len(self.df)} rows."
        )

    def remove_junk_and_outliers(self, min_words: int = 3) -> None:
        """
        Removes reviews that are too short (< min_words) to provide context.
        Addresses the need to remove 'junk' reviews.
        """

        if self.df is None:
            return

        initial_count = len(self.df)

        # Calculate word count safely
        self.df["word_count"] = self.df["review"].progress_apply(
            lambda x: len(str(x).split())
        )

        # Filter
        self.df = self.df[self.df["word_count"] >= min_words].copy()

        print(
            f"--> [Filter] Removed short outliers (< {min_words} words). Dropped {initial_count - len(self.df)} rows."
        )

    def filter_non_italian(self) -> None:
        """
        filter_non_italian using langdetect.
        Based on dataset being 'mostly in Italian'.
        """
        if self.df is None:
            return

        initial_count = len(self.df)

        def is_italian_safe(text: str) -> bool:
            try:
                # If text is very short (<20 chars), language detection is unreliable.
                # We assume it is valid to avoid dropping valid short Italian phrases.
                if len(str(text)) < 20:
                    return True
                return detect(text) == "it"
            except:
                return False

        print("--> [Filter] Running language detection (this may take a moment)...")
        self.df["is_italian"] = self.df["Review_Text"].progress_apply(is_italian_safe)
        self.df = self.df[self.df["is_italian"]].copy()

        print(
            f"--> [Filter] Removed non-Italian texts. Dropped {initial_count - len(self.df)} rows."
        )

        # Set the final cleaned dataframe and reset index
        self.df_clean = self.df.reset_index(drop=True)

    def get_cleaned_corpus(self) -> List[str]:
        """Returns the list of text strings ready for BERTopic."""
        if self.df_clean is None:
            raise ValueError("Data processing not complete.")
        return self.df_clean["Review_Text"].astype(str).tolist()

    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """Returns the full cleaned dataframe (useful for saving later)."""
        return self.df_clean
