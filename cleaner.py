import re
from typing import Optional

import emoji
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from tqdm import tqdm

# Ensure reproducibility for language detection
DetectorFactory.seed = 0


class DataProcessor:
    """
    Handles data loading, language detection, and text cleaning
    (including emoji-to-text conversion).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        tqdm.pandas()

    def load_data(self) -> pd.DataFrame:
        """Loads data from Excel file."""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"--> [Loader] Data loaded. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

    def detect_language(self, column: str = "review") -> pd.DataFrame:
        """Detects language for the specified column."""
        if self.df is None:
            raise ValueError("Data not loaded.")

        print("--> [LangDetect] Detecting languages...")

        def safe_detect(text):
            try:
                return detect(text)
            except:
                return np.nan

        self.df["detected_lang"] = self.df[column].progress_apply(safe_detect)
        return self.df

    def basic_cleaning(
        self, text_column: str = "review", target_column: str = "clean_text"
    ) -> pd.DataFrame:
        """
        Performs cleaning:
        1. Emojis -> Text (CRITICAL for sentiment)
        2. Normalize spaces
        3. Mask URLs/Mentions
        4. Reduce repeated chars
        """
        print("--> [Cleaner] Performing text cleaning & emoji conversion...")

        def clean(text):
            if not isinstance(text, str):
                return ""

            # 1. Convert Emojis to text (e.g. 👍 -> :thumbs_up:)
            # This logic matches your original sentimentanalysis.py logic
            # We use demojize to convert emoji to text string
            text = emoji.replace_emoji(
                text, replace=lambda e, data: f" {emoji.demojize(e)} "
            )

            # 2. Normalize spaces
            text = text.strip()
            text = re.sub(r"\s+", " ", text)

            # 3. Mask URLs and Usernames
            text = re.sub(r"http\S+", "<URL>", text)
            text = re.sub(r"@\w+", "<USER>", text)

            # 4. Normalize repeated characters (e.g., "nooooo" -> "noo")
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

            return text

        self.df[target_column] = self.df[text_column].progress_apply(clean)
        return self.df

    def remove_junk_reviews(self, column: str = "clean_text") -> pd.DataFrame:
        """
        Removes meaningless reviews (e.g., single words like 'ok', 'boh').
        Can be toggled in main.py if needed.
        """
        if self.df is None:
            return pd.DataFrame()

        junk_words = [
            "ok",
            "pessima",
            "bella",
            "sconsigliata",
            "bene",
            "boh",
            "bleah",
            "male",
            "top",
        ]

        # Calculate word counts
        word_counts = self.df[column].apply(lambda x: len(str(x).split()))

        # Identify junk
        is_junk = (word_counts <= 1) & (
            self.df[column].str.lower().str.strip().isin(junk_words)
        )

        initial_count = len(self.df)
        self.df = self.df[~is_junk].copy()

        print(f"--> [Cleaner] Removed {initial_count - len(self.df)} junk reviews.")
        return self.df
