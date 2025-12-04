from typing import List, Optional
import pandas as pd
from langdetect import DetectorFactory, detect
import re
import numpy as np
from tqdm import tqdm

DetectorFactory.seed = 0  # Ensure reproducibility for language detection

class ReviewCleaner:
    """
    Handles data loading and pre-processing
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

    def filter_by_sentiment(self, target_sentiments: List[str] = ["negative", "neutral"]) -> None:
        """
        Filters the dataset to keep only specific sentiments.
        Goal: Detect 'negative topics (issues)'.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        initial_count = len(self.df)

        # Normalize sentiment column to lowercase for comparison
        mask = (
            self.df["sentiment"]
            .str.lower()
            .isin([s.lower() for s in target_sentiments])
        )
        self.df = self.df[mask].copy()

        print(
            f"--> [Filter] Kept sentiments {target_sentiments}. Dropped {initial_count - len(self.df)} rows."
        )

    #DONT USE
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

    
    def detect_language(self) -> None:
        """
        Detects the language of the review using langdetect.
        Based on dataset being 'mostly in Italian'.
        """
        if self.df is None:
            return

        def safe_detect_language(text:str) -> None:
            try:
                return detect(text)
            except:
                return np.nan
        
        print("--> [Language Detector] Running language detection (this may take a moment)...")
        self.df["language"] = self.df["review"].progress_apply(safe_detect_language)

    
    def clean_text(self) -> pd.DataFrame:
        """
        Performs pre-processing to clean the text before rearranging the sentiment:
        1. Normalize spacing 
        2. Remove URLs, mentions 
        3. Reduce repeated characters
        """

        def clean(text:str) -> str:
            if not isinstance(text, str): # Handle non-string inputs
                return ""
            # Replace repeated spaces with one space only ' '
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            # Replace URLs, mentions 
            text = re.sub(r'http\S+', '<URL> ', text)
            text = re.sub(r'@\w+', '<USER>', text)
            # Reduce repeated charachters ('bellooo' -> 'bello')
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

            return text

        # Apply the function to the 'review' column
        self.df['cleaned_review'] = self.df['review'].progress_apply(clean)

        print(f"--> [Cleaner] Cleaning reviews text.")

        return self.df
    
    
    def remove_junk(self) -> pd.DataFrame:
        """
        Removes meaningless reviews:
        1. Reviews that contain only meaningless words like 'ok', 'pessima', etc.
        2. Reviews with empty text or whitespace or punctuation 
        """
        if self.df is None:
            return
        
        # Define meaningless words
        WORDS = ['ok', 'pessima', 'bella', 'sconsigliata', 'bene', 'boh', 'bleah', 'male']

        # Calculate word count safely
        self.df["word_count"] = self.df["cleaned_review"].progress_apply(
            lambda x: len(str(x).split())
        )

        # Filter 
        mask = ((self.df["word_count"] == 1) & (self.df["cleaned_review"].str.lower().str.strip().isin(WORDS)))
        self.df = self.df[~mask].copy()

        print(f"--> [Filter] Removed short reviews with just 1 meaningless word. Dropped {mask.sum()} rows.")

        # Remove rows where cleaned_review is empty or whitespace
        before = len(self.df)
        self.df = self.df[self.df["cleaned_review"].str.strip() != ""].copy()
        # Remove rows with ONLY punctuation *excluding tags <URL> and <USER>*
        pattern_only_punct = r'^(?!<URL>|<USER>)[\W_]+$'
        self.df = self.df[~self.df["cleaned_review"].str.match(pattern_only_punct)]
        after = len(self.df)

        print(f"--> [Cleaner] Removed {before - after} empty/meaningless reviews.")

        # Save cleaned df 
        self.df_clean = self.df.copy()

        return self.df_clean


    def get_cleaned_corpus(self) -> List[str]:
        """Returns the list of text strings ready for BERTopic."""
        if self.df_clean is None:
            raise ValueError("Data processing not complete.")
        return self.df_clean["cleaned_review"].astype(str).tolist()

    def get_cleaned_dataframe(self) -> pd.DataFrame:
        """Returns the full cleaned dataframe (useful for saving later)."""
        return self.df_clean
