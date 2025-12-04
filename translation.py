import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

class ReviewTranslation:
    """
    Handles language detection and machine translation to Italian
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.mask_non_it = None
        self.non_italian_languages = None

        tqdm.pandas()

    def analyze_languages(self) -> None:
        """
        Analyzes 
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.mask_non_it = self.df["language"]!="it"
        self.non_italian_languages = self.df[self.mask_non_it]['language'].dropna().unique().tolist()

        print(f"--> Detected {self.df[self.mask_non_it].shape[0]} non-Italian reviews")
        print(f"--> There are {len(self.non_italian_languages)} non-Italian languages:", self.non_italian_languages)

    
    def translate_to_italian(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df["review_translated"] = self.df["review"]

        for lang in tqdm(self.non_italian_languages, desc=" --> [Translator] Translation to Italian using Google Translator"):
            mask = self.df["language"] == lang
            texts = self.df.loc[mask, "review"].tolist()

            translator = GoogleTranslator(source=lang, target="it")

            translations = []
            for t in texts:
                try:
                    translated = translator.translate(t)
                    translations.append(translated)
                except Exception as e:
                    print(f"Error with language: ({lang}): {e}")
                    translations.append(f"TRANSLATION_ERROR_{lang}")

            self.df.loc[mask, "review_translated"] = translations

        return self.df

