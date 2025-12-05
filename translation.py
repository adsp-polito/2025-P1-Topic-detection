import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


class TranslatorModule:
    """
    Handles translation of non-Italian reviews into Italian.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.translator = None

    def translate_non_italian(
        self, text_col: str = "review", lang_col: str = "detected_lang"
    ) -> pd.DataFrame:
        """
        Iterates through non-Italian rows and translates them.
        Creates a 'final_text' column containing original Italian + Translated text.
        """
        # Initialize target column with original text
        self.df["final_text"] = self.df[text_col]

        # Identify non-Italian rows that are valid languages
        # We ignore NaN or 'it'
        mask = (self.df[lang_col] != "it") & (self.df[lang_col].notna())
        non_it_langs = self.df.loc[mask, lang_col].unique()

        print(
            f"--> [Translator] Found {len(non_it_langs)} foreign languages to translate."
        )

        for lang in non_it_langs:
            lang_mask = self.df[lang_col] == lang
            texts_to_translate = self.df.loc[lang_mask, text_col].tolist()

            # Initialize translator for this batch
            # Note: GoogleTranslator is rate-limited; for huge datasets consider alternatives
            self.translator = GoogleTranslator(source=lang, target="it")

            translated_batch = []
            print(f"    Translating {len(texts_to_translate)} reviews from {lang}...")

            for text in tqdm(texts_to_translate, leave=False):
                try:
                    res = self.translator.translate(text)
                    translated_batch.append(res)
                except Exception:
                    # Fallback to original text if translation fails
                    translated_batch.append(text)

            self.df.loc[lang_mask, "final_text"] = translated_batch

        return self.df
