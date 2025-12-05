import logging

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Mute standard Hugging Face warnings to keep output clean
logging.getLogger("transformers").setLevel(logging.ERROR)


class ReviewCleaner:
    """
    Handles data loading, filtering, and SOTA language processing (Detection + Translation).
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.df_clean = None

        self.device = 0 if torch.cuda.is_available() else -1
        print(f"--> [Device] Running on {'GPU (CUDA)' if self.device == 0 else 'CPU'}")

        # Initialize tqdm for pandas operations
        tqdm.pandas()

    def load_data(self):
        """Loads the dataset from CSV or Excel."""
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"--> [Loader] Data loaded successfully. Total rows: {len(self.df)}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def filter_by_sentiment(self, target_sentiments=["negative", "neutral"]):
        """
        Filters dataset to keep only 'negative' and 'neutral' reviews
        to focus on issues/bugs.
        """
        if self.df is None:
            return
        initial = len(self.df)

        mask = self.df["sentiment"].str.lower().isin(target_sentiments)
        self.df = self.df[mask].copy()

        print(
            f"--> [Filter] Kept {target_sentiments}. Dropped {initial - len(self.df)} rows."
        )

    def remove_junk(self, min_words=2):
        """Removes reviews too short to be meaningful."""
        if self.df is None:
            return
        initial = len(self.df)

        self.df["word_count"] = self.df["review"].apply(lambda x: len(str(x).split()))
        self.df = self.df[self.df["word_count"] >= min_words].copy()

        print(
            f"--> [Filter] Removed short outliers (< {min_words} words). Dropped {initial - len(self.df)} rows."
        )

    def _get_nllb_code(self, iso_code: str) -> str:
        """
        Dynamically maps standard 2-letter ISO codes (from detection)
        to NLLB's 600+ language codes.
        Default fallback: 'eng_Latn' (English) to prevent crashes.
        """
        # Common mappings for European languages likely in app reviews
        mapping = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ro": "ron_Latn",
            "nl": "nld_Latn",
            "pl": "pol_Latn",
            "ru": "rus_Cyrl",
            "ar": "arb_Arab",
            "zh": "zho_Hans",
        }
        return mapping.get(iso_code, "eng_Latn")

    def detect_and_translate(self):
        """
        1. Detects language using XLM-RoBERTa.
        2. Translates non-Italian reviews to Italian using NLLB-200.
        """
        if self.df is None:
            return

        # --- STEP 1: LANGUAGE DETECTION ---
        print("--> [Pipeline] Loading Language Detector (XLM-RoBERTa)...")
        # pipeline automatically handles device placement
        detector = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=self.device,
            truncation=True,
            top_k=1,
        )

        def detect_lang(text):
            try:
                # Returns [{'label': 'fr', 'score': 0.98}]
                return detector(str(text))[0]["label"]
            except:
                return "unknown"

        print("--> [Step 1/2] Detecting languages (this may take a moment)...")
        self.df["detected_lang"] = self.df["review"].progress_apply(detect_lang)

        # Log distribution
        top_langs = self.df["detected_lang"].value_counts().head(5).to_dict()
        print(f"    Language Distribution found: {top_langs}")

        # --- STEP 2: TRANSLATION ---
        # Identify rows that need translation:
        # Not Italian ('it') AND Not 'unknown'
        mask_translate = (self.df["detected_lang"] != "it") & (
            self.df["detected_lang"] != "unknown"
        )
        rows_to_translate = self.df[mask_translate]

        if rows_to_translate.empty:
            print("--> [Pipeline] All reviews are Italian. No translation needed.")
            self.df["Final_Text"] = self.df["review"]
        else:
            print(
                f"--> [Step 2/2] Translating {len(rows_to_translate)} reviews to Italian..."
            )
            print("    Loading NLLB-200 Model... (Download ~1.2GB first run)")

            model_name = "facebook/nllb-200-distilled-600M"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                self.device if self.device == 0 else "cpu"
            )

            # Helper function for translation
            def translate_wrapper(text, source_iso):
                try:
                    # 1. Get correct source code (e.g., 'fra_Latn')
                    src_code = self._get_nllb_code(source_iso)
                    tokenizer.src_lang = src_code

                    # 2. Tokenize
                    inputs = tokenizer(str(text), return_tensors="pt").to(model.device)

                    # 3. Generate Italian ('ita_Latn')
                    generated = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"],
                        max_length=128,
                    )

                    # 4. Decode
                    return tokenizer.batch_decode(generated, skip_special_tokens=True)[
                        0
                    ]
                except Exception:
                    # Fallback: return original text if translation fails
                    return text

            # Apply translation
            tqdm.pandas(desc="Translating Rows")
            translated_texts = rows_to_translate.progress_apply(
                lambda row: translate_wrapper(row["review"], row["detected_lang"]),
                axis=1,
            )

            # Merge translated text back
            self.df["Final_Text"] = self.df["review"]  # Start with original
            self.df.loc[mask_translate, "Final_Text"] = translated_texts

            print("--> [Pipeline] Translation complete.")

        # Final Cleanup: Make 'review' the clean Italian version
        self.df["Original_Review"] = self.df["review"]  # Backup
        self.df["review"] = self.df["Final_Text"]

        self.df_clean = self.df.reset_index(drop=True)

    def get_cleaned_corpus(self):
        """Returns the list of text for BERTopic."""
        if self.df_clean is None:
            raise ValueError("Pipeline not run. Call detect_and_translate() first.")
        return self.df_clean["review"].astype(str).tolist()

    def get_cleaned_dataframe(self):
        """Returns the full DF with metadata."""
        return self.df_clean
