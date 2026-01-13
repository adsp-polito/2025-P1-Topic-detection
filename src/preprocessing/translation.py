import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from src.utils.config import cfg


class TranslatorModule:
    """
    Handles SOTA language detection and translation using configurations
    defined in config.yaml.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.device = 0 if torch.cuda.is_available() else -1

        # Load Configs
        self.conf = cfg.get("translation")
        self.batch_size = self.conf.get("batch_size", 16)
        self.mapping = self.conf.get("iso_to_nllb_mapping", {})
        self.target_lang = self.conf.get("target_lang", "ita_Latn")

        # Initialize Models
        print(f"--> [Translator] Initializing models on device: {self.device}...")

        # 1. Detection Pipeline
        self.detector = pipeline(
            "text-classification",
            model=self.conf.get("detection_model"),
            device=self.device,
            truncation=True,
            top_k=1,
        )

        # 2. Translation Model (NLLB)
        model_name = self.conf.get("translation_model")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if self.device == 0:
            self.nllb_model = self.nllb_model.cuda()

    def detect_and_translate(self, text_col: str = "review") -> pd.DataFrame:
        """
        1. Detects language in batches.
        2. Groups by language and translates in batches.
        """
        # --- PHASE 1: BATCH DETECTION ---
        print("--> [Step 1/2] Batch Language Detection...")
        texts = self.df[text_col].astype(str).tolist()

        # Run pipeline
        results = self.detector(texts, batch_size=self.batch_size, truncation=True)
        self.df["detected_lang"] = [res[0]["label"] for res in results]

        top_langs = self.df["detected_lang"].value_counts().head(5).to_dict()
        print(f"    Top languages found: {top_langs}")

        # --- PHASE 2: GROUPED BATCH TRANSLATION ---
        self.df["final_text"] = self.df[text_col]  # Default to original

        # Filter for rows that need translation
        # (Not 'it' AND exists in our config mapping)
        mask = (self.df["detected_lang"] != "it") & (
            self.df["detected_lang"].isin(self.mapping.keys())
        )

        to_translate_df = self.df[mask].copy()

        if to_translate_df.empty:
            print("--> [Translator] No foreign languages detected to translate.")
            return self.df

        print(f"--> [Step 2/2] Translating {len(to_translate_df)} reviews...")

        # Group by detected language to batch efficiently
        grouped = to_translate_df.groupby("detected_lang")

        for lang_iso, group_df in tqdm(grouped, desc="Translating by Language Group"):
            # Get NLLB code from Config
            nllb_code = self.mapping[lang_iso]

            src_texts = group_df[text_col].tolist()
            indices = group_df.index

            # Set tokenizer source language
            self.nllb_tokenizer.src_lang = nllb_code

            translated_batch = []

            # Sub-batching within the language group
            for i in range(0, len(src_texts), self.batch_size):
                batch_texts = src_texts[i : i + self.batch_size]

                inputs = self.nllb_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.conf.get("max_length", 128),
                ).to(self.nllb_model.device)

                with torch.no_grad():
                    generated_tokens = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(
                            self.target_lang
                        ),
                        max_length=self.conf.get("max_length", 128),
                    )

                decoded = self.nllb_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                translated_batch.extend(decoded)

            # Assign back
            self.df.loc[indices, "final_text"] = translated_batch

        print("--> [Translator] Pipeline complete.")
        return self.df
