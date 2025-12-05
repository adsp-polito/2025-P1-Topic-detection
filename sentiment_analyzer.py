import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class SentimentEnsemble:
    """
    Implements the 3-way voting system for Sentiment Analysis:
    1. FEEL-IT (Manual inference)
    2. XLM-RoBERTa (Manual inference)
    3. FEEL-IT (Pipeline inference)

    Final sentiment is based on a composition score.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> [Sentiment] Initializing Voting Ensemble on {self.device}...")

        # --- MODEL 1 & 3: FEEL-IT ---
        self.feelit_name = "MilaNLProc/feel-it-italian-sentiment"
        self.feelit_tokenizer = AutoTokenizer.from_pretrained(self.feelit_name)
        self.feelit_model = AutoModelForSequenceClassification.from_pretrained(
            self.feelit_name
        ).to(self.device)

        # Pipeline version (for the 3rd vote)
        self.feelit_pipeline = pipeline(
            "text-classification",
            model=self.feelit_name,
            tokenizer=self.feelit_name,
            top_k=None,
            device=0 if self.device == "cuda" else -1,
        )

        # --- MODEL 2: XLM-RoBERTa ---
        self.xlm_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.xlm_tokenizer = AutoTokenizer.from_pretrained(self.xlm_name)
        self.xlm_model = AutoModelForSequenceClassification.from_pretrained(
            self.xlm_name
        ).to(self.device)

        # Labels mapping
        self.label_map_numeric = {"negative": -1, "neutral": 0, "positive": 1}

    def _predict_torch(self, texts, model, tokenizer, labels_order):
        """Helper for batch inference with PyTorch models"""
        preds = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = model(**inputs).logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

            preds.extend([labels_order[p] for p in batch_preds])
        return preds

    def get_ensemble_sentiment(
        self, df: pd.DataFrame, text_col: str = "review_translated"
    ) -> pd.DataFrame:
        """
        Applies the 3 approaches and calculates the composition score.
        Only runs on 'neutral' rows to re-classify them, or can run on all if needed.
        """
        # We only want to process the rows that are currently 'neutral'
        mask = df["sentiment"].astype(str).str.lower() == "neutral"
        target_df = df[mask].copy()

        if len(target_df) == 0:
            print("--> [Sentiment] No neutral reviews to re-classify.")
            return df

        texts = target_df[text_col].astype(str).tolist()
        print(
            f"--> [Sentiment] Re-classifying {len(texts)} neutral reviews with Ensemble..."
        )

        # --- VOTE 1: FEEL-IT Manual ---
        # Labels for FEEL-IT: ['negative', 'positive'] (Note: It usually only has 2, but let's check config if it has neutral.
        # The specific model 'MilaNLProc/feel-it-italian-sentiment' is trained on 2 classes: neg, pos)
        # If the model has 2 classes, index 0=neg, 1=pos
        v1_labels = ["negative", "positive"]
        v1_preds = self._predict_torch(
            texts, self.feelit_model, self.feelit_tokenizer, v1_labels
        )

        # --- VOTE 2: XLM-RoBERTa ---
        # Labels for cardiffnlp: ['negative', 'neutral', 'positive'] -> 0, 1, 2
        v2_labels = ["negative", "neutral", "positive"]
        v2_preds = self._predict_torch(
            texts, self.xlm_model, self.xlm_tokenizer, v2_labels
        )

        # --- VOTE 3: FEEL-IT Pipeline ---
        # The pipeline output structure is complex, we simplify extraction
        print("    ...Running pipeline inference...")
        pipe_out = self.feelit_pipeline(texts, batch_size=16, truncation=True)
        v3_preds = [res[0]["label"] for res in pipe_out]  # Taking top label

        # --- AGGREGATION ---
        target_df["v1"] = [self.label_map_numeric.get(x, 0) for x in v1_preds]
        target_df["v2"] = [self.label_map_numeric.get(x, 0) for x in v2_preds]
        target_df["v3"] = [self.label_map_numeric.get(x, 0) for x in v3_preds]

        # Composition Score
        target_df["sentiment_composition"] = (
            target_df["v1"] + target_df["v2"] + target_df["v3"]
        )

        # Final Decision Logic
        def decide(score):
            if score > 0:
                return "positive"
            else:
                return "negative"

        target_df["new_sentiment"] = target_df["sentiment_composition"].apply(decide)

        # Merge back into main dataframe
        df.loc[mask, "sentiment"] = target_df["new_sentiment"]

        print("--> [Sentiment] Re-classification complete.")
        return df
