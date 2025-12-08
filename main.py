import os

import pandas as pd
import wandb

from cleaner import DataProcessor
from config import cfg
from duplicate_remover import DuplicateRemover
from evaluation import TaxonomyMapper, calculate_coherence_metrics
from logger import WandBLogger
from sentiment_analyzer import SentimentEnsemble
from topic_modeler import TopicModeler
from translation import TranslatorModule
from utils import ensure_directories, load_taxonomy, seed_everything


def main():
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    print("=== HYPE TOPIC DETECTION PIPELINE ===")

    # INITIALIZE WANDB (Global Run)
    if cfg.get("project.wandb_logging"):
        main_logger = WandBLogger(
            job_type="full_pipeline", run_name="hype_analysis_master"
        )
        print("--> [WandB] Pipeline run started.")

    # Create 'out' folder if it doesn't exist
    os.makedirs("out", exist_ok=True)
    ensure_directories(
        [
            cfg.get("paths.cache"),
            cfg.get("paths.output_topics"),
            cfg.get("paths.output_mapping"),
            "./out/bertopic_model/",
        ]
    )

    # Load Paths from Config
    data_path = cfg.get("paths.data")
    cache_path = cfg.get("paths.cache")
    use_cache = cfg.get("preprocessing.use_cache")

    loader = DataProcessor(data_path)
    df = None

    # --- CHECKPOINT LOGIC ---
    if use_cache and os.path.exists(cache_path):
        print(f"--> [Cache] Found cached file '{cache_path}'. Loading...")
        df = pd.read_pickle(cache_path)
        print(f"--> [Cache] Loaded {len(df)} reviews from cache.")

        loader.df = df
    else:
        print("--> [Cache] No cache found. Running full preprocessing...")

        # 1. LOAD DATA
        df = loader.load_data()

        # 2. & 3. SOTA DETECTION & TRANSLATION (Batch Processed)
        translator = TranslatorModule(df)
        df = translator.detect_and_translate(text_col="review")

        # 4. TEXT CLEANING & EMOJI CONVERSION
        # Clean *before* sentiment analysis so emojis become text (e.g., ":thumbs_down:")
        loader.df = df
        df = loader.basic_cleaning(text_column="final_text", target_column="clean_text")

        # 5. RE-CLASSIFY SENTIMENT (Ensemble)
        sentiment_engine = SentimentEnsemble()
        df = sentiment_engine.get_ensemble_sentiment(df, text_col="clean_text")

        # SAVE CACHE
        print(f"--> [Cache] Saving to '{cache_path}'...")
        df.to_pickle(cache_path)

    # --- EDA LOGGING (Sentiment Distribution) ---
    if cfg.get("project.wandb_logging"):
        print("--> [WandB] Logging Sentiment Distribution (EDA)...")
        eda_logger = WandBLogger()

        # Prepare Data
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]

        # Log Table
        table = wandb.Table(dataframe=sent_counts)
        eda_logger.log_plot("sentiment_data", table, plot_type="table")

        # Log Bar Chart
        bar_plot = wandb.plot.bar(
            table, "sentiment", "count", title="Sentiment Distribution"
        )
        eda_logger.log_plot("sentiment_dist_plot", bar_plot, plot_type="chart")

    # 6. FILTER DATASET (STRICTLY NEGATIVE)
    print("--> [Filter] Keeping ONLY Negative reviews for Topic Detection...")
    df = df[df["sentiment"] == "negative"].reset_index(drop=True)

    print(f"--> [Filter] {len(df)} negative reviews remaining.")

    loader.df = df

    # Junk Removal
    df = loader.remove_junk_reviews(column="clean_text")

    # 7. DEDUPLICATION (TF-IDF)
    deduplicator = DuplicateRemover()
    df = deduplicator.remove_duplicates(df, text_col="clean_text")

    # 8. TOPIC DETECTION (BERTopic)
    docs = df["clean_text"].tolist()
    print(f"--> [Topic Modeling] Starting run on {len(docs)} negative reviews...")

    if len(docs) > 10:
        tm = TopicModeler()

        # Run Modeling
        model, topics, probs = tm.run(docs)

        # Save Basic Results
        df["topic"] = topics
        out_file_topics = cfg.get("paths.output_topics")
        os.makedirs(os.path.dirname(out_file_topics), exist_ok=True)

        df.to_excel(out_file_topics, index=False)
        print(f"--> [Done] Basic results saved to {out_file_topics}")

        if main_logger:
            main_logger.log_artifact(
                out_file_topics,
                "dataset",
                "labeled_reviews",
                "Negative reviews with Topic IDs",
            )

        model_save_path = "./out/bertopic_model"
        tm.save_model(model_save_path)

        if main_logger:
            main_logger.log_artifact(
                model_save_path,
                "model",
                "hype_bertopic_model",
                "BERTopic model trained on negative reviews",
            )

        # --- EVALUATION PHASE ---

        # A. Quantitative Metric (Silhouette Score)
        # We re-encode docs to get embeddings (fast with cached model)
        print("--> [Evaluation] Generating embeddings for scoring...")
        embeddings = tm.embedding_model.encode(docs, show_progress_bar=False)

        # Calculate Coherence (Logs to WandB if enabled)
        calculate_coherence_metrics(model, docs, embeddings, topics)

        # B. Taxonomy Mapping
        tax_path = cfg.get("paths.taxonomy")
        provided_labels = load_taxonomy(tax_path)

        if provided_labels:
            mapper = TaxonomyMapper(embedding_model=tm.embedding_model)
            mapping_df = mapper.map_topics_to_taxonomy(model, provided_labels)

            print(mapping_df.head())
            out_file_map = cfg.get("paths.output_mapping")
            mapping_df.to_excel(out_file_map, index=False)
            print(f"--> [Done] Taxonomy comparison saved to {out_file_map}")

            if main_logger:
                main_logger.log_artifact(
                    out_file_map,
                    "dataset",
                    "taxonomy_mapping",
                )

    else:
        print("--> [Error] Not enough data for topic modeling.")

    if cfg.get("project.wandb_logging"):
        print("--> [WandB] Run finished.")
        wandb.finish()


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
