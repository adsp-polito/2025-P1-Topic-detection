import os

import pandas as pd
import wandb

from baseline import BaselineModeler
from cleaner import DataProcessor
from config import cfg
from duplicate_remover import DuplicateRemover
from logger import WandBLogger
from mwe import MWEExtractor
from sentiment_analyzer import SentimentEnsemble
from translation import TranslatorModule
from utils import ensure_directories, seed_everything


def plot_top_words(model_name, results_df):
    """
    Helper to create WandB bar charts for top words in each topic.
    Parses the "Top_Words" string (comma-separated) back into counts/lists.
    Since we don't have exact counts from the summary string,
    we visualizes the rank or just list them in a table.

    For a better visual, we will rely on the Table we log,
    but we can try to log a basic chart if needed.
    Here we stick to a rich Table which is best for text lists.
    """
    # Filter for the specific model
    df = results_df[results_df["Model"] == model_name].copy()

    # Create a WandB Table
    table = wandb.Table(dataframe=df)
    return table


def run_baselines():
    # 1. SETUP & REPRODUCIBILITY
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    print("=== HYPE BASELINE RUNNER (LDA/NMF) ===")

    # 2. INITIALIZE WANDB
    # We use a distinct job_type so these runs don't mess up your BERTopic charts
    logger = None
    if cfg.get("project.wandb_logging"):
        logger = WandBLogger(
            job_type="baseline_comparison", run_name="hype_baseline_run"
        )
        print("--> [WandB] Baseline run started.")

    # Ensure output directories exist
    ensure_directories([cfg.get("paths.cache"), "./out/"])

    # 3. FULL DATA PIPELINE (Copy of main.py logic)
    # We want to ensure we are working on the EXACT same processed data.

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

        # A. Load
        df = loader.load_data()

        # B. Detect & Translate
        translator = TranslatorModule(df)
        df = translator.detect_and_translate(text_col="review")

        # C. Clean
        loader.df = df
        df = loader.basic_cleaning(text_column="final_text", target_column="clean_text")

        # D. Sentiment Re-classification
        sentiment_engine = SentimentEnsemble()
        df = sentiment_engine.get_ensemble_sentiment(df, text_col="clean_text")

        # Save Cache
        print(f"--> [Cache] Saving to '{cache_path}'...")
        df.to_pickle(cache_path)

    # 4. FILTERING & REFINEMENT
    print("--> [Filter] Keeping ONLY Negative reviews for Baselines...")
    df = df[df["sentiment"] == "negative"].reset_index(drop=True)
    print(f"--> [Filter] {len(df)} negative reviews remaining.")

    loader.df = df

    # Junk Removal
    df = loader.remove_junk_reviews(column="clean_text")

    # Deduplication
    deduplicator = DuplicateRemover()
    df = deduplicator.remove_duplicates(df, text_col="clean_text")

    # Multi-Word Expressions
    mwe_extractor = MWEExtractor(df=df)
    _ = mwe_extractor.extract_mwe()
    df = mwe_extractor.apply_mwe()

    # 5. PREPARE DOCUMENTS
    docs = df["clean_text_mwe"].tolist()

    if len(docs) < 10:
        print("--> [Error] Not enough documents to run baselines.")
        if logger:
            logger.finish()
        return

    # 6. RUN BASELINES (LDA & NMF)
    baseline_modeler = BaselineModeler()
    df_results = baseline_modeler.run(docs)

    # 7. LOGGING & SAVING
    out_dir = "./out"
    out_path = f"{out_dir}/baseline_comparison.xlsx"
    df_results.to_excel(out_path, index=False)
    print(f"--> [Done] Baseline results saved to: {out_path}")

    if logger:
        # A. Log the Excel file as an Artifact
        logger.log_artifact(
            out_path,
            type="results",
            name="baseline_topics_excel",
            description="LDA and NMF topics (Top 15 words)",
        )

        # B. Log Tables for Visualization in Dashboard
        # We split them to make the dashboard cleaner
        lda_table = plot_top_words("LDA", df_results)
        nmf_table = plot_top_words("NMF", df_results)

        logger.log_plot("lda_topics_table", lda_table, plot_type="table")
        logger.log_plot("nmf_topics_table", nmf_table, plot_type="table")

        # C. Log Metrics
        logger.log_metrics(
            {
                "n_docs_baseline": len(docs),
                "n_topics_configured": baseline_modeler.n_topics,
            }
        )

        print("--> [WandB] Run finished.")
        logger.finish()


if __name__ == "__main__":
    run_baselines()
