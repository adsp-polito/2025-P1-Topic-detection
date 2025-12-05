import pandas as pd
import wandb

from cleaner import DataProcessor
from duplicate_remover import DuplicateRemover
from evaluation import TaxonomyMapper, calculate_coherence_metrics
from sentiment_analyzer import SentimentEnsemble
from topic_modeler import TopicModeler
from translation import TranslatorModule

# ================= CONFIGURATION =================
DATA_PATH = "./data/dataset_v2.xlsx"
TAXONOMY_PATH = "./data/taxonomy_v2.xlsx"  # Ensure this matches your filename
WANDB_PROJECT = "hype-project-topics"

# Options: "multilingual" (SOTA Generic) OR "italian_social" (AlBERTo - Tweets/Reviews)
MODEL_CHOICE = "italian_social"
# =================================================


def load_taxonomy(path):
    """Loads the unique labels from the taxonomy CSV."""
    try:
        # Assuming the CSV has a header. We look for the first column.
        df_tax = pd.read_csv(path)
        # Get the first column as a list of strings
        labels = df_tax.iloc[:, 0].astype(str).unique().tolist()
        print(f"--> [Taxonomy] Loaded {len(labels)} labels from {path}")
        return labels
    except Exception as e:
        print(f"--> [Error] Could not load taxonomy: {e}")
        return []


def main():
    print("=== HYPE TOPIC DETECTION PIPELINE ===")
    print(f"=== Selected Model: {MODEL_CHOICE} ===")

    # 1. LOAD DATA
    loader = DataProcessor(DATA_PATH)
    df = loader.load_data()

    # 2. DETECT LANGUAGE
    df = loader.detect_language(column="review")

    # 3. TRANSLATE (Non-IT -> IT)
    translator = TranslatorModule(df)
    df = translator.translate_non_italian(text_col="review")

    # 4. TEXT CLEANING & EMOJI CONVERSION
    # Clean *before* sentiment analysis so emojis become text (e.g., ":thumbs_down:")
    loader.df = df
    df = loader.basic_cleaning(text_column="final_text", target_column="clean_text")

    # 5. RE-CLASSIFY SENTIMENT (Ensemble)
    sentiment_engine = SentimentEnsemble()
    df = sentiment_engine.get_ensemble_sentiment(df, text_col="clean_text")

    # 6. FILTER DATASET (STRICTLY NEGATIVE)
    print("--> [Filter] Keeping ONLY Negative reviews for Topic Detection...")
    df = df[df["sentiment"] == "negative"].reset_index(drop=True)

    print(f"--> [Filter] {len(df)} negative reviews remaining.")

    # Junk Removal
    df = loader.remove_junk_reviews(column="clean_text")

    # 7. DEDUPLICATION (TF-IDF)
    deduplicator = DuplicateRemover(threshold=0.92)
    df = deduplicator.remove_duplicates(df, text_col="clean_text")

    # 8. TOPIC DETECTION (BERTopic)
    docs = df["clean_text"].tolist()
    print(f"--> [Topic Modeling] Starting run on {len(docs)} negative reviews...")

    if len(docs) > 50:
        tm = TopicModeler(project_name=WANDB_PROJECT)
        run_name = f"negative_issues_{MODEL_CHOICE}"

        # Run Modeling
        model, topics, probs = tm.run(docs, run_name=run_name, model_type=MODEL_CHOICE)

        # Save Basic Results
        df["topic"] = topics
        output_filename = f"results_topics_{MODEL_CHOICE}.xlsx"
        df.to_excel(output_filename, index=False)
        print(f"--> [Done] Basic results saved to {output_filename}")

        # --- EVALUATION PHASE ---

        # A. Quantitative Metric (Silhouette Score)
        # We re-encode docs to get embeddings (fast with cached model)
        print("--> [Evaluation] Generating embeddings for scoring...")
        embeddings = tm.embedding_model.encode(docs, show_progress_bar=False)

        silhouette = calculate_coherence_metrics(model, docs, embeddings, topics)
        if wandb.run is not None:
            wandb.log({"silhouette_score": silhouette})

        # B. Taxonomy Mapping
        provided_labels = load_taxonomy(TAXONOMY_PATH)

        if provided_labels:
            mapper = TaxonomyMapper(embedding_model=tm.embedding_model)
            mapping_df = mapper.map_topics_to_taxonomy(model, provided_labels)

            print(mapping_df.head())
            mapping_filename = f"taxonomy_comparison_{MODEL_CHOICE}.xlsx"
            mapping_df.to_excel(mapping_filename, index=False)
            print(f"--> [Done] Taxonomy comparison saved to {mapping_filename}")

    else:
        print("--> [Error] Not enough data for topic modeling.")


if __name__ == "__main__":
    main()
