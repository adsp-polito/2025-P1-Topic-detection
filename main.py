from cleaner import DataProcessor
from duplicate_remover import DuplicateRemover
from sentiment_analyzer import SentimentEnsemble
from topic_modeler import TopicModeler
from translation import TranslatorModule

# ================= CONFIGURATION =================
DATA_PATH = "dataset_v2.xlsx"
WANDB_PROJECT = "hype-project-topics"

# SELECT MODEL HERE:
# Options: "multilingual" (SOTA Generic) OR "italian_social" (AlBERTo - Tweets/Reviews)
MODEL_CHOICE = "italian_social"
# =================================================


def main():
    print("=== HYPE TOPIC DETECTION PIPELINE ===")
    print(f"=== Selected Model: {MODEL_CHOICE} ===")

    # 1. LOAD
    loader = DataProcessor(DATA_PATH)
    df = loader.load_data()

    # 2. DETECT LANGUAGE
    df = loader.detect_language(column="review")

    # 3. TRANSLATE (Non-IT -> IT)
    translator = TranslatorModule(df)
    # This assumes your translation.py creates a 'final_text' or similar column
    # If using the class provided earlier, it creates 'final_text' containing the translated version
    df = translator.translate_non_italian(text_col="review")

    # 4. TEXT CLEANING & EMOJI CONVERSION
    # Clean *before* sentiment analysis so emojis become text (e.g., ":thumbs_down:")
    loader.df = df
    df = loader.basic_cleaning(text_column="final_text", target_column="clean_text")

    # 5. RE-CLASSIFY SENTIMENT (The "Ensemble" step)
    sentiment_engine = SentimentEnsemble()
    # Re-classify 'neutral' reviews into 'positive' or 'negative'
    df = sentiment_engine.get_ensemble_sentiment(df, text_col="clean_text")

    # 6. FILTER DATASET (STRICTLY NEGATIVE)
    # We drop Positive and remaining Neutrals/Errors immediately.
    print("--> [Filter] Keeping ONLY Negative reviews for Topic Detection...")
    df = df[df["sentiment"] == "negative"].reset_index(drop=True)

    print(f"--> [Filter] {len(df)} negative reviews remaining.")

    # Junk Removal
    df = loader.remove_junk_reviews(column="clean_text")

    # 7. DEDUPLICATION (TF-IDF)
    # Now running only on the negative subset
    deduplicator = DuplicateRemover(threshold=0.92)
    df = deduplicator.remove_duplicates(df, text_col="clean_text")

    # 8. TOPIC DETECTION (BERTopic)
    # We use the entire remaining dataframe since it is already filtered
    docs = df["clean_text"].tolist()

    print(f"--> [Topic Modeling] Starting run on {len(docs)} negative reviews...")

    if len(docs) > 50:
        tm = TopicModeler(project_name=WANDB_PROJECT)

        run_name = f"negative_issues_{MODEL_CHOICE}"
        model, topics, probs = tm.run(docs, run_name=run_name, model_type=MODEL_CHOICE)

        # Save Results
        df["topic"] = topics
        output_filename = f"results_topics_{MODEL_CHOICE}.xlsx"
        df.to_excel(output_filename, index=False)
        print(f"--> [Done] Results saved to {output_filename}")

    else:
        print("--> [Error] Not enough data for topic modeling.")


if __name__ == "__main__":
    main()
