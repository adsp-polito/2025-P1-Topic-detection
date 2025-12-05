from cleaner2 import ReviewCleaner
from topic_model import TopicDetector

# --- CONFIGURATION ---
DATA_PATH = "HYPE-dataset.csv"  # Ensure this matches your file name

# Hyperparameters for the experiment
CONFIG = {
    "min_topic_size": 30,  # Start with 30. Lower to 15 if you get too few topics.
    "embedding_model": "nickprock/sentence-bert-base-italian-uncased",  # Robust default
    "sentiment_filter": ["negative", "neutral"],  # Focus on issues
    "min_word_count": 3,  # Filter "junk"
}


def main():
    print("=== STARTING HYPE TOPIC DETECTION PROJECT ===")

    # ---------------------------------------------
    # PHASE 1: EDA & SOTA PRE-PROCESSING
    # ---------------------------------------------
    cleaner = ReviewCleaner(DATA_PATH)

    # 1. Load Data
    df = cleaner.load_data()
    if df.empty:
        print("CRITICAL ERROR: Data not found. Stopping.")
        return

    # 2. Apply Business Logic Filters
    cleaner.filter_by_sentiment(CONFIG["sentiment_filter"])
    cleaner.remove_junk(min_words=CONFIG["min_word_count"])

    # 3. Translation Pipeline: Detect Language -> Translate to Italian
    # This uses the Transformer models (XLM-R + NLLB)
    cleaner.detect_and_translate()

    # Prepare corpus for training
    docs = cleaner.get_cleaned_corpus()

    # Update config with actual data size for tracking
    CONFIG["final_dataset_size"] = len(docs)
    print(f"\n--> [Status] Ready for training with {len(docs)} documents.")

    # ---------------------------------------------
    # PHASE 2: TOPIC DETECTION (WANDB ENABLED)
    # ---------------------------------------------

    # Initialize detector with our configuration
    detector = TopicDetector(CONFIG)

    # Train the model
    detector.fit(docs)

    # ---------------------------------------------
    # PHASE 3: OUTPUTS & DELIVERABLES
    # ---------------------------------------------

    # 1. Save Topic Report
    topic_info = detector.get_topic_overview()
    print("\n--- TOP DETECTED TOPICS ---")
    print(topic_info.head(10))

    topic_info.to_csv("HYPE_detected_topics.csv", index=False)
    print("--> Report saved to 'HYPE_detected_topics.csv'")

    # 2. Save the actual model (for reuse later)
    detector.save_model("hype_model_output")

    # 3. Clean up WandB
    detector.finish_run()

    print("\n=== PROJECT EXECUTION COMPLETE ===")
    print("Check your WandB dashboard for interactive charts.")


if __name__ == "__main__":
    main()
