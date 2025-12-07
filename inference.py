from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from config import cfg


def load_and_predict():
    # 1. PATHS
    model_path = "./out/bertopic_model"

    # 2. LOAD EMBEDDING MODEL
    # CRITICAL: We must use the exact same embedding model used during training.
    embed_model_name = cfg.get("topic_modeling.embedding_model")
    print(f"--> Loading embedding model: {embed_model_name}...")
    embedding_model = SentenceTransformer(embed_model_name)

    # 3. LOAD BERTOPIC MODEL
    print(f"--> Loading BERTopic model from: {model_path}...")
    # We pass embedding_model here to ensure consistency
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model)

    print("--> Model loaded successfully!")

    # --- EXAMPLE USE CASE 1: PREDICT ON NEW DATA ---
    print("\n--- TEST PREDICTIONS ---")
    new_reviews = [
        "L'applicazione si blocca sempre quando provo a fare un bonifico.",
        "Non riesco ad accedere, credenziali non valide.",
        "Il servizio clienti non risponde mai al telefono.",
        "Tutto perfetto, app bellissima!",
    ]

    # .transform() returns the topic IDs and probabilities
    topics, probs = topic_model.transform(new_reviews)

    # Display results
    print(f"{'Review':<60} | {'Topic ID':<10} | {'Topic Name'}")
    print("-" * 100)

    for review, topic_id in zip(new_reviews, topics):
        # Handle outlier topic (-1)
        if topic_id == -1:
            topic_name = "Outlier / Noise"
        else:
            # Get the top 3 words to represent the topic name
            topic_info = topic_model.get_topic(topic_id)
            if topic_info:
                topic_name = "_".join([word for word, _ in topic_info[:3]])
            else:
                topic_name = "Unknown"

        print(f"{review[:58]:<60} | {topic_id:<10} | {topic_name}")

    # --- EXAMPLE USE CASE 2: INSPECT MODEL INFO ---
    print("\n--- MODEL INFO ---")
    freq = topic_model.get_topic_info()
    print(f"Total topics found: {len(freq) - 1}")  # -1 because of outlier topic
    print(freq.head())


if __name__ == "__main__":
    load_and_predict()
