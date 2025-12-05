import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def calculate_coherence_metrics(topic_model, docs, embeddings, topics):
    """
    Calculates the Silhouette Score to measure cluster separation.
    Excludes the -1 (noise) topic to get a fair metric of the actual clusters.
    """
    print("--> [Evaluation] Calculating Silhouette Score...")

    # Convert topics to numpy array for boolean indexing
    topics = np.array(topics)

    # Filter out noise (-1)
    mask = topics != -1

    if np.sum(mask) < 2:
        print("    [Warning] Not enough clustered data points to calculate Silhouette.")
        return -1.0

    clean_embeddings = embeddings[mask]
    clean_topics = topics[mask]

    # Calculate Score
    # Range: -1 to 1. Higher is better (more distinct topics).
    score = silhouette_score(clean_embeddings, clean_topics)

    print(f"    Silhouette Score: {score:.4f}")
    return score


class TaxonomyMapper:
    """
    Compares discovered BERTopic topics with a provided set of 'Golden Labels' (Taxonomy)
    using Semantic Similarity.
    """

    def __init__(self, embedding_model):
        # We reuse the same embedding model used for BERTopic
        self.embedding_model = embedding_model

    def map_topics_to_taxonomy(self, topic_model, taxonomy_labels: list):
        """
        Returns a DataFrame showing the best match for each discovered topic.
        """
        print("--> [Evaluation] Mapping discovered topics to provided Taxonomy...")

        # 1. Get Discovered Topic Representations
        topic_info = topic_model.get_topic_info()
        # Filter out Topic -1 (Noise)
        topic_info = topic_info[topic_info["Topic"] != -1]

        discovered_texts = []
        topic_ids = []

        # Construct a string representation for each topic (using top 5 words)
        for t_id in topic_info["Topic"]:
            words = [word for word, _ in topic_model.get_topic(t_id)[:5]]
            discovered_texts.append(" ".join(words))
            topic_ids.append(t_id)

        if not topic_ids:
            print("    [Warning] No topics found (only noise). Skipping mapping.")
            return pd.DataFrame()

        # 2. Embed Both Lists
        print("    Embedding topics and taxonomy labels...")
        dt_embeddings = self.embedding_model.encode(discovered_texts)
        tax_embeddings = self.embedding_model.encode(taxonomy_labels)

        # 3. Calculate Cosine Similarity Matrix
        similarity_matrix = cosine_similarity(dt_embeddings, tax_embeddings)

        # 4. Find Best Matches
        results = []

        for idx, t_id in enumerate(topic_ids):
            # Find index of highest score in the row
            best_match_idx = similarity_matrix[idx].argmax()
            best_score = similarity_matrix[idx][best_match_idx]
            best_label = taxonomy_labels[best_match_idx]

            results.append(
                {
                    "Topic_ID": t_id,
                    "Top_Words": discovered_texts[idx],
                    "Best_Match_Label": best_label,
                    "Similarity_Score": round(best_score, 4),
                    "Match_Type": "Strong" if best_score > 0.6 else "Weak/New",
                }
            )

        df_mapping = pd.DataFrame(results)
        df_mapping = df_mapping.sort_values(by="Similarity_Score", ascending=False)

        return df_mapping
