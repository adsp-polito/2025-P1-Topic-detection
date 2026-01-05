import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from config import cfg
from logger import WandBLogger


def calculate_coherence_metrics(topic_model, docs, embeddings, topics, embedding_model, logger=None):
    """
    Calculates: 
    - Silhouette Score --> to measure cluster separation.
    - Topic Coherence (embedding-based)
    - Topic Diversity
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

    # Silhouette Score
    # Range: -1 to 1. Higher is better (more distinct topics).
    silhouette = silhouette_score(clean_embeddings, clean_topics)

    print(f"    Silhouette Score: {silhouette:.4f}")

    # Topic Coherence
    topic_ids = sorted(set(clean_topics))
    topic_coherences = []

    for t_id in topic_ids:
        words = [w for w, _ in topic_model.get_topic(t_id)[:10]]
        
        if len(words) < 2:
            continue

        word_embeddings = embedding_model.encode(words)
        sim_matrix = cosine_similarity(word_embeddings)
        mean_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        topic_coherences.append(mean_sim)

    topic_coherence = float(np.mean(topic_coherences)) if topic_coherences else 0.0
    print(f"    Topic Coherence: {topic_coherence:.4f}")

    # Topic Diversity
    all_words = []
    for t_id in topic_ids:
        
        all_words.extend([w for w, _ in topic_model.get_topic(t_id)[:10]])

    unique_words = set(all_words)
    topic_diversity = len(unique_words) / len(all_words) if all_words else 0.0

    print(f"    Topic Diversity: {topic_diversity:.4f}")

    # Log to WandB
    if cfg.get("project.wandb_logging"):
        if logger is None:
            print("    Warning: WandBLogger must be passed from main()")
        else:
            logger.log_metrics({
                    "silhouette_score": silhouette,
                    "topic_coherence": topic_coherence,
                    "topic_diversity": topic_diversity,
                    })

    return {
        "silhouette": silhouette,
        "topic_coherence": topic_coherence,
        "topic_diversity": topic_diversity,
    }


class TaxonomyMapper:
    """
    Compares discovered BERTopic topics with a provided set of labels (Taxonomy)
    using Semantic Similarity.
    """

    def __init__(self, embedding_model):
        # We reuse the same embedding model used for BERTopic
        self.embedding_model = embedding_model

    def map_topics_to_taxonomy(self, topic_model, taxonomy_df: pd.DataFrame):
        """
        Returns a DataFrame showing the best match for each discovered topic.
        Expects taxonomy_df to have columns: ['Label', 'Embedding_Text']
        """
        print("--> [Evaluation] Mapping discovered topics to provided Taxonomy...")

        if taxonomy_df.empty:
            print("    [Warning] Taxonomy DataFrame is empty. Skipping.")
            return pd.DataFrame()

        # 1. Get Discovered Topic Representations
        topic_info = topic_model.get_topic_info()
        # Filter out Topic -1 (Noise)
        topic_info = topic_info[topic_info["Topic"] != -1]

        discovered_texts = []
        topic_ids = []
        custom_labels = []

        # Construct a string representation for each topic (using top 10 words for better context)
        topic_info = topic_info.reset_index(drop=True)

        for idx, t_id in enumerate(topic_info["Topic"]):

            words = [word for word, _ in topic_model.get_topic(t_id)[:10]]
            discovered_texts.append(" ".join(words))
            topic_ids.append(t_id)

            if hasattr(topic_model, "custom_labels_") and topic_model.custom_labels_:
                if idx < len(topic_model.custom_labels_):
                    label = topic_model.custom_labels_[idx]
                else:
                    label = topic_info.loc[
                        topic_info["Topic"] == t_id, "Name"
                    ].values[0]
            else:
                label = topic_info.loc[
                    topic_info["Topic"] == t_id, "Name"
                ].values[0]

            custom_labels.append(label)

        if not topic_ids:
            print("    [Warning] No topics found (only noise). Skipping mapping.")
            return pd.DataFrame()

        # 2. Embed Both Lists, we embed the "Combined" text (Label + Description) from the taxonomy
        print("    Embedding topics and taxonomy descriptions...")
        dt_embeddings = self.embedding_model.encode(discovered_texts)
        tax_embeddings = self.embedding_model.encode(
            taxonomy_df["Embedding_Text"].tolist()
        )

        # 3. Calculate Cosine Similarity Matrix
        similarity_matrix = cosine_similarity(dt_embeddings, tax_embeddings)

        # 4. Find Best Matches
        results = []
        taxonomy_labels = taxonomy_df["Label"].tolist()

        for idx, t_id in enumerate(topic_ids):
            # Find index of highest score in the row
            best_match_idx = similarity_matrix[idx].argmax()
            best_score = similarity_matrix[idx][best_match_idx]
            best_label = taxonomy_labels[best_match_idx]

            results.append(
                {
                    "Topic_ID": t_id,
                     "LLM_Label": custom_labels[idx],
                    "Top_Words": discovered_texts[idx],
                    "Best_Match_Label": best_label,
                    "Similarity_Score": round(best_score, 4),
                    "Match_Type": "Strong" if best_score > 0.55 else "Weak/New",
                }
            )

        df_mapping = pd.DataFrame(results)
        df_mapping = df_mapping.sort_values(by="Similarity_Score", ascending=False)

        return df_mapping


class ExactMatcher:
    """
    Compares discovered BERTopic topics assignment with the HYPE assignments, i
    in different modalities
    """

    def __init__(self, df):
        self.df = df
        self.n_rows = len(df)

    def compute_exact_match_mono(self):

        count = 0

        mask = self.df.apply(
            lambda row: included_or_equal(
                row["taxonomy_label"], row["labels_list"]
            ),
            axis=1,
        )
        count = mask.sum()
        match = count / self.n_rows
        print("Exact Match: ", match)

        return match

        
    def compute_precision(self, mode="mono"):

        if mode=="mono":
            pred_column = "taxonomy_label"
        elif mode=="multi":
            pred_column = "taxonomy_labels_multi"
        else:
            print("Please provide a correct modality: mono/multi. Cuntinuing with mono")

        correct = 0
        predicted = 0

        for _, row in self.df.iterrows():
            y_pred = to_set(row[pred_column])
            y_true = to_set(row["labels_list"])

            correct += len(y_pred & y_true)
            predicted += len(y_pred)

        if predicted:
            precision = round(correct / predicted, 4) 
        else:
            precision = 0.0
        print(f"Precision using {mode} mode for topics: ", precision)

        return precision
    

    def compute_precision_translation(self, mode="mono"):

        if mode=="mono":
            pred_column = "taxonomy_label"
        elif mode=="multi":
            pred_column = "taxonomy_labels_multi"
        else:
            print("Please provide a correct modality: mono/multi. Cuntinuing with mono")

        correct = 0
        predicted = 0

        mask = self.df["detected_lang"] != "it"
        df_translated = self.df[mask]

        for _, row in df_translated.iterrows():
            y_pred = to_set(row[pred_column])
            y_true = to_set(row["labels_list"])

            correct += len(y_pred & y_true)
            predicted += len(y_pred)

        if predicted:
            precision = round(correct / predicted, 4) 
        else:
            precision = 0.0
        print(f"Precision on TRANSLATED reviews only, using {mode} mode for topics: ", precision)

        return precision
    




def to_set(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return set()
    if isinstance(x, list):
        return set(x)
    if isinstance(x, str):
        return {x}
    return set()


def included_or_equal(a, b):
    """
    Check if taxonomy_label (a) matches any of the labels (b).
    Returns True if a is equal to any element in b (exact match only).
    """
    # --- check a ---
    if a is None:
        return False

    # --- check b ---
    if b is None:
        return False

    # b is a numpy array
    if isinstance(b, np.ndarray):
        b = b.tolist()
    
    # case 1: both a and b dont have labels
    if a=="No Match (Outlier)" and b==[]:
        return True

    # case 2: b is a list - check for match in list
    if isinstance(b, list):
        return a in b

    # case 3: b is a string - check for match in string
    if isinstance(b, str):
        return a == b or a in b

    # fallback
    return False    


