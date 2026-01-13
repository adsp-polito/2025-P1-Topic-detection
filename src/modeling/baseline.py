import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import cfg


def calculate_baseline_topic_metrics(
    model, vectorizer, embedding_model, top_k=10, logger=None, model_name="baseline"
):
    """
    Computes:
    - Topic Coherence (embedding-based)
    - Topic Diversity
    For LDA / NMF models.
    """

    print(f"--> [Evaluation] Calculating metrics for {model_name}...")

    feature_names = vectorizer.get_feature_names_out()
    topic_words = []

    # Extract top-k words per topic
    for topic in model.components_:
        top_indices = topic.argsort()[: -top_k - 1 : -1]
        words = [feature_names[i] for i in top_indices]
        topic_words.append(words)

    # Topic Coherence
    topic_coherences = []

    for words in topic_words:
        if len(words) < 2:
            continue

        embeddings = embedding_model.encode(words)
        sim_matrix = cosine_similarity(embeddings)
        mean_sim = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
        topic_coherences.append(mean_sim)

    topic_coherence = float(np.mean(topic_coherences)) if topic_coherences else 0.0
    print(f"    Topic Coherence: {topic_coherence:.4f}")

    # Topic Diversity
    all_words = [w for topic in topic_words for w in topic]
    topic_diversity = len(set(all_words)) / len(all_words) if all_words else 0.0
    print(f"    Topic Diversity: {topic_diversity:.4f}")

    # Wandb
    if cfg.get("project.wandb_logging") and logger is not None:
        logger.log_metrics(
            {
                f"{model_name}_topic_coherence": topic_coherence,
                f"{model_name}_topic_diversity": topic_diversity,
            }
        )

    return {
        "topic_coherence": topic_coherence,
        "topic_diversity": topic_diversity,
    }


class BaselineModeler:
    """
    Runs standard baseline models (LDA and NMF) for comparison against BERTopic.
    """

    def __init__(self):
        self.conf = cfg.get("baselines")
        self.n_topics = self.conf.get("n_topics", 15)
        self.n_top_words = self.conf.get("n_top_words", 10)
        self.random_state = self.conf.get("random_state", 42)

        """
        # Setup Stopwords (Matching the main TopicModeler)
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words("italian")
        extras = cfg.get("topic_modeling.extra_stopwords", [])
        self.stop_words.extend(extras)
        """

    def run(self, docs: list) -> pd.DataFrame:
        """
        Runs both LDA and NMF on the provided documents.
        Returns a DataFrame containing the top words for each topic for both models.
        """
        print(f"--> [Baselines] Running LDA and NMF with {self.n_topics} topics...")

        results = []

        # --- 1. LDA (Latent Dirichlet Allocation) ---
        # LDA requires raw counts (CountVectorizer)
        # max_df=0.95: ignore words appearing in >95% of docs (too common)
        # min_df=2: ignore words appearing in <2 docs
        print("    [LDA] Vectorizing...")
        tf_vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            # stop_words=self.stop_words
        )
        tf = tf_vectorizer.fit_transform(docs)

        print("    [LDA] Fitting model...")
        lda = LatentDirichletAllocation(
            n_components=self.n_topics, random_state=self.random_state, n_jobs=-1
        )
        lda.fit(tf)

        self._extract_topics(lda, tf_vectorizer, "LDA", results)

        # --- 2. NMF (Non-negative Matrix Factorization) ---
        # NMF works best with normalized data (TF-IDF)
        print("    [NMF] Vectorizing...")
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            # stop_words=self.stop_words
        )
        tfidf = tfidf_vectorizer.fit_transform(docs)

        print("    [NMF] Fitting model...")
        nmf = NMF(
            n_components=self.n_topics, random_state=self.random_state, init="nndsvd"
        )
        nmf.fit(tfidf)

        self._extract_topics(nmf, tfidf_vectorizer, "NMF", results)

        # Create DataFrame
        df_res = pd.DataFrame(results)
        print("--> [Baselines] Finished.")

        # Assign topics to documents
        print("    [LDA] Assigning topics to documents...")
        lda_docs_df = self.assign_topics_to_docs(lda, tf_vectorizer, docs, "LDA")

        print("    [NMF] Assigning topics to documents...")
        nmf_docs_df = self.assign_topics_to_docs(nmf, tfidf_vectorizer, docs, "NMF")

        return {
            "topics_df": df_res,
            "lda_docs_topics_df": lda_docs_df,
            "lda_model": lda,
            "lda_vectorizer": tf_vectorizer,
            "nmf_docs_topics_df": nmf_docs_df,
            "nmf_model": nmf,
            "nmf_vectorizer": tfidf_vectorizer,
        }

    def _extract_topics(self, model, vectorizer, model_name, results_list):
        feature_names = vectorizer.get_feature_names_out()

        for topic_idx, topic in enumerate(model.components_):
            # Get indices of top words
            top_indices = topic.argsort()[: -self.n_top_words - 1 : -1]
            top_words = [feature_names[i] for i in top_indices]

            results_list.append(
                {
                    "Model": model_name,
                    "Topic_ID": topic_idx,
                    "Top_Words": ", ".join(top_words),
                }
            )

    def assign_topics_to_docs(self, model, vectorizer, docs, model_name):
        """
        Assigns a primary topic to each document.
        Returns a DataFrame with review-level topic assignments.
        """

        X = vectorizer.transform(docs)
        doc_topic_dist = model.transform(X)

        rows = []

        for i, topic_scores in enumerate(doc_topic_dist):
            topic_id = int(np.argmax(topic_scores))
            score = float(topic_scores[topic_id])

            rows.append(
                {
                    "review_idx": i,
                    "document": docs[i],
                    "model": model_name,
                    "assigned_topic": topic_id,
                    "topic_score": score,
                }
            )

        return pd.DataFrame(rows)
