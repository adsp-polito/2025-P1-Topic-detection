import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import cfg


class BaselineModeler:
    """
    Runs standard baseline models (LDA and NMF) for comparison against BERTopic.
    """

    def __init__(self):
        self.conf = cfg.get("baselines")
        self.n_topics = self.conf.get("n_topics", 10)
        self.n_top_words = self.conf.get("n_top_words", 15)
        self.random_state = self.conf.get("random_state", 42)

        # Setup Stopwords (Matching the main TopicModeler)
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words("italian")
        extras = cfg.get("topic_modeling.extra_stopwords", [])
        self.stop_words.extend(extras)

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
            max_df=0.95, min_df=2, stop_words=self.stop_words
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
            max_df=0.95, min_df=2, stop_words=self.stop_words
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
        return df_res

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
