import nltk
import wandb
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from config import cfg
from logger import WandBLogger


class TopicModeler:
    """
    BERTopic wrapper with SOTA Italian configurations and WandB logging.
    """

    def __init__(self):
        self.config = cfg.get("topic_modeling")
        self.project_name = cfg.get("project.name")
        self.embedding_model = None
        self.topic_model = None

        # Setup Italian Stopwords
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words("italian")
        # Add project-specific noise words
        extras = self.config.get("extra_stopwords", [])
        self.stop_words.extend(extras)

    def run(self, docs: list, run_name: str = "bertopic_run"):
        """
        Executes the topic modeling pipeline.
        """
        model_type = self.config.get("model_choice")
        print(f"--> [BERTopic] Starting run: {run_name} using [{model_type}] config")

        # 1. Initialize WandB
        logger = WandBLogger()

        # 2. Select and Store Embedding Model
        # We store it in 'self' to access it later for evaluation
        model_name = self.config.get("embedding_model")
        print(f"    Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Encode embeddings
        print("    Encoding embeddings...")
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)

        # 3. Dimensionality Reduction (UMAP)
        umap_conf = self.config.get("umap")
        umap_model = UMAP(
            n_neighbors=umap_conf["n_neighbors"],
            n_components=umap_conf["n_components"],
            min_dist=umap_conf["min_dist"],
            metric=umap_conf["metric"],
            random_state=cfg.get("project.seed", 42),
        )

        # 4. Clustering (HDBSCAN)
        hdb_conf = self.config.get("hdbscan")
        hdbscan_model = HDBSCAN(
            min_cluster_size=hdb_conf["min_cluster_size"],
            metric="euclidean",
            cluster_selection_method=hdb_conf["cluster_selection_method"],
            prediction_data=True,
        )

        # 5. Vectorizer (Topic Representation) min_df=2 ensures words appear in at least 2 topics (or docs), preventing the crash if topics are few.
        vectorizer_model = CountVectorizer(
            stop_words=self.stop_words, min_df=self.config.get("min_df", 2)
        )

        # 6. Initialize and Fit BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language=self.config.get("language", "multilingual"),
            calculate_probabilities=True,
            verbose=True,
        )

        print("--> [BERTopic] Fitting model...")
        topics, probs = self.topic_model.fit_transform(docs, embeddings=embeddings)

        # 7. Logging to WandB
        freq = self.topic_model.get_topic_info()
        n_topics = len(freq) - 1
        print(f"--> [BERTopic] Generated {n_topics} topics.")

        logger.log_metrics({"n_topics": n_topics, "n_docs": len(docs)})

        # Log Topic Info Table
        logger.log_plot("topic_info", wandb.Table(dataframe=freq), plot_type="table")

        # --- ADVANCED PLOTS ---
        try:
            # A. Intertopic Distance
            logger.log_plot("plot_intertopic", self.topic_model.visualize_topics())

            # B. Bar Chart (Top 15)
            logger.log_plot(
                "plot_barchart", self.topic_model.visualize_barchart(top_n_topics=15)
            )

            # C. Hierarchy (Tree) - Addresses Task 3b
            # Limit to top 50 topics to keep it readable
            fig_hierarchy = self.topic_model.visualize_hierarchy(top_n_topics=50)
            logger.log_plot("plot_hierarchy", fig_hierarchy)

            # D. Similarity Heatmap - Addresses Topic Separation
            # We filter out Topic -1 (Noise) for the heatmap
            if n_topics > 1:
                fig_heatmap = self.topic_model.visualize_heatmap(
                    n_clusters=n_topics - 1
                )
                logger.log_plot("plot_heatmap", fig_heatmap)

        except Exception as e:
            print(f"--> [Warning] Could not log plots: {e}")

        return self.topic_model, topics, probs

    def save_model(self, path: str):
        """Saves the trained model to disk."""
        if self.topic_model:
            # Safetensors is preferred for security and speed
            self.topic_model.save(path, serialization="safetensors", save_ctfidf=True)
            print(f"--> [BERTopic] Model saved locally to {path}")
        else:
            print("--> [Error] No model to save. Run .run() first.")
