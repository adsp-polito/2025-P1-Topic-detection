import nltk
import wandb
from bertopic import BERTopic
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class TopicModeler:
    """
    BERTopic wrapper with SOTA Italian configurations and WandB logging.
    """

    def __init__(self, project_name="hype-topic-detection"):
        self.project_name = project_name
        self.embedding_model = None

        # Setup Italian Stopwords
        nltk.download("stopwords", quiet=True)
        self.stop_words = stopwords.words("italian")
        # Add project-specific noise words
        self.stop_words.extend(
            [
                "app",
                "applicazione",
                "hype",
                "solo",
                "sempre",
                "fatto",
                "fare",
                "avere",
                "essere",
                "paga",
            ]
        )

    def run(
        self,
        docs: list,
        run_name: str = "bertopic_run",
        model_type: str = "multilingual",
    ):
        """
        Executes the topic modeling pipeline.

        Args:
            docs: List of strings (reviews)
            run_name: Name for the WandB run
            model_type: 'multilingual' or 'italian_social'
        """
        print(f"--> [BERTopic] Starting run: {run_name} using [{model_type}] model")

        # 1. Initialize WandB
        run = wandb.init(project=self.project_name, name=run_name, job_type="modeling")
        wandb.config.update({"model_type": model_type})

        # 2. Select and Store Embedding Model
        # CRITICAL CHANGE: We store it in 'self' to access it later for evaluation
        if model_type == "italian_social":
            print("    Loading Italian Social model (AlBERTo)...")
            model_name = (
                "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
            )
        else:
            print("    Loading SOTA Multilingual model (MiniLM)...")
            model_name = "paraphrase-multilingual-MiniLM-L12-v2"

        self.embedding_model = SentenceTransformer(model_name)

        # Encode embeddings
        print("    Encoding embeddings...")
        embeddings = self.embedding_model.encode(docs, show_progress_bar=True)

        # 3. Dimensionality Reduction (UMAP)
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        # 4. Clustering (HDBSCAN)
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # 5. Vectorizer (Topic Representation)
        vectorizer_model = CountVectorizer(stop_words=self.stop_words, min_df=5)

        # 6. Initialize and Fit BERTopic
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            language="italian" if model_type == "italian_social" else "multilingual",
            calculate_probabilities=True,
            verbose=True,
        )

        print("--> [BERTopic] Fitting model...")
        topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

        # 7. Logging to WandB
        freq = topic_model.get_topic_info()
        n_topics = len(freq) - 1
        print(f"--> [BERTopic] Generated {n_topics} topics.")

        wandb.log(
            {
                "n_topics": n_topics,
                "n_docs": len(docs),
                "topic_info": wandb.Table(dataframe=freq),
            }
        )

        try:
            wandb.log(
                {"intertopic_map": wandb.Html(topic_model.visualize_topics().to_html())}
            )
            wandb.log(
                {
                    "barchart": wandb.Html(
                        topic_model.visualize_barchart(top_n_topics=10).to_html()
                    )
                }
            )
        except Exception as e:
            print(f"--> [Warning] Could not log plots: {e}")

        run.finish()
        return topic_model, topics, probs
