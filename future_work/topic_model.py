from typing import List, Optional

import pandas as pd
import wandb
from bertopic import BERTopic


class TopicDetector:
    """
    Wrapper for BERTopic with WandB experiment tracking.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Dictionary containing hyperparameters.
                    Keys: 'min_topic_size', 'language', 'embedding_model'
        """
        self.config = config
        self.topics: Optional[List[int]] = None
        self.probs: Optional[List[float]] = None

        # Initialize WandB run
        # We use reinit=True so we can run multiple experiments in one script if needed
        self.run = wandb.init(
            project="HYPE-Topic-Detection", config=config, reinit=True
        )

        print(f"--> [WandB] Run initialized: {self.run.name}")

        # Initialize BERTopic
        # Since we translated everything to Italian, we use a specific Italian model.
        self.model = BERTopic(
            language="italian",
            embedding_model=config.get(
                "embedding_model", "nickprock/sentence-bert-base-italian-uncased"
            ),
            min_topic_size=config.get("min_topic_size", 30),
            verbose=True,
            calculate_probabilities=False,  # Set True only if soft-clustering is needed
        )

    def fit(self, documents: List[str]) -> None:
        """
        Fits the model to the documents and logs metrics/plots to WandB.
        """

        print(
            "--> [Model] Fitting BERTopic model (this includes embedding & clustering)..."
        )

        self.topics, self.probs = self.model.fit_transform(documents)

        # --- METRICS CALCULATION ---
        topic_info = self.model.get_topic_info()

        # -1 is the outlier cluster
        n_topics = len(topic_info) - 1

        # Count outliers safely
        if -1 in topic_info["Topic"].values:
            n_outliers = topic_info.loc[topic_info["Topic"] == -1, "Count"].values[0]
        else:
            n_outliers = 0

        dataset_size = len(documents)
        outlier_pct = (n_outliers / dataset_size) * 100

        print("--> [Model] Training Complete.")
        print(f"    Topics Found: {n_topics}")
        print(f"    Outliers: {n_outliers} ({outlier_pct:.2f}%)")

        # --- WANDB LOGGING ---
        # 1. Log scalar metrics
        wandb.log(
            {
                "n_topics": n_topics,
                "n_outliers": n_outliers,
                "outlier_percentage": outlier_pct,
            }
        )

        # 2. Log Interactive Plots
        # Intertopic Distance Map (Essential for 'separation' check)
        fig_dist = self.model.visualize_topics()
        wandb.log({"intertopic_distance": fig_dist})

        # Barchart (Essential for 'what topics emerge?' check)
        fig_bar = self.model.visualize_barchart(top_n_topics=15)
        wandb.log({"topic_word_scores": fig_bar})

        # Heatmap (Optional: shows similarity between topics)
        try:
            fig_heat = self.model.visualize_heatmap()
            wandb.log({"topic_heatmap": fig_heat})
        except:
            pass  # Fails if too few topics

        print("--> [WandB] Metrics and Visualizations logged to dashboard.")

    def get_topic_overview(self) -> pd.DataFrame:
        """Returns the dataframe with topic info."""
        return self.model.get_topic_info()

    def save_model(self, path: str = "hype_bertopic_model"):
        """Saves the trained model locally."""
        self.model.save(path)
        print(f"--> [Model] Saved locally to {path}")

        # Optional: Save model artifact to WandB
        artifact = wandb.Artifact("model", type="model")
        artifact.add_dir(path)
        self.run.log_artifact(artifact)

    def finish_run(self):
        """Closes the WandB connection."""
        self.run.finish()
