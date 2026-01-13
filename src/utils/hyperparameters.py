import os
from itertools import product

import nltk
import pandas as pd
import wandb
from nltk.corpus import stopwords
from tqdm import tqdm

from src.evaluation.evaluation import calculate_coherence_metrics
from src.modeling.multilabel import MultiLabelModeler
from src.modeling.topic_modeler import TopicModeler
from src.preprocessing.cleaner import DataProcessor
from src.preprocessing.duplicate_remover import DuplicateRemover
from src.preprocessing.mwe import MWEExtractor
from src.preprocessing.sentiment_analyzer import SentimentEnsemble
from src.preprocessing.translation import TranslatorModule
from src.utils.config import cfg
from src.utils.logger import WandBLogger
from src.utils.utils import ensure_directories, seed_everything

UMAP_GRID = {
    "n_neighbors": [35, 50, 60],
    "n_components": [5],
    "min_dist": [0.0],
}

HDBSCAN_GRID = {
    "min_cluster_size": [30, 40],
    "min_samples": [1, 2],
}


def run_tuning():
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    print("=== HYPE BERTOPIC HYPERPARAMETER TUNING ===")
    # 1. SETUP & REPRODUCIBILITY
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    # 2. INITIALIZE WANDB
    # We use a distinct job_type so these runs don't mess up your BERTopic charts
    logger = None
    if cfg.get("project.wandb_logging"):
        logger = WandBLogger(
            job_type="hyperparameter_tuning", run_name="hype_tuning_run"
        )
        print("--> [WandB] Hyperparamter tuning run started.")

    wandb_table = None

    if cfg.get("project.wandb_logging"):
        wandb_table = wandb.Table(
            columns=[
                "n_neighbors",
                "n_components",
                "min_dist",
                "min_cluster_size",
                "min_samples",
                "n_topics",
                "outlier_pct",
                "silhouette",
                "topic_coherence",
                "topic_diversity",
                "outlier_pctNew",
                "silhouetteNew",
                "topic_coherenceNew",
                "topic_diversityNew",
            ]
        )

    # Create 'out/tuning' folder if it doesn't exist
    os.makedirs("out/tuning", exist_ok=True)
    ensure_directories(
        [
            cfg.get("paths.cache"),
            cfg.get("paths.output_tuning"),
            "./out/tuning/",
        ]
    )

    # Load Paths from Config
    data_path = cfg.get("paths.data")
    cache_path = cfg.get("paths.cache")
    final_cache_path = "./out/cache_preprocessing_full.pkl"
    use_cache = cfg.get("preprocessing.use_cache")

    loader = DataProcessor(data_path)
    df = None

    # --- CHECKPOINT LOGIC ---
    # Check if the cache with full preprocessing is available
    if use_cache and os.path.exists(final_cache_path):
        print(f"--> [Cache] Found cached file '{final_cache_path}'. Loading...")
        cache_obj = pd.read_pickle(final_cache_path)
        df = cache_obj["df"]
        docs = cache_obj["docs"]

        print(f"--> [Cache] Loaded {len(docs)} preprocessed documents from cache.")
    else:
        # Check if the cache with partial preprocessing is available
        if use_cache and os.path.exists(cache_path):
            print(f"--> [Cache] Found cached file '{cache_path}'. Loading...")
            df = pd.read_pickle(cache_path)
            print(f"--> [Cache] Loaded {len(df)} reviews from cache.")

            loader.df = df
        else:
            print("--> [Cache] No cache found. Running full preprocessing...")

            # 1. LOAD DATA
            df = loader.load_data()

            # 2. & 3. SOTA DETECTION & TRANSLATION (Batch Processed)
            translator = TranslatorModule(df)
            df = translator.detect_and_translate(text_col="review")

            # 4. TEXT CLEANING & EMOJI CONVERSION
            # Clean *before* sentiment analysis so emojis become text (e.g., ":thumbs_down:")
            loader.df = df
            df = loader.basic_cleaning(
                text_column="final_text", target_column="clean_text"
            )

            # 5. RE-CLASSIFY SENTIMENT (Ensemble)
            sentiment_engine = SentimentEnsemble()
            df = sentiment_engine.get_ensemble_sentiment(df, text_col="clean_text")

        # 6. FILTER DATASET (STRICTLY NEGATIVE)
        print("--> [Filter] Keeping ONLY Negative reviews for Topic Detection...")
        df = df[df["sentiment"] == "negative"].reset_index(drop=True)

        print(f"--> [Filter] {len(df)} negative reviews remaining.")

        loader.df = df

        # Junk Removal
        df = loader.remove_junk_reviews(column="clean_text")

        # 7. DEDUPLICATION (TF-IDF)
        deduplicator = DuplicateRemover()
        df = deduplicator.remove_duplicates(df, text_col="clean_text")

        # 8. MULTI-WORD EXPRESSIONS
        mwe_extractor = MWEExtractor(df=df)
        mwe_list = mwe_extractor.extract_mwe()
        mwe_list.to_excel("./data/mwe_list.xlsx", index=False)
        df = mwe_extractor.apply_mwe()

        # 9. STOPWORD REMOVAL COMPARISON

        nltk.download("stopwords", quiet=True)
        italian_stopwords = set(stopwords.words("italian"))

        # Load TF-IDF stopwords (domain-specific)
        tfidf_stopwords_path = "./data/tfidf_stopwords.txt"
        with open(tfidf_stopwords_path, "r", encoding="utf-8") as f:
            tfidf_stopwords = set(line.strip() for line in f if line.strip())

        print(f"[Stopwords] Italian: {len(italian_stopwords)}")
        print(f"[Stopwords] TF-IDF: {len(tfidf_stopwords)}")

        def remove_stopwords(text: str, stopword_set: set):
            return " ".join([w for w in text.split() if w.lower() not in stopword_set])

        # -----------------------------------------------------
        # CHOOSE ONE SETTING (comment / uncomment)
        # -----------------------------------------------------

        # A) NO stopword removal (baseline)
        docs = df["clean_text_mwe"].tolist()
        print("[Stopwords] CLASSIC: no stopword removal")

        # B) Italian stopwords only (classic NLP)
        # docs = [
        #     remove_stopwords(text, italian_stopwords)
        #     for text in df["clean_text_mwe"]
        # ]
        # print("[Stopwords] CLASSIC: Italian stopwords removed")

        # C) TF-IDF stopwords only (domain-driven)
        # docs = [
        #     remove_stopwords(text, tfidf_stopwords)
        #     for text in df["clean_text_mwe"]
        # ]
        # print("[Stopwords] CLASSIC: TF-IDF stopwords removed")

        # D) Italian − TF-IDF (delta)
        # delta_stopwords = tfidf_stopwords - italian_stopwords
        # docs = [
        #     remove_stopwords(text, delta_stopwords)
        #     for text in df["clean_text_mwe"]
        # ]
        # print("[Stopwords] DELTA:  TF-IDF minus Italian stopwords removed")

        # E) Italian + TF-IDF (delta)
        # union_stopwords = set(italian_stopwords) | set(tfidf_stopwords)
        # docs = [
        #    remove_stopwords(text, union_stopwords)
        #    for text in df["clean_text_mwe"]
        # ]
        # print("[Stopwords] UNION:  TF-IDF and Italian stopwords removed")

        os.makedirs("./out", exist_ok=True)
        pd.to_pickle({"df": df, "docs": docs}, final_cache_path)
        print(f"--> [Cache] Saved final preprocessing cache to {final_cache_path}")

    # 10. HYPERPARAMETER TUNING (BERTopic)
    print(f"--> [TUNING] Starting BERTopic hyperparameter tuning on {len(docs)} docs")

    results = []

    n_umap = len(list(product(*UMAP_GRID.values())))
    n_hdb = len(list(product(*HDBSCAN_GRID.values())))
    total_runs = n_umap * n_hdb

    print(f"--> [TUNING] Total configurations: {total_runs}")

    run_id = 0

    for umap_vals in tqdm(
        product(*UMAP_GRID.values()),
        total=n_umap,
        desc="UMAP grid",
    ):
        umap_params = dict(zip(UMAP_GRID.keys(), umap_vals))

        for hdb_vals in tqdm(
            product(*HDBSCAN_GRID.values()),
            total=n_hdb,
            desc="HDBSCAN grid",
            leave=False,
        ):
            hdb_params = dict(zip(HDBSCAN_GRID.keys(), hdb_vals))

            run_id += 1

            print("\n========================================")
            print(
                f"[RUN {run_id}/{total_runs}] UMAP={umap_params} | HDBSCAN={hdb_params}"
            )

            tm = TopicModeler(
                umap_params=umap_params,
                hdbscan_params=hdb_params,
            )

            model, topics, probs = tm.run(
                docs, architecture_name="umap_hdbscan", logger=None
            )

            n_topics = len(set(topics)) - (1 if -1 in topics else 0)

            # ---------------- EVALUATION ----------------

            embeddings = tm.embedding_model.encode(docs, show_progress_bar=False)
            multi_label_modeler = MultiLabelModeler(model, docs, topics, probs)
            results_df = multi_label_modeler.get_top3_topics_per_review(
                indices=None, top_words=5, alpha=0.85, min_abs_score=0.20, max_labels=3
            )
            results_df.to_excel("reviews_top3_topics.xlsx", index=False)
            print(f"Salvato {len(results_df)} review con top 3 topic")

            updated_topics = results_df["assigned_topic_primary"]

            df_tmp = df.copy()
            df_tmp["topic"] = topics

            n_outliers = len(df_tmp[df_tmp["topic"] == -1])
            outlier_perc = n_outliers / len(df_tmp)

            print(f"--> [Evaluation] Outliers: {n_outliers} ({outlier_perc:.2%})")

            n_outliersNew = len(results_df[results_df["assigned_topic_primary"] == -1])
            outlier_percNew = n_outliersNew / len(results_df)

            print(
                f"--> [Evaluation] NewOutliers: {n_outliersNew} ({outlier_percNew:.2%})"
            )

            scores = calculate_coherence_metrics(
                model,
                docs,
                embeddings,
                topics,
                embedding_model=tm.embedding_model,
                logger=None,
            )

            scoresWithLesstOutlier = calculate_coherence_metrics(
                model,
                docs,
                embeddings,
                updated_topics,
                embedding_model=tm.embedding_model,
                logger=None,
            )

            silhouette = scores.get("silhouette")
            topic_coherence = scores.get("topic_coherence")
            topic_diversity = scores.get("topic_diversity")

            silhouetteNew = scoresWithLesstOutlier.get("silhouette")
            topic_coherenceNew = scoresWithLesstOutlier.get("topic_coherence")
            topic_diversityNew = scoresWithLesstOutlier.get("topic_diversity")

            if cfg.get("project.wandb_logging"):
                wandb_table.add_data(
                    umap_params["n_neighbors"],
                    umap_params["n_components"],
                    umap_params["min_dist"],
                    hdb_params["min_cluster_size"],
                    hdb_params["min_samples"],
                    n_topics,
                    outlier_perc,
                    silhouette,
                    topic_coherence,
                    topic_diversity,
                    outlier_percNew,
                    silhouetteNew,
                    topic_coherenceNew,
                    topic_diversityNew,
                )

            results.append(
                {
                    **umap_params,
                    **hdb_params,
                    "n_topics": n_topics,
                    "outlier_pct": outlier_perc,
                    "silhouette": silhouette,
                    "topic_coherence": topic_coherence,
                    "topic_diversity": topic_diversity,
                    "outlier_pctNew": outlier_percNew,
                    "silhouetteNew": silhouetteNew,
                    "topic_coherenceNew": topic_coherenceNew,
                    "topic_diversityNew": topic_diversityNew,
                }
            )

            print(
                f"[CHECK] topics={n_topics} | "
                f"outliers={outlier_perc:.2%} | "
                f"outliersNew={outlier_percNew:.2%} | "
            )

            if run_id % 5 == 0:
                pd.DataFrame(results).to_excel(
                    "./out/tuning/results_partial.xlsx", index=False
                )

    results_df = pd.DataFrame(results)

    os.makedirs("./out/tuning", exist_ok=True)
    out_path = "./out/tuning/results_tuning.xlsx"
    results_df.to_excel(out_path, index=False)

    print(f"\n[SAVED] Hyperparameter tuning results → {out_path}")

    if cfg.get("project.wandb_logging"):
        print("--> [WandB] Run finished.")
        wandb.finish()


if __name__ == "__main__":
    run_tuning()
