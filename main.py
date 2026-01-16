import ast
import os
import random

import nltk
import numpy as np
import pandas as pd
import wandb
from nltk.corpus import stopwords

from src.evaluation.evaluation import (
    ExactMatcher,
    HierarchyAnalyzer,
    TaxonomyMapper,
    calculate_coherence_metrics,
)
from src.modeling.multilabel import MultiLabelModeler, map_topics_to_taxonomy_list
from src.modeling.topic_modeler import TopicModeler
from src.preprocessing.cleaner import DataProcessor
from src.preprocessing.duplicate_remover import DuplicateRemover
from src.preprocessing.mwe import MWEExtractor
from src.preprocessing.sentiment_analyzer import SentimentEnsemble
from src.preprocessing.translation import TranslatorModule
from src.utils.config import cfg
from src.utils.logger import WandBLogger
from src.utils.utils import ensure_directories, load_taxonomy, seed_everything


def main():
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    print("=== HYPE TOPIC DETECTION PIPELINE ===")

    # Load pipeline settings from config
    definedArchitecture_name = cfg.get("pipeline.architecture", "umap_hdbscan")
    definedRunningType_name = cfg.get("pipeline.run_type", "unsupervised")
    stopword_strategy = cfg.get("pipeline.stopword_strategy", "none")

    print(f"--> [Config] Architecture: {definedArchitecture_name}")
    print(f"--> [Config] Run Type: {definedRunningType_name}")
    print(f"--> [Config] Stopword Strategy: {stopword_strategy}")

    main_logger = None

    # INITIALIZE WANDB (Global Run)
    if cfg.get("project.wandb_logging"):
        main_logger = WandBLogger(
            job_type="full_pipeline", run_name="hype_analysis_UnionStopWord"
        )
        print("--> [WandB] Pipeline run started.")

    # Create 'out' folder if it doesn't exist
    os.makedirs("out", exist_ok=True)
    ensure_directories(
        [
            cfg.get("paths.cache"),
            cfg.get("paths.output_topics"),
            cfg.get("paths.output_mapping"),
            "./out/bertopic_model/",
        ]
    )

    # Load Paths from Config
    data_path = cfg.get("paths.data")
    cache_path = cfg.get("paths.cache")
    use_cache = cfg.get("preprocessing.use_cache")

    loader = DataProcessor(data_path)
    df = None

    # --- CHECKPOINT LOGIC ---
    if use_cache and os.path.exists(cache_path):
        print(f"--> [Cache] Found cached file '{cache_path}'. Loading...")
        cache_obj = pd.read_pickle(cache_path)

        # Handle both dictionary format (from hyperparameters.py) and DataFrame format
        if isinstance(cache_obj, dict):
            df = cache_obj["df"]
            print(f"--> [Cache] Loaded {len(df)} reviews from cache (dict format).")
        else:
            df = cache_obj
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
        df = loader.basic_cleaning(text_column="final_text", target_column="clean_text")

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

        # SAVE CACHE
        print(f"--> [Cache] Saving to '{cache_path}'...")
        df.to_pickle(cache_path)

    # --- EDA LOGGING (Sentiment Distribution) ---
    if cfg.get("project.wandb_logging"):
        print("--> [WandB] Logging Sentiment Distribution (EDA)...")
        eda_logger = WandBLogger()

        # Prepare Data
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]

        # Log Table
        table = wandb.Table(dataframe=sent_counts)
        eda_logger.log_plot("sentiment_data", table, plot_type="table")

        # Log Bar Chart
        bar_plot = wandb.plot.bar(
            table, "sentiment", "count", title="Sentiment Distribution"
        )
        eda_logger.log_plot("sentiment_dist_plot", bar_plot, plot_type="chart")

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

    texts = df["clean_text_mwe"].tolist()

    # All possible strategies: "none", "italian", "tfidf", "delta", "union"

    if stopword_strategy == "none":
        docs = texts
        print("[Stopwords] NONE: no stopword removal")

    elif stopword_strategy == "italian":
        docs = [remove_stopwords(text, italian_stopwords) for text in texts]
        print("[Stopwords] ITALIAN: Italian stopwords removed")

    elif stopword_strategy == "tfidf":
        docs = [remove_stopwords(text, tfidf_stopwords) for text in texts]
        print("[Stopwords] TF-IDF: TF-IDF stopwords removed")

    elif stopword_strategy == "delta":
        delta_stopwords = set(tfidf_stopwords) - set(italian_stopwords)
        docs = [remove_stopwords(text, delta_stopwords) for text in texts]
        print("[Stopwords] DELTA: TF-IDF minus Italian stopwords removed")

    elif stopword_strategy == "union":
        union_stopwords = set(italian_stopwords) | set(tfidf_stopwords)
        docs = [remove_stopwords(text, union_stopwords) for text in texts]
        print("[Stopwords] UNION: Italian + TF-IDF stopwords removed")

    else:
        raise ValueError(
            f"Unknown stopwords_strategy '{stopword_strategy}'. "
            "Choose among: none, italian, tfidf, delta, union."
        )

    # 10. TOPIC DETECTION (BERTopic)
    print(f"--> [Topic Modeling] Starting run on {len(docs)} negative reviews...")

    if len(docs) > 10:
        # PREPARE SEMI-SUPERVISED y

        # Adjust the labels columns
        if "labels" in df.columns:
            df["labels_list"] = df["labels"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        else:
            df["labels_list"] = [[] for _ in range(len(df))]

        # Consider only mono-label reviews (take the first label)
        df_single = df[df["labels_list"].apply(len) == 1].copy()
        df_single["label"] = df_single["labels_list"].apply(lambda x: x[0])

        # Keep only labels with >= min_label_examples examples
        min_label_examples = cfg.get("pipeline.min_label_examples", 25)
        label_counts = df_single["label"].value_counts()
        valid_labels = label_counts[label_counts >= min_label_examples].index.tolist()
        df_single = df_single[df_single["label"].isin(valid_labels)]

        print(f"Valid supervised labels: {len(valid_labels)}")

        # Sample supervised_sample_ratio of labeled data
        supervised_ratio = cfg.get("pipeline.supervised_sample_ratio", 0.15)
        n_supervised = int(supervised_ratio * len(df_single))
        random.seed(seed_val)
        supervised_idx = random.sample(list(df_single.index), n_supervised)

        # Numerical encoding for labels
        label2id = {lbl: i for i, lbl in enumerate(sorted(valid_labels))}

        # Document of all -1, since it indicate unsupervised
        y = np.full(len(docs), -1, dtype=int)

        # Populate thw document with supervised labels
        for idx in supervised_idx:
            lbl = df.loc[idx, "labels_list"][0]
            y[idx] = label2id[lbl]

        print(f"Supervised docs: {(y != -1).sum()}")
        print(f"Unsupervised docs: {(y == -1).sum()}")

        # Topic modeling
        tm = TopicModeler()

        model, topics, probs, topic_descriptions = tm.run(
            docs,
            y=y,
            run_name="bertopic_run",
            architecture_name=definedArchitecture_name,
            type_name=definedRunningType_name,
            logger=main_logger,
        )

        # MULTILABELING
        if definedArchitecture_name == "umap_hdbscan":
          multi_label_modeler = MultiLabelModeler(model, docs, topics, probs)
          results_df = multi_label_modeler.get_top3_topics_per_review(
              indices=None, top_words=5, alpha=0.85, min_abs_score=0.20, max_labels=3
          )
          results_df.to_excel("./out/reviews_top3_topics.xlsx", index=False)
          print(f"Saved {len(results_df)} reviews with 3 top topics")

          updated_topics = results_df["assigned_topic_primary"]
          df["updated_topic"] = updated_topics
          df["multi_topics"] = results_df["multi_topics"]

        # Save Basic Results
        if isinstance(topics, tuple):
            topics = topics[0]
        df["topic"] = topics

        n_outliers = len(df[df["topic"] == -1])
        outlier_perc = (n_outliers / len(df)) * 100
        print(
            f"--> [Evaluation] Outliers (Topic -1): {n_outliers} ({outlier_perc:.2f}%)"
        )

        if definedArchitecture_name == "umap_hdbscan":
          n_outliers = len(df[df["updated_topic"] == -1])
          outlier_perc = (n_outliers / len(df)) * 100
          print(
              f"--> [Evaluation] Updated outliers after outlier reduction (Topic -1): {n_outliers} ({outlier_perc:.2f}%)"
          )

        if cfg.get("project.wandb_logging") and main_logger:
            main_logger.log_metrics({"outlier_percentage": outlier_perc})

        # If outliers > 40%, warn the user
        if outlier_perc > 40:
            print(
                "    [WARNING] High outlier count! Consider lowering 'min_cluster_size' in config.yaml"
            )

        out_file_topics = cfg.get("paths.output_topics")
        os.makedirs(os.path.dirname(out_file_topics), exist_ok=True)

        df.to_excel(out_file_topics, index=False)
        print(f"--> [Done] Basic results saved to {out_file_topics}")

        if main_logger:
            main_logger.log_artifact(
                out_file_topics,
                "dataset",
                "labeled_reviews",
                "Negative reviews with Topic IDs",
            )

        model_save_path = "./out/bertopic_model"
        tm.save_model(model_save_path)

        if main_logger:
            main_logger.log_artifact(
                model_save_path,
                "model",
                "hype_bertopic_model",
                "BERTopic model trained on negative reviews",
            )

        # --- EVALUATION PHASE ---

        # A. Quantitative Metric (Silhouette Score)
        # Re-encode docs to get embeddings (fast with cached model)
        print("--> [Evaluation] Generating embeddings for scoring...")
        embeddings = tm.embedding_model.encode(docs, show_progress_bar=False)

        # Calculate Coherence (Logs to WandB if enabled)
        if definedArchitecture_name == "umap_hdbscan":
          topics = updated_topics

        scores = calculate_coherence_metrics(
            model,
            docs,
            embeddings,
            topics,
            embedding_model=tm.embedding_model,
            logger=main_logger,
        )

        # B. Taxonomy Mapping
        tax_path = cfg.get("paths.taxonomy")

        taxonomy_df = load_taxonomy(tax_path)
        if not taxonomy_df.empty:
            mapper = TaxonomyMapper(embedding_model=tm.embedding_model)

            mapping_df = mapper.map_topics_to_taxonomy(
                model, taxonomy_df, topic_descriptions
            )
            print(mapping_df.head())

            out_file_map = cfg.get("paths.output_mapping")
            mapping_df = mapping_df.replace(
                {
                    "\r": "",
                    "\n": "",
                    "_x00d_": "",
                    "_x00d": "",
                },
                regex=True,
            )
            mapping_df.to_excel(out_file_map, index=False)
            print(f"--> [Done] Taxonomy comparison saved to {out_file_map}")

            # Add best matching label to the main dataframe with topics
            # Create a mapping dictionary: Topic_ID -> Best_Match_Label
            topic_to_label = dict(
                zip(mapping_df["Topic_ID"], mapping_df["Best_Match_Label"])
            )

            # Add the label column to the main dataframe
            if definedArchitecture_name == "umap_hdbscan":
              df["taxonomy_label"] = df["updated_topic"].map(topic_to_label)
              df["taxonomy_labels_multi"] = df["multi_topics"].apply(
                lambda x: map_topics_to_taxonomy_list(x, topic_to_label)
                )
            else:
              df["taxonomy_label"] = df["topic"].map(topic_to_label)
            
            df["taxonomy_label"] = df["taxonomy_label"].apply(
                lambda x: [] if pd.isna(x) else x
              )

            # Re-save the updated dataframe with taxonomy labels
            final_path = "./out/resultswithtaxonomy.xlsx"
            df = df.replace(
                {
                    "\r": "",
                    "\n": "",
                    "_x00d_": "",
                    "_x00d": "",
                },
                regex=True,
            )
            df.to_excel(final_path, index=False)

            print(
                f"--> [Done] Updated results with taxonomy labels saved to {final_path}"
            )

            # Match old labels to newer ones to ensure consistency
            mapping_df_old_to_new = taxonomy_df[taxonomy_df["Old_Label"].notna()][
                ["Old_Label", "Label"]
            ]
            old_to_new = dict(
                zip(mapping_df_old_to_new["Old_Label"], mapping_df_old_to_new["Label"])
            )

            def convert_tags(tag_list):
                new_list = []
                for tag in list(tag_list):
                    if tag in old_to_new:
                        new_list.append(old_to_new[tag])
                    else:
                        new_list.append(tag)
                return new_list

            df["labels_list"] = df["labels_list"].apply(convert_tags)

            # ExactMatch count
            matcher = ExactMatcher(df)
            exact_match = matcher.compute_exact_match_mono()
            precision_mono = matcher.compute_precision(mode="mono")
            predcision_silhouette_translations_mono = (
                matcher.compute_precision_silhouette_translation(
                    embeddings, topics, mode="mono"
                )
            )
            if definedArchitecture_name == "umap_hdbscan":
              precision_multi = matcher.compute_precision(mode="multi")
              predcision_silhouette_translations_multi = (
                matcher.compute_precision_silhouette_translation(
                    embeddings, topics, mode="multi"
                )
            )
            
            if main_logger:
                main_logger.log_artifact(
                    out_file_map,
                    "dataset",
                    "taxonomy_mapping",
                )
        else:
            print("--> [Warning] No taxonomy loaded. Skipping mapping.")

        # 11. HIERARCHICAL TOPIC DETECTION (Task 3b)
        print("\n--> [Task 3b] Running Hierarchical Topic Detection...")

        # Initialize the analyzer
        hierarchy_analyzer = HierarchyAnalyzer(model, docs)

        # Compute and Save Artifacts (Tree, Plot, Data)
        hierarchy_analyzer.compute_hierarchy()
        hierarchy_analyzer.save_artifacts("./out/hierarchy")

        # Compare with the provided Taxonomy (if it exists)
        # We check if taxonomy_df was loaded earlier
        if "taxonomy_df" in locals() and not taxonomy_df.empty:
            hierarchy_analyzer.compare_with_taxonomy(taxonomy_df)

    if cfg.get("project.wandb_logging"):
        print("--> [WandB] Run finished.")
        wandb.finish()


if __name__ == "__main__":
    main()
