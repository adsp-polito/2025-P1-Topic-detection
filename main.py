import os

import pandas as pd
import random
import numpy as np
import wandb
import ast
import nltk

from cleaner import DataProcessor
from config import cfg
from duplicate_remover import DuplicateRemover
from evaluation import TaxonomyMapper, calculate_coherence_metrics
from logger import WandBLogger
from mwe import MWEExtractor
from sentiment_analyzer import SentimentEnsemble
from topic_modeler import TopicModeler
from translation import TranslatorModule
from utils import ensure_directories, load_taxonomy, seed_everything, save_reviews_with_topic_probabilities
from nltk.corpus import stopwords
from multilabel import get_top3_topics_per_review


def main():
    seed_val = cfg.get("project.seed", 42)
    seed_everything(seed_val)

    print("=== HYPE TOPIC DETECTION PIPELINE ===")

    #EVALUATE IF MOVE IT TO config.yaml

    #["umap_hdbscan", "kernelpca_spectral", "kernelpca_kmeans", "umap_spectral", "umap_kmeans"]
    definedArchitecture_name="umap_hdbscan"
    #["unsupervised", "guided", "semi_supervised"]
    definedRunningType_name= "unsupervised"
    #["none", "italian", "tfidf", "delta", "union"]
    stopword_strategy= "none"

    main_logger=None

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
    final_cache_path = "./out/cache_preprocessing_full.pkl"

    loader = DataProcessor(data_path)
    df = None

    # --- CHECKPOINT LOGIC ---
    if use_cache and os.path.exists(final_cache_path):
        print(f"--> [Cache] Found cached file '{final_cache_path}'. Loading...")
        df = pd.read_pickle(final_cache_path)
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
        print(f"--> [Cache] Saving to '{final_cache_path}'...")
        df.to_pickle(final_cache_path)

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
        return " ".join(
            [w for w in text.split() if w.lower() not in stopword_set]
        )
    
    texts = df["clean_text_mwe"].tolist()

    if stopword_strategy == "none":
        docs = texts
        print("[Stopwords] NONE: no stopword removal")

    elif stopword_strategy == "italian":
        docs = [
            remove_stopwords(text, italian_stopwords)
            for text in texts
        ]
        print("[Stopwords] ITALIAN: Italian stopwords removed")

    elif stopword_strategy == "tfidf":
        docs = [
            remove_stopwords(text, tfidf_stopwords)
            for text in texts
        ]
        print("[Stopwords] TF-IDF: TF-IDF stopwords removed")

    elif stopword_strategy == "delta":
        delta_stopwords = set(tfidf_stopwords) - set(italian_stopwords)
        docs = [
            remove_stopwords(text, delta_stopwords)
            for text in texts
        ]
        print("[Stopwords] DELTA: TF-IDF minus Italian stopwords removed")

    elif stopword_strategy == "union":
        union_stopwords = set(italian_stopwords) | set(tfidf_stopwords)
        docs = [
            remove_stopwords(text, union_stopwords)
            for text in texts
        ]
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

        #Adjust the labels columns
        if "labels" in df.columns:
            df["labels_list"] = df["labels"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        else:
            df["labels_list"] = [[] for _ in range(len(df))]

        #Consider only mono-label reviews (take the first label)
        df_single = df[df["labels_list"].apply(len) == 1].copy()
        df_single["label"] = df_single["labels_list"].apply(lambda x: x[0])

        # Keep only labels with >= 25 examples
        label_counts = df_single["label"].value_counts()
        valid_labels = label_counts[label_counts >= 25].index.tolist()
        df_single = df_single[df_single["label"].isin(valid_labels)]

        print(f"Valid supervised labels: {len(valid_labels)}")

        # Sample 15%
        n_supervised = int(0.15 * len(df_single))
        random.seed(seed_val)
        supervised_idx = random.sample(list(df_single.index), n_supervised)

        #Numerical encoding for labels
        label2id = {lbl: i for i, lbl in enumerate(sorted(valid_labels))}

        #Document of all -1, since it indicate unsupervised
        y = np.full(len(docs), -1, dtype=int)

        #Populate thw document with supervised labels
        for idx in supervised_idx:
            lbl = df.loc[idx, "labels_list"][0]
            y[idx] = label2id[lbl]


        print(f"Supervised docs: {(y != -1).sum()}")
        print(f"Unsupervised docs: {(y == -1).sum()}")


        tm = TopicModeler()

        model, topics, probs = tm.run(
            docs,
            y=y,
            run_name="bertopic_run",
            architecture_name=definedArchitecture_name,
            type_name=definedRunningType_name,
            logger=main_logger,
        )

        output_probs_path = "./out/reviews_topic_probabilities.xlsx"

        save_reviews_with_topic_probabilities(
            docs=docs,
            topics=topics,
            probs=probs,
            output_path=output_probs_path,
            top_k=3   
        )
        

        # 3 TOPICS FOR REVIEW
        if (definedArchitecture_name=="umap_hdbscan"):
          results_df = get_top3_topics_per_review(model, docs, topics, probs)
          results_df.to_csv("reviews_top3_topics.csv", index=False)
          df["multi_topics"]=results_df["multi_topics"]
          print(f"Salvato {len(results_df)} review con top 3 topic")

        # Save Basic Results
        if isinstance(topics, tuple):
           topics = topics[0]
        df["topic"] = topics


        n_outliers = len(df[df["topic"] == -1])
        outlier_perc = (n_outliers / len(df)) * 100
        print(
            f"--> [Evaluation] Outliers (Topic -1): {n_outliers} ({outlier_perc:.2f}%)"
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
        scores = calculate_coherence_metrics(model, docs, embeddings, topics, embedding_model=tm.embedding_model, logger=main_logger)

        # B. Taxonomy Mapping
        tax_path = cfg.get("paths.taxonomy")

        taxonomy_df = load_taxonomy(tax_path)
        if not taxonomy_df.empty:
              mapper = TaxonomyMapper(embedding_model=tm.embedding_model)

              mapping_df = mapper.map_topics_to_taxonomy(model, taxonomy_df)

              print(mapping_df.head())
              out_file_map = cfg.get("paths.output_mapping")
              mapping_df.to_excel(out_file_map, index=False)
              print(f"--> [Done] Taxonomy comparison saved to {out_file_map}")


              # Add best matching label to the main dataframe with topics
              # Create a mapping dictionary: Topic_ID -> Best_Match_Label
              topic_to_label = dict(zip(mapping_df["Topic_ID"], mapping_df["Best_Match_Label"]))

              # Add the label column to the main dataframe
              df["taxonomy_label"] = df["topic"].map(topic_to_label)
              df["taxonomy_label"] = df["taxonomy_label"].fillna("No Match (Outlier)")

              # Re-save the updated dataframe with taxonomy labels
              final_path = "resultswithtaxonomy.xlsx"
              df.to_excel(final_path, index=False)

              print(f"--> [Done] Updated results with taxonomy labels saved to {final_path}")

              #ExactMatch count
              count=0
              tot=len(df)

              def included_or_equal(a, b):
                if pd.isna(a) or pd.isna(b):
                    return False

                # caso: b è lista
                if isinstance(b, list):
                    return a in b

                # caso: b è stringa
                if isinstance(b, str):
                    return a in b

                return False

              mask = df.apply(lambda row: included_or_equal(row["taxonomy_label"], row["labels"]), axis=1)
              count = mask.sum()
              print(count/tot)



              if main_logger:
                  main_logger.log_artifact(
                      out_file_map,
                      "dataset",
                      "taxonomy_mapping",
                  )
        else:
              print("--> [Warning] No taxonomy loaded. Skipping mapping.")

    if cfg.get("project.wandb_logging"):
          print("--> [WandB] Run finished.")
          wandb.finish()


if __name__ == "__main__":
    main()
