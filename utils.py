import os
import random

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure
    reproducibility of results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic algorithms (may impact performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"--> [Reproducibility] Global seed set to {seed}")


def load_taxonomy(path: str) -> pd.DataFrame:
    """
    Loads the taxonomy and prepares it for semantic comparison.
    Filters out rows marked as DEPRECATED.
    """
    if not os.path.exists(path):
        print(f"--> [Error] Taxonomy file not found at: {path}")
        return pd.DataFrame()

    try:
        df_tax = pd.read_excel(path)

        if "DEPRECATED" in df_tax.columns:
            df_tax = df_tax[~df_tax["DEPRECATED"]]

        # Intentional skip of first row as per user requirement
        df_tax = df_tax.iloc[1:]

        # Standardize columns
        # We rename the first two columns to ensure consistent access
        df_tax.rename(
            columns={df_tax.columns[0]: "Label", df_tax.columns[2]: "Description"},
            inplace=True,
        )

        # Create a "Combined" text field for better embedding
        # Format: "Label: Description"
        df_tax["Embedding_Text"] = df_tax.apply(
            lambda x: f"{x['Label']}: {x['Description']}"
            if pd.notnull(x["Description"])
            else str(x["Label"]),
            axis=1,
        )

        print(f"--> [Taxonomy] Loaded {len(df_tax)} labels from {path}")
        return df_tax[["Label", "Description", "Embedding_Text"]]
    except Exception as e:
        print(f"--> [Error] Could not load taxonomy: {e}")
        return pd.DataFrame()


def ensure_directories(paths: list):
    """
    Ensures that the directory structures for the given file paths exist.
    """
    for path in paths:
        if path:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"--> [System] Created directory: {directory}")
                print(f"--> [System] Created directory: {directory}")


def save_reviews_with_topic_probabilities(
    docs,
    topics,
    probs,
    output_path,
    top_k=None
):
    """
    Save an Excel file with:
    - review text
    - assigned topic
    - probability for each topic
    - optional top-k topics per review
    """

    n_topics = probs.shape[1]

    # Base dataframe
    df = pd.DataFrame({
        "review": docs,
        "assigned_topic": topics
    })

    # Add probability columns
    for t in range(n_topics):
        df[f"prob_topic_{t}"] = probs[:, t]

    # Optional: top-k topics
    if top_k is not None:
        topk_idx = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]
        topk_probs = np.take_along_axis(probs, topk_idx, axis=1)

        for k in range(top_k):
            df[f"top{k+1}_topic"] = topk_idx[:, k]
            df[f"top{k+1}_prob"] = topk_probs[:, k]

    # Save
    df.to_excel(output_path, index=False)
    print(f"[Saved] Review-topic probabilities → {output_path}")