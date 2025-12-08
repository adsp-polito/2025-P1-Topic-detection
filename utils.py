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


def load_taxonomy(path: str) -> list:
    """
    Loads the unique labels from the taxonomy Excel file.
    Filters out rows marked as DEPRECATED.
    """
    if not os.path.exists(path):
        print(f"--> [Error] Taxonomy file not found at: {path}")
        return []

    try:
        df_tax = pd.read_excel(path)

        if "DEPRECATED" in df_tax.columns:
            df_tax = df_tax[~df_tax["DEPRECATED"]]

        # Get the first column as a list of strings
        labels = df_tax.iloc[1:, 0].astype(str).unique().tolist()
        print(f"--> [Taxonomy] Loaded {len(labels)} labels from {path}")
        return labels
    except Exception as e:
        print(f"--> [Error] Could not load taxonomy: {e}")
        return []


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
