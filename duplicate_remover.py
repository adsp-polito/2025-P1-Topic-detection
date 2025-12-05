import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateRemover:
    """
    Identifies and removes duplicate reviews using TF-IDF and Cosine Similarity.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def remove_duplicates(
        self, df: pd.DataFrame, text_col: str = "final_text"
    ) -> pd.DataFrame:
        """
        Calculates similarity matrix and drops duplicates.
        """
        print(
            f"--> [Deduplication] Running TF-IDF similarity (Threshold: {self.threshold})..."
        )

        texts = df[text_col].astype(str).tolist()

        # Using simple params to keep it fast but effective
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.85)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle case where vocabulary is empty
            print(
                "--> [Deduplication] Warning: Empty vocabulary (dataset too small?). Skipping."
            )
            return df

        # Calculate Cosine Similarity
        # Note: For very large datasets (50k+), this matrix approach might OOM.
        # For 5.5k it is perfectly fine.
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Identify duplicates
        # We look at the upper triangle of the matrix
        to_drop = set()
        num_docs = len(texts)

        # Iterate efficiently
        # We use np.argwhere to find indices > threshold without double loop in Python
        # Mask lower triangle and diagonal
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)

        # Find pairs where similarity > threshold
        rows, cols = np.where((sim_matrix > self.threshold) & mask)

        for r, c in zip(rows, cols):
            # Mark the second occurrence (c) for deletion
            to_drop.add(df.index[c])

        print(f"--> [Deduplication] Found {len(to_drop)} duplicates.")

        df_clean = df.drop(index=list(to_drop)).reset_index(drop=True)
        return df_clean
