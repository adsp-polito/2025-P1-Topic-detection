import numpy as np
import pandas as pd


def get_top3_topics_per_review(
    model,
    docs,
    topics,
    probs,
    indices=None,
    top_words=5,
    alpha=0.85,        # threshold relativo
    max_labels=3,      # massimo numero di topic assegnabili
):

    results = []

    if indices is None:
        indices = range(len(docs))

    for i in indices:
        review_scores = probs[i]

        # Ordina topic per score decrescente
        sorted_idx = np.argsort(review_scores)[::-1]
        sorted_scores = review_scores[sorted_idx]

        max_score = sorted_scores[0]

        # Multi-label selection (relative threshold)
        selected = [
            (int(tid), float(score))
            for tid, score in zip(sorted_idx, sorted_scores)
            if score >= alpha * max_score and tid != -1
        ][:max_labels]

        # Lista dei topic assegnati (MULTI-TOPIC)
        multi_topics = [tid for tid, _ in selected]

        row = {
            "review_idx": i,
            "document": docs[i],
            "assigned_topic_primary": int(topics[i]),
            "multi_topics": multi_topics,         
            "n_assigned_topics": len(multi_topics),
        }

        # Dettaglio top-k topic
        for rank, (topic_id, score) in enumerate(selected, start=1):
            row[f"topic_{rank}"] = topic_id
            row[f"topic_{rank}_score"] = score

            words = model.get_topic(topic_id)
            row[f"topic_{rank}_words"] = (
                ", ".join([w[0] for w in words[:top_words]])
                if words else "N/A"
            )

        results.append(row)

    return pd.DataFrame(results)
