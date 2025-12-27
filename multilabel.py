import numpy as np
import pandas as pd


def get_top3_topics_per_review(
    model,
    docs,
    topics,
    probs,
    indices=None,
    top_words=5,
):

    results = []

    if indices is None:
        indices = range(len(docs))

    for i in indices:
        review_probs = probs[i]

        # Top-3 topic indices (descending probability)
        top3_indices = np.argsort(review_probs)[-3:][::-1]
        top3_probs = review_probs[top3_indices]

        row = {
            "review_idx": i,
            "document": docs[i],
            "assigned_topic": int(topics[i]),
        }

        for rank in range(3):
            topic_id = int(top3_indices[rank])
            row[f"topic_{rank+1}"] = topic_id
            row[f"topic_{rank+1}_prob"] = float(top3_probs[rank])

            if topic_id == -1:
                row[f"topic_{rank+1}_words"] = "Outlier"
            else:
                topic_words = model.get_topic(topic_id)
                if topic_words:
                    row[f"topic_{rank+1}_words"] = ", ".join(
                        [w[0] for w in topic_words[:top_words]]
                    )
                else:
                    row[f"topic_{rank+1}_words"] = "N/A"

        results.append(row)

    return pd.DataFrame(results)
