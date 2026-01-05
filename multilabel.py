import numpy as np
import pandas as pd

def map_topics_to_taxonomy_list(multi_topic_list, topic_to_label):

    taxonomy_labels = []
    for t in multi_topic_list:
        if t in topic_to_label:
            taxonomy_labels.append(topic_to_label[t])

    # remove duplicates while preserving order
    return list(dict.fromkeys(taxonomy_labels))


class MultiLabelModeler:
    """
    
    """

    def __init__(self, model, docs, topics, probs):
        self.model = model
        self.docs = docs
        self.topics = topics
        self.probs = probs

    def get_top3_topics_per_review(self,
        indices=None,
        top_words=5,
        alpha=0.85,
        min_abs_score = 0.30,
        max_labels=3,
    ):

        results = []

        if indices is None:
            indices = range(len(self.docs))

        for i in indices:
            review_scores = self.probs[i]

            # Ordina topic per score decrescente
            sorted_idx = np.argsort(review_scores)[::-1]
            sorted_scores = review_scores[sorted_idx]

            max_score = sorted_scores[0]

            # Multi-label selection (include -1 if relevant)
            selected = [
                (int(tid), float(score))
                for tid, score in zip(sorted_idx, sorted_scores)
                if score >= alpha * max_score
            ][:max_labels]

            # Lista dei topic selezionati
            multi_topics = [tid for tid, _ in selected]

            old_topic = int(self.topics[i])

            if old_topic != -1:
                assigned_primary = old_topic
            else:
                # primo topic valido (≠ -1) tra quelli selezionati
                assigned_primary = next(
                    (tid for tid, score in selected if tid != -1 and score >= min_abs_score),
                    -1
                )

            row = {
                "review_idx": i,
                "document": self.docs[i],
                "assigned_topic_primary": assigned_primary,
                "topic_raw": old_topic,
                "multi_topics": multi_topics,
                "n_assigned_topics": len(multi_topics),
            }

            # Dettaglio top-k topic
            for rank, (topic_id, score) in enumerate(selected, start=1):
                row[f"topic_{rank}"] = topic_id
                row[f"topic_{rank}_score"] = score

                words = self.model.get_topic(topic_id)
                row[f"topic_{rank}_words"] = (
                    ", ".join([w[0] for w in words[:top_words]])
                    if words else "N/A"
                )

            results.append(row)

        return pd.DataFrame(results)




