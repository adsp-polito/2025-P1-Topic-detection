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
            sorted_idx = np.argsort(review_scores)[::-1]
            sorted_scores = review_scores[sorted_idx]

            max_score = sorted_scores[0]
            candidates = [
                (int(tid), float(score))
                for tid, score in zip(sorted_idx, sorted_scores)
                if score >= alpha * max_score
            ][:max_labels]

        
            multi_topics = [tid for tid, _ in candidates if tid != -1]

            old_topic = int(self.topics[i])

            if old_topic != -1:
                assigned_primary = old_topic
            else:
              
                if len(candidates) > 0 and candidates[0][1] >= min_abs_score:
                    assigned_primary = multi_topics[0] if len(multi_topics) > 0 else -1
                else:
                    assigned_primary = -1
                    multi_topics = []

            """
            # DEBUG
            if i<30:
                print("Candidates:", candidates)
                print("Multilabeling:", multi_topics)
                print("Old topic: ", old_topic)
                print("Assigned topic", assigned_primary)
                print(f"SCORE {alpha}*{max_score} = {alpha * max_score}")
            """

    
            row = {
                "review_idx": i,
                "document": self.docs[i],
                "assigned_topic_primary": assigned_primary,
                "topic_raw": old_topic,
                "multi_topics": multi_topics,
                "n_assigned_topics": len(multi_topics),
            }

          
            for rank, (topic_id, score) in enumerate(candidates, start=1):
                row[f"topic_{rank}"] = topic_id
                row[f"topic_{rank}_score"] = score

                words = self.model.get_topic(topic_id)
                row[f"topic_{rank}_words"] = (
                    ", ".join([w[0] for w in words[:top_words]])
                    if words else "N/A"
                )

            results.append(row)

        return pd.DataFrame(results)




