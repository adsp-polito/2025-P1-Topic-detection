import spacy
import pandas as pd
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from collections import Counter

class MWEExtractor:
    def __init__(self, df=pd.DataFrame, spacy_model="it_core_news_lg", freq_filter=20, top_k=50):
        """
        Initialize the extractor.
        - freq_filter: minimum frequency for a bigram to be considered
        - top_k: number of top PMI bigrams to keep
        """
        self.nlp = spacy.load(spacy_model)
        self.freq_filter = freq_filter
        self.top_k = top_k
        self.bigrams_lemma = set()
        self.df = df

    def extract_mwe(self):

        self.df["review_translated"] = self.df["review_translated"].astype(str)
        texts = self.df["review_translated"].str.lower().tolist()

        # 1) Preprocess sentences into list of (lemma, pos)
        processed_sentences = []
        for doc in self.nlp.pipe(texts, disable=["parser", "ner"]):
            tokens = [
                (token.lemma_.lower(), token.pos_)
                for token in doc
                if token.is_alpha
                and not token.is_stop
                and len(token) > 2
            ]
            processed_sentences.append(tokens)

        # 2) Extract bigram candidates (frequency + PMI + POS patterns)
        bigram_measures = BigramAssocMeasures()

        all_tokens = [lemma for sent in processed_sentences for (lemma, pos) in sent]

        finder = BigramCollocationFinder.from_words(all_tokens)
        finder.apply_freq_filter(self.freq_filter)

        bigrams_pmi = finder.nbest(bigram_measures.pmi, self.top_k)

        # POS-based filtered bigrams
        valid_bigrams = []
        for sent in processed_sentences:
            for i in range(len(sent)-1):
                w1, p1 = sent[i]
                w2, p2 = sent[i+1]

                # keep only NOUN + NOUN or NOUN + ADJ
                if (p1 == "NOUN" and p2 == "NOUN") or (p1 == "NOUN" and p2 == "ADJ"):
                    valid_bigrams.append((w1, w2))

        counts = Counter(valid_bigrams)

        # Intersection: only bigrams with PMI + pattern frequency
        mwe_candidates = []
        for bg in bigrams_pmi:
            if bg in counts:
                lemma1, lemma2 = bg
                freq = counts[bg]
                mwe_candidates.append((lemma1, lemma2, freq))

        # Save only lemma tuples
        self.bigrams_lemma = {(lemma1, lemma2) for lemma1, lemma2, freq in mwe_candidates}

        return pd.DataFrame(mwe_candidates, columns=["lemma1", "lemma2", "count"])
        


    def merge_sentence(self, text):
    # 3) Merge lemma-based bigrams inside a sentence
        doc = self.nlp(text)
        lemmas = [t.lemma_.lower() for t in doc]

        merged = []
        i = 0
        while i < len(lemmas):
            if (
                i < len(lemmas) - 1
                and (lemmas[i], lemmas[i+1]) in self.bigrams_lemma
            ):
                merged.append(lemmas[i] + "-" + lemmas[i+1])
                i += 2
            else:
                merged.append(lemmas[i])
                i += 1
        return " ".join(merged)
        

    # 4) Apply MWE merging to an entire dataframe column
    def apply_mwe(self, df, text_column="review_translated", output_column="review_mwe"):
        df[output_column] = df[text_column].astype(str).apply(self.merge_sentence)
        return df


if __name__ == "__main__":
    df = pd.read_excel("./data/reviews_cleaned_translated.xlsx")

    extractor = MWEExtractor(df)

    print("Extracting MWEs...")
    candidates = extractor.extract_mwe()
    print("MWE candidates:", candidates)

    mwe_list = f"./data/mwe_list.xlsx"
    candidates.to_excel(mwe_list, index=False)

    print("Applying MWE merging...")
    df = extractor.apply_mwe(df)

    print(df.head())
