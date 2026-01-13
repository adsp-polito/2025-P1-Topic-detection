import spacy
import pandas as pd
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from collections import Counter

class MWEExtractor:
    def __init__(self, df : pd.DataFrame, spacy_model="it_core_news_lg", freq_filter=20, top_k=50):
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

    def extract_mwe(self) -> pd.DataFrame:
        """
        Extracts the list of MWEs (pairs of words that commonly appear together)
        """
        print("--> [Extractor] Extracting MWEs...")

        self.df["clean_text"] = self.df["clean_text"].astype(str)
        texts = self.df["clean_text"].str.lower().tolist()

        # Preprocess sentences into list of (lemma, pos)
        processed_sentences = []
        for doc in self.nlp.pipe(texts, disable=["parser", "ner"]):
            tokens = [
                (token.lemma_.lower(), token.pos_)
                for token in doc
                if token.is_alpha
                and not token.is_stop
                and len(token.text) > 2
            ]
            processed_sentences.append(tokens)

        # Extract bigram candidates (frequency + PMI + POS patterns)
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
        

    def merge_sentence(self, text: str) -> str:
        """
        Merges lemma-based bigrams inside a sentence
        """
        doc = self.nlp(text)
        original_tokens = [t.text for t in doc]                
        lemmas = [t.lemma_.lower() for t in doc]               

        merged = []
        i = 0

        while i < len(doc):
            # if consecutives lemmas form a MWE --> merges the lemmas to form the MWE with original tokens
            if (
                i < len(doc) - 1
                and (lemmas[i], lemmas[i+1]) in self.bigrams_lemma
            ):
                merged.append(original_tokens[i] + "_" + original_tokens[i+1])
                i += 2

            else:
                merged.append(original_tokens[i])
                i += 1

        return " ".join(merged)

        
    def apply_mwe(self) -> pd.DataFrame:
        """
        Applies MWE merging to an entire dataframe column
        """
        print("--> [Extractor] Applying MWEs...")

        self.df["clean_text_mwe"] = self.df["clean_text"].astype(str).apply(self.merge_sentence)
        changed_rows = (self.df["clean_text_mwe"] != self.df["clean_text"]).sum()

        print(f"--> [Extractor] {changed_rows} reviews where updated with MWEs.")

        return self.df
