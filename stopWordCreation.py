import os

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from translation import TranslatorModule


def main():
    print("=== STOPWORD EXTRACTION USING TF-IDF ===")

    # Download required NLTK data
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

    # 1. LOAD DATASET
    data_path = "./data/banking-app-reviews-train.csv"
    print(f"--> [Step 1/4] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"    Loaded {len(df)} reviews")

    # 2. TRANSLATE TO ITALIAN
    print("--> [Step 2/4] Translating reviews to Italian...")
    translator = TranslatorModule(df)
    df = translator.detect_and_translate(text_col="review_text")
    print("    Translation completed")

    # 3. PREPROCESSING (lowercase + lemmatization)
    print("--> [Step 3/4] Preprocessing text (lowercase + lemmatization)...")
    
    # Use the translated text (final_text column from translator)
    df['lowerReview'] = df['final_text'].str.lower().str.split()

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(text):
        return [lemmatizer.lemmatize(w) for w in text]

    df['finalReview'] = df['lowerReview'].apply(lemmatize_text)
    
    # Create corpus
    corpus = list(map(' '.join, df["finalReview"]))
    print(f"    Preprocessed {len(corpus)} documents")

    # 4. TF-IDF APPROACH
    print("--> [Step 4/4] Applying TF-IDF to identify stopwords...")
    
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(corpus)

    # Create dictionary to find TF-IDF value for each word
    word_lst = vectorizer.get_feature_names_out()
    count_lst = x.toarray().sum(axis=0)

    vocab_df = pd.DataFrame(
        list(zip(word_lst, count_lst)),
        columns=["vocab", "tfidf_value"]
    )

    sorted_df = vocab_df.sort_values(by="tfidf_value", ascending=False)

    # 5. SAVE TOP 200 WORDS
    output_file = "./data/tfidf_stopwords.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    textfile = open(output_file, "w", encoding="utf-8")
    for element in sorted_df.vocab.head(200):
        textfile.write(element + "\n")
    textfile.close()

    print(f"\n--> [Done] Top 200 stopwords saved to {output_file}")
    print(f"    Top 10 stopwords: {list(sorted_df.vocab.head(10))}")


if __name__ == "__main__":
    main()
