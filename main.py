from cleaner import ReviewCleaner
from translation import ReviewTranslation
import pandas as pd

def main():

    FILE = "data\dataset_v2.xlsx"  

    # PRE-PROCESSING PHASE

    print("STEP 1: Loading dataset")
    cleaner = ReviewCleaner(FILE)
    df = cleaner.load_data()

    print("STEP 2: Filtering sentiments")
    cleaner.filter_by_sentiment()

    print("STEP 3: Language detection")
    cleaner.detect_language()

    print("STEP 4: Text cleaning")
    cleaner.clean_text()

    print("STEP 5: Remove junk reviews")
    df_clean  = cleaner.remove_junk()

    print("STEP 6: Translation")
    translator = ReviewTranslation(df_clean)
    translator.analyze_languages()
    df_translated = translator.translate_to_italian()

    print("STEP 7: Saving output")
    df_translated.to_excel("reviews_cleaned_translated_2.xlsx", index=False)

    print("\n>>> DONE! Saved to reviews_cleaned_translated.xlsx")


if __name__ == "__main__":
    main()
