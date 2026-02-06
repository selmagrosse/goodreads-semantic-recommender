"""
Extends a Goodreads dataset with book descriptions.

This script:
1. Loads a cleaned Goodreads dataset
2. Fetches book descriptions from:
   - Google Books API
   - OpenLibrary API
3. Saves the extended dataset.
"""

import pandas as pd
from fetch_descriptions_google import add_google_descriptions
from fetch_descriptions_openlibrary import add_openlibrary_descriptions
from clean_description_columns import clean_description_columns

INPUT_PATH = "data/interim/goodreads_library_cleaned.csv"
OUTPUT_PATH = "data/interim/goodreads_library_cleaned_descriptions.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} books")

    # Fetch descriptions from Google Books API
    df = add_google_descriptions(df)
    # Fetch descriptions from OpenLibrary API
    df = add_openlibrary_descriptions(df)
    # Clean description columns
    df = clean_description_columns(df, ["Description Google", "Description OpenLibrary"])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved extended dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
