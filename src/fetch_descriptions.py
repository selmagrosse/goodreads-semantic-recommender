"""
Extends a Goodreads dataset with book descriptions.

This script:
1. Loads a cleaned Goodreads dataset
2. Fetches missing book descriptions from:
   - Google Books API
   - OpenLibrary API
3. Uses an LLM to generate descriptions for any remaining missing entries
4. Saves the extended dataset.
"""

import pandas as pd
from fetch_descriptions_google import add_google_descriptions
from fetch_descriptions_openlibrary import add_openlibrary_descriptions
from fetch_descriptions_llm import add_llm_descriptions

INPUT_PATH = "data/interim/goodreads_library_cleaned.csv"
OUTPUT_PATH = "data/interim/goodreads_library_cleaned_descriptions.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} books")

    # Fetch descriptions from Google Books API
    df = add_google_descriptions(df)
    # Fetch descriptions from OpenLibrary API
    df = add_openlibrary_descriptions(df)
    # Generate descriptions using LLM for any remaining missing entries
    df = add_llm_descriptions(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved extended dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
