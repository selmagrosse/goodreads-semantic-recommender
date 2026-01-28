'''
Fetch book descriptions from Google Books and OpenLibrary APIs and add them to a Goodreads dataset.
'''

import pandas as pd
from fetch_descriptions_google import add_google_descriptions
from fetch_descriptions_openlibrary import add_openlibrary_descriptions

INPUT_PATH = "data/interim/goodreads_library_cleaned.csv"
OUTPUT_PATH = "data/interim/goodreads_library_cleaned_descriptions.csv"

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} books")

    df = add_google_descriptions(df)
    df = add_openlibrary_descriptions(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved extended dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
