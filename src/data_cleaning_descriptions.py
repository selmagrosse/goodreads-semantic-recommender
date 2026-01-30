"""
Clean and normalize book description columns.

This script:
- Normalizes whitespace and punctuation
- Removes URLs and non-printable characters
- Lowercases text
- Ensures empty descriptions are truly empty strings
"""

import pandas as pd

INPUT_PATH = "data/interim/goodreads_library_cleaned_descriptions.csv"
OUTPUT_PATH = "data/interim/goodreads_library_descriptions_cleaned.csv"

DESCRIPTION_COLS = [
    "Description Google",
    "Description OpenLibrary",
    "Description LLM",
]

def clean_description(series: pd.Series) -> pd.Series:
    """
    Clean and normalize a description text column.
    """
    return (
        series
        .fillna("")                                            # fill NaNs with empty strings
        .astype(str)                                           # Enforce string type
        .str.strip()                                           # Strip leading/trailing whitespace
        .str.lower()                                           # Lowercase all text
        .str.replace(r"http\S+|www\.\S+", "", regex=True)      # Remove URLs
        .str.replace(r"[\x00-\x1f\x7f-\x9f]", "", regex=True)  # Remove non-printable / OCR chars
        .str.replace(r"\s+", " ", regex=True)                  # Replace repeated whitespace with single space
        .str.replace(r"([!?,;:-])\1+", r"\1", regex=True)      # Replace repeated punctuation with single punctuation, excluding dots
        .str.replace(r"(\.\s){2,}\.", "...", regex=True)       # Normalize multiple spaced dots to standard ellipsis
        .str.strip()                                           # Strip again after replacements to clean up any leftover spaces
    )

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} books")

    for col in DESCRIPTION_COLS:
        df[col] = clean_description(df[col])

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()