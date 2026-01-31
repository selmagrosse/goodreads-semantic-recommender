"""
Quality evaluation of book descriptions using an LLM.

This script:
- Loads cleaned Goodreads dataset enriched with book descriptions from three sources: Google Books, OpenLibrary, and LLM-generated descriptions
- Normalizes description text across three sources
- Uses an LLM (GPT-4o-mini) to score the quality of each description on a scale from 1 to 10 
- Outputs a dataset with per-source quality scores
"""

import openai
import pandas as pd
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "/content/goodreads_library_descriptions_cleaned.csv"
OUTPUT_FILE = "/content/goodreads_library_descriptions_quality_scored.csv"

DESCRIPTION_COLS = {
    "Description Google": "Quality Score Google",
    "Description OpenLibrary": "Quality Score OpenLibrary",
    "Description LLM": "Quality Score LLM",
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0
MIN_WORDS = 10

def normalize_text(text: str) -> str:
    """
    Normalize description text by collapsing whitespace
    and converting it into a single-line string.
    """
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

def score_with_gpt(description: str) -> int:
    """
    Score a book description using an LLM.

    Parameters
    ----------
    description : str
        Book description text.

    Returns
    -------
    int
        Quality score in [1, 10], or 0 if invalid or missing.
    """
    # Empty or trivial descriptions
    if not description or len(description.split()) < 10:
        return 0

    prompt = f"""You are a professional librarian. Rate the quality of the following book description on a scale from 1 (very bad) to 10 (excellent).

    Description:
    "{description}"

    Guidelines:
    - 1 = gibberish, irrelevant, or extremely short
    - 5 = somewhat informative but limited or generic
    - 10 = informative, coherent, and meaningful

    Return ONLY the number 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()
        # Return a number between 1 and 10, or 0 if invalid
        return int(text) if text.isdigit() and 1 <= int(text) <= 10 else 0

    except Exception as e:
        print("GPT error:", e)
        return 0
    
def score_description_column(
    df: pd.DataFrame, 
    desc_col: str, 
    score_col: str) -> pd.Series:
    """Score a single description column using GPT."""
    scores = []

    for desc in tqdm(df[desc_col], desc=f"Scoring {desc_col}"):
        scores.append(score_with_gpt(desc))

    return pd.Series(scores, name=score_col)

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # Normalize descriptions
    for col in DESCRIPTION_COLS:
        df[col] = df[col].apply(normalize_text)

    # Score each source
    for desc_col, score_col in DESCRIPTION_COLS.items():
        print(f"\nScoring {desc_col} → {score_col}")
        df[score_col] = score_description_column(df, desc_col, score_col)

    # Inspect
    print(
        df[
            [
                "Title",
                "Author",
                "Quality Score Google",
                "Quality Score OpenLibrary",
                "Quality Score LLM",
            ]
        ].head(20)
    )

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved scored dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

