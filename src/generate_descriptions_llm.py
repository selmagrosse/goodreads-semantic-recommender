"""
Generates book descriptions for a Goodreads dataset using a Large Language Model (LLM).
Only fills in missing descriptions or descriptions with low quality scores.

For each book without a description or with low-quality fetched descriptions:
- Uses title + author as input
- Produces short, neutral descriptions (3–5 sentences)

Saves descriptions in the 'Description LLM' column.
"""

from typing import Optional
import pandas as pd
import time
from openai import OpenAI
from clean_description_columns import clean_description_columns
import os
from dotenv import load_dotenv

from fetch_descriptions import INPUT_PATH, OUTPUT_PATH

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 2
SLEEP_SECONDS = 0.5
THRESHOLD = 4 # Minimum quality score


def build_prompt(title: str, author: str) -> str:
    """
    Build a prompt for generating a book description.
    """
    return (
        "Write a concise, neutral book description (3–5 sentences).\n"
        "Do not invent plot details.\n\n"
        f"Title: {title}\n"
        f"Author: {author}"
    )

def generate_description(
    client: OpenAI,
    title: str,
    author: str
) -> Optional[str]:
    """
    Generate a single book description using an LLM.

    Parameters
    ----------
    client : OpenAI
        Initialized OpenAI client.
    title : str
        Book title.
    author : str
        Book author.

    Returns
    -------
    str or None
        Generated description, or None if generation failed.
    """
    prompt = build_prompt(title, author)

    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"LLM error for '{title}': {e}")
            time.sleep(1)

    return None

def add_llm_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LLM-generated descriptions to a Goodreads dataset.

    Only fills descriptions when:
    - 'Description Google' is missing
    - 'Description OpenLibrary' is missing
    - 'Description LLM' is missing

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Goodreads dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'Description LLM' column.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    client = OpenAI(api_key=OPENAI_API_KEY)

    descriptions = []
    total = len(df)

    for i, row in df.iterrows():
        print(f"[LLM] {i+1}/{total}: {row['Title']}")

        if pd.notna(row.get("Description LLM")):
            descriptions.append(row["Description LLM"])
            continue

        # Only generate if other sources failed
        if (
            pd.notna(row.get("Description Google"))
            and (row.get("Quality Score Google", 0) > THRESHOLD)
            or pd.notna(row.get("Description OpenLibrary"))
            and (row.get("Quality Score OpenLibrary", 0) > THRESHOLD)
        ):
            descriptions.append(None)
            continue

        description = generate_description(
            client,
            row["Title"],
            row["Author"]
        )

        descriptions.append(description)
        time.sleep(SLEEP_SECONDS)

    df["Description LLM"] = descriptions
    return df

def main():
    input_path = "data/interim/goodreads_library_fetched_descriptions_quality_scored.csv"
    output_path = "data/interim/goodreads_library_descriptions_with_llm.csv"
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} books")

    # Generate descriptions using LLM
    df = add_llm_descriptions(df)
    # Clean description columns
    df = clean_description_columns(df, ["Description LLM"])

    df.to_csv(output_path, index=False)
    print(f"Saved extended dataset to {output_path}")

if __name__ == "__main__":
    main()