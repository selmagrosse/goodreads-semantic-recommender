"""
Generates book descriptions for a Goodreads dataset using a Large Language Model (LLM).
Only fills in missing descriptions.

For each book without a description:
- Uses title + author as input
- Produces short, neutral descriptions (3–5 sentences)

Saves descriptions in the 'Description LLM' column.
"""

from typing import Optional
import pandas as pd
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
MAX_RETRIES = 2
SLEEP_SECONDS = 0.5


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
            or pd.notna(row.get("Description OpenLibrary"))
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