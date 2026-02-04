"""
Module for evaluating the quality of book descriptions using a language model (LLM).

This module provides reusable functions to:
- Normalize and clean description text from multiple sources
- Score the quality of descriptions using GPT-4o-mini on a scale from 1 (very bad) to 10 (excellent)
- Apply scoring to one or more columns in a pandas DataFrame

Functions:
- normalize_text(text: str) -> str
    Normalize a single description by collapsing whitespace and stripping extra spaces.

- score_description(description: str) -> int
    Use the LLM to rate a single description, returning an integer score between 1 and 10.
    Returns 0 for invalid, missing, or too short descriptions.

- score_description_column(df: pd.DataFrame, desc_col: str, score_col: str) -> pd.Series
    Apply `score_description` to a full DataFrame column and return a Series of scores.

- normalize_descriptions(df: pd.DataFrame, description_cols: list[str]) -> pd.DataFrame
    Normalize multiple description columns in a DataFrame.

- evaluate_descriptions(df: pd.DataFrame, description_cols: dict[str, str]) -> pd.DataFrame
    Normalize and score multiple description columns, returning a DataFrame with
    new quality score columns.
"""

import openai
import pandas as pd
from tqdm import tqdm
import re
import os
from dotenv import load_dotenv

load_dotenv()

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

def score_description(description: str) -> int:
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
        scores.append(score_description(desc))

    return pd.Series(scores, name=score_col)

def normalize_descriptions(df: pd.DataFrame, description_cols: list[str]) -> pd.DataFrame:
    for col in description_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    return df

def evaluate_descriptions(df: pd.DataFrame, description_cols: dict[str, str]) -> pd.DataFrame:
    """
    Score each description in a given column and add a corresponding score column.

    Parameters
    ----------
    df: DataFrame with description columns
    description_cols: dict mapping description column → score column name

    Returns
    -------
    DataFrame with added score columns
    """
    df = normalize_descriptions(df, list(description_cols.keys()))
    # Score each source
    for desc_col, score_col in description_cols.items():
        df[score_col] = score_description_column(df, desc_col, score_col)
    return df

