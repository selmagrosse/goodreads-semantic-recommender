"""
Clean and normalize book description columns.

This script:
- Normalizes whitespace and punctuation
- Removes URLs and non-printable characters
- Lowercases text
- Ensures empty descriptions are truly empty strings
"""

import pandas as pd

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

def clean_description_columns(
    df: pd.DataFrame,
    columns: list[str]) -> pd.DataFrame:
    """
    Clean specified description columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list[str]
        List of column names to clean

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned columns
    """

    for col in columns:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found, skipping")
            continue

        df[col] = clean_description(df[col])

    return df