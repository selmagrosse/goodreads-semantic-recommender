"""
Fetches book descriptions for a Goodreads dataset using OpenLibrary API.

For each book:
- Attempts ISBN13 lookup
- Falls back to ISBN

Saves descriptions in the 'Description OpenLibrary' column.
"""

from typing import Optional
import pandas as pd
import requests
import time

API_URL = "https://openlibrary.org/isbn/{}.json"

def get_openlibrary_description(isbn: Optional[str]) -> Optional[str]:
    """
    Fetch a single book description from OpenLibrary.

    Parameters
    ----------
    isbn : str or None
        ISBN or ISBN13 identifier.

    Returns
    -------
    str or None
        Book description if found, otherwise None.
    """
    if not isbn:
        return None

    try:
        response = requests.get(API_URL.format(isbn), timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()

        description = data.get("description")
        # Handle cases where description is a dict with 'value' key
        if isinstance(description, dict):
            return description.get("value")
        return description

    except Exception as e:
        print(f"OpenLibrary error for ISBN {isbn}: {e}")

    return None


def add_openlibrary_descriptions(
    df: pd.DataFrame,
    sleep_time: float = 0.2,
) -> pd.DataFrame:
    """
    Add OpenLibrary descriptions to a Goodreads dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Goodreads dataset.
    sleep_time : float, default=0.2
        Time to sleep between API requests (seconds).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'Description OpenLibrary' column.
    """
    descriptions = []
    total = len(df)

    for i, row in df.iterrows():
        print(f"[OpenLibrary] {i+1}/{total}: {row['Title']}")

        description = (
            get_openlibrary_description(row.get("ISBN13"))
            or get_openlibrary_description(row.get("ISBN"))
        )

        descriptions.append(description)
        time.sleep(sleep_time)

    df["Description OpenLibrary"] = descriptions
    return df
