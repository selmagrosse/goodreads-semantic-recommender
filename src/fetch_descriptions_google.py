"""
Fetches book descriptions for a Goodreads dataset using Google Books API.

For each book:
- Attempts ISBN13 lookup
- Falls back to ISBN
- Falls back to title + author search

Saves descriptions in the 'Description Google' column.

"""

from typing import Optional
import pandas as pd
import requests
import time

API_URL = "https://www.googleapis.com/books/v1/volumes"

def get_google_description(
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> Optional[str]:
    """
    Fetch a single book description from Google Books.

    Parameters
    ----------
    isbn : str, optional
        ISBN or ISBN13 identifier.
    title : str, optional
        Book title (used as fallback).
    author : str, optional
        Book author (used with title fallback).

    Returns
    -------
    str or None
        Book description if found, otherwise None.
    """

    if isbn:
        params = {"q": f"isbn:{isbn}"}
    elif title and author:
        params = {"q": f"intitle:{title}+inauthor:{author}"}
    else:
        return None

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "items" in data:
            return data["items"][0]["volumeInfo"].get("description")

    except Exception as e:
        print(f"Google API error: {e}")

    return None


def add_google_descriptions(
    df: pd.DataFrame,
    sleep_time: float = 0.2,
) -> pd.DataFrame:
    """
    Add Google Books descriptions to a Goodreads dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Goodreads dataset.
    sleep_time : float, default=0.2
        Time to sleep between API requests (seconds).

    Returns
    -------
    pd.DataFrame
        DataFrame with a new 'Description Google' column.
    """
    descriptions = []

    total = len(df)

    for i, row in df.iterrows():
        print(f"[Google] {i+1}/{total}: {row['Title']}")

        description = (
            get_google_description(isbn=row.get("ISBN13"))
            or get_google_description(isbn=row.get("ISBN"))
            or get_google_description(title=row["Title"], author=row["Author"])
        )

        descriptions.append(description)
        time.sleep(sleep_time)

    df["Description Google"] = descriptions
    return df

