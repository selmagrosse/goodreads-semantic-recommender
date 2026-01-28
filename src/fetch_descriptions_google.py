"""
Fetches book descriptions for a Goodreads dataset using Google Books API.

For each book:
- Attempts ISBN13 lookup
- Falls back to ISBN
- Falls back to title + author search

Saves descriptions in 'Description Google' column.

"""

import pandas as pd
import requests
import time

API_URL = "https://www.googleapis.com/books/v1/volumes"

def get_google_description(isbn=None, title=None, author=None):
    params = None

    if isbn:
        params = {"q": f"isbn:{isbn}"}
    elif title and author:
        params = {"q": f"intitle:{title}+inauthor:{author}"}
    else:
        return None

    try:
        r = requests.get(API_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "items" in data:
            return data["items"][0]["volumeInfo"].get("description")

    except Exception as e:
        print(f"Google API error: {e}")

    return None


def add_google_descriptions(df, sleep_time=0.2):
    descriptions = []

    for i, row in df.iterrows():
        print(f"[Google] {i+1}/{len(df)}: {row['Title']}")

        desc = (
            get_google_description(isbn=row.get("ISBN13"))
            or get_google_description(isbn=row.get("ISBN"))
            or get_google_description(title=row["Title"], author=row["Author"])
        )

        descriptions.append(desc)
        time.sleep(sleep_time)

    df["Description_Google"] = descriptions
    return df

