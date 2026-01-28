"""
Fetches book descriptions for a Goodreads dataset using OpenLibrary API.

For each book:
- Attempts ISBN13 lookup
- Falls back to ISBN

Saves descriptions in 'Description OpenLibrary' column.

"""

import pandas as pd
import requests
import time

API_URL = "https://openlibrary.org/isbn/{}.json"

def get_openlibrary_description(isbn):
    if not isbn:
        return None

    try:
        r = requests.get(API_URL.format(isbn), timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        desc = data.get("description")
        if isinstance(desc, dict):
            return desc.get("value")
        return desc

    except Exception as e:
        print(f"OpenLibrary error {isbn}: {e}")

    return None


def add_openlibrary_descriptions(df, sleep_time=0.2):
    descriptions = []

    for i, row in df.iterrows():
        print(f"[OpenLibrary] {i+1}/{len(df)}: {row['Title']}")

        desc = (
            get_openlibrary_description(row.get("ISBN13"))
            or get_openlibrary_description(row.get("ISBN"))
        )

        descriptions.append(desc)
        time.sleep(sleep_time)

    df["Description_OpenLibrary"] = descriptions
    return df
