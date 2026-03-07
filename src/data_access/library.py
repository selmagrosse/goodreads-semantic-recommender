import os
import pandas as pd
import requests
import numpy as np
from pathlib import Path

# Resolve project root dynamically
BASE_DIR = Path(__file__).resolve()
while not (BASE_DIR / "data").exists():
    BASE_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "goodreads_library_final_descriptions.csv"
COVERS_DIR = BASE_DIR / "app"

# Explicit tag columns present in the dataset
TAG_COLUMNS = [
    "american",
    "auto-biography",
    "children",
    "engineering",
    "ex-yu",
    "favorites",
    "feminism",
    "immigration",
    "indian",
    "latin-america",
    "lists",
    "own",
    "philosophy",
    "play",
    "poetry",
    "psychology",
    "re-read",
    "read",
    "read-in-german",
    "russian",
    "sci-fi",
    "science",
    "to-read",
    "travel",
]

class Library:
    def __init__(self, path=DATA_PATH):
        self.df = pd.read_csv(path)
        self.df.columns = self.df.columns.str.strip()
        self.tag_columns = [
            col for col in self.df.columns
            if col in TAG_COLUMNS
        ]

    def get_available_tags(self):
        return sorted(self.tag_columns)

    def get_books_by_tag(self, tag):
        if tag not in self.tag_columns:
            return pd.DataFrame()
        books = self.df[self.df[tag] == 1].copy()
        return books

    @staticmethod
    def get_cover_path(row):
        path = row.get("Cover Path", "")
        if isinstance(path, str) and path.strip():
            return str(COVERS_DIR / path)
        return str(COVERS_DIR / "no_cover.jpg")
    
    # --------------------------
    # URL-based cover retrieval
    # --------------------------
    @staticmethod
    def get_cover_url(isbn):
        """
        Returns a direct URL to the book cover image.
        Example using Open Library Covers API:
        https://covers.openlibrary.org/b/isbn/{ISBN}-M.jpg
        """
        isbn_str = str(isbn).strip()
        if not isbn_str:
            return None
        return f"https://covers.openlibrary.org/b/isbn/{isbn_str}-M.jpg"
    
    def get_seed_books(self):
        """
        Returns books suitable for seeding recommendations:
        - already read (not tagged 'to-read')
        - personal rating >= 4
        """
        df = self.df.copy()

        seed_books = df[
            (df["to-read"] != 1) &
            (df["My Rating"] >= 4)
        ]

        return seed_books

    def compute_seed_embedding(self, collection):
        seeds = self.get_seed_books()

        if seeds.empty:
            return None

        # use dataframe index as IDs (same as used in Chroma)
        ids = seeds.index.astype(str).tolist()

        if not ids:
            print("No valid seed IDs found")
            return None

        # retrieve embeddings from Chroma
        result = collection.get(
            ids=ids,
            include=["embeddings"]
        )

        embeddings = np.array(result["embeddings"])

        if embeddings.size == 0:
            print("No embeddings retrieved for seeds")
            return None

        seed_vector = embeddings.mean(axis=0)
        return seed_vector
