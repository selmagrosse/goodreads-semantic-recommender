import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/goodreads_library_final_descriptions.csv")

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
            return f"/file={path}"
        return "/file=covers/no_cover.jpg"