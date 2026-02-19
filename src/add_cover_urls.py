import pandas as pd
from pathlib import Path
from src.data_access.library import Library

DATA_PATH = Path("data/processed/goodreads_library_final_descriptions.csv")

def add_cover_urls():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Cover URL" not in df.columns:
        print("Creating 'Cover URL' column...")
        df["Cover URL"] = df["ISBN"].apply(Library.get_cover_url)
    else:
        print("'Cover URL' already exists — rebuilding values")
        df["Cover URL"] = df["ISBN"].apply(Library.get_cover_url)

    df.to_csv(DATA_PATH, index=False)
    print("Dataset updated safely.")

if __name__ == "__main__":
    add_cover_urls()