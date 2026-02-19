import pandas as pd
import requests
from pathlib import Path
from time import sleep

DATA_PATH = Path("data/processed/goodreads_library_final_descriptions.csv")
COVER_DIR = Path("covers")
COVER_DIR.mkdir(parents=True, exist_ok=True)

def filename_from_isbn(isbn):
    if not isbn or str(isbn) == "0":
        return None
    return COVER_DIR / f"{isbn}.jpg"

def download_cover(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 500:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def cache_covers():
    df = pd.read_csv(DATA_PATH)

    local_paths = []

    for i, row in df.iterrows():
        isbn = row.get("ISBN")
        url = row.get("Cover URL")

        path = filename_from_isbn(isbn)

        if path is None:
            local_paths.append("")
            continue

        if not path.exists():
            print(f"Downloading {isbn}...")
            success = download_cover(url, path)
            sleep(0.2)  

            if not success:
                path = ""

        local_paths.append(str(path))

    df["Cover Path"] = local_paths
    df.to_csv(DATA_PATH, index=False)
    print("All covers cached!")

if __name__ == "__main__":
    cache_covers()