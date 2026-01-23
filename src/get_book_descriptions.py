import pandas as pd
import requests
import time

INPUT_PATH = "data/goodreads_library_cleaned.csv"
OUTPUT_PATH = "data/goodreads_library_cleaned_descriptions.csv"

def get_description_by_isbn(isbn):
    if pd.isna(isbn) or str(isbn).strip() == "":
        return None

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"isbn:{isbn}"}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if "items" in data:
            volume = data["items"][0]["volumeInfo"]
            return volume.get("description", None)

    except Exception as e:
        print(f"ISBN error {isbn}: {e}")

    return None


def get_description_by_title_author(title, author):
    query = f'intitle:{title}+inauthor:{author}'
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if "items" in data:
            volume = data["items"][0]["volumeInfo"]
            return volume.get("description", None)

    except Exception as e:
        print(f"Title/Author error {title}: {e}")

    return None


def fetch_description(row):
    # Try ISBN13 first
    desc = get_description_by_isbn(row["ISBN13"])

    # Fallback to ISBN
    if not desc:
        desc = get_description_by_isbn(row["ISBN"])

    # Fallback to title + author
    if not desc:
        desc = get_description_by_title_author(row["Title"], row["Author"])

    return desc


def main():
    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded {len(df)} books")

    descriptions = []

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] Fetching: {row['Title']}")

        desc = fetch_description(row)
        descriptions.append(desc)

        time.sleep(0.2)

    df["Description"] = descriptions

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
