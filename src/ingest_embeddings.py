"""
Book embedding and ingestion script for a book recommender.

This script:
- loads a processed Goodreads dataset
- builds semantic embedding text from title, author, description, and selected tags
- generates sentence embeddings
- stores embeddings and metadata in a persistent Chroma vector store

Semantic tags are embedded to support taste-aware recommendations,
while logistical tags are kept only as metadata for filtering.
"""

from http import client
from xml.parsers.expat import model
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

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

# Semantic expensions for tags used in embeddings (own, re-read, read, to-read, read-in-german are not embedding-expanded as they are more about personal reading status rather than content)
TAG_EXPANSIONS = {
    "american": "American literature, books by authors from the United States",
    "auto-biography": "autobiography, memoir, personal life stories written by the author",
    "children": "children's books, literature for young readers, child development",
    "engineering": "engineering, technical books, applied science and technology",
    "ex-yu": "former Yugoslavia, Balkan literature, Ex-Yugoslav authors",
    "favorites": "personal favorites, books I especially liked",
    "feminism": "feminist literature, women's rights, gender equality",
    "immigration": "immigration, migration, identity, displacement and belonging, expatriate experiences",
    "indian": "Indian literature, books by authors from India",
    "latin-america": "Latin American literature, South American authors",
    "lists": "curated lists, book recommendations, thematic collections",
    "philosophy": "philosophy, philosophical thinking, ethics, metaphysics",
    "play": "theatre plays, drama written for stage performance",
    "poetry": "poetry, poems, lyrical literature",
    "psychology": "psychology, human behavior, mental processes",
    "russian": "Russian literature, books by Russian authors",
    "sci-fi": "science fiction, speculative and futuristic literature",
    "science": "science, popular science, scientific topics",
    "travel": "travel literature, journeys, places and cultures"
}

def get_active_tags(row, tag_columns):
    """
    Return a list of tag names that are active (True) for a given row.
    """
    return [tag for tag in tag_columns if row.get(tag, False)]

def build_embedding_text(row):
    """
    Build embedding text for a single book.

    Includes:
    - title
    - author
    - description
    - semantic and preference tags (expanded)
    Logistical tags are not included in the embedding text, but can be used for filtering in Chroma.
    """
    parts = [
        f"Title: {row['Title']}",
        f"Author: {row['Author']}",
        f"Description: {row['Final Description']}",
    ]

    # Only include tags that have semantic expansions
    embed_tags = [
        tag for tag in TAG_EXPANSIONS
        if row.get(tag, False)
    ]

    if embed_tags:
        parts.append("User tags: " + ", ".join(embed_tags))

        for tag in embed_tags:
            parts.append(TAG_EXPANSIONS[tag])

    return "\n".join(parts)

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Generate sentence embeddings for a list of texts using a SentenceTransformer model.
    """
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)


def ingest_into_chroma(df, embeddings, persist_path):
    """
    Store embeddings, metadata, and documents into a persistent Chroma collection.
    """
    client = chromadb.PersistentClient(path=persist_path)

    collection = client.get_or_create_collection(
        name="books",
        embedding_function=None
    )

    metadata_columns = [
        "Title",
        "Author",
        "ISBN",
        "ISBN13",
        "Number of Pages",
        "Original Publication Year",
        "My Rating",
        "Average Rating",
        "Final Description Source",
        "Final Description Score",
    ] + TAG_COLUMNS

    metadatas = df[metadata_columns].to_dict(orient="records")

    # Ensure unique IDs for Chroma
    df = df.reset_index()  # ensures unique integer index
    ids = df.index.astype(str).tolist()  # unique string IDs for Chroma

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        documents=df["Final Description"].tolist(),
    )

    return collection


def main():
    """
    Main ingestion pipeline:
    - load data
    - prepare embedding text
    - generate embeddings
    - ingest into Chroma
    """
    data_path = "data/processed/goodreads_library_final_descriptions.csv"
    persist_path = os.path.abspath("vectorstore/chroma")

    df = pd.read_csv(data_path)

    # Ensure boolean tag columns (required for Chroma filtering)
    df[TAG_COLUMNS] = df[TAG_COLUMNS].fillna(0).astype(bool)

    # Drop rows without a final description
    df = df.dropna(subset=["Final Description"])

    # Build embedding text
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)

    # Generate embeddings
    embeddings = generate_embeddings(df["embedding_text"].tolist())

    # Ingest into Chroma
    ingest_into_chroma(df, embeddings, persist_path)

    print(f"Successfully ingested {len(df)} books into Chroma!")


if __name__ == "__main__":
    main()
