import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

df = pd.read_csv("data/processed/goodreads_library_final_descriptions.csv")
persist_path = os.path.abspath("vectorstore/chroma")

# Ensure no missing final_description
df = df.dropna(subset=["Final Description"])

# Combine fields for embedding
# Example: "Title by Author: Description"
df["embedding_text"] = df.apply(
    lambda row: f"{row['Title']} by {row['Author']}: {row['Final Description']}", axis=1
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(df["embedding_text"].tolist(), show_progress_bar=True)

# Initialize Chroma
client = chromadb.PersistentClient(
    path=persist_path  # folder for persistence
)

collection = client.get_or_create_collection(
    name="books",
    embedding_function=None  # we provide our own embeddings
)

# Store metadata
metadata_columns = ["Title", "Author", "ISBN", "ISBN13","Number of Pages", "Original Publication Year", 
                    "My Rating", "Average Rating", "Final Description Source", 
                    "Final Description Score", "Tag List"]
metadata = df[metadata_columns].to_dict(orient="records")

# Ensure unique IDs for Chroma
df = df.reset_index()  # ensures unique integer index
ids = df.index.astype(str).tolist()  # unique string IDs for Chroma

# Insert into Chroma
collection.add(
    ids=ids,   # unique IDs
    embeddings=embeddings.tolist(),
    metadatas=metadata,
    documents=df["Final Description"].tolist()  # optional, raw text
)

print(f"Successfully ingested {len(df)} books into Chroma!")