import chromadb
import numpy as np
from src.embedding_function import embed_query

class VectorStore:
    def __init__(self, persist_path="vectorstore/chroma", collection_name="books"):
        # Load persistent Chroma DB
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_collection(collection_name)

    def recommend(self, text_query, repo=None, use_seed=True, alpha=0.7, n_results=5):
        """
        Hybrid recommendation:
        semantic query + seed profile
        """

        # Step 1 — embed user query
        query_embedding = embed_query(text_query)

        combined_embedding = np.array(query_embedding)

        # Step 2 — compute seed embedding (if repo provided)
        if use_seed and repo is not None:
            seed_embedding = repo.compute_seed_embedding(self.collection)

            if seed_embedding is not None:
                combined_embedding = (
                    alpha * np.array(query_embedding)
                    + (1 - alpha) * np.array(seed_embedding)
                )
        combined_embedding = np.array(combined_embedding)

        # Step 3 — query Chroma
        results = self.collection.query(
            query_embeddings=[combined_embedding.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        return results