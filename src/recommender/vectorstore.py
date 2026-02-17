import chromadb
from src.embedding_function import embed_query

class VectorStore:
    def __init__(self, persist_path="vectorstore/chroma", collection_name="books"):
        # Load persistent Chroma DB
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_collection(collection_name)

    def recommend(self, text_query, n_results=5):
        """
        Embed the text_query using the shared embedding function
        and return top-n similar books with metadata.
        """
        # embed query
        query_embedding = embed_query(text_query)

        # query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        return results