import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import umap
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/goodreads_library_descriptions_cleaned.csv")

# Fill missing descriptions with empty string and ensure string type
df['Description'] = df['Description'].fillna("").astype(str)

# Compute embeddings in batches
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

batch_size = 64
embeddings_list = []
descriptions = df['Description'].tolist()

for i in range(0, len(descriptions), batch_size):
    batch = descriptions[i:i+batch_size]
    batch_emb = model.encode(batch, show_progress_bar=True, normalize_embeddings=True)
    embeddings_list.append(batch_emb)

embeddings = np.vstack(embeddings_list).astype('float32')  # Convert to float32 for FAISS

os.makedirs("data/embeddings", exist_ok=True)
np.save("data/embeddings/goodreads_descriptions_embeddings.npy", embeddings)

# Compute k-nearest neighbor embedding density (memory-safe)
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)  # Inner product = cosine if embeddings are normalized
index.add(embeddings)

faiss.write_index(index, "data/embeddings/goodreads_embeddings_index.faiss")

k = 10  # number of neighbors to consider
densities = []

batch_size_knn = 100  # batch size for k-NN search
for i in range(0, embeddings.shape[0], batch_size_knn):
    batch = embeddings[i:i+batch_size_knn]
    sims, _ = index.search(batch, k+1)  # +1 because first neighbor is self
    # Average similarity to neighbors excluding self
    densities.extend(sims[:, 1:].mean(axis=1))

df['embedding_density'] = densities

df.to_csv("data/goodreads_library_descriptions_with_density.csv", index=False)

# Sort and inspect
df_sorted = df.sort_values("embedding_density", ascending=True)  # low density first

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

print("=== 30 lowest-density descriptions ===")
print(df_sorted[['Title', 'Author', 'Description_single_line', 'embedding_density']].head(30))

# Filter low-quality descriptions (bottom 5%)
threshold = df['embedding_density'].quantile(0.05)
low_quality = df[df['embedding_density'] < threshold]

print(f"\nNumber of low-quality descriptions (bottom 5%): {len(low_quality)}")

# Optional: UMAP visualization
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
coords = reducer.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df['embedding_density'], s=10, cmap='viridis')
plt.colorbar(scatter, label="Embedding density")
plt.title("Book Description Embedding Space")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.show()




