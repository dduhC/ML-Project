"""
Encode papers into TF-IDF and SBERT embeddings, save for downstream FAISS indexing.

Input:
    df with columns ["paper_id", "content"]  -- already preprocessed

Outputs:
    SBERTEmbeddings.npy        -- dense (N, 384) float32, L2-normalized for cosine via inner product
    TfidfEmbeddings.npy        -- dense (N, n_components) float32, SVD-reduced, L2-normalized
    TfidfSparse.npz            -- raw sparse TF-IDF matrix (N, vocab_size), for keyword-level explainability
    tfidf_vectorizer.pkl       -- fitted vectorizer (needed to inspect vocab / top terms later)
    svd_model.pkl              -- fitted SVD model (needed if you want to transform new docs later)
    paper_id_order.csv         -- paper_id in the exact row order of the above arrays
"""

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("./data/train.csv")  # must have columns: paper_id, content
print(df.head())
df = df.reset_index(drop=True)  # CRITICAL: ensures array row i <-> df.iloc[i]

texts = df["content"].tolist()
n = len(texts)
print(f"Loaded {n} documents")

# Save paper_id order for later lookup (embeddings[i] <-> paper_id_order.iloc[i])
df[["paper_id"]].to_csv("paper_id_order.csv", index=False)

# ------------------------------------------------------------------
# 2. SBERT embeddings
# ------------------------------------------------------------------
print("Encoding SBERT embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

# Normalize so inner product == cosine similarity (for FAISS IndexFlatIP)
faiss.normalize_L2(sbert_embeddings)
sbert_embeddings = sbert_embeddings.astype("float32")
np.save("SBERTEmbeddings.npy", sbert_embeddings)
print(f"Saved SBERTEmbeddings.npy with shape {sbert_embeddings.shape}")

# ------------------------------------------------------------------
# 3. TF-IDF embeddings
# ------------------------------------------------------------------
print("Fitting TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=50000,   # cap vocab size to keep sparse matrix manageable
    stop_words="english",
)
tfidf_sparse = tfidf_vectorizer.fit_transform(texts)  # (N, vocab_size) sparse
print(f"TF-IDF sparse matrix shape: {tfidf_sparse.shape}")

# Save sparse matrix (for keyword-level overlap / explainability later)
sp.save_npz("TfidfSparse.npz", tfidf_sparse)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Reduce dimensionality for FAISS (dense, manageable size)
print("Reducing TF-IDF dimensionality with SVD...")
n_components = 256
svd = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_sparse)  # (N, 256) dense

faiss.normalize_L2(tfidf_reduced)
tfidf_reduced = tfidf_reduced.astype("float32")
np.save("TfidfEmbeddings.npy", tfidf_reduced)
print(f"Saved TfidfEmbeddings.npy with shape {tfidf_reduced.shape}")

with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd, f)

print("Done.")