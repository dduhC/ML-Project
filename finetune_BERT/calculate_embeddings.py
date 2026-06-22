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

def make_content_df(df: pd.DataFrame) -> pd.DataFrame:
    content = df['title'] + " " + df['primary_category'] + " " + df['abstract']
    return pd.DataFrame({'paper_id': df['paper_id'], 'content': content})

df = pd.read_csv("arxiv_dataset/train.csv")
df_test = pd.read_csv("arxiv_dataset/test.csv")

df = make_content_df(df)
df_test = make_content_df(df_test)

df.to_csv("finetune_BERT/train_content.csv")
df_test.to_csv("finetune_BERT/test_content.csv")

texts = df["content"].tolist()
texts_test = df_test['content'].tolist()
n = len(texts)
print(f"Loaded {n} documents")

# Save paper_id order for later lookup (embeddings[i] <-> paper_id_order.iloc[i])
df[["paper_id"]].to_csv("finetune_BERT/paper_id_order.csv", index=False)

# ------------------------------------------------------------------
# 2. SBERT embeddings
# ------------------------------------------------------------------
print("Encoding SBERT embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.device)
sbert_embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

# Normalize so inner product == cosine similarity (for FAISS IndexFlatIP)
faiss.normalize_L2(sbert_embeddings)
sbert_embeddings = sbert_embeddings.astype("float32")
np.save("arxiv_dataset/SBERTEmbeddings.npy", sbert_embeddings)
print(f"Saved SBERTEmbeddings.npy with shape {sbert_embeddings.shape}")

print("Encoding SBERT embeddings for test set...")
n_test = len(texts_test)
print(f"Loaded {n_test} documents")
sbert_embeddings_test = model.encode(
    texts_test,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

# Normalize so inner product == cosine similarity (for FAISS IndexFlatIP)
faiss.normalize_L2(sbert_embeddings_test)
sbert_embeddings_test = sbert_embeddings_test.astype("float32")
np.save("arxiv_dataset/SBERTEmbeddings_test.npy", sbert_embeddings_test)
print(f"Saved SBERTEmbeddings_test.npy with shape {sbert_embeddings_test.shape}")

# ------------------------------------------------------------------
# 3. TF-IDF embeddings
# ------------------------------------------------------------------
# print("Fitting TF-IDF...")
# tfidf_vectorizer = TfidfVectorizer(
#     max_features=50000,   # cap vocab size to keep sparse matrix manageable
#     stop_words="english",
# )
# tfidf_sparse = tfidf_vectorizer.fit_transform(texts)  # (N, vocab_size) sparse
# print(f"TF-IDF sparse matrix shape: {tfidf_sparse.shape}")

# # Save sparse matrix (for keyword-level overlap / explainability later)
# sp.save_npz("finetune_BERT/TfidfSparse.npz", tfidf_sparse)

# with open("finetune_BERT/tfidf_vectorizer.pkl", "wb") as f:
#     pickle.dump(tfidf_vectorizer, f)

# # Reduce dimensionality for FAISS (dense, manageable size)
# print("Reducing TF-IDF dimensionality with SVD...")
# n_components = 256
# svd = TruncatedSVD(n_components=n_components, random_state=42)
# tfidf_reduced = svd.fit_transform(tfidf_sparse)  # (N, 256) dense

# tfidf_reduced = np.ascontiguousarray(tfidf_reduced.astype("float32"))
# faiss.normalize_L2(tfidf_reduced)
# np.save("finetune_BERT/TfidfEmbeddings.npy", tfidf_reduced)
# print(f"Saved TfidfEmbeddings.npy with shape {tfidf_reduced.shape}")

# with open("finetune_BERT/svd_model.pkl", "wb") as f:
#     pickle.dump(svd, f)

# print("Done.")