import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

df = pd.read_csv("finetune_BERT/train_content.csv")
texts = df["content"].tolist()
n = len(texts)
print(f"Loaded {n} documents")

# Save paper_id order for later lookup (embeddings[i] <-> paper_id_order.iloc[i])
df[["paper_id"]].to_csv("finetune_BERT/paper_id_order.csv", index=False)

# ------------------------------------------------------------------
# 2. SBERT embeddings
# ------------------------------------------------------------------
print("Encoding SBERT embeddings...")
model = SentenceTransformer("finetune_BERT/model_checkpoint/finetuned-arxiv-sbert_v2")
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
print(sbert_embeddings.shape)
np.save("finetune_BERT/SBERTEmbeddings_final.npy", sbert_embeddings)
print(f"Saved SBERTEmbeddings_final.npy with shape {sbert_embeddings.shape}")
