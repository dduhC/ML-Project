import pandas as pd
import numpy as np
import scipy.sparse as sp
import faiss
import pickle
import random
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. Load embeddings và metadata
# ─────────────────────────────────────────────

sbert_embeddings = np.load("SBERTEmbeddings.npy")  #?: shape (N, 384), đã L2-normalize
tfidf_embeddings = np.load("TfidfEmbeddings.npy")  #?: shape (N, 256), đã L2-normalize
tfidf_sparse     = sp.load_npz("TfidfSparse.npz")  #?: sparse (N, vocab), dùng để tính raw TF-IDF similarity nếu cần

paper_id_order = pd.read_csv("paper_id_order.csv")  #?: ánh xạ row index → paper_id
df = pd.read_csv("./data/train.csv").reset_index(drop=True)
#! df phải reset_index và giữ đúng thứ tự như lúc encode — không được shuffle hay filter sau bước này

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

n = len(df)
print(f"Loaded {n} papers")

# ─────────────────────────────────────────────
# 2. Build FAISS index
# ─────────────────────────────────────────────

sbert_index = faiss.IndexFlatIP(sbert_embeddings.shape[1])
sbert_index.add(sbert_embeddings)  #?: IndexFlatIP + L2-normalized vector → inner product = cosine similarity

tfidf_index = faiss.IndexFlatIP(tfidf_embeddings.shape[1])
tfidf_index.add(tfidf_embeddings)

print("FAISS indexes built")

# ─────────────────────────────────────────────
# 3. Precompute lookup structures
# ─────────────────────────────────────────────

#? group paper index theo category để filter nhanh, không cần scan toàn bộ df mỗi lần
category_to_indices = defaultdict(list)
for i, row in df.iterrows():
    category_to_indices[row["primary_category"]].append(i)

#? group paper index theo author — dùng first_author để tránh ambiguity khi paper có nhiều author
author_to_indices = defaultdict(list)
for i, row in df.iterrows():
    author_to_indices[row["first_author"]].append(i)

# ─────────────────────────────────────────────
# 4. Config
# ─────────────────────────────────────────────

K_NEIGHBORS      = 50    #?: lấy dư neighbor để sau filter còn đủ candidate
SBERT_POS3_THRESH = 0.85 #?: threshold cao cho cross-category positive — chỉ lấy cặp thực sự gần
N_ANCHORS        = 30000 #?: sample subset anchor, không cần tất cả 130k
RATIO_POS1       = 0.40  #?: category + TF-IDF cao
RATIO_POS2       = 0.30  #?: author + category
RATIO_POS3       = 0.15  #?: cross-category SBERT
RATIO_NEG1       = 0.30  #?: hard negative: TF-IDF gần nhưng khác category
RATIO_NEG2       = 0.10  #?: easy negative: random sample
RATIO_NEG3       = 0.30  #?: same-category hard negative: cùng category nhưng TF-IDF thấp
#! tổng ratio positive và negative không cần bằng nhau — mỗi bên normalize riêng khi sample

# ─────────────────────────────────────────────
# 5. Helper functions
# ─────────────────────────────────────────────

#@ Query FAISS index, trả về (indices, similarities) đã loại bỏ chính anchor
#@ Trả về arrays shape (k,) thay vì (1, k) để tiện dùng
def query_neighbors(index, embedding, anchor_idx, k=K_NEIGHBORS):
    sims, idxs = index.search(embedding.reshape(1, -1), k + 1)  #?: +1 vì kết quả luôn include chính nó
    sims, idxs = sims[0], idxs[0]
    mask = idxs != anchor_idx  #?: loại anchor ra khỏi kết quả
    return idxs[mask][:k], sims[mask][:k]

#@ Chọn positive theo strategy 1: cùng category, TF-IDF similarity cao (top half của neighbors)
#@ Trả về index của positive paper, hoặc None nếu không tìm được
def get_positive_cat_tfidf(anchor_idx):
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    idxs, sims = query_neighbors(tfidf_index, tfidf_embeddings[anchor_idx], anchor_idx)
    for idx, sim in zip(idxs, sims):
        if df.iloc[idx]["primary_category"] == anchor_cat:
            return int(idx)  #?: lấy neighbor gần nhất cùng category theo TF-IDF
    return None

#@ Chọn positive theo strategy 2: cùng author + cùng category
#@ Skip nếu author chỉ có 1 paper (không có candidate nào khác)
def get_positive_author(anchor_idx):
    author = df.iloc[anchor_idx]["first_author"]
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    candidates = [i for i in author_to_indices[author]
                  if i != anchor_idx and df.iloc[i]["primary_category"] == anchor_cat]
    #! cần check cả điều kiện category — cùng author khác category là false positive
    if len(candidates) < 1:
        return None
    return random.choice(candidates)

#@ Chọn positive theo strategy 3: khác category nhưng SBERT similarity cao (cross-category)
#@ Chỉ lấy nếu similarity vượt threshold cao để giảm risk false positive
def get_positive_cross_category(anchor_idx):
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    idxs, sims = query_neighbors(sbert_index, sbert_embeddings[anchor_idx], anchor_idx)
    for idx, sim in zip(idxs, sims):
        if df.iloc[idx]["primary_category"] != anchor_cat and sim >= SBERT_POS3_THRESH:
            return int(idx)
    return None  #?: nếu không có neighbor nào vượt threshold thì bỏ qua anchor này cho strategy 3

#@ Chọn negative theo strategy 1: TF-IDF gần (top neighbors) nhưng khác category và khác author — hard negative
def get_negative_hard_tfidf(anchor_idx):
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    anchor_author = df.iloc[anchor_idx]["first_author"]
    idxs, _ = query_neighbors(tfidf_index, tfidf_embeddings[anchor_idx], anchor_idx)
    for idx in idxs:
        row = df.iloc[idx]
        if row["primary_category"] != anchor_cat and row["first_author"] != anchor_author:
            return int(idx)
    return None

#@ Chọn negative theo strategy 2: easy negative — random sample từ toàn dataset
#@ Với 130k paper, random pair gần như chắc chắn có similarity thấp
def get_negative_easy(anchor_idx):
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    for _ in range(20):  #?: thử tối đa 20 lần để tránh vô tình lấy cùng category
        idx = random.randint(0, n - 1)
        if idx != anchor_idx and df.iloc[idx]["primary_category"] != anchor_cat:
            return idx
    return None

#@ Chọn negative theo strategy 3: cùng category nhưng TF-IDF thấp + SBERT thấp — same-category hard negative
#@ Lấy từ bottom neighbors (TF-IDF xa) trong cùng category
def get_negative_same_cat_hard(anchor_idx):
    anchor_cat = df.iloc[anchor_idx]["primary_category"]
    same_cat = [i for i in category_to_indices[anchor_cat] if i != anchor_idx]
    if len(same_cat) < 10:
        return None
    #? sample ngẫu nhiên từ pool cùng category — không lấy top TF-IDF neighbor (đã là positive)
    #? lấy từ random sample thay vì bottom FAISS vì FAISS không tối ưu cho "xa nhất"
    candidates = random.sample(same_cat, min(20, len(same_cat)))
    sims = [float(np.dot(tfidf_embeddings[anchor_idx], tfidf_embeddings[c])) for c in candidates]
    #? chọn candidate có TF-IDF similarity thấp nhất trong pool sample
    worst_idx = candidates[int(np.argmin(sims))]
    #? kiểm tra thêm SBERT similarity cũng thấp để đảm bảo đây thực sự là hard negative
    sbert_sim = float(np.dot(sbert_embeddings[anchor_idx], sbert_embeddings[worst_idx]))
    if sbert_sim < 0.6:
        return worst_idx
    return None

# ─────────────────────────────────────────────
# 6. Generate triplets
# ─────────────────────────────────────────────

anchor_indices = random.sample(range(n), min(N_ANCHORS, n))
#? sample anchor subset — không cần tất cả 130k, 20-30k là đủ để train

triplets = []

for anchor_idx in anchor_indices:
    anchor_paper_id = df.iloc[anchor_idx]["paper_id"]

    # --- positive candidates theo từng strategy ---
    pos_candidates = []

    if random.random() < RATIO_POS1:
        p = get_positive_cat_tfidf(anchor_idx)
        if p is not None:
            pos_candidates.append(("pos1_cat_tfidf", p))

    if random.random() < RATIO_POS2:
        p = get_positive_author(anchor_idx)
        if p is not None:
            pos_candidates.append(("pos2_author_cat", p))

    if random.random() < RATIO_POS3:
        p = get_positive_cross_category(anchor_idx)
        if p is not None:
            pos_candidates.append(("pos3_cross_cat", p))

    # --- negative candidates theo từng strategy ---
    neg_candidates = []

    if random.random() < RATIO_NEG1:
        neg = get_negative_hard_tfidf(anchor_idx)
        if neg is not None:
            neg_candidates.append(("neg1_hard_tfidf", neg))

    if random.random() < RATIO_NEG2:
        neg = get_negative_easy(anchor_idx)
        if neg is not None:
            neg_candidates.append(("neg2_easy", neg))

    if random.random() < RATIO_NEG3:
        neg = get_negative_same_cat_hard(anchor_idx)
        if neg is not None:
            neg_candidates.append(("neg3_same_cat_hard", neg))

    # --- tạo tất cả cặp (positive, negative) cho anchor này ---
    for pos_type, pos_idx in pos_candidates:
        for neg_type, neg_idx in neg_candidates:
            triplets.append({
                "anchor_id"   : anchor_paper_id,
                "positive_id" : df.iloc[pos_idx]["paper_id"],
                "negative_id" : df.iloc[neg_idx]["paper_id"],
                "pos_strategy": pos_type,
                "neg_strategy": neg_type,
            })

print(f"Generated {len(triplets)} triplets")

# ─────────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────────

triplets_df = pd.DataFrame(triplets)
triplets_df.to_csv("triplets.csv", index=False)
print("Saved triplets.csv")
print(triplets_df["pos_strategy"].value_counts())
print(triplets_df["neg_strategy"].value_counts())
#? in distribution để verify ratio giữa các strategy trước khi dùng