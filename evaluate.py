import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("Loading data for evaluation...")
df = pd.read_csv('arxiv_dataset/train.csv')
sbert_embeddings = np.load('arxiv_dataset/SBERTEmbeddings_final.npy')

ltr_model = joblib.load('ltr_model.pkl')

paper_ids = df['paper_id'].values
paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
paper_to_cat = pd.Series(df['primary_category'].values, index=df['paper_id']).to_dict()

recency_scores = df['recency_score'].values

with open('UserDataGenerator/synthetic_users.json', 'r') as f:
    users_data = json.load(f)
    users = users_data.get('test', users_data['train'][:100])

vector_data = np.load('user_profiles_SBERT/user_vectors_halflife_90.npz')
users_vector = vector_data['vectors']

def history_set(user_history):
    his = set()
    for paper, _ in user_history:
        if paper in paper_to_cat:
            his.add(paper_to_cat[paper])
    return his

K = 10
hit_rate_sum = 0
ndcg_sum = 0
precision_sum = 0
sem_coherence_sum = 0
ild_sum = 0

n_eval_users = len(users)

print(f"Evaluating on {n_eval_users} users...")

for u_idx, user in enumerate(users):
    user_id = user['user_id']
    user_his_cat = history_set(user['train_history'])
    user_vec = users_vector[u_idx].reshape(1, -1)
    
    target_paper = user['target_paper']
    
    candidate_ids = [target_paper] + user['negative_papers']
    if len(candidate_ids) < 100:
        random_candidates = list(np.random.choice(paper_ids, 100 - len(candidate_ids), replace=False))
        candidate_ids.extend(random_candidates)
    
    candidate_ids = [pid for pid in candidate_ids if pid in paper_id_to_idx]
    cand_indices = [paper_id_to_idx[pid] for pid in candidate_ids]
    
    cand_embeds = sbert_embeddings[cand_indices]
    
    cos_sims = cosine_similarity(user_vec, cand_embeds).flatten()
    recs = recency_scores[cand_indices]
    cat_matches = [int(paper_to_cat.get(pid, '') in user_his_cat) for pid in candidate_ids]
    
    X_test = np.column_stack((cos_sims, recs, cat_matches))
    preds = ltr_model.predict_proba(X_test)[:, 1]
    
    top_k_ranks = np.argsort(preds)[::-1][:K]
    top_k_indices = [cand_indices[i] for i in top_k_ranks]
    top_k_ids = [candidate_ids[i] for i in top_k_ranks]
    
    hit = 1 if target_paper in top_k_ids else 0
    hit_rate_sum += hit
    
    dcg = 0
    idcg = 1.0
    for rank, pid in enumerate(top_k_ids, start=1):
        if pid == target_paper:
            dcg += 1.0 / np.log2(rank + 1)
    ndcg_sum += (dcg / idcg)
    
    cat_hits = sum([1 for pid in top_k_ids if paper_to_cat.get(pid, '') in user_his_cat])
    precision_sum += (cat_hits / K)
    
    top_k_embeds = sbert_embeddings[top_k_indices]
    coherence = cosine_similarity(user_vec, top_k_embeds).mean()
    sem_coherence_sum += coherence
    
    if K > 1:
        sim_matrix = cosine_similarity(top_k_embeds)
        np.fill_diagonal(sim_matrix, 0)
        ild = 1.0 - (sim_matrix.sum() / (K * (K - 1)))
        ild_sum += ild

print("\n--- EVALUATION METRICS ---")
print(f"Category Precision@{K}: {precision_sum / n_eval_users:.4f}")
print(f"NDCG@{K}            : {ndcg_sum / n_eval_users:.4f}")
print(f"Hit Rate@{K}        : {hit_rate_sum / n_eval_users:.4f}")
print(f"Semantic Coherence : {sem_coherence_sum / n_eval_users:.4f}")
print(f"Intra-list Diversity: {ild_sum / n_eval_users:.4f}")
print("--------------------------")
