import pandas as pd
import numpy as np
import json
import scipy.sparse
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import GradientBoostingClassifier
import os

print("Loading data...")
df = pd.read_csv('arxiv_dataset/train.csv')
sbert_embeddings = np.load('arxiv_dataset/SBERTEmbeddings_final.npy')

paper_ids = df['paper_id'].values
paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
paper_to_cat = pd.Series(df['primary_category'].values, index=df['paper_id']).to_dict()

recency_scores = df['recency_score'].values

with open('UserDataGenerator/synthetic_users.json', 'r') as f:
    users = json.load(f)['train']

users_id = [u['user_id'] for u in users]
target = {u['user_id']: u['target_paper'] for u in users}
negative = {u['user_id']: u['negative_papers'] for u in users}
history = {u['user_id']: [x[0] for x in u['train_history']] for u in users}

vector_data = np.load('user_profiles_SBERT/user_vectors_halflife_90.npz')
users_vector = vector_data['vectors']

def history_set(user):
    his = set()
    for paper in history[user]:
        if paper in paper_to_cat:
            his.add(paper_to_cat[paper])
    return his

print("Generating training data for LTR...")
X_train = []
y_train = []

for u_idx, user in enumerate(users_id):
    user_his_cat = history_set(user)
    user_vec = users_vector[u_idx].reshape(1, -1)
    
    # Target paper
    t_paper = target[user]
    if t_paper in paper_id_to_idx:
        p_idx = paper_id_to_idx[t_paper]
        cos = cosine_similarity(user_vec, sbert_embeddings[p_idx].reshape(1, -1))[0][0]
        rec = recency_scores[p_idx]
        cat_match = int(paper_to_cat.get(t_paper, '') in user_his_cat)
        
        X_train.append([cos, rec, cat_match])
        y_train.append(1)
        
    # Negative papers
    for n_paper in negative.get(user, []):
        if n_paper in paper_id_to_idx:
            p_idx = paper_id_to_idx[n_paper]
            cos = cosine_similarity(user_vec, sbert_embeddings[p_idx].reshape(1, -1))[0][0]
            rec = recency_scores[p_idx]
            cat_match = int(paper_to_cat.get(n_paper, '') in user_his_cat)
            
            X_train.append([cos, rec, cat_match])
            y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training LTR Model on {X_train.shape[0]} samples...")
ltr_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
ltr_model.fit(X_train, y_train)

importances = ltr_model.feature_importances_
print("Feature Importances:", importances)

joblib.dump(ltr_model, 'ltr_model.pkl')
print("Model saved to ltr_model.pkl")
