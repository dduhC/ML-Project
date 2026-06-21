import streamlit as st
import pandas as pd
import numpy as np
import json
import scipy.sparse
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ArXiv Recommender", page_icon="📚", layout="wide")

LAMBDA_PARAM = 0.7
HALF_LIFE = 90

@st.cache_resource(show_spinner=False)
def load_models():
    # Cache busted to reload the 3-feature model
    ltr_model = joblib.load('ltr_model.pkl')
    preprocessor = joblib.load('arxiv_dataset/text_preprocessor.pkl')
    tfidf_matrix = scipy.sparse.load_npz('arxiv_dataset/tfidf_matrix.npz')
    feature_names = preprocessor.vectorizer.get_feature_names_out()
    return ltr_model, tfidf_matrix, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv('arxiv_dataset/train.csv')
    sbert_embeddings = np.load('arxiv_dataset/SBERTEmbeddings_final.npy')
    
    with open('UserDataGenerator/synthetic_users.json', 'r') as f:
        synthetic_users = json.load(f)['train']
        
    vector_data = np.load(f'user_profiles_SBERT/user_vectors_halflife_{HALF_LIFE}.npz')
    users_vector = vector_data['vectors']
    
    with open(f'user_profiles_SBERT/user_metadata_halflife_{HALF_LIFE}.json', 'r') as f:
        user_metadata = json.load(f)
        
    return df, sbert_embeddings, synthetic_users, users_vector, user_metadata

# Load everything
df, sbert_embeddings, synthetic_users, users_vector, user_metadata = load_data()
ltr_model, tfidf_matrix, feature_names = load_models()

paper_ids = df['paper_id'].values
paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
paper_to_cat = pd.Series(df['primary_category'].values, index=df['paper_id']).to_dict()

recency_scores = df['recency_score'].values

def get_keywords(paper_idx):
    row = tfidf_matrix[paper_idx].toarray()[0]
    top_idx = np.argsort(row)[::-1][:5]
    return ", ".join([feature_names[i] for i in top_idx if row[i] > 0])

def mmr_rerank(candidate_indices, relevance_scores, embeddings, top_k=10, lambda_param=0.7):
    selected = []
    remaining = list(candidate_indices)
    while len(selected) < top_k and remaining:
        if not selected:
            best_idx = remaining[int(np.argmax([relevance_scores[i] for i in remaining]))]
        else:
            selected_embeds = embeddings[selected]
            mmr_scores = []
            for idx in remaining:
                relevance = relevance_scores[idx]
                sims = cosine_similarity(embeddings[idx].reshape(1, -1), selected_embeds)[0]
                redundancy = sims.max()
                mmr_scores.append(lambda_param * relevance - (1 - lambda_param) * redundancy)
            best_idx = remaining[int(np.argmax(mmr_scores))]
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def get_explainable_recommendations(user_vector, history_paper_ids, top_k=10):
    history_indices = set(paper_id_to_idx[pid] for pid in history_paper_ids if pid in paper_id_to_idx)
    user_his_cat = set()
    for pid in history_paper_ids:
        if pid in paper_to_cat:
            user_his_cat.add(paper_to_cat[pid])
            
    # Pool all candidates (exclude history)
    valid_indices = [i for i in range(len(paper_ids)) if i not in history_indices]
    
    # Calculate features for LTR
    cand_embeds = sbert_embeddings[valid_indices]
    cos_sims = cosine_similarity(user_vector.reshape(1, -1), cand_embeds).flatten()
    recs = recency_scores[valid_indices]
    cat_matches = [int(paper_to_cat.get(paper_ids[i], '') in user_his_cat) for i in valid_indices]
    
    # Filter top 100 via basic hybrid heuristic before LTR (to save time), or just predict all
    # To keep it rigorous, predict all with LTR:
    X_candidates = np.column_stack((cos_sims, recs, cat_matches))
    ltr_probs = ltr_model.predict_proba(X_candidates)[:, 1]
    
    # Select top 100 for MMR
    pool_size = 100
    top_100_idx_in_valid = np.argsort(ltr_probs)[::-1][:pool_size]
    
    pool_indices = [valid_indices[i] for i in top_100_idx_in_valid]
    relevance_dict = {idx: ltr_probs[i] for i, idx in zip(top_100_idx_in_valid, pool_indices)}
    
    # MMR
    selected_indices = mmr_rerank(
        candidate_indices=pool_indices,
        relevance_scores=relevance_dict,
        embeddings=sbert_embeddings,
        top_k=top_k,
        lambda_param=LAMBDA_PARAM
    )
    
    # Build explanation table
    recs_list = []
    for rank, idx in enumerate(selected_indices, start=1):
        pid = paper_ids[idx]
        
        # We need the original index inside valid_indices to extract feature values
        v_i = valid_indices.index(idx)
        
        sim_score = cos_sims[v_i]
        rec_score = recs[v_i]
        ltr_score = ltr_probs[v_i]
        kw = get_keywords(idx)
        
        recs_list.append({
            'Rank': rank,
            'Paper ID': pid,
            'Title': df.iloc[idx]['title'],
            'Category': df.iloc[idx]['primary_category'],
            'TF-IDF Keywords': kw,
            'LTR Score': round(ltr_score, 4),
            'Semantic Similarity': round(sim_score, 4),
            'Recency Score': round(rec_score, 4)
        })
        
    return pd.DataFrame(recs_list)

st.title("📚 Personalized ArXiv Paper Recommender")
st.markdown("A recommendation system featuring LTR (Gradient Boosting), MMR Diversification, and Explainable AI.")

user_type = st.radio("Select User Type:", ["Existing User", "New User (Cold Start)"], horizontal=True)

if user_type == "Existing User":
    user_choices = [u['user_id'] for u in user_metadata]
    selected_user_id = st.selectbox("Select User:", user_choices)
    
    if st.button("Generate Explainable Recommendations", type="primary"):
        with st.spinner("Executing LTR Pipeline and Explanations..."):
            user_idx = user_choices.index(selected_user_id)
            user_vec = users_vector[user_idx]
            synth_user = next((u for u in synthetic_users if u['user_id'] == selected_user_id), None)
            history_ids = [x[0] for x in synth_user['train_history']] if synth_user else []
            
            st.markdown("### 📖 Reading History")
            history_df = df[df['paper_id'].isin(history_ids)]
            st.dataframe(history_df[['paper_id', 'title', 'primary_category']], use_container_width=True)
            
            st.markdown(f"### 🎯 Top 10 Recommendations & Explanations")
            recs_df = get_explainable_recommendations(user_vec, history_ids)
            st.dataframe(recs_df, use_container_width=True)
            
            # Show feature importances from LTR model
            st.markdown("#### 🧠 LTR Model Feature Importance Breakdown")
            fi = ltr_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": ["Cosine Similarity", "Recency Score", "Category Match"],
                "Importance Weight": [f"{v*100:.2f}%" for v in fi]
            })
            st.table(fi_df)

else:
    st.subheader("Cold-Start Category Onboarding")
    all_categories = sorted(df['primary_category'].dropna().unique())
    selected_cats = st.multiselect("Favorite Categories:", all_categories, max_selections=3)
    
    if st.button("Generate Explainable Recommendations", type="primary"):
        if not selected_cats:
            st.error("Please select at least one category.")
        else:
            with st.spinner("Bootstrapping profile..."):
                cat_df = df[df['primary_category'].isin(selected_cats)]
                top_recent_idx = cat_df.sort_values(by='recency_score', ascending=False).head(10).index
                pseudo_vec = np.mean(sbert_embeddings[top_recent_idx], axis=0)
                
                # Treat top_recent_idx as "history" to set categories
                pseudo_history = df.iloc[top_recent_idx]['paper_id'].tolist()
                
                st.markdown("### 🎯 Top 10 Recommendations & Explanations")
                recs_df = get_explainable_recommendations(pseudo_vec, pseudo_history)
                st.dataframe(recs_df, use_container_width=True)
