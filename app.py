import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from cold_start import build_cold_start_profile
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="ArXiv Recommender", page_icon="📚", layout="wide")

ALPHA = 0.7
HALF_LIFE = 90
LAMBDA_PARAM = 0.7

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

df, sbert_embeddings, synthetic_users, users_vector, user_metadata = load_data()

paper_ids = df['paper_id'].values
paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
recency_scores = df['recency_score'].values

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

def get_recommendations(user_vector, history_paper_ids, top_k=10):
    # final_score = alpha * cosine_sim + (1 - alpha) * recency_score
    cos_scores = cosine_similarity(user_vector.reshape(1, -1), sbert_embeddings).flatten()
    final_scores = ALPHA * cos_scores + (1 - ALPHA) * recency_scores
    
    # filter history
    history_indices = set(paper_id_to_idx[pid] for pid in history_paper_ids if pid in paper_id_to_idx)
    
    # Pre-filter top pool_size candidates
    pool_size = 100
    valid_indices = [i for i in range(len(paper_ids)) if i not in history_indices]
    
    valid_scores = final_scores[valid_indices]
    top_pool_valid_idx = np.argsort(valid_scores)[::-1][:pool_size]
    
    pool_indices = [valid_indices[i] for i in top_pool_valid_idx]
    relevance_dict = {idx: final_scores[idx] for idx in pool_indices}
    
    # Apply MMR
    selected_indices = mmr_rerank(
        candidate_indices=pool_indices,
        relevance_scores=relevance_dict,
        embeddings=sbert_embeddings,
        top_k=top_k,
        lambda_param=LAMBDA_PARAM
    )
    
    rec_df = df.iloc[selected_indices].copy()
    rec_df['hybrid_score'] = [final_scores[i] for i in selected_indices]
    rec_df['Rank'] = range(1, len(selected_indices) + 1)
    
    return rec_df[['Rank', 'paper_id', 'title', 'primary_category', 'hybrid_score']]

st.title("📚 Personalized ArXiv Paper Recommender")
st.markdown("A hybrid recommendation system utilizing SBERT embeddings, temporal weighting, and MMR diversification.")

user_type = st.radio("Select User Type:", ["Existing User", "New User (Cold Start)"], horizontal=True)

if user_type == "Existing User":
    st.subheader("Select an Existing User")
    user_choices = [u['user_id'] for u in user_metadata]
    selected_user_id = st.selectbox("User ID:", user_choices)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing user profile and computing hybrid scores..."):
            user_idx = user_choices.index(selected_user_id)
            user_vec = users_vector[user_idx]
            
            synth_user = next((u for u in synthetic_users if u['user_id'] == selected_user_id), None)
            history_ids = [x[0] for x in synth_user['train_history']] if synth_user else []
            
            st.markdown("### 📖 Reading History")
            history_df = df[df['paper_id'].isin(history_ids)]
            st.dataframe(history_df[['paper_id', 'title', 'primary_category']], use_container_width=True)
            
            st.markdown(f"### 🎯 Top 10 Recommendations for `{selected_user_id}`")
            recs = get_recommendations(user_vec, history_ids)
            st.dataframe(recs, use_container_width=True)

else:
    st.subheader("Cold-Start Category Onboarding")
    st.markdown("Select 1 to 3 categories that interest you to bootstrap your profile.")
    
    all_categories = sorted(df['primary_category'].dropna().unique())
    selected_cats = st.multiselect("Favorite Categories:", all_categories, max_selections=3)
    
    if st.button("Generate Recommendations", type="primary"):
        if not selected_cats:
            st.error("Please select at least one category to continue.")
        else:
            with st.spinner("Bootstrapping profile and computing hybrid scores..."):
                pseudo_vec, seed_indices = build_cold_start_profile(
                    df,
                    sbert_embeddings,
                    selected_cats,
                    papers_per_category=20,
                )

                seed_counts = (
                    df.iloc[seed_indices]['primary_category']
                    .value_counts()
                    .reindex(selected_cats, fill_value=0)
                )
                seed_summary = ", ".join(
                    f"{category}: {count}"
                    for category, count in seed_counts.items()
                )
                st.caption(f"Profile initialized from {seed_summary} recent papers.")
                
                st.markdown("### 🎯 Top 10 Recommendations for You")
                recs = get_recommendations(pseudo_vec, history_paper_ids=[])
                st.dataframe(recs, use_container_width=True)
