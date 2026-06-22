import streamlit as st
import pandas as pd
import numpy as np
import json
import scipy.sparse
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import TextPreprocessor
from cold_start import build_cold_start_profile
from user_tsne import create_user_profile_tsne
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="ArXiv Recommender", page_icon="📚", layout="wide")

LAMBDA_PARAM = 0.7
HALF_LIFE = 90

@st.cache_resource(show_spinner=False)
def load_models():
    ltr_sbert = joblib.load('ltr_sbert.pkl')
    ltr_tfidf = joblib.load('ltr_tfidf.pkl')
    preprocessor = joblib.load('arxiv_dataset/text_preprocessor.pkl')
    feature_names = preprocessor.vectorizer.get_feature_names_out()
    return ltr_sbert, ltr_tfidf, feature_names

@st.cache_data
def load_data():
    # Use TEST set for evaluation and recommendation pool
    df = pd.read_csv('arxiv_dataset/test.csv')
    
    # Load TF-IDF test matrix (Sparse)
    tfidf_test_sparse = scipy.sparse.load_npz('arxiv_dataset/tfidf_test_matrix.npz')
    
    # Load SBERT test matrix (Dense)
    sbert_test = np.load('arxiv_dataset/SBERTEmbeddings_test_final.npy')
    
    with open('UserDataGenerator/synthetic_users.json', 'r') as f:
        synthetic_users = json.load(f)['train']
        
    # User profiles
    vector_data_tfidf = np.load(f'user_profiles/user_vectors_halflife_{HALF_LIFE}.npz')
    users_vector_tfidf = vector_data_tfidf['vectors']
    
    vector_data_sbert = np.load(f'user_profiles_SBERT/user_vectors_halflife_{HALF_LIFE}.npz')
    users_vector_sbert = vector_data_sbert['vectors']
    
    with open(f'user_profiles/user_metadata_halflife_{HALF_LIFE}.json', 'r') as f:
        user_metadata = json.load(f)
        
    return df, tfidf_test_sparse, sbert_test, synthetic_users, users_vector_tfidf, users_vector_sbert, user_metadata

df, tfidf_test_sparse, sbert_test, synthetic_users, users_vector_tfidf, users_vector_sbert, user_metadata = load_data()
ltr_sbert, ltr_tfidf, feature_names = load_models()

paper_ids = df['paper_id'].values
paper_id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
paper_to_cat = pd.Series(df['primary_category'].values, index=df['paper_id']).to_dict()
recency_scores = df['recency_score'].values

def get_keywords(paper_idx):
    row = tfidf_test_sparse[paper_idx]
    if scipy.sparse.issparse(row):
        row = row.toarray()[0]
    top_idx = np.argsort(row)[::-1][:5]
    return ", ".join([feature_names[i] for i in top_idx if row[i] > 0])

def mmr_rerank(candidate_indices, relevance_scores, embeddings, top_k=10, lambda_param=0.7):
    selected = []
    remaining = list(candidate_indices)

    while len(selected) < top_k and remaining:
        if not selected:
            best_idx = remaining[int(np.argmax([relevance_scores[i] for i in remaining]))]
        else:
            if scipy.sparse.issparse(embeddings):
                selected_embeds = scipy.sparse.vstack([embeddings[i] for i in selected])
            else:
                selected_embeds = embeddings[selected]
            mmr_scores = []
            for idx in remaining:
                relevance = relevance_scores[idx]
                sims = cosine_similarity(embeddings[idx].reshape(1, -1) if not scipy.sparse.issparse(embeddings) else embeddings[idx], selected_embeds)[0]
                redundancy = sims.max()
                mmr_scores.append(lambda_param * relevance - (1 - lambda_param) * redundancy)
            best_idx = remaining[int(np.argmax(mmr_scores))]

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

def get_explainable_recommendations(user_vector, history_categories, test_embeddings, ltr_model, top_k=10):
    # Candidate pool is the entire test set
    valid_indices = list(range(len(paper_ids)))
    
    # Calculate features for LTR
    cand_embeds = test_embeddings[valid_indices]
    cos_sims = cosine_similarity(user_vector.reshape(1, -1), cand_embeds).flatten()
    recs = recency_scores[valid_indices]
    cat_matches = [int(paper_to_cat.get(paper_ids[i], '') in history_categories) for i in valid_indices]
    
    # Predict with LTR
    X_candidates = np.column_stack((cos_sims, recs, cat_matches))
    ltr_probs = ltr_model.predict_proba(X_candidates)[:, 1]
    
    # Select top pool_size for MMR
    pool_size = 100
    top_pool_idx = np.argsort(ltr_probs)[::-1][:pool_size]
    
    pool_indices = [valid_indices[i] for i in top_pool_idx]
    relevance_dict = {zip_idx: ltr_probs[i] for i, zip_idx in zip(top_pool_idx, pool_indices)}
    
    # MMR
    selected_indices = mmr_rerank(
        candidate_indices=pool_indices,
        relevance_scores=relevance_dict,
        embeddings=test_embeddings,
        top_k=top_k,
        lambda_param=LAMBDA_PARAM
    )
    
    # Build explanation table
    recs_list = []
    for rank, idx in enumerate(selected_indices, start=1):
        pid = paper_ids[idx]
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
        
    return pd.DataFrame(recs_list), selected_indices

st.title("📚 Personalized ArXiv Paper Recommender (Test Set)")
st.markdown("A recommendation system evaluating on the Test set using LTR.")

embedding_type = st.radio("Select Embedding Model for Recommendations:", ["SBERT (Dense)", "TF-IDF (Sparse)"], horizontal=True)

if "SBERT" in embedding_type:
    active_test_embeddings = sbert_test
    active_users_vector = users_vector_sbert
    active_ltr_model = ltr_sbert
else:
    active_test_embeddings = tfidf_test_sparse
    active_users_vector = users_vector_tfidf
    active_ltr_model = ltr_tfidf

user_type = st.radio("Select User Type:", ["Existing User", "New User (Cold Start)"], horizontal=True)

if user_type == "Existing User":
    st.subheader("Select an Existing User")
    user_choices = [u['user_id'] for u in user_metadata]
    selected_user_id = st.selectbox("User ID:", user_choices)
    
    if st.button("Generate Explainable Recommendations", type="primary"):
        with st.spinner("Executing LTR Pipeline on Test Set..."):
            user_idx = user_choices.index(selected_user_id)
            user_vec = active_users_vector[user_idx]
            
            synth_user = next((u for u in synthetic_users if u['user_id'] == selected_user_id), None)
            history_ids = [x[0] for x in synth_user['train_history']] if synth_user else []
            
            train_df = pd.read_csv('arxiv_dataset/train.csv')
            train_cat_map = pd.Series(train_df['primary_category'].values, index=train_df['paper_id']).to_dict()
            
            history_categories = set(train_cat_map.get(pid) for pid in history_ids if pid in train_cat_map)
            
            st.markdown("### 📖 Reading History (from Train Set)")
            history_df = train_df[train_df['paper_id'].isin(history_ids)]
            st.dataframe(history_df[['paper_id', 'title', 'primary_category']], use_container_width=True)
            
            st.markdown(f"### 🎯 Top 10 Recommendations from Test Set ({embedding_type})")
            recs_df, recommendation_indices = get_explainable_recommendations(user_vec, history_categories, active_test_embeddings, active_ltr_model)
            st.dataframe(recs_df, use_container_width=True)

            st.markdown("#### 🧠 LTR Model Feature Importance Breakdown")
            fi = active_ltr_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": ["Cosine Similarity", "Recency Score", "Category Match"],
                "Importance Weight": [f"{v*100:.2f}%" for v in fi]
            })
            st.table(fi_df)

            st.markdown("### User Profile and Recommendations in t-SNE Space")
            try:
                figure = create_user_profile_tsne(
                    df=df,
                    embeddings=active_test_embeddings,
                    user_vector=user_vec,
                    recommendation_indices=recommendation_indices,
                    context_indices=[], # No history in test set
                    context_label="Reading history",
                )
                st.pyplot(figure, width="stretch")
                plt.close(figure)
                st.caption(
                    "t-SNE is a qualitative visualization; distances can be "
                    "distorted and are not used as an evaluation metric."
                )
            except Exception as e:
                st.warning(f"Could not generate t-SNE plot: {e}")

else:
    st.subheader("Cold-Start Category Onboarding")
    st.markdown("Select 1 to 3 categories that interest you to bootstrap your profile.")
    
    all_categories = sorted(df['primary_category'].dropna().unique())
    selected_cats = st.multiselect("Favorite Categories:", all_categories, max_selections=3)
    
    if st.button("Generate Explainable Recommendations", type="primary"):
        if not selected_cats:
            st.error("Please select at least one category to continue.")
        else:
            with st.spinner("Bootstrapping profile and running LTR on Test Set..."):
                pseudo_vec, seed_indices = build_cold_start_profile(
                    df,
                    active_test_embeddings,
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
                st.caption(f"Profile initialized from {seed_summary} recent test papers.")
                
                history_categories = set(selected_cats)
                
                st.markdown(f"### 🎯 Top 10 Recommendations from Test Set ({embedding_type})")
                recs_df, recommendation_indices = get_explainable_recommendations(
                    pseudo_vec, history_categories, active_test_embeddings, active_ltr_model
                )
                st.dataframe(recs_df, use_container_width=True)

                st.markdown("#### 🧠 LTR Model Feature Importance Breakdown")
                fi = active_ltr_model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": ["Cosine Similarity", "Recency Score", "Category Match"],
                    "Importance Weight": [f"{v*100:.2f}%" for v in fi]
                })
                st.table(fi_df)

                st.markdown("### User Profile and Recommendations in t-SNE Space")
                try:
                    figure = create_user_profile_tsne(
                        df=df,
                        embeddings=active_test_embeddings,
                        user_vector=pseudo_vec,
                        recommendation_indices=recommendation_indices,
                        context_indices=seed_indices,
                        context_label="Cold-start seed papers",
                    )
                    st.pyplot(figure, width="stretch")
                    plt.close(figure)
                    st.caption(
                        "t-SNE is a qualitative visualization; distances can be "
                        "distorted and are not used as an evaluation metric."
                    )
                except Exception as e:
                    st.warning(f"Could not generate t-SNE plot: {e}")
