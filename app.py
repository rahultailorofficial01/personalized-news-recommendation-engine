import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="AI News Recommender", layout="wide", page_icon="📰")

# 2. Artifacts Load Karna
@st.cache_resource
def load_nlp_artifacts():
    df = pickle.load(open('news_df.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    return df, tfidf, tfidf_matrix

try:
    df, tfidf, tfidf_matrix = load_nlp_artifacts()
except Exception as e:
    st.error(f"Error loading NLP files: {e}")

st.title("📰 Personalized News Recommendation Engine")
st.write("3rd Year AI & DS Engineering Project — Content-Based Filtering using TF-IDF & Cosine Similarity")
st.markdown("---")

# 3. Sidebar Configuration for Interaction
st.sidebar.header("👤 User Interest Profile")
user_mode = st.sidebar.radio("Choose Recommendation Strategy:", ["Select Favorite Article", "Type Custom Search / Query"])

# 4. Engine Core Logic
if user_mode == "Select Favorite Article":
    st.subheader("🎯 Based on an Article You Liked")
    selected_title = st.selectbox("Pick an article to simulate your reading history:", df['Title'].values)
    
    if st.button("🚀 Fetch Personalized Feed", type="primary"):
        # Selected article ka index nikalna
        idx = df[df['Title'] == selected_title].index[0]
        
        # Pure matrix ke sath similarity calculate karna
        # Cosine similarity text vectors ke beech ka angle nikalta hai
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        # Scores ko sort karke top 3 recommendations nikalna (khud ko chhodkar)
        related_indices = sim_scores.argsort()[::-1][1:4]
        
        st.write("### Recommended For You:")
        for i in related_indices:
            score_percentage = sim_scores[i] * 100
            with st.container():
                st.info(f"**📖 {df['Title'].iloc[i]}** | Match Score: **{score_percentage:.1f}%**")
                st.caption(f"Category: {df['Category'].iloc[i]} | Content Preview: {df['Content'].iloc[i]}")
                st.markdown("---")

else:
    st.subheader("🔍 Real-time Semantic Search Interface")
    user_query = st.text_input("What are you interested in reading today? (e.g., 'muscle gym split', 'nvidia ai models', 'placement strategy')")
    
    if st.button("🔎 Run NLP Engine", type="primary"):
        if user_query.strip() == "":
            st.warning("Please enter a valid query string.")
        else:
            # User query ko transform karna usi vector space me
            query_vector = tfidf.transform([user_query])
            
            # Distance measure karna
            sim_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Sorting logic
            top_indices = sim_scores.argsort()[::-1][:3]
            
            st.write("### Top Semantic Matches:")
            for i in top_indices:
                score_percentage = sim_scores[i] * 100
                if score_percentage > 0:
                    with st.container():
                        st.success(f"**📰 {df['Title'].iloc[i]}** | Relevance: **{score_percentage:.1f}%**")
                        st.caption(f"Category: {df['Category'].iloc[i]} | Snippet: {df['Content'].iloc[i]}")
                        st.markdown("---")
                else:
                    st.write("No direct high-relevance match found for this token sequence. Try other technical keywords!")
                    break