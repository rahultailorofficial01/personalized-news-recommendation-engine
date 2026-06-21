import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("⏳ Step 1: Creating Production-Grade News Dataset...")

# Real-world mock news data
news_data = {
    'Article_ID': [1, 2, 3, 4, 5, 6, 7, 8],
    'Title': [
        "Next-Gen NVIDIA GPUs Set to Revolutionize AI Model Training",
        "Top 10 High-Protein Diets for Muscle Building and Gym Fitness",
        "Cracking Product-Based MNC Interviews: The Ultimate Roadmap",
        "OpenAI Launches Advanced GPT Models with Reasoning Capabilities",
        "Creative Gym Workouts: How to Build a Perfect Aesthetic Physique",
        "How to Get Referrals on LinkedIn for Software Engineering Roles",
        "Global Markets Face Correction as Tech Stocks Witness Sharp Dip",
        "The Rise of Anime Culture: How Action Series Drive Discipline"
    ],
    'Category': ['Tech/AI', 'Fitness', 'Careers', 'Tech/AI', 'Fitness', 'Careers', 'Finance', 'Entertainment'],
    'Content': [
        "Nvidia announced new Blackwell chips architecture maximizing deep learning matrix multiplication speed.",
        "To maximize muscle hypertrophy, balance whey protein, soya chunks, and clean fats in your daily macro intake.",
        "Mastering Data Structures, Algorithms, and system design is crucial to clear top IT giant placement interviews.",
        "The new reasoning LLM models show human-like processing in complex math, coding, and logical debugging.",
        "A structured push-pull-legs gym split combined with progressive overload ensures high quality lean muscle gains.",
        "Building a strong personal brand on tech Twitter and LinkedIn is the fastest way to land software engineering referrals.",
        "Wall street indexes dropped by 3% today as semiconductor and tech heavy shares faced liquidations.",
        "Shonen anime and historical fighter narratives are surging in popularity, inspiring youth towards focus and fitness."
    ]
}

df = pd.DataFrame(news_data)

print("⏳ Step 2: Initializing Text Vectorization Pipeline (TF-IDF)...")
# TF-IDF text patterns ko high-dimensional vectors me convert karta hai
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Content'])

print("💾 Step 3: Serializing NLP Artifacts for Real-time Streaming...")
# Artifacts save kar rahe hain taaki runtime UI par cosine similarity turant calculate ho sake
with open('news_df.pkl', 'wb') as f:
    pickle.dump(df, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("🚀 NLP Backend Engine Saved Successfully!")