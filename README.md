# 📰 Personalized News Recommendation Engine (NLP)

An end-to-end Content-Based Recommendation system powered by Natural Language Processing (NLP). This repository contains a production-ready textual ranking pipeline designed to evaluate metadata and generate semantic article recommendations based on user reading profiles or real-time string searches.

---

## 🚀 Key Features

* **Text Vectorization (TF-IDF):** Leverages `TfidfVectorizer` to capture textual semantics and weights across article datasets, filtering out boilerplate stopwords.
* **Mathematical Semantic Search:** Employs high-dimensional **Cosine Similarity** matrices to calculate real-time contextual distances between user queries and stored content.
* **Dual-Mode Recommendation Engine:** Features twin evaluation paths: recommendation based on historic article interactions or custom interactive unstructured search input.
* **Interactive Data Interface:** Clean front-end interface built with **Streamlit** to demonstrate ranking updates live.

---

## 🛠️ Tech Stack & Project Structure

* **Core Stack:** Python, Scikit-Learn, Pandas, NumPy
* **NLP Workflow:** Text Tokenization, Vectorization, Similarity Ranking Matrices
* **UI Interface:** Streamlit Framework

```text
news_recommendation_project/
│
├── app.py                     # Streamlit Main UI Engine
├── train_recommendation.py    # Core Backend NLP Vectorization Script
├── news_df.pkl                # Serialized Article Dataframe Binaries
├── tfidf_matrix.pkl           # High-dimensional Mathematical Vector Matrices
├── tfidf_vectorizer.pkl       # Serialized Vocabulary Mapping Configurations
└── README.md                  # System Documentation Blueprint