import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

# Title
st.title("📄 Job Recommendation System with TF-IDF & Clustering")

# Sidebar to upload job dataset
st.sidebar.header("Upload Datasets")
job_data_file = st.sidebar.file_uploader("Upload Job Postings Dataset (CSV)", type=["csv"])
cv_files = st.sidebar.file_uploader("Upload up to 5 CVs (TXT)", type=["txt"], accept_multiple_files=True)

# Load and show job dataset
if job_data_file:
    job_df = pd.read_csv(job_data_file)
    st.subheader("📊 Job Postings Dataset")
    st.dataframe(job_df.head())

    if 'description' not in job_df.columns:
        st.error("Job dataset must include a 'description' column.")
    else:
        # TF-IDF vectorization
        st.subheader("🔍 TF-IDF Vectorization of Job Descriptions")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform(job_df['description'])
        st.success("TF-IDF embedding created!")

        # Clustering jobs
        num_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        job_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

        # Show cluster distribution
        st.subheader("📌 Job Cluster Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=job_df['cluster'], ax=ax)
        st.pyplot(fig)

        # Process CVs
        if cv_files:
            st.subheader("🧑‍💼 CV-Based Recommendations")
            for i, cv_file in enumerate(cv_files[:5]):
                cv_text = cv_file.read().decode("utf-8")
                cv_vec = vectorizer.transform([cv_text])
                similarity_scores = cosine_similarity(cv_vec, tfidf_matrix).flatten()
                top_indices = similarity_scores.argsort()[-5:][::-1]
                top_jobs = job_df.iloc[top_indices]

                with st.expander(f"CV {i+1} - Top 5 Recommendations"):
                    st.write("**Uploaded CV:**")
                    st.code(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text, language='text')
                    st.write("**Top Job Matches:**")
                    st.dataframe(top_jobs[['title', 'company', 'location', 'description', 'cluster']])
