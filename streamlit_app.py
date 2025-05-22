import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import gensim.downloader as api

st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

st.title("🚀 Job Recommendation Dashboard with FastText Embeddings")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", (
    "Upload CVs",
    "Job Posting Dataset",
    "Embeddings",
    "Clustering",
    "CV Analysis"
))

# Sidebar: Upload CVs widget for relevant sections
if section == "Upload CVs" or section == "CV Analysis":
    uploaded_files = st.sidebar.file_uploader("Upload up to 5 CV files (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
else:
    uploaded_files = None

@st.cache_data(show_spinner=True)
def load_job_dataset():
    url = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
    df = pd.read_csv(url)
    expected_cols = {"Job.ID", "text", "Title"}
    if not expected_cols.issubset(set(df.columns)):
        st.error("Dataset missing required columns: Job.ID, text, Title")
        return pd.DataFrame()
    return df

def extract_text_from_txt(file) -> str:
    try:
        return file.read().decode("utf-8")
    except Exception:
        return ""

def extract_text_from_pdf(file) -> str:
    return "[PDF content parsing not implemented in this demo]"

def analyze_cv_text(text, job_keywords_set):
    from collections import Counter

    stopwords = set([
        "and","or","the","a","an","of","to","in","with","for","on","at",
        "by","from","as","is","are","was","were","be","been","have","has","had","I","you",
        "he","she","it","they","we","us","our","your","their","this","that","these","those",
        "will","would","can","could","should"
    ])

    text_lower = text.lower()
    words = re.findall(r"\b[a-z]{2,}\b", text_lower)
    words_filtered = [w for w in words if w not in stopwords]
    word_count = len(words_filtered)
    word_freq = Counter(words_filtered)
    top_keywords = [w for w, c in word_freq.most_common(5)]

    skill_matches = job_keywords_set.intersection(set(words_filtered))
    skill_match_count = len(skill_matches)

    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    summary = " ".join(sentences[:2]) if len(sentences) >= 2 else text.strip()

    return {
        "Word Count": word_count,
        "Top Keywords": top_keywords,
        "Skill Match Count": skill_match_count,
        "Matched Skills": list(skill_matches),
        "Summary": summary
    }

@st.cache_resource(show_spinner=True)
def load_fasttext_model():
    # This will download the model on first run (~1GB), then cache it.
    ft_model = api.load("fasttext-wiki-news-subwords-300")
    return ft_model

def embed_texts_fasttext(texts, ft_model, embedding_dim=300):
    embeddings = []
    for text in texts:
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        valid_vectors = []
        for w in words:
            if w in ft_model:
                valid_vectors.append(ft_model[w])
        if valid_vectors:
            mean_vector = np.mean(valid_vectors, axis=0)
        else:
            mean_vector = np.zeros(embedding_dim, dtype=float)
        embeddings.append(mean_vector)
    return np.vstack(embeddings)

# Load dataset once
df_jobs = load_job_dataset()

if section == "Upload CVs":
    st.header("1️⃣ Upload Your CVs (Max 5)")

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Please upload a maximum of 5 CV files.")
        else:
            st.success(f"{len(uploaded_files)} CV files uploaded successfully.")
    else:
        st.info("Use the sidebar to upload up to 5 CV files.")

elif section == "Job Posting Dataset":
    st.header("2️⃣ Job Posting Dataset")

    st.dataframe(df_jobs[['Job.ID', 'Title', 'text']])

elif section == "Embeddings":
    st.header("3️⃣ Create Embeddings of Job Posting Dataset Using FastText")

    if df_jobs.empty:
        st.error("Job posting dataset failed to load.")
    else:
        with st.spinner("Loading FastText model..."):
            ft_model = load_fasttext_model()
        texts = df_jobs["text"].fillna("").tolist()
        with st.spinner("Computing FastText embeddings for job postings..."):
            job_embeddings = embed_texts_fasttext(texts, ft_model)

        st.write("Embeddings created for", len(texts), "job postings.")
        st.write("Embedding vector shape:", job_embeddings.shape)

        st.session_state['job_embeddings'] = job_embeddings

elif section == "Clustering":
    st.header("4️⃣ Clustering Job Postings Using FastText Embeddings")

    if "job_embeddings" not in st.session_state:
        st.warning("Please generate embeddings first in the 'Embeddings' section.")
    else:
        job_embeddings = st.session_state['job_embeddings']
        num_clusters = 5
        model = KMeans(n_clusters=num_clusters, random_state=42)
        with st.spinner("Clustering job postings..."):
            cluster_labels = model.fit_predict(job_embeddings)

        df_jobs['cluster'] = cluster_labels

        st.write(f"Job postings clustered into {num_clusters} clusters.")
        st.dataframe(df_jobs[['Job.ID', 'Title', 'cluster']])

        # PCA visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(job_embeddings)
        df_jobs["x"] = coords[:, 0]
        df_jobs["y"] = coords[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.get_cmap('tab10', num_clusters)
        for cluster_id in range(num_clusters):
            cluster_points = df_jobs[df_jobs["cluster"] == cluster_id]
            ax.scatter(cluster_points["x"], cluster_points["y"],
                       color=colors(cluster_id), label=f"Cluster {cluster_id}", s=80, alpha=0.8)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Job Postings Clusters Visualization (FastText Embeddings)")
        ax.legend()
        st.pyplot(fig)

elif section == "CV Analysis":
    st.header("5️⃣ CV Analysis Results")

    if not uploaded_files:
        st.info("Upload up to 5 CV files in the sidebar to see analyses here.")
    else:
        if len(uploaded_files) > 5:
            st.error("Please upload a maximum of 5 CV files.")
        else:
            all_job_keywords = set()
            for txt in df_jobs["text"].fillna(""):
                kws = set(re.findall(r"\b[a-z]{2,}\b", txt.lower()))
                all_job_keywords.update(kws)
            for title in df_jobs["Title"].fillna(""):
                kws = set(re.findall(r"\b[a-z]{2,}\b", title.lower()))
                all_job_keywords.update(kws)

            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"CV {i+1}: {uploaded_file.name}")

                if uploaded_file.type == "text/plain":
                    content = extract_text_from_txt(uploaded_file)
                elif uploaded_file.type == "application/pdf":
                    content = extract_text_from_pdf(uploaded_file)
                else:
                    content = ""

                if not content:
                    st.warning("Could not extract text from this file or file type not supported in this demo.")
                    continue

                analysis_results = analyze_cv_text(content, all_job_keywords)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Word Count:** {analysis_results['Word Count']}")
                    st.markdown(f"**Skill Match Count:** {analysis_results['Skill Match Count']}")
                    st.markdown(f"**Matched Skills:** {', '.join(analysis_results['Matched Skills']) if analysis_results['Matched Skills'] else 'None'}")
                with col2:
                    st.markdown("**Top Keywords:**")
                    st.write(", ".join(analysis_results["Top Keywords"]))

                st.markdown("**Summary from CV:**")
                st.write(analysis_results["Summary"])

                st.markdown("----")

st.markdown(
"""
---
*This dashboard demonstrates a job recommendation system using CV uploads, FastText pretrained embeddings via gensim to create embeddings of job posting texts, clustering job postings, and basic CV analyses.*

**Requirements:**

- Install required packages via:

