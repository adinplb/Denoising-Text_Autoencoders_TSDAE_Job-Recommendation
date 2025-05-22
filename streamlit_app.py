import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

st.title("🚀 Job Recommendation Dashboard")

# Sidebar Navigation

st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", (
    "Upload CVs",
    "Job Posting Dataset",
    "Embeddings",
    "Clustering",
    "CV Analysis"
))

# Sidebar: Upload CVs widget
if section == "Upload CVs" or section == "CV Analysis":
    uploaded_files = st.sidebar.file_uploader("Upload up to 5 CV files (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
else:
    uploaded_files = None

# Sample job posting dataset (small demo)
job_postings = [
    {
        "job_id": 1,
        "title": "Data Scientist",
        "company": "Tech Innovations",
        "location": "San Francisco, CA",
        "description": "Looking for a Data Scientist experienced in Python, machine learning, data analysis, and statistics.",
        "requirements": "Python, Machine Learning, Statistics, Data Analysis, SQL"
    },
    {
        "job_id": 2,
        "title": "Software Engineer",
        "company": "Dev Solutions",
        "location": "New York, NY",
        "description": "Seeking a software engineer skilled in Java, Spring, cloud computing, and microservices.",
        "requirements": "Java, Spring, Cloud, Microservices, REST APIs"
    },
    {
        "job_id": 3,
        "title": "Frontend Developer",
        "company": "Creative Labs",
        "location": "Austin, TX",
        "description": "Frontend developer needed with expertise in React, JavaScript, CSS, and UI/UX principles.",
        "requirements": "JavaScript, React, CSS, HTML, UI/UX"
    },
    {
        "job_id": 4,
        "title": "DevOps Engineer",
        "company": "CloudWorks",
        "location": "Seattle, WA",
        "description": "DevOps Engineer with skills in Docker, Kubernetes, CI/CD pipelines, and AWS cloud services.",
        "requirements": "Docker, Kubernetes, AWS, CI/CD, Terraform"
    },
    {
        "job_id": 5,
        "title": "Business Analyst",
        "company": "Enterprise Corp",
        "location": "Boston, MA",
        "description": "Looking for a Business Analyst with strong communication skills and experience gathering requirements.",
        "requirements": "Business Analysis, Communication, Requirements Gathering, Agile"
    }
]

# Convert to DataFrame
df_jobs = pd.DataFrame(job_postings)

def extract_text_from_txt(file) -> str:
    try:
        return file.read().decode("utf-8")
    except Exception:
        return ""

def extract_text_from_pdf(file) -> str:
    # For demonstration, we will not do real pdf parsing as it requires external libs
    # Instead, show placeholder text
    return "[PDF content parsing not implemented in this demo]"

def analyze_cv_text(text, job_keywords_set):
    """
    Perform 5 analyses on CV text:
    1. Word count
    2. Top 5 keywords (using simple frequency excluding stop words)
    3. Skill matching count (overlap with job keywords)
    4. Most common words (top 5)
    5. Summary - first 2 sentences
    """
    from collections import Counter
    import string

    stopwords = set([
        "and","or","the","a","an","of","to","in","with","for","on","at",
        "by","from","as","is","are","was","were","be","been","have","has","had","I","you",
        "he","she","it","they","we","us","our","your","their","this","that","these","those",
        "will","would","can","could","should"
    ])

    # sanitize text
    text_lower = text.lower()
    # tokenize
    words = re.findall(r"\\b[a-z]{2,}\\b", text_lower)
    # filter stop words
    words_filtered = [w for w in words if w not in stopwords]
    word_count = len(words_filtered)
    word_freq = Counter(words_filtered)
    top_keywords = [w for w, c in word_freq.most_common(5)]

    # Skill match count: count how many job keywords appear in CV
    skill_matches = job_keywords_set.intersection(set(words_filtered))
    skill_match_count = len(skill_matches)

    # Most common words (top 5)
    most_common = top_keywords

    # Summary: first two sentences
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    summary = " ".join(sentences[:2]) if len(sentences) >= 2 else text.strip()

    return {
        "Word Count": word_count,
        "Top Keywords": top_keywords,
        "Skill Match Count": skill_match_count,
        "Matched Skills": list(skill_matches),
        "Summary": summary
    }

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

    st.dataframe(df_jobs[['job_id', 'title', 'company', 'location']])

elif section == "Embeddings":
    st.header("3️⃣ Embeddings of Job Posting Dataset")

    # Use TF-IDF on job descriptions + requirements combined text
    df_jobs["text"] = df_jobs["description"] + " " + df_jobs["requirements"]

    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(df_jobs["text"])

    st.write("TF-IDF vectorizer created job posting embeddings with shape:", X_tfidf.shape)

elif section == "Clustering":
    st.header("4️⃣ Clustering Job Postings with KMeans")

    # Use TF-IDF on job descriptions + requirements combined text
    df_jobs["text"] = df_jobs["description"] + " " + df_jobs["requirements"]
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(df_jobs["text"])

    # Use KMeans to cluster job postings, K=3
    num_clusters = 3
    model = KMeans(n_clusters=num_clusters, random_state=42)
    df_jobs["cluster"] = model.fit_predict(X_tfidf)

    st.write("Job postings clustered into", num_clusters, "clusters.")
    clustered_jobs = df_jobs[["job_id", "title", "company", "location", "cluster"]]
    st.dataframe(clustered_jobs)

    # Visualize clusters using PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_tfidf.toarray())
    df_jobs["x"] = coords[:,0]
    df_jobs["y"] = coords[:,1]

    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']
    for cluster_id in range(num_clusters):
        cluster_points = df_jobs[df_jobs["cluster"]==cluster_id]
        ax.scatter(cluster_points["x"], cluster_points["y"], c=colors[cluster_id], label=f"Cluster {cluster_id}", s=100, alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Job Postings Clusters Visualization")
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
            # Aggregate all job keywords as a set
            all_job_keywords = set()
            for req in df_jobs["requirements"]:
                kws = set([kw.strip().lower() for kw in req.split(",")])
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

                # Display analyses in columns
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

st.markdown("""
---
*This dashboard demonstrates a basic job recommendation system using CV uploads, job posting data visualization, embedding creation with TF-IDF, and job posting clustering with KMeans.*
""")

