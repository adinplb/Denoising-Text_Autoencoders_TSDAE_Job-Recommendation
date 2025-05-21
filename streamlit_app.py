import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper functions ---
@st.cache_data
def load_user_data():
    url = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/user_applicant_jobs.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def load_jobs_data():
    url = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/tech_jobs.csv"
    df = pd.read_csv(url)
    return df

def extract_text_from_uploaded_file(uploaded_file):
    # Simple text extraction for .txt files or .pdf if needed
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        else:
            return "Unsupported file format. Please upload a .txt or .pdf file."
    return None

# --- Streamlit app ---
st.set_page_config(page_title="CV Job Matcher Dashboard", layout="wide")

st.title("🚀 CV Job Matcher Dashboard")
st.markdown("""
Upload your CV, explore job listings, and find best matches based on text similarity.
""")

# Sidebar for file upload
st.sidebar.header("Upload your CV")
uploaded_file = st.sidebar.file_uploader("Upload CV (TXT or PDF)", type=["txt", "pdf"])

cv_text = extract_text_from_uploaded_file(uploaded_file)
if cv_text:
    st.subheader("Your Uploaded CV Text")
    st.text_area("CV Content", cv_text, height=300)

# Load datasets
user_data = load_user_data()
jobs_data = load_jobs_data()

st.sidebar.markdown("---")
st.sidebar.header("Job Listings")

# Show some job title filters or search
job_titles = jobs_data['job_title'].unique()
selected_jobs = st.sidebar.multiselect("Filter jobs by title", options=job_titles, default=job_titles[:5])

filtered_jobs = jobs_data[jobs_data['job_title'].isin(selected_jobs)]

st.subheader(f"Job Listings ({len(filtered_jobs)})")
st.dataframe(filtered_jobs[['job_id', 'job_title']].reset_index(drop=True))

# If CV text uploaded, calculate similarity and show top matches
if cv_text:
    st.subheader("Top Job Matches Based on Your CV")

    # Vectorize CV text + jobs combined_text for similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    combined_corpus = [cv_text] + filtered_jobs['combined_text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_corpus)

    # Cosine similarity of CV against jobs (first row vs rest)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Attach similarity score and sort descending
    filtered_jobs = filtered_jobs.copy()
    filtered_jobs['similarity'] = cosine_sim
    top_matches = filtered_jobs.sort_values(by='similarity', ascending=False).head(10)

    for idx, row in top_matches.iterrows():
        st.markdown(f"### {row['job_title']} (Similarity: {row['similarity']:.2f})")
        st.write(row['combined_text'][:300] + "...")
        st.markdown("---")
else:
    st.info("Upload your CV on the left sidebar to see job matches.")

# Optional: Show user applicant job data
st.subheader("User Applicant Job Data Sample")
st.dataframe(user_data.head(10))

# Footer
st.markdown("""
---
*Dashboard created with Streamlit*  
""")
