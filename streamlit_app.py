import streamlit as st
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper functions ---
@st.cache_data
def load_user_data():
    url = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/user_applicant_jobs.csv"
    df = pd.read_csv(url)
    return df

@st.cache_data
def load_jobs_data():
    url = "https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/combined_jobs_2000.csv"
    df = pd.read_csv(url)
    return df

def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        else:
            return "Unsupported file format. Please upload a .txt or .pdf file."
    return None

@st.cache_resource(show_spinner=True)
def load_tsdae_model():
    model_path = "./tsdae_model"  # <-- Your saved TSDAE model folder here
    model = SentenceTransformer(model_path)
    return model

# --- Streamlit app ---
st.set_page_config(page_title="CV Job Matcher Dashboard", layout="wide")

st.title("🚀 CV Job Matcher Dashboard")
st.markdown("""
Upload your CV, explore job listings, and find best matches based on semantic similarity using your pre-trained TSDAE model.
""")

# Sidebar for file upload
st.sidebar.header("Upload your CV")
uploaded_file = st.sidebar.file_uploader("Upload CV (TXT or PDF)", type=["txt", "pdf"])

cv_text = extract_text_from_uploaded_file(uploaded_file)
if cv_text and "Unsupported file format" not in cv_text:
    st.subheader("Your Uploaded CV Text")
    st.text_area("CV Content", cv_text, height=300)

# Load datasets
user_data = load_user_data()
jobs_data = load_jobs_data()

st.sidebar.markdown("---")
st.sidebar.header("Job Listings")

job_titles = jobs_data['job_title'].unique()
selected_jobs = st.sidebar.multiselect("Filter jobs by title", options=job_titles, default=job_titles[:5])

filtered_jobs = jobs_data[jobs_data['job_title'].isin(selected_jobs)]

st.subheader(f"Job Listings ({len(filtered_jobs)})")
st.dataframe(filtered_jobs[['job_id', 'job_title']].reset_index(drop=True))

if cv_text and "Unsupported file format" not in cv_text:
    st.subheader("Top Job Matches Based on Your CV")

    model = load_tsdae_model()

    # Encode CV and job combined_text using TSDAE model
    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    job_texts = filtered_jobs['combined_text'].tolist()
    job_embeddings = model.encode(job_texts, convert_to_tensor=True)

    # Compute cosine similarity (using torch)
    cosine_scores = torch.nn.functional.cosine_similarity(cv_embedding.unsqueeze(0), job_embeddings)

    # Add similarity scores to DataFrame and sort
    filtered_jobs = filtered_jobs.copy()
    filtered_jobs['similarity'] = cosine_scores.cpu().numpy()
    top_matches = filtered_jobs.sort_values(by='similarity', ascending=False).head(10)

    for idx, row in top_matches.iterrows():
        st.markdown(f"### {row['job_title']} (Similarity: {row['similarity']:.3f})")
        st.write(row['combined_text'][:300] + "...")
        st.markdown("---")
else:
    st.info("Upload your CV (txt or pdf) on the left sidebar to see job matches.")

# Optional: Show user applicant job data sample
st.subheader("User Applicant Job Data Sample")
st.dataframe(user_data.head(10))

# Footer
st.markdown("""
---
*Dashboard created with Streamlit*  
""")

