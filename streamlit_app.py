import streamlit as st
import pandas as pd
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer
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
st.set_page_config(page_title="CV & User Profile Job Matcher", layout="wide")

st.title("🚀 CV & User Profile Job Matcher Dashboard")
st.markdown("""
Upload your CV or use your profile data to explore job listings and find best matches based on semantic similarity using your pre-trained TSDAE model.
""")

# Sidebar for CV upload
st.sidebar.header("Upload your CV")
uploaded_file = st.sidebar.file_uploader("Upload CV (TXT or PDF)", type=["txt", "pdf"])
cv_text = extract_text_from_uploaded_file(uploaded_file)
if cv_text and "Unsupported file format" not in cv_text:
    st.subheader("Your Uploaded CV Text")
    st.text_area("CV Content", cv_text, height=300)

# Load datasets
user_data = load_user_data()
jobs_data = load_jobs_data()

# Sidebar job title filtering
st.sidebar.markdown("---")
st.sidebar.header("Job Listings")
job_titles = jobs_data['job_title'].unique()
selected_jobs = st.sidebar.multiselect("Filter jobs by title", options=job_titles, default=job_titles[:5])
filtered_jobs = jobs_data[jobs_data['job_title'].isin(selected_jobs)]
st.subheader(f"Job Listings ({len(filtered_jobs)})")
st.dataframe(filtered_jobs[['job_id', 'job_title']].reset_index(drop=True))

# Load your TSDAE model
model = load_tsdae_model()

def semantic_similarity_recommendation(input_text, jobs_df):
    # Encode input text and job combined_texts
    input_emb = model.encode(input_text, convert_to_tensor=True)
    job_texts = jobs_df['combined_text'].tolist()
    job_embs = model.encode(job_texts, convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(input_emb.unsqueeze(0), job_embs)
    jobs_df = jobs_df.copy()
    jobs_df['similarity'] = cosine_scores.cpu().numpy()
    return jobs_df.sort_values(by='similarity', ascending=False).head(10)

# --- Match using CV text ---
if cv_text and "Unsupported file format" not in cv_text:
    st.subheader("Top Job Matches Based on Your Uploaded CV")
    top_matches_cv = semantic_similarity_recommendation(cv_text, filtered_jobs)
    for idx, row in top_matches_cv.iterrows():
        st.markdown(f"### {row['job_title']} (Similarity: {row['similarity']:.3f})")
        st.write(row['combined_text'][:300] + "...")
        st.markdown("---")
else:
    st.info("Upload your CV (txt or pdf) on the left sidebar to see job matches.")

# --- Match using raw user profile text ---
st.markdown("---")
st.subheader("User Profile-Based Job Recommendations")

# For demo: Combine some user profile columns into a text profile representation
# Adjust the columns used here depending on your actual user_data structure
if not user_data.empty:
    user_profile_texts = []
    for idx, row in user_data.iterrows():
        # Example: concatenating user skills, job history, education, etc.
        text_profile = f"{row.get('skills', '')} {row.get('job_experience', '')} {row.get('education', '')}"
        user_profile_texts.append((row['user_id'], text_profile.strip()))

    user_id = st.selectbox("Select User ID from dataset", [uid for uid, _ in user_profile_texts])
    selected_profile_text = next(text for uid, text in user_profile_texts if uid == user_id)

    st.write(f"User Profile Text (combined): {selected_profile_text[:500]}...")

    top_matches_profile = semantic_similarity_recommendation(selected_profile_text, filtered_jobs)
    for idx, row in top_matches_profile.iterrows():
        st.markdown(f"### {row['job_title']} (Similarity: {row['similarity']:.3f})")
        st.write(row['combined_text'][:300] + "...")
        st.markdown("---")
else:
    st.warning("User profile data is empty or missing required fields.")

# --- Show user applicant job data sample ---
st.subheader("User Applicant Job Data Sample")
st.dataframe(user_data.head(10))

# Footer
st.markdown("""
---
*Dashboard created with Streamlit*  
""")
