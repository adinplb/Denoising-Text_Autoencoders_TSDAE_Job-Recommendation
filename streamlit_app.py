import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import fitz  # PyMuPDF for PDF parsing
import docx
import os

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('output/tsdae_model')  # Change to your path

# Extract text from uploaded file
def extract_text(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith('.docx'):
        return "\n".join([p.text for p in docx.Document(uploaded_file).paragraphs])
    else:
        return uploaded_file.read().decode('utf-8')

# Compute recommendations
def get_recommendations(resume_text, model, jobs_df, job_embeddings, top_k=20):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    similarities = util.cos_sim(resume_embedding, job_embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)
    recommended_jobs = jobs_df.iloc[top_results.indices.cpu()]
    recommended_jobs['Score'] = top_results.values.cpu().numpy()
    return recommended_jobs

# UI
st.title("Job Recommendation System (TSDAE)")
st.markdown("Upload your **CV/Resume** to get personalized job matches!")

uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    st.text_area("Resume Text", resume_text, height=200)

    model = load_model()
    jobs_df = pd.read_csv("data/jobs.csv")  # Adjust path
    job_embeddings = torch.load("data/job_embeddings.pt")  # Adjust path

    with st.spinner("Generating recommendations..."):
        recommendations = get_recommendations(resume_text, model, jobs_df, job_embeddings)
        st.success("Top 20 Job Recommendations")
        st.dataframe(recommendations[['title', 'company', 'location', 'Score']])
