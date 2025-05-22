import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
import torch

st.set_page_config(page_title="Job Recommendation Dashboard", layout="wide")

st.title("🚀 Job Recommendation Dashboard with TechWolf JobBERT-v2 Embeddings")

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
    # ensure expected columns
    expected_cols = {"Job.ID", "text", "Title"}
    if not expected_cols.issubset(set(df.columns)):
        st.error("Dataset does not contain required columns: Job.ID, text, Title")
        return pd.DataFrame()
    return df

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
def load_jobbert_model():
    model_name = "TechWolf/JobBERT-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def embed_texts(texts, tokenizer, model, device='cpu', batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded_input = {k: v.to(device) for k,v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Use pooled output (corresponds to CLS token representation)
            if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
                pooled = model_output.pooler_output
            else:
                # fallback to mean pooling if pooler_output is not available
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                masked_embeddings = token_embeddings * attention_mask
                summed = torch.sum(masked_embeddings, dim=1)
                counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                pooled = summed / counts
            pooled = pooled.cpu().numpy()
            embeddings.extend(pooled)
    return np.vstack(embeddings)

# Load Dataset once
df_jobs = load_job_dataset()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    st.header("3️⃣ Create Embeddings of Job Posting Dataset Using TechWolf JobBERT-v2")

    if df_jobs.empty:
        st.error("Job posting dataset failed to load.")
    else:
        with st.spinner("Loading JobBERT-v2 model..."):
            tokenizer, model = load_jobbert_model()
            model.to(device)

        texts = df_jobs["text"].fillna("").tolist()
        with st.spinner("Computing embeddings for job postings..."):
            job_embeddings = embed_texts(texts, tokenizer, model, device=device, batch_size=32)

        st.write("Embeddings created for", len(texts), "job postings.")
        st.write("Embedding vector shape:", job_embeddings.shape)

        st.session_state['job_embeddings'] = job_embeddings

elif section == "Clustering":
    st.header("4️⃣ Clustering Job Postings Using TechWolf JobBERT-v2 Embeddings")

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
        df_jobs["x"] = coords[:,0]
        df_jobs["y"] = coords[:,1]

        fig, ax = plt.subplots(figsize=(8,6))
        colors = plt.cm.get_cmap('tab10', num_clusters)
        for cluster_id in range(num_clusters):
            cluster_points = df_jobs[df_jobs["cluster"] == cluster_id]
            ax.scatter(cluster_points["x"], cluster_points["y"],
                       color=colors(cluster_id), label=f"Cluster {cluster_id}", s=80, alpha=0.8)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Job Postings Clusters Visualization (TechWolf JobBERT-v2 Embeddings)")
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
*This dashboard demonstrates a job recommendation system using CV uploads, the [TechWolf JobBERT-v2](https://huggingface.co/TechWolf/JobBERT-v2) pretrained model from Hugging Face to create embeddings of job posting texts, clustering job postings, and basic CV analyses.*

**Requirements:**
- `transformers` and `torch` Python packages. Install via:
