import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from tqdm import tqdm  # Import tqdm for progress bar
from sentence_transformers import SentenceTransformer, util # Import SentenceTransformer
from sklearn.cluster import KMeans
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.decomposition import PCA

# Download necessary NLTK resources (run once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
except LookupError:
    nltk.download('punkt_tab')

# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv'
RELEVANT_FEATURES = ['Job.ID', 'text', 'Title']
N_CLUSTERS = 20

# --- Global Data Storage (using Streamlit Session State) ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None
if 'job_clusters' not in st.session_state:
    st.session_state['job_clusters'] = None
if 'cv_text' not in st.session_state:
    st.session_state['cv_text'] = ""

# --- Helper Functions --- (Keep existing helper functions)

# --- Text Denoising Function ---
def denoise_text(text, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text

    if method == 'a':
        # === (a) Random Deletion ===
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True
        result = np.array(words)[keep_or_not]

    elif method == 'b':
        # === (b) Remove high-frequency words ===
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'")
        high_freq_words = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        to_remove = set(random.sample(high_freq_words, int(del_ratio * len(high_freq_words)))) if high_freq_words else set()
        result = [w for i, w in enumerate(words) if i not in to_remove]

    elif method == 'c':
        # === (c) Based on (b) + shuffle remaining words ===
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'")
        high_freq_words = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        to_remove = set(random.sample(high_freq_words, int(del_ratio * len(high_freq_words)))) if high_freq_words else set()
        result = [w for i, w in enumerate(words) if i not in to_remove]
        random.shuffle(result)

    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")

    return TreebankWordDetokenizer().detokenize(result)

# --- Embedding Generation Functions --- (Keep existing embedding functions)

# --- Page Functions --- (Keep existing page functions)

def tsdae_page():
    st.header("TSDAE (Noise Injection)")
    st.write("This page applies noise to the preprocessed text and generates embeddings.")

    if st.session_state['data'] is None or 'processed_text' not in st.session_state['data'].columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page.")
        return

    data = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is not None:
        st.subheader("TSDAE Settings")
        denoising_method = st.selectbox("Denoising Method", ['a', 'b', 'c'], index=0,
                                         help="Method 'a': Random deletion. 'b': Remove high-frequency words. 'c': Based on 'b' + shuffle.")
        deletion_ratio = st.slider("Deletion Ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.1)

        word_freq_dict = None
        if denoising_method in ['b', 'c']:
            # Create a word frequency dictionary from the processed text
            all_words = []
            for text in data['processed_text'].fillna('').tolist():
                all_words.extend(word_tokenize(text))
            word_freq_dict = {word.lower(): all_words.count(word.lower()) for word in set(all_words)}

        if st.button("Apply Noise and Generate Embeddings"):
            with st.spinner("Applying noise..."):
                noisy_texts = [denoise_text(text, method=denoising_method, del_ratio=deletion_ratio, word_freq_dict=word_freq_dict)
                               for text in tqdm(data['processed_text'].fillna('').tolist(), desc="Applying Noise")]
                st.session_state['data']['noisy_text'] = noisy_texts

            st.subheader("Noisy Text (Preview)")
            st.dataframe(st.session_state['data'][['processed_text', 'noisy_text']].head(), use_container_width=True)

            st.subheader("Generating Embeddings for Original and Noisy Text")
            original_embeddings = generate_embeddings_with_progress(bert_model, data['processed_text'].fillna('').tolist())
            noisy_embeddings = generate_embeddings_with_progress(bert_model, st.session_state['data']['noisy_text'].tolist())

            if original_embeddings.size > 0 and noisy_embeddings.size > 0:
                # Combine embeddings (averaging as per your example)
                tsdae_embeddings = (original_embeddings + noisy_embeddings) / 2.0
                st.session_state['tsdae_embeddings'] = tsdae_embeddings
                st.subheader("Combined TSDAE Embeddings (Preview)")
                st.write("Shape of combined embeddings:", tsdae_embeddings.shape)
                st.write("Preview of the first 3 combined embeddings:")
                st.write(tsdae_embeddings[:3])
            else:
                st.warning("Failed to generate embeddings for original or noisy text.")
    else:
        st.warning("BERT model not loaded. Cannot proceed with TSDAE.")

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", "Clustering Job2Vec", "Upload CV"])

if page == "Home":
    home_page()
elif page == "Preprocessing":









'''
import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Loading ---
@st.cache_data
def load_job_data(url):
    try:
        df = pd.read_csv(url)
        if 'text' in df.columns and 'Title' in df.columns:
            return df[['Job.ID', 'text', 'Title']].rename(columns={'text': 'description', 'Title': 'title'})
        else:
            st.error("Error: 'text' and 'Title' columns not found in the job data.")
            return None
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

job_data_url = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
job_df = load_job_data(job_data_url)

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head())

# --- CV Upload Functionality ---
def extract_text_from_pdf(uploaded_file):
    try:
        text = pdf_extract_text(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(uploaded_file):
    try:
        document = DocxDocument(uploaded_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None

uploaded_cv = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
cv_text = ""
if uploaded_cv is not None:
    file_extension = uploaded_cv.name.split(".")[-1].lower()
    if file_extension == "pdf":
        cv_text = extract_text_from_pdf(uploaded_cv)
    elif file_extension == "docx":
        cv_text = extract_text_from_docx(uploaded_cv)

if cv_text:
    st.subheader("Uploaded CV Content (Preview)")
    st.text_area("CV Text", cv_text, height=300)

# --- Embedding using BERT ---
@st.cache_resource
def load_bert_model(model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data
def generate_embeddings(_model, texts):
    embeddings = _model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

bert_model = load_bert_model()
job_embeddings = None
normalized_job_embeddings = None
cv_embedding = None
normalized_cv_embedding = None

if job_df is not None and 'description' in job_df.columns:
    job_descriptions = job_df['description'].fillna('').tolist()
    job_embeddings = generate_embeddings(bert_model, job_descriptions)
    normalized_job_embeddings = normalize(job_embeddings)

if cv_text and bert_model is not None:
    cv_embedding = generate_embeddings(bert_model, [cv_text])[0]
    normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))

# --- Similarity Matching and Recommendations ---
if normalized_cv_embedding is not None and normalized_job_embeddings is not None and job_df is not None:
    st.subheader("Top 20 Job Recommendations")
    cosine_similarities = cosine_similarity(normalized_cv_embedding, normalized_job_embeddings)[0]
    similarity_df = pd.DataFrame({'title': job_df['title'], 'similarity_score': cosine_similarities})
    top_recommendations = similarity_df.sort_values(by='similarity_score', ascending=False).head(20)
    st.dataframe(top_recommendations)
elif uploaded_cv is not None:
    st.info("Please wait while job embeddings are being generated.")
elif job_df is not None:
    st.info("Upload your CV to get job recommendations.")
else:
    st.info("Job data not loaded.")

# --- Basic Evaluation (Illustrative - Simplified) ---
if normalized_cv_embedding is not None and normalized_job_embeddings is not None and job_df is not None:
    st.subheader("Basic Recommendation Statistics")
    avg_similarity = cosine_similarities.mean()
    max_similarity = cosine_similarities.max()
    min_similarity = cosine_similarities.min()

    st.write(f"Average Similarity: {avg_similarity:.4f}")
    st.write(f"Maximum Similarity: {max_similarity:.4f}")
    st.write(f"Minimum Similarity: {min_similarity:.4f}")
'''
'''
UMAP USED
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import plotly.express as px
import umap

# --- Constants ---
JOB_DATA_URL = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"

# --- Data Loading ---
@st.cache_data
def load_job_data(url):
    try:
        df = pd.read_csv(url)
        if 'text' in df.columns and 'Title' in df.columns:
            return df[['Job.ID', 'text', 'Title']].rename(columns={'text': 'description', 'Title': 'title'})
        else:
            st.error("Error: 'text' and 'Title' columns not found in the job data.")
            return None
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

job_df = load_job_data(JOB_DATA_URL)

# --- Sidebar for CV Upload ---
with st.sidebar:
    st.header("Upload Your CV")
    uploaded_cv = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_cv:
        st.success("CV uploaded successfully!")

# --- Main Dashboard ---
st.title("Job Posting Embedding Visualization (UMAP)")

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head())

    # --- Embedding Generation ---
    @st.cache_resource
    def load_bert_model(model_name="all-mpnet-base-v2"):
        model = SentenceTransformer(model_name)
        return model

    @st.cache_data
    def generate_job_embeddings(_model, df):
        if df is not None and 'description' in df.columns:
            job_descriptions = df['description'].fillna('').tolist()
            embeddings = _model.encode(job_descriptions, convert_to_tensor=True).cpu().numpy()
            normalized_embeddings = normalize(embeddings)
            return normalized_embeddings
        return None

    bert_model = load_bert_model()
    job_embeddings = generate_job_embeddings(bert_model, job_df)

    if job_embeddings is not None:
        st.subheader("Job Posting Embeddings (Normalized - Preview)")
        st.write(job_embeddings[:5])  # Display a snippet of the embeddings

        # --- Visualization of Embeddings (using UMAP for dimensionality reduction to 3D) ---
        st.subheader("Visualize Job Posting Embeddings (UMAP 3D)")
        @st.cache_data
        def reduce_dimensionality_umap(embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings

        reduced_embeddings_umap = reduce_dimensionality_umap(job_embeddings)

        fig_umap = px.scatter_3d(
            job_df,
            x=reduced_embeddings_umap[:, 0],
            y=reduced_embeddings_umap[:, 1],
            z=reduced_embeddings_umap[:, 2],
            hover_data=['title', 'description'],
            title='3D Visualization of Job Posting Embeddings (UMAP)'
        )
        st.plotly_chart(fig_umap)
    else:
        st.info("Job embeddings could not be generated.")

else:
    st.info("Job data not loaded. Please ensure the URL is correct.")

'''
