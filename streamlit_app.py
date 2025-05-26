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
from tqdm import tqdm  # Used for local progress bar simulation, not directly visible in Streamlit's st.progress
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Resource Downloads ---
# This block ensures necessary NLTK data is available when the app runs,
# especially important for Streamlit Cloud deployments.
@st.cache_resource
def download_nltk_resources():
    try:
        stopwords.words('english')
    except LookupError:
        st.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    try:
        word_tokenize("example text")
    except LookupError:
        st.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
    except LookupError:
        st.info("Downloading NLTK punkt_tab resource...")
        nltk.download('punkt_tab')
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv'
RELEVANT_FEATURES = ['Job.ID', 'text', 'Title']
N_CLUSTERS = 20 # Default number of clusters for KMeans
ANNOTATORS_NAMES = ["Annotator 1", "Annotator 2", "Annotator 3"] # Reduced for brevity

# --- Global Data Storage (using Streamlit Session State) ---
# This helps share data between pages without re-running heavy computations
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None
if 'job_clusters' not in st.session_state:
    st.session_state['job_clusters'] = None
if 'cv_text' not in st.session_state: # Kept for single CV context if needed elsewhere, but multi-CV is primary
    st.session_state['cv_text'] = ""
if 'tsdae_embeddings' not in st.session_state:
    st.session_state['tsdae_embeddings'] = None
if 'cv_embedding' not in st.session_state: # Kept for single CV context if needed elsewhere
    st.session_state['cv_embedding'] = None
if 'uploaded_cvs_data' not in st.session_state: # New: stores list of {'filename', 'text', 'embedding'} for multiple CVs
    st.session_state['uploaded_cvs_data'] = []
if 'all_recommendations_for_annotation' not in st.session_state: # Stores recommendations for annotation page
    st.session_state['all_recommendations_for_annotation'] = {} # Format: {cv_filename: DataFrame of top 20 recs}
if 'collected_annotations' not in st.session_state: # Stores collected annotations
    st.session_state['collected_annotations'] = pd.DataFrame()


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading data...')
def load_data_from_url(url):
    """Loads job data from a given URL and selects relevant features."""
    try:
        df = pd.read_csv(url)
        st.success('Successfully loaded data!')
        return df[RELEVANT_FEATURES].copy()
    except Exception as e:
        st.error(f'Error loading data from URL: {e}')
        return None

def extract_text_from_pdf(uploaded_file):
    """Extracts text from a PDF file."""
    try:
        text = pdf_extract_text(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(uploaded_file):
    """Extracts text from a DOCX file."""
    try:
        document = Document(uploaded_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None

# --- Standalone Text Preprocessing Function (for individual text like CV) ---
def preprocess_text(text):
    """
    Performs text preprocessing steps: symbol removal, case folding, tokenization,
    stopwords removal, and stemming.
    """
    if isinstance(text, str):
        # Symbol Removal
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)
        # Case Folding
        text = text.lower()
        # Stopwords Removal and Tokenization
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_words = [w for w in word_tokens if w not in stop_words]
        # Stemming
        porter = PorterStemmer()
        stemmed_words = [porter.stem(w) for w in filtered_words]
        return " ".join(stemmed_words)
    return ""

# --- Text Preprocessing Function with Intermediate Results and Progress Bar ---
def preprocess_text_with_intermediate(data_df):
    """
    Performs text preprocessing steps (symbol removal, case folding, tokenization,
    stopwords removal, stemming) and stores intermediate results.
    Includes a Streamlit progress bar.
    """
    processed_results = []
    if 'text' in data_df.columns:
        with st.spinner("Preprocessing 'text' column... This might take a moment."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_rows = len(data_df)
            for i, text in enumerate(data_df['text'].fillna('')):
                intermediate = {}
                if isinstance(text, str):
                    intermediate['original'] = text
                    symbol_removed = text.translate(str.maketrans('', '', string.punctuation))
                    symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed)
                    intermediate['symbol_removed'] = symbol_removed
                    case_folded = symbol_removed.lower()
                    intermediate['case_folded'] = case_folded
                    word_tokens = word_tokenize(case_folded)
                    intermediate['tokenized'] = " ".join(word_tokens)
                    stop_words = set(stopwords.words('english'))
                    filtered = [w for w in word_tokens if w not in stop_words]
                    intermediate['stopwords_removed'] = " ".join(filtered)
                    porter = PorterStemmer()
                    stemmed = [porter.stem(w) for w in filtered]
                    intermediate['stemmed'] = " ".join(stemmed)
                    processed_results.append(intermediate)
                else:
                    processed_results.append({
                        'original': '', 'symbol_removed': '', 'case_folded': '',
                        'tokenized': '', 'stopwords_removed': '', 'stemmed': ''
                    })
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
            data_df['preprocessing_steps'] = processed_results
            data_df['processed_text'] = [d['stemmed'] for d in processed_results]
            st.success("Preprocessing of 'text' column complete!")
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("The 'text' column was not found in the dataset.")
    return data_df

# --- Text Denoising Function ---
def denoise_text(text, method='a', del_ratio=0.6, word_freq_dict=No
