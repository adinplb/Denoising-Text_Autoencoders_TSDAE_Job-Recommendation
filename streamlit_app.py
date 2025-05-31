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
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer
# from sentence_transformers.evaluation import RerankingEvaluator # Not used in the final eval as per last discussion
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score 

# --- NLTK Resource Downloads ---
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
        st.info("Downloading NLTK punkt_tab resource (if available)...")
        try:
            nltk.download('punkt_tab')
        except Exception:
            st.warning("NLTK punkt_tab resource not found or download failed, usually not critical.")
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'

FEATURES_TO_COMBINE = [
    'Status', 'Title', 'Position', 'Company', 
    'City', 'State.Name', 'Industry', 'Job.Description', 
    'Employment.Type', 'Education.Required'
]
N_CLUSTERS = 20 
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
if 'data' not in st.session_state:
    st.session_state['data'] = None 
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None 
if 'job_text_embedding_job_ids' not in st.session_state: 
    st.session_state['job_text_embedding_job_ids'] = None
if 'tsdae_embeddings' not in st.session_state:
    st.session_state['tsdae_embeddings'] = None 
if 'tsdae_embedding_job_ids' not in st.session_state: 
    st.session_state['tsdae_embedding_job_ids'] = None
if 'job_clusters_raw' not in st.session_state: 
    st.session_state['job_clusters_raw'] = None
if 'uploaded_cvs_data' not in st.session_state:
    st.session_state['uploaded_cvs_data'] = [] 
if 'all_recommendations_for_annotation' not in st.session_state:
    st.session_state['all_recommendations_for_annotation'] = {} 
if 'collected_annotations' not in st.session_state:
    st.session_state['collected_annotations'] = pd.DataFrame()
if 'annotator_details' not in st.session_state:
    st.session_state['annotator_details'] = {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS}
if 'current_annotator_slot_for_input' not in st.session_state: 
    st.session_state['current_annotator_slot_for_input'] = ANNOTATORS[0] if ANNOTATORS else None
if 'annotators_saved_status' not in st.session_state: 
    st.session_state['annotators_saved_status'] = set()


# --- Helper Functions ---
@st.cache_data(show_spinner='Memuat data pekerjaan...')
def load_and_combine_data_from_url(url, features_to_combine):
    try:
        df_full = pd.read_csv(url) 
        st.success('Berhasil memuat data dari URL!')

        if 'Job.ID' in df_full.columns:
            df_full['Job.ID'] = df_full['Job.ID'].astype(str)
        else:
            st.error("Kolom 'Job.ID' tidak ditemukan dalam dataset.")
            return None

        # Determine which of the features_to_combine are actually in the loaded dataframe
        existing_features_to_combine = [col for col in features_to_combine if col in df_full.columns]
        missing_features = [col for col in features_to_combine if col not in df_full.columns]
        if missing_features:
            st.warning(f"Fitur berikut tidak ditemukan di dataset dan akan diabaikan dalam penggabungan: {', '.join(missing_features)}")

        # Select Job.ID and only the existing features to combine for the working dataframe
        cols_to_keep_initially = ['Job.ID'] + existing_features_to_combine
        # Also keep 'Title' if it's not already in existing_features_to_combine but is in df_full
        if 'Title' in df_full.columns and 'Title' not in cols_to_keep_initially:
            cols_to_keep_initially.append('Title')
            
        df = df_full[list(set(cols_to_keep_initially))].copy() # Use set to avoid duplicate columns if Title was in features_to_combine

        for feature in existing_features_to_combine:
            df[feature] = df[feature].fillna('').astype(str)
        
        df['combined_jobs'] = df[existing_features_to_combine].agg(' '.join, axis=1)
        df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        st.success("Kolom 'combined_jobs' berhasil dibuat.")
        # Ensure 'Title' exists for downstream use, even if it wasn't part of combine (it usually is)
        if 'Title' not in df.columns and 'Title' in df_full.columns:
            df['Title'] = df_full['Title']

        return df
    except Exception as e:
        st.error(f'Error memuat atau menggabungkan data: {e}')
        return None

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

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip(): 
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words and w.isalnum()]
    if not filtered_words: 
        return ""
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in filtered_words]
    return " ".join(stemmed_words)

def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'):
    processed_results_intermediate = [] 
    if text_column_to_process not in data_df.columns:
        st.warning(f"Kolom '{text_column_to_process}' tidak ditemukan untuk preprocessing.")
        # Add empty columns if they don't exist to prevent downstream errors
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = ""
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = [{} for _ in range(len(data_df))]
        return data_df 

    with st.spinner(f"Preprocessing kolom '{text_column_to_process}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)
        
        processed_texts_col = [] # To build the new 'processed_text' column

        for i, text_content in enumerate(data_df[text_column_to_process].fillna('').astype(str)):
            intermediate = {'original': text_content}
            symbol_removed = text_content.translate(str.maketrans('', '', string.punctuation))
            symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed)
            intermediate['symbol_removed'] = symbol_removed
            case_folded = symbol_removed.lower()
            intermediate['case_folded'] = case_folded
            word_tokens_temp = word_tokenize(case_folded)
            intermediate['tokenized'] = " ".join(word_tokens_temp)
            stop_words_temp = set(stopwords.words('english'))
            valid_tokens_for_stop_stem = [w for w in word_tokens_temp if w.isalnum()]
            filtered_temp = [w for w in valid_tokens_for_stop_stem if w not in stop_words_temp]
            intermediate['stopwords_removed'] = " ".join(filtered_temp)
            porter_temp = PorterStemmer()
            stemmed_temp = [porter_temp.stem(w) for w in filtered_temp]
            final_processed_text = " ".join(stemmed_temp)
            intermediate['stemmed'] = final_processed_text
            
            processed_results_intermediate.append(intermediate)
            processed_texts_col.append(final_processed_text)

            if total_rows > 0:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
        
        data_df['processed_text'] = processed_texts_col # Assign the collected processed texts
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success(f"Preprocessing kolom '{text_column_to_process}' selesai! Kolom 'processed_text' telah dibuat/diperbarui.")
        progress_bar.empty()
        status_text.empty()
    return data_df

@st.cache_resource
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

@st.cache_data
def generate_embeddings_with_progress(_model, texts_list_to_embed): 
    if _model is None:
        st.error("BERT model is not loaded for embedding generation.")
        return np.array([]) 
    if not texts_list_to_embed: 
        st.warning("Input text list for embedding is empty.")
        return np.array([])
    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = [] 
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size] 
                batch_embeddings_np_array = _model.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False) 
                embeddings_result_list.extend(batch_embeddings_np_array) 
                if total_texts_to_embed > 0:
                    progress_val = (i + len(batch_texts_segment)) / total_texts_to_embed
                    embedding_progress_bar.progress(progress_val)
                    embedding_status_text.text(f"Embedded {i + len(batch_texts_segment)}/{total_texts_to_embed} texts.")
            st.success("Embedding generation complete!")
            embedding_progress_bar.empty()
            embedding_status_text.empty()
            return np.array(embeddings_result_list)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

@st.cache_data
def cluster_embeddings_with_progress(embeddings_to_cluster_param, n_clusters_for_algo): 
    if embeddings_to_cluster_param is None or embeddings_to_cluster_param.size == 0:
        st.warning("No embeddings provided for clustering.")
        return None
    if n_clusters_for_algo > embeddings_to_cluster_param.shape[0]:
        st.warning(f"K ({n_clusters_for_algo}) > samples ({embeddings_to_cluster_param.shape[0]}). Adjusting K.")
        n_clusters_for_algo = embeddings_to_cluster_param.shape[0]
    if n_clusters_for_algo < 1 : 
         st.error("Not enough samples to cluster (K < 1).")
         return None
    if n_clusters_for_algo == 1 and embeddings_to_cluster_param.shape[0] > 1 : 
        st.warning(f"K=1 requested for >1 samples. Setting K=2 for meaningful clustering.")
        n_clusters_for_algo = 2
    elif embeddings_to_cluster_param.shape[0] == 1 and n_clusters_for_algo >1: 
        st.warning(f"Only 1 sample available. Setting K=1.")
        n_clusters_for_algo = 1

    try:
        with st.spinner(f"Clustering {embeddings_to_cluster_param.shape[0]} embeddings into {n_clusters_for_algo} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters_for_algo, random_state=42, n_init='auto')
            clusters_assigned = kmeans.fit_predict(embeddings_to_cluster_param) 
            st.success(f"Clustering complete!")
            return clusters_assigned
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

# --- Page Functions ---
def home_page():
    st.header("Beranda: Analisis Data Eksploratif")
    st.write("Halaman ini menyediakan ringkasan dataset pekerjaan dan memungkinkan Anda untuk menjelajahi fiturnya.")

    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE)
    
    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Pratinjau Data (termasuk `combined_jobs`)')
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        
        # Add other features from FEATURES_TO_COMBINE if they exist and are not already included
        for col in FEATURES_TO_COMBINE:
            if col in data_df.columns and col not in cols_to_preview:
                cols_to_preview.append(col)
        
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Ringkasan Data')
        st.write(f'Jumlah baris: {len(data_df)}')
        st.write(f'Jumlah kolom: {len(data_df.columns)}')
        
        if 'combined_jobs' in data_df.columns:
            st.subheader('Contoh Isi Kolom `combined_jobs`')
            for i in range(min(3, len(data_df))):
                title_display = data_df.iloc[i]['Title'] if 'Title' in data_df.columns else "N/A"
                with st.expander(f"Job.ID: {data_df.iloc[i]['Job.ID']} - {title_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Kolom 'combined_jobs' belum dibuat atau tidak ada dalam data.")

        st.subheader('Cari Kata dalam Fitur')
        search_word = st.text_input("Masukkan kata untuk dicari:", key="home_search_word_new")
        available_cols_search = [col for col in ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE if col in data_df.columns]
        search_column = st.selectbox("Pilih fitur untuk dicari:", [''] + available_cols_search, key="home_search_column_new")

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column)

                if not search_results.empty:
                    st.write(f"Ditemukan {len(search_results)} entri untuk '{search_word}' di '{search_column}':")
                    st.dataframe(search_results[display_search_cols].head(), use_container_width=True) 
                else:
                    st.info(f"Tidak ada entri ditemukan untuk '{search_word}' di '{search_column}'.")
        
        st.subheader('Informasi Fitur')
        st.write('**Fitur yang tersedia (setelah pemrosesan):**', data_df.columns.tolist())
    else:
        st.error("Data tidak dapat dimuat. Mohon periksa sumber data atau koneksi Anda.")
    return

# CORRECTED preprocessing_page DEFINITION
def preprocessing_page():
    st.header("Preprocessing Data Pekerjaan")
    st.write("Halaman ini melakukan preprocessing pada kolom 'combined_jobs' dari dataset pekerjaan.")

    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Data pekerjaan atau kolom 'combined_jobs' belum tersedia. Silakan kembali ke halaman 'Home' untuk memuat data terlebih dahulu.")
        if st.button("Kembali ke Home untuk Memuat Data"):
            # This button doesn't automatically switch pages, but signals user action
            st.info("Silakan pilih 'Home' dari navigasi sidebar.")
        return
    
    data_df_to_preprocess = st.session_state['data']

    # Display info about the column to be processed
    st.info("Kolom 'combined_jobs' akan diproses untuk membuat kolom 'processed_text'.")
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("Lihat contoh 'combined_jobs' (sebelum diproses)"):
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())

    if st.button("Jalankan Preprocessing pada Kolom 'combined_jobs'", key="run_job_col_prep_btn"):
        with st.spinner("Sedang melakukan preprocessing pada 'combined_jobs'..."):
            # Always work on a copy if modifying and reassigning to session state
            data_copy = data_df_to_preprocess.copy()
            # Pass 'combined_jobs' as the column to process
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy, text_column_to_process='combined_jobs')
        st.success("Preprocessing kolom 'combined_jobs' selesai! Kolom 'processed_text' telah dibuat/diperbarui.")
    
    # Display results if 'processed_text' (result of preprocessing 'combined_jobs') exists
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Preprocessing pada 'combined_jobs' telah dilakukan.")
        display_data_processed = st.session_state['data'] # Renamed for clarity
        
        if 'preprocessing_steps' in display_data_processed.columns:
            st.subheader("Hasil Preprocessing (Langkah Menengah dari proses terakhir)")
            valid_intermediate_steps = [s for s in display_data_processed['preprocessing_steps'] if isinstance(s, dict)] # Renamed
            if valid_intermediate_steps:
                st.dataframe(pd.DataFrame(valid_intermediate_steps).head(), use_container_width=True)
            else:
                st.warning("Data langkah menengah preprocessing tidak dalam format yang diharapkan atau kosong.")
        
        st.subheader("Teks Akhir Hasil Preprocessing ('processed_text') (Pratinjau)")
        st.dataframe(display_data_processed[['Job.ID', 'combined_jobs', 'processed_text']].head(), use_container_width=True)
        
        search_word_in_processed = st.text_input("Cari kata dalam 'processed_text':", key="prep_job_proc_search") # Renamed
        if search_word_in_processed:
            search_results_in_processed = display_data_processed[display_data_processed['processed_text'].astype(str).str.contains(search_word_in_processed, na=False, case=False)] # Renamed
            if not search_results_in_processed.empty:
                st.dataframe(search_results_in_processed[['Job.ID', 'Title' if 'Title' in display_data_processed else 'Job.ID', 'processed_text']].head(), use_container_width=True)
            else:
                st.info(f"Tidak ada hasil untuk '{search_word_in_processed}' dalam 'processed_text'.")
    else:
        st.info("Kolom 'combined_jobs' tersedia, tetapi preprocessing belum dijalankan. Klik tombol di atas.")
    return

# ... (other page functions: tsdae_page, bert_model_page, etc. need to use 'processed_text') ...
# Ensure these functions are defined and correctly use 'processed_text'

def tsdae_page():
    st.header("TSDAE (Noise Injection & Embedding for Job Text)")
    st.write("Applies sequential noise and generates TSDAE embeddings for preprocessed job text.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed (from 'combined_jobs') first. Visit 'Preprocessing' page.")
        return
    # ... rest of TSDAE logic using st.session_state.data['processed_text'] ...
    st.info("TSDAE page implementation using 'processed_text'.")
    return

def bert_model_page():
    st.header("Standard BERT Embeddings (Job Descriptions)")
    st.write("Generates standard BERT embeddings from preprocessed job descriptions ('processed_text').")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed (from 'combined_jobs'). Visit 'Preprocessing' page.")
        return
    # ... rest of BERT model page logic using st.session_state.data['processed_text'] ...
    st.info("BERT Model page implementation using 'processed_text'.")
    return

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings generated from 'processed_text'.")
    if st.session_state.get('data') is None or \
       (st.session_state.get('job_text_embeddings') is None and st.session_state.get('tsdae_embeddings') is None):
        st.warning("Embeddings not generated yet. Please run 'BERT Model' or 'TSDAE' page first after preprocessing.")
        return
    # ... rest of Clustering logic ...
    st.info("Clustering page implementation.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    # ... (Upload CV logic as previously defined, it's independent of job data's combined_jobs) ...
    st.info("Upload CV page implementation.")
    return

def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations based on CVs and job embeddings (from 'processed_text').")
    # ... (Job Recommendation logic as previously defined, ensure it uses job embeddings from 'processed_text') ...
    st.info("Job Recommendation page implementation.")
    return

def annotation_page():
    st.header("Annotation of Job Recommendations")
    # ... (Annotation logic as previously defined) ...
    st.info("Annotation page implementation.")
    return

def _calculate_average_precision(ranked_relevance_binary, k_val):
    if not ranked_relevance_binary: return 0.0
    ranked_relevance_binary = ranked_relevance_binary[:k_val] 
    relevant_hits, sum_precisions = 0, 0.0
    for i, is_relevant in enumerate(ranked_relevance_binary):
        if is_relevant:
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1)
    return sum_precisions / relevant_hits if relevant_hits > 0 else 0.0

def evaluation_page():
    st.header("Model Evaluation")
    # ... (Evaluation logic as previously defined) ...
    st.info("Evaluation page implementation.")
    return


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio")

if page == "Home":
    home_page()
elif page == "Preprocessing":
    preprocessing_page() # Ensure this is now correctly defined and called
elif page == "TSDAE (Noise Injection)":
    tsdae_page()
elif page == "BERT Model":
    bert_model_page()
elif page == "Clustering Job2Vec":
    clustering_page()
elif page == "Upload CV": 
    upload_cv_page()
elif page == "Job Recommendation":
    job_recommendation_page()
elif page == "Annotation":
    annotation_page()
elif page == "Evaluation":
    evaluation_page()
