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
@st.cache_data(show_spinner='Loading job data...') # English
def load_and_combine_data_from_url(url, features_to_combine):
    try:
        df_full = pd.read_csv(url) 
        st.success('Successfully loaded data from URL!') # English

        if 'Job.ID' in df_full.columns:
            df_full['Job.ID'] = df_full['Job.ID'].astype(str)
        else:
            st.error("Column 'Job.ID' not found in the dataset.") # English
            return None

        existing_features_to_combine = [col for col in features_to_combine if col in df_full.columns]
        missing_features = [col for col in features_to_combine if col not in df_full.columns]
        if missing_features:
            st.warning(f"The following features were not found in the dataset and will be ignored in the combination: {', '.join(missing_features)}") # English

        cols_to_keep_initially = ['Job.ID'] + existing_features_to_combine
        if 'Title' in df_full.columns and 'Title' not in cols_to_keep_initially:
            cols_to_keep_initially.append('Title')
            
        df = df_full[list(set(cols_to_keep_initially))].copy() 

        for feature in existing_features_to_combine:
            df[feature] = df[feature].fillna('').astype(str)
        
        df['combined_jobs'] = df[existing_features_to_combine].agg(' '.join, axis=1)
        df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        st.success("Column 'combined_jobs' created successfully.") # English
        if 'Title' not in df.columns and 'Title' in df_full.columns:
            df['Title'] = df_full['Title']

        return df
    except Exception as e:
        st.error(f'Error loading or combining data: {e}') # English
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
        st.warning(f"Column '{text_column_to_process}' not found for preprocessing.") # English
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = ""
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = [{} for _ in range(len(data_df))]
        return data_df 

    with st.spinner(f"Preprocessing column '{text_column_to_process}'..."): # English
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)
        
        processed_texts_col = [] 

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
                status_text.text(f"Processed {i + 1}/{total_rows} entries.") # English
        
        data_df['processed_text'] = processed_texts_col 
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success(f"Preprocessing of column '{text_column_to_process}' complete! Column 'processed_text' has been created/updated.") # English
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
    st.header("Home: Exploratory Data Analysis") # English
    st.write("This page provides an overview of the job dataset and allows you to explore its features.") # English

    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE)
    
    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Data Preview (including `combined_jobs`)') # English
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        
        for col in FEATURES_TO_COMBINE:
            if col in data_df.columns and col not in cols_to_preview:
                cols_to_preview.append(col)
        
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Data Summary') # English
        st.write(f'Number of rows: {len(data_df)}') # English
        st.write(f'Number of columns: {len(data_df.columns)}') # English
        
        if 'combined_jobs' in data_df.columns:
            st.subheader('Sample Content of `combined_jobs` Column') # English
            for i in range(min(3, len(data_df))):
                title_display = data_df.iloc[i]['Title'] if 'Title' in data_df.columns else "N/A"
                with st.expander(f"Job.ID: {data_df.iloc[i]['Job.ID']} - {title_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Column 'combined_jobs' has not been created or is not in the data.") # English

        st.subheader('Search Word in Feature') # English
        search_word = st.text_input("Enter word to search:", key="home_search_word_new") # English
        available_cols_search = [col for col in ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE if col in data_df.columns]
        search_column = st.selectbox("Select feature to search in:", [''] + available_cols_search, key="home_search_column_new") # English

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column)

                if not search_results.empty:
                    st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':") # English
                    st.dataframe(search_results[display_search_cols].head(), use_container_width=True) 
                else:
                    st.info(f"No entries found for '{search_word}' in '{search_column}'.") # English
        
        st.subheader('Feature Information') # English
        st.write('**Available Features (after processing):**', data_df.columns.tolist()) # English
    else:
        st.error("Data could not be loaded. Please check the data source or your connection.") # English
    return

def preprocessing_page():
    st.header("Job Data Preprocessing") # English
    st.write("This page performs preprocessing on the 'combined_jobs' column of the job dataset.") # English

    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or 'combined_jobs' column not available. Please return to the 'Home' page to load data first.") # English
        if st.button("Return to Home to Load Data"): # English
            st.info("Please select 'Home' from the sidebar navigation.") # English
        return
    
    data_df_to_preprocess = st.session_state['data']

    st.info("The 'combined_jobs' column will be processed to create the 'processed_text' column.") # English
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before processing)"): # English
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())

    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_col_prep_btn"): # English
        with st.spinner("Preprocessing 'combined_jobs'..."): # English
            data_copy = data_df_to_preprocess.copy()
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy, text_column_to_process='combined_jobs')
        st.success("Preprocessing of 'combined_jobs' column complete! 'processed_text' column has been created/updated.") # English
    
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Preprocessing has been performed on 'combined_jobs'.") # English
        display_data_processed = st.session_state['data'] 
        
        if 'preprocessing_steps' in display_data_processed.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)") # English
            valid_intermediate_steps = [s for s in display_data_processed['preprocessing_steps'] if isinstance(s, dict)] 
            if valid_intermediate_steps:
                st.dataframe(pd.DataFrame(valid_intermediate_steps).head(), use_container_width=True)
            else:
                st.warning("Intermediate preprocessing steps data is not in the expected format or is empty.") # English
        
        st.subheader("Final Preprocessed Text ('processed_text') (Preview)") # English
        st.dataframe(display_data_processed[['Job.ID', 'combined_jobs', 'processed_text']].head(), use_container_width=True)
        
        search_word_in_processed = st.text_input("Search word in 'processed_text':", key="prep_job_proc_search") # English
        if search_word_in_processed:
            search_results_in_processed = display_data_processed[display_data_processed['processed_text'].astype(str).str.contains(search_word_in_processed, na=False, case=False)] 
            if not search_results_in_processed.empty:
                st.dataframe(search_results_in_processed[['Job.ID', 'Title' if 'Title' in display_data_processed else 'Job.ID', 'processed_text']].head(), use_container_width=True)
            else:
                st.info(f"No results for '{search_word_in_processed}' in 'processed_text'.") # English
    else:
        st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet. Click the button above.") # English
    return

def tsdae_page():
    st.header("TSDAE (Noise Injection & Embedding for Job Text)")
    st.write("Applies sequential noise and generates TSDAE embeddings for preprocessed job text ('processed_text').") # English
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed (from 'combined_jobs') first. Visit 'Preprocessing' page.") # English
        return
    bert_model = load_bert_model()
    if bert_model is None: return

    st.subheader("TSDAE Settings")
    del_ratio = st.slider("Deletion Ratio", 0.1, 0.9, 0.6, 0.1, key="tsdae_del_r")
    freq_thresh = st.slider("High Freq Threshold", 10, 500, 100, 10, key="tsdae_freq_t")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="tsdae_run_all_btn"):
        data_tsdae = st.session_state['data'].copy()
        # Ensure 'processed_text' is used for generating word frequencies
        if 'processed_text' not in data_tsdae.columns or data_tsdae['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Cannot proceed with TSDAE.")
            return

        words_for_freq = [w for txt in data_tsdae['processed_text'].fillna('').astype(str) for w in word_tokenize(txt)]
        word_freq = {w.lower(): words_for_freq.count(w.lower()) for w in set(words_for_freq)}
        if not word_freq: st.warning("Word frequency dictionary for TSDAE is empty.")

        with st.spinner("Applying Noise A..."): data_tsdae['noisy_text_a'] = data_tsdae['processed_text'].fillna('').astype(str).apply(lambda x: denoise_text(x, 'a', del_ratio))
        with st.spinner("Applying Noise B..."): data_tsdae['noisy_text_b'] = data_tsdae['noisy_text_a'].astype(str).apply(lambda x: denoise_text(x, 'b', del_ratio, word_freq, freq_thresh))
        with st.spinner("Applying Noise C..."): data_tsdae['final_noisy_text'] = data_tsdae['noisy_text_b'].astype(str).apply(lambda x: denoise_text(x, 'c', del_ratio, word_freq, freq_thresh))
        st.session_state['data'] = data_tsdae
        st.success("Noise application complete.")
        st.dataframe(data_tsdae[['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        noisy_series = data_tsdae['final_noisy_text'].fillna('').astype(str)
        mask = noisy_series.str.strip() != ''
        valid_noisy_texts = noisy_series[mask].tolist()
        valid_noisy_job_ids = data_tsdae.loc[mask, 'Job.ID'].tolist()
        if not valid_noisy_texts: st.warning("No valid noisy texts for TSDAE embedding.")
        else:
            st.session_state['tsdae_embeddings'] = generate_embeddings_with_progress(bert_model, valid_noisy_texts)
            st.session_state['tsdae_embedding_job_ids'] = valid_noisy_job_ids
            if st.session_state.get('tsdae_embeddings', np.array([])).size > 0: st.success(f"TSDAE embeddings generated for {len(valid_noisy_job_ids)} jobs!")
            else: st.warning("TSDAE embedding output empty.")
    
    if st.session_state.get('tsdae_embeddings') is not None:
        st.subheader("Current TSDAE Embeddings")
        st.write(f"Shape: {st.session_state['tsdae_embeddings'].shape} (for {len(st.session_state.get('tsdae_embedding_job_ids',[]))} jobs)")
    if 'final_noisy_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader("Current Noisy Text Columns (Preview)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)
    return

def bert_model_page():
    st.header("Standard BERT Embeddings (Job Descriptions)")
    st.write("Generates standard BERT embeddings from 'processed_text' (derived from 'combined_jobs').") # English
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed. Visit 'Preprocessing' page.") # English
        return
    bert_model = load_bert_model()
    if bert_model is None: return

    if st.button("Generate/Regenerate Standard Job Embeddings", key="gen_std_emb_btn"):
        data_bert = st.session_state['data']
        # Ensure 'processed_text' is used
        if 'processed_text' not in data_bert.columns or data_bert['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Please run preprocessing first.")
            return

        proc_series = data_bert['processed_text'].fillna('').astype(str)
        mask = proc_series.str.strip() != ''
        valid_texts = proc_series[mask].tolist()
        valid_job_ids = data_bert.loc[mask, 'Job.ID'].tolist()
        if not valid_texts: st.warning("No valid processed job texts for embedding.")
        else:
            st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, valid_texts)
            st.session_state['job_text_embedding_job_ids'] = valid_job_ids
            if st.session_state.get('job_text_embeddings', np.array([])).size > 0: st.success(f"Standard job embeddings generated for {len(valid_job_ids)} jobs!")
            else: st.warning("Standard job embedding output empty.")

    job_emb = st.session_state.get('job_text_embeddings')
    job_ids = st.session_state.get('job_text_embedding_job_ids')
    if job_emb is not None and job_emb.size > 0 and job_ids:
        st.subheader(f"Current Standard Job Embeddings ({len(job_ids)} jobs)")
        st.write(f"Shape: {job_emb.shape}")
        st.subheader("2D Visualization (PCA)")
        if len(job_emb) >= 2: 
            try:
                pca_2d = PCA(n_components=2).fit_transform(job_emb)
                plot_pca_df = pd.DataFrame(pca_2d, columns=['PC1','PC2'])
                plot_pca_df['Job.ID'] = job_ids
                # Ensure 'text' (original description) and 'Title' are present for hover data
                hover_data_cols = ['Job.ID']
                if 'Title' in st.session_state['data'].columns: hover_data_cols.append('Title')
                if 'text' in st.session_state['data'].columns: hover_data_cols.append('text') # original 'text' if available
                elif 'Job.Description' in st.session_state['data'].columns: hover_data_cols.append('Job.Description') # fallback
                
                hover_df = st.session_state['data'][st.session_state['data']['Job.ID'].isin(job_ids)][hover_data_cols]
                plot_pca_df = pd.merge(plot_pca_df, hover_df, on='Job.ID', how='left')
                
                hover_text_col = 'text' if 'text' in plot_pca_df.columns else 'Job.Description' if 'Job.Description' in plot_pca_df.columns else None

                if not plot_pca_df.empty and 'Title' in plot_pca_df.columns:
                    fig_pca = px.scatter(plot_pca_df, 'PC1','PC2', 
                                         hover_name='Title', 
                                         hover_data={hover_text_col: True, 'Job.ID': True, 'PC1':False, 'PC2':False} if hover_text_col else {'Job.ID': True, 'PC1':False, 'PC2':False},
                                         title='2D PCA of Standard Job Embeddings')
                    st.plotly_chart(fig_pca, use_container_width=True)
                else: st.warning("PCA plot data incomplete (Title or hover text missing).")
            except Exception as e: st.error(f"PCA Error: {e}")
        else: st.warning("Need >= 2 data points for PCA.")
    else: st.info("Standard job embeddings not generated yet.")
    return

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings generated from 'processed_text'.") # English
    if st.session_state.get('data') is None: st.error("Job data not loaded."); return

    emb_to_cluster, job_ids_clust, src_name_clust = None, None, ""
    choice = st.radio("Embeddings for clustering:", ("TSDAE", "Standard BERT"), key="clust_emb_choice", horizontal=True)

    if choice == "TSDAE":
        if st.session_state.get('tsdae_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['tsdae_embeddings']
            job_ids_clust = st.session_state.get('tsdae_embedding_job_ids')
            src_name_clust = "TSDAE Embeddings"
            if not job_ids_clust: st.error("TSDAE Job IDs missing."); return
        else: st.warning("TSDAE embeddings unavailable."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['job_text_embeddings']
            job_ids_clust = st.session_state.get('job_text_embedding_job_ids')
            src_name_clust = "Standard BERT Job Embeddings"
            if not job_ids_clust: st.error("Std BERT Job IDs missing."); return
        else: st.warning("Std BERT embeddings unavailable."); return
    
    st.info(f"Using: {src_name_clust} ({len(job_ids_clust)} items)")
    if emb_to_cluster is not None and job_ids_clust:
        max_k_val = emb_to_cluster.shape[0]
        if max_k_val < 2: st.error("Need >= 2 items to cluster."); return
        k_val = st.slider("Number of Clusters (K)", 2, min(50, max_k_val), min(N_CLUSTERS, max_k_val), key="k_slider_c")
        if st.button(f"Run K-Means (K={k_val}) on {src_name_clust}", key="run_kmeans_c_btn"):
            labels = cluster_embeddings_with_progress(emb_to_cluster, k_val)
            if labels is not None:
                if len(job_ids_clust) == len(labels):
                    info_df = pd.DataFrame({'Job.ID': job_ids_clust, 'cluster': labels})
                    data_copy = st.session_state['data'].copy()
                    if 'cluster' in data_copy.columns: data_copy = data_copy.drop(columns=['cluster'])
                    st.session_state['data'] = pd.merge(data_copy, info_df, on='Job.ID', how='left')
                    st.success(f"'cluster' column updated for {len(job_ids_clust)} jobs.")
                else: st.error("Job ID / cluster label length mismatch.")
            else: st.error("Clustering failed.")

    if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader(f"Current Clustering (K={st.session_state['data']['cluster'].nunique(dropna=True)})")
        # Use 'combined_jobs' or 'processed_text' for display if 'text' is no longer primary
        display_text_col = 'combined_jobs' if 'combined_jobs' in st.session_state['data'].columns else 'processed_text'
        st.dataframe(st.session_state['data'][['Job.ID', 'Title', display_text_col, 'cluster']].head(10), height=300)
        valid_cl = st.session_state['data']['cluster'].dropna().unique()
        if valid_cl.size > 0:
            st.subheader("Sample Job Descriptions per Cluster")
            for c_num in sorted(valid_cl):
                st.write(f"**Cluster {int(c_num)}:**")
                subset = st.session_state['data'][st.session_state['data']['cluster'] == c_num]
                if not subset.empty: st.dataframe(subset[['Job.ID', 'Title', display_text_col]].sample(min(3,len(subset)),random_state=1), height=150)
                st.write("---")
    else: st.info("No 'cluster' column in dataset or no clusters assigned.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5).") # English
    uploaded_cv_files = st.file_uploader("Choose CV files:", type=["pdf","docx"], accept_multiple_files=True, key="cv_upload_widget_main") # English
    if uploaded_cv_files:
        if len(uploaded_cv_files) > 5:
            st.warning("Max 5 CVs. Processing first 5.") # English
            uploaded_cv_files = uploaded_cv_files[:5]
        if st.button("Process Uploaded CVs", key="proc_cv_btn_main"): # English
            cv_data_batch = []
            bert_model_for_cv = load_bert_model()
            if not bert_model_for_cv: 
                st.error("BERT model load failed for CVs."); return 
            with st.spinner("Processing CVs..."): # English
                for i, cv_file in enumerate(uploaded_cv_files):
                    o_txt, p_txt, cv_e = "", "", None
                    try:
                        file_ext = cv_file.name.split(".")[-1].lower()
                        if file_ext == "pdf": o_txt = extract_text_from_pdf(cv_file)
                        elif file_ext == "docx": o_txt = extract_text_from_docx(cv_file)
                        if o_txt and o_txt.strip():
                            p_txt = preprocess_text(o_txt) 
                            if p_txt and p_txt.strip():
                                e_arr = generate_embeddings_with_progress(bert_model_for_cv, [p_txt])
                                cv_e = e_arr[0] if (e_arr is not None and e_arr.size > 0) else None
                        cv_data_batch.append({'filename':cv_file.name, 'original_text':o_txt or "", 
                                              'processed_text':p_txt or "", 'embedding':cv_e})
                        if cv_e is not None: st.success(f"Processed & embedded: {cv_file.name}") # English
                        else: st.warning(f"Failed to process/embed: {cv_file.name}") # English
                    except Exception as e:
                        st.error(f"Error with {cv_file.name}: {e}") # English
                st.session_state['uploaded_cvs_data'] = cv_data_batch
                st.success(f"CV batch processing done.") # English

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs:") # English
        for i, cv_d in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_d.get('filename', 'N/A')}"):
                st.text_area(f"Original:", cv_d.get('original_text',''), height=70, disabled=True, key=f"disp_cv_o_{i}")
                st.text_area(f"Processed:", cv_d.get('processed_text',''), height=70, disabled=True, key=f"disp_cv_p_{i}")
                st.success("Embedding OK.") if cv_d.get('embedding') is not None and cv_d.get('embedding').size > 0 else st.warning("Embedding missing.") # English
    return


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations based on CVs and job embeddings (from 'processed_text').") # English
    if not st.session_state.get('uploaded_cvs_data'): 
        st.warning("Upload & process CVs first."); return
    main_data = st.session_state.get('data')
    # Ensure 'processed_text' (derived from 'combined_jobs') is used for job embeddings
    if main_data is None or 'processed_text' not in main_data.columns:
        st.error("Job data with 'processed_text' (from 'combined_jobs') not available. Load & preprocess first."); return
    
    # ... (rest of job_recommendation_page logic, ensure it uses job embeddings from 'processed_text') ...
    st.info("Job Recommendation page implementation using 'processed_text' for jobs.") # English
    return


def annotation_page():
    st.header("Annotation of Job Recommendations")
    st.write("Annotate relevance and provide feedback for recommended jobs.") # English
    # ... (Annotation logic as previously defined) ...
    st.info("Annotation page implementation.") # English
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
    st.write("Evaluates top 20 recommendations based on human annotations.") # English
    # ... (Evaluation logic as previously defined) ...
    st.info("Evaluation page implementation.") # English
    return


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio")

if page == "Home":
    home_page()
elif page == "Preprocessing":
    preprocessing_page() 
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
