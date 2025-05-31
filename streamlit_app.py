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
# from sentence_transformers.evaluation import RerankingEvaluator 
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
JOB_DETAIL_FEATURES_TO_DISPLAY = [
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type', 
    'Position', 'Industry', 'Education.Required', 'State.Name'
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
if 'uploaded_cvs_data' not in st.session_state:
    st.session_state['uploaded_cvs_data'] = [] 
if 'all_recommendations_for_annotation' not in st.session_state:
    st.session_state['all_recommendations_for_annotation'] = {} 
if 'collected_annotations' not in st.session_state: 
    st.session_state['collected_annotations'] = pd.DataFrame()
if 'uploaded_annotation_data' not in st.session_state: # For storing DF from uploaded annotation CSV
    st.session_state['uploaded_annotation_data'] = None
if 'annotator_details' not in st.session_state:
    st.session_state['annotator_details'] = {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS}
if 'current_annotator_slot_for_input' not in st.session_state: 
    st.session_state['current_annotator_slot_for_input'] = ANNOTATORS[0] if ANNOTATORS else None
if 'annotators_saved_status' not in st.session_state: 
    st.session_state['annotators_saved_status'] = set()


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading job data...')
def load_and_combine_data_from_url(url, features_to_combine_list, detail_features_to_ensure):
    try:
        df_full = pd.read_csv(url) 
        st.success('Successfully loaded data from URL!')

        if 'Job.ID' not in df_full.columns:
            st.error("Column 'Job.ID' not found in the dataset.")
            return None
        df_full['Job.ID'] = df_full['Job.ID'].astype(str)

        existing_features_to_combine = [col for col in features_to_combine_list if col in df_full.columns]
        missing_features_for_combine = [col for col in features_to_combine_list if col not in df_full.columns]
        if missing_features_for_combine:
            st.warning(f"The following features intended for combination were not found: {', '.join(missing_features_for_combine)}")

        cols_to_load_set = set(['Job.ID', 'Title']) 
        cols_to_load_set.update(existing_features_to_combine)
        cols_to_load_set.update(detail_features_to_ensure)
        
        actual_cols_to_load = [col for col in list(cols_to_load_set) if col in df_full.columns]
        df = df_full[actual_cols_to_load].copy() 

        for feature in existing_features_to_combine: 
            if feature in df.columns: 
                df[feature] = df[feature].fillna('').astype(str)
        
        df['combined_jobs'] = df[existing_features_to_combine].agg(' '.join, axis=1)
        df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        st.success("Column 'combined_jobs' created successfully.")
        return df
    except Exception as e:
        st.error(f'Error loading or combining data: {e}')
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

def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    if not isinstance(text_to_denoise, str) or not text_to_denoise.strip():
        return "" 
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return "" 
    result_words = [] 
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0 and n > 0 : 
            idx_to_keep = np.random.choice(n) 
            keep_or_not[idx_to_keep] = True
        result_words = np.array(words)[keep_or_not].tolist() 
    elif method in ('b', 'c'):
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'.")
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0 and num_to_remove <= len(high_freq_indices):
             to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result_words and words: 
            result_words = [random.choice(words)]
        if method == 'c' and result_words: 
            random.shuffle(result_words)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result_words)

def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'):
    processed_results_intermediate = [] 
    if text_column_to_process not in data_df.columns:
        st.warning(f"Column '{text_column_to_process}' not found for preprocessing.")
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = ""
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = [{} for _ in range(len(data_df))]
        return data_df 

    with st.spinner(f"Preprocessing column '{text_column_to_process}'..."):
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
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
        
        data_df['processed_text'] = processed_texts_col 
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success(f"Preprocessing of column '{text_column_to_process}' complete! Column 'processed_text' has been created/updated.")
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
    st.header("Home: Exploratory Data Analysis") 
    st.write("This page provides an overview of the job dataset and allows you to explore its features.") 

    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES_TO_DISPLAY)
    
    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Data Preview (including `combined_jobs`)') 
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Data Summary') 
        st.write(f'Number of rows: {len(data_df)}') 
        st.write(f'Number of columns: {len(data_df.columns)}') 
        
        if 'combined_jobs' in data_df.columns:
            st.subheader('Sample Content of `combined_jobs` Column') 
            for i in range(min(3, len(data_df))):
                title_display = data_df.iloc[i]['Title'] if 'Title' in data_df.columns else "N/A"
                with st.expander(f"Job.ID: {data_df.iloc[i]['Job.ID']} - {title_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Column 'combined_jobs' has not been created or is not in the data.") 

        st.subheader('Search Word in Feature') 
        search_word = st.text_input("Enter word to search:", key="home_search_word_new") 
        all_available_cols_for_search = ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE + JOB_DETAIL_FEATURES_TO_DISPLAY
        searchable_cols = sorted(list(set(col for col in all_available_cols_for_search if col in data_df.columns)))
        search_column = st.selectbox("Select feature to search in:", [''] + searchable_cols, key="home_search_column_new") 

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column)

                if not search_results.empty:
                    st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':") 
                    st.dataframe(search_results[display_search_cols].head(), use_container_width=True) 
                else:
                    st.info(f"No entries found for '{search_word}' in '{search_column}'.") 
        
        st.subheader('Feature Information') 
        st.write('**Available Features (after processing):**', data_df.columns.tolist()) 
    else:
        st.error("Data could not be loaded. Please check the data source or your connection.") 
    return

def preprocessing_page():
    st.header("Job Data Preprocessing") 
    st.write("This page performs preprocessing on the 'combined_jobs' column of the job dataset.") 

    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or 'combined_jobs' column not available. Please return to the 'Home' page to load data first.") 
        if st.button("Return to Home to Load Data"): 
            st.info("Please select 'Home' from the sidebar navigation.") 
        return
    
    data_df_to_preprocess = st.session_state['data']

    st.info("The 'combined_jobs' column will be processed to create the 'processed_text' column.") 
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before processing)"): 
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())

    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_col_prep_btn"): 
        with st.spinner("Preprocessing 'combined_jobs'..."): 
            data_copy = data_df_to_preprocess.copy()
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy, text_column_to_process='combined_jobs')
        st.success("Preprocessing of 'combined_jobs' column complete! 'processed_text' column has been created/updated.") 
    
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Preprocessing has been performed on 'combined_jobs'.") 
        display_data_processed = st.session_state['data'] 
        
        if 'preprocessing_steps' in display_data_processed.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)") 
            valid_intermediate_steps = [s for s in display_data_processed['preprocessing_steps'] if isinstance(s, dict)] 
            if valid_intermediate_steps:
                st.dataframe(pd.DataFrame(valid_intermediate_steps).head(), use_container_width=True)
            else:
                st.warning("Intermediate preprocessing steps data is not in the expected format or is empty.") 
        
        st.subheader("Final Preprocessed Text ('processed_text') (Preview)") 
        st.dataframe(display_data_processed[['Job.ID', 'combined_jobs', 'processed_text']].head(), use_container_width=True)
        
        search_word_in_processed = st.text_input("Search word in 'processed_text':", key="prep_job_proc_search") 
        if search_word_in_processed:
            search_results_in_processed = display_data_processed[display_data_processed['processed_text'].astype(str).str.contains(search_word_in_processed, na=False, case=False)] 
            if not search_results_in_processed.empty:
                display_cols_search_proc = ['Job.ID']
                if 'Title' in display_data_processed.columns: display_cols_search_proc.append('Title')
                display_cols_search_proc.append('processed_text')
                st.dataframe(search_results_in_processed[display_cols_search_proc].head(), use_container_width=True)
            else:
                st.info(f"No results for '{search_word_in_processed}' in 'processed_text'.") 
    else:
        st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet. Click the button above.") 
    return

def tsdae_page():
    st.header("TSDAE (Noise Injection & Embedding for Job Text)")
    st.write("Applies sequential noise and generates TSDAE embeddings for 'processed_text' (derived from 'combined_jobs').") 
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed (from 'combined_jobs') first. Visit 'Preprocessing' page.") 
        return
    
    bert_model = load_bert_model() 
    if bert_model is None: 
        st.error("BERT model could not be loaded for TSDAE page."); return

    st.subheader("TSDAE Settings")
    deletion_ratio = st.slider("Deletion Ratio", 0.1, 0.9, 0.6, 0.1, key="tsdae_del_ratio_main")
    freq_threshold = st.slider("High Frequency Threshold", 10, 500, 100, 10, key="tsdae_freq_thresh_main")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="tsdae_run_button_main"):
        data_tsdae_local = st.session_state['data'].copy()
        if 'processed_text' not in data_tsdae_local.columns or data_tsdae_local['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Cannot proceed with TSDAE.")
            return

        all_words = [w for txt_proc in data_tsdae_local['processed_text'].fillna('').astype(str) for w in word_tokenize(txt_proc)]
        word_freq_dict_tsdae = {word.lower(): all_words.count(word.lower()) for word in set(all_words)}
        if not word_freq_dict_tsdae: st.warning("Word frequency dictionary for TSDAE is empty (all processed texts might be empty).")

        st.markdown("---")
        st.markdown("##### Applying Noise Method A (Random Deletion)")
        noisy_text_stage_a = []
        source_texts_a = data_tsdae_local['processed_text'].fillna('').astype(str).tolist()
        total_items_a = len(source_texts_a)
        progress_bar_a = st.progress(0)
        status_text_a = st.empty()
        for idx, text_content in enumerate(source_texts_a):
            noisy_text_stage_a.append(denoise_text(text_content, method='a', del_ratio=deletion_ratio))
            if total_items_a > 0:
                progress_bar_a.progress((idx + 1) / total_items_a)
                status_text_a.text(f"Method A: Processed {idx + 1}/{total_items_a} entries.")
        data_tsdae_local['noisy_text_a'] = noisy_text_stage_a
        progress_bar_a.empty(); status_text_a.empty()
        st.success("Method A noise application complete.")

        st.markdown("---")
        st.markdown("##### Applying Noise Method B (High-Frequency Word Removal)")
        noisy_text_stage_b = []
        source_texts_b = data_tsdae_local['noisy_text_a'].tolist() 
        total_items_b = len(source_texts_b)
        progress_bar_b = st.progress(0)
        status_text_b = st.empty()
        for idx, text_content in enumerate(source_texts_b):
            noisy_text_stage_b.append(denoise_text(text_content, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict_tsdae, freq_threshold=freq_threshold))
            if total_items_b > 0:
                progress_bar_b.progress((idx + 1) / total_items_b)
                status_text_b.text(f"Method B: Processed {idx + 1}/{total_items_b} entries.")
        data_tsdae_local['noisy_text_b'] = noisy_text_stage_b
        progress_bar_b.empty(); status_text_b.empty()
        st.success("Method B noise application complete.")

        st.markdown("---")
        st.markdown("##### Applying Noise Method C (High-Frequency Word Removal + Shuffle)")
        final_noisy_texts_list = [] 
        source_texts_c = data_tsdae_local['noisy_text_b'].tolist() 
        total_items_c = len(source_texts_c)
        progress_bar_c = st.progress(0)
        status_text_c = st.empty()
        for idx, text_content in enumerate(source_texts_c):
            final_noisy_texts_list.append(denoise_text(text_content, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict_tsdae, freq_threshold=freq_threshold))
            if total_items_c > 0:
                progress_bar_c.progress((idx + 1) / total_items_c)
                status_text_c.text(f"Method C: Processed {idx + 1}/{total_items_c} entries.")
        data_tsdae_local['final_noisy_text'] = final_noisy_texts_list
        progress_bar_c.empty(); status_text_c.empty()
        st.success("Method C noise application complete.")
        
        st.session_state['data'] = data_tsdae_local 
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        final_noisy_texts_series = st.session_state['data']['final_noisy_text'].fillna('').astype(str)
        non_empty_mask = final_noisy_texts_series.str.strip() != ''
        valid_final_noisy_texts = final_noisy_texts_series[non_empty_mask].tolist()
        job_ids_for_tsdae_embeddings = st.session_state['data'].loc[non_empty_mask, 'Job.ID'].tolist()

        if not valid_final_noisy_texts:
            st.warning("No valid (non-empty) 'final_noisy_text' found to generate TSDAE embeddings.")
        else:
            tsdae_embeddings_generated = generate_embeddings_with_progress(bert_model, valid_final_noisy_texts)
            if tsdae_embeddings_generated.size > 0:
                st.session_state['tsdae_embeddings'] = tsdae_embeddings_generated
                st.session_state['tsdae_embedding_job_ids'] = job_ids_for_tsdae_embeddings
                st.success(f"TSDAE embeddings generated for {len(job_ids_for_tsdae_embeddings)} jobs!")
            else:
                st.warning("TSDAE embedding generation resulted in an empty output.")
    
    if 'tsdae_embeddings' in st.session_state and st.session_state.tsdae_embeddings is not None:
        st.subheader("Current TSDAE Embeddings (Preview)")
        st.write(f"Shape: {st.session_state.tsdae_embeddings.shape}")
        st.write(st.session_state.tsdae_embeddings[:3])
    return

def bert_model_page():
    st.header("Standard BERT Embeddings (Job Descriptions)")
    st.write("Generates standard BERT embeddings from 'processed_text' (derived from 'combined_jobs').") 
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed. Visit 'Preprocessing' page.") 
        return
    bert_model = load_bert_model()
    if bert_model is None: return

    if st.button("Generate/Regenerate Standard Job Embeddings", key="gen_std_emb_btn"):
        data_bert = st.session_state['data']
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
            if st.session_state.get('job_text_embeddings', np.array([])).size > 0: st.success(f"Std job embeddings generated for {len(valid_job_ids)} jobs!")
            else: st.warning("Std job embedding output empty.")

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
                
                hover_data_cols = ['Job.ID']
                main_data_for_hover = st.session_state['data']
                if 'Title' in main_data_for_hover.columns: hover_data_cols.append('Title')
                
                description_col_for_hover = 'combined_jobs' if 'combined_jobs' in main_data_for_hover.columns else 'text' 
                if description_col_for_hover in main_data_for_hover.columns : hover_data_cols.append(description_col_for_hover)
                
                hover_df = main_data_for_hover[main_data_for_hover['Job.ID'].isin(job_ids)][hover_data_cols]
                plot_pca_df = pd.merge(plot_pca_df, hover_df, on='Job.ID', how='left')
                
                hover_data_config = {'Job.ID':True, 'PC1':False, 'PC2':False}
                if description_col_for_hover in plot_pca_df.columns:
                    hover_data_config[description_col_for_hover] = True


                if not plot_pca_df.empty and 'Title' in plot_pca_df.columns:
                    fig_pca = px.scatter(plot_pca_df, 'PC1','PC2', 
                                         hover_name='Title', 
                                         hover_data=hover_data_config,
                                         title='2D PCA of Standard Job Embeddings')
                    st.plotly_chart(fig_pca, use_container_width=True)
                else: st.warning("PCA plot data incomplete (Title or hover text missing).")
            except Exception as e: st.error(f"PCA Error: {e}")
        else: st.warning("Need >= 2 data points for PCA.")
    else: st.info("Standard job embeddings not generated yet.")
    return

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings generated from 'processed_text'.")
    if st.session_state.get('data') is None: 
        st.error("Job data not loaded. Please go to Home page first."); return

    emb_to_cluster, job_ids_clust, src_name_clust = None, None, ""
    choice = st.radio("Embeddings for clustering:", ("TSDAE", "Standard BERT"), key="clust_emb_choice_main", horizontal=True)

    if choice == "TSDAE":
        if st.session_state.get('tsdae_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['tsdae_embeddings']
            job_ids_clust = st.session_state.get('tsdae_embedding_job_ids')
            src_name_clust = "TSDAE Embeddings"
            if not job_ids_clust: st.error("TSDAE Job IDs missing. Please run TSDAE embedding generation."); return
        else: st.warning("TSDAE embeddings unavailable. Please generate them on the TSDAE page."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['job_text_embeddings']
            job_ids_clust = st.session_state.get('job_text_embedding_job_ids')
            src_name_clust = "Standard BERT Job Embeddings"
            if not job_ids_clust: st.error("Standard BERT Job IDs missing. Please run BERT Model embedding generation."); return
        else: st.warning("Standard BERT job embeddings unavailable. Please generate them on the BERT Model page."); return
    
    st.info(f"Using: {src_name_clust} ({len(job_ids_clust)} items for potential clustering)")
    
    if emb_to_cluster is not None and job_ids_clust:
        max_k_val = emb_to_cluster.shape[0]
        if max_k_val < 2: st.error("Need at least 2 embedded items to cluster."); return
        
        num_clusters_input = st.slider("Number of Clusters (K)", 2, min(50, max_k_val), min(N_CLUSTERS, max_k_val), key="k_slider_cluster_main")
        
        if st.button(f"Run K-Means (K={num_clusters_input}) on {src_name_clust}", key="run_kmeans_button_main"):
            cluster_labels = cluster_embeddings_with_progress(emb_to_cluster, num_clusters_input)
            if cluster_labels is not None:
                if len(job_ids_clust) == len(cluster_labels):
                    cluster_info_df = pd.DataFrame({'Job.ID': job_ids_clust, 'cluster_temp': cluster_labels})
                    data_df_with_clusters = st.session_state['data'].copy()
                    if 'cluster' in data_df_with_clusters.columns: 
                        data_df_with_clusters = data_df_with_clusters.drop(columns=['cluster'])
                    st.session_state['data'] = pd.merge(data_df_with_clusters, cluster_info_df, on='Job.ID', how='left')
                    if 'cluster_temp' in st.session_state['data'].columns:
                        st.session_state['data'].rename(columns={'cluster_temp': 'cluster'}, inplace=True)
                    st.success(f"'cluster' column updated in the main dataset for {len(job_ids_clust)} jobs.")
                else:
                    st.error("Mismatch between number of Job IDs and generated cluster labels. Cannot merge.")
            else:
                st.error("Clustering algorithm failed to return labels.")

    if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns and \
       emb_to_cluster is not None and job_ids_clust is not None and \
       len(job_ids_clust) == emb_to_cluster.shape[0]: 

        st.subheader("2D Visualization of Clustered Embeddings (PCA)")
        if emb_to_cluster.shape[0] >= 2: 
            try:
                pca_cluster = PCA(n_components=2)
                reduced_embeddings_for_plot = pca_cluster.fit_transform(emb_to_cluster)
                plot_df_cluster = pd.DataFrame(reduced_embeddings_for_plot, columns=['PC1', 'PC2'])
                plot_df_cluster['Job.ID'] = job_ids_clust 
                
                data_for_plot_merge = st.session_state['data'][st.session_state['data']['Job.ID'].isin(job_ids_clust)].copy()
                cols_for_merge = ['Job.ID', 'Title', 'cluster']
                text_col_for_hover = 'combined_jobs' if 'combined_jobs' in data_for_plot_merge.columns else 'Job.Description'
                if text_col_for_hover not in cols_for_merge and text_col_for_hover in data_for_plot_merge.columns:
                    cols_for_merge.append(text_col_for_hover)
                
                data_for_plot_merge = data_for_plot_merge[cols_for_merge]
                plot_df_cluster = pd.merge(plot_df_cluster, data_for_plot_merge, on='Job.ID', how='left')
                
                if 'cluster' in plot_df_cluster.columns:
                    plot_df_cluster['cluster'] = plot_df_cluster['cluster'].astype('category') 
                    hover_data_plot = {'Job.ID': True, 'cluster': True, 'PC1': False, 'PC2': False}
                    if text_col_for_hover in plot_df_cluster.columns:
                         hover_data_plot[text_col_for_hover] = True

                    if not plot_df_cluster.empty and 'Title' in plot_df_cluster.columns and 'cluster' in plot_df_cluster.columns:
                        fig_cluster_pca = px.scatter(
                            plot_df_cluster, x='PC1', y='PC2', color='cluster',
                            hover_name='Title', hover_data=hover_data_plot,
                            title=f'2D PCA of Clustered {src_name_clust}'
                        )
                        st.plotly_chart(fig_cluster_pca, use_container_width=True)
                    else:
                        st.warning("Could not generate cluster visualization. Required data missing.")
                else:
                    st.warning("Cluster information not found for plot DataFrame.")
            except Exception as e_pca_plot:
                st.error(f"Error during PCA visualization of clusters: {e_pca_plot}")
        else:
            st.warning("Not enough data points for PCA visualization.")
            
    elif 'cluster' in st.session_state.get('data', pd.DataFrame()).columns: 
        st.info("Cluster information is present. To re-visualize, run clustering again.")
    else:
        st.info("No cluster information to visualize. Run clustering first.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5).")
    uploaded_cv_files = st.file_uploader("Choose CV files:", type=["pdf","docx"], accept_multiple_files=True, key="cv_upload_widget_main")
    if uploaded_cv_files:
        if len(uploaded_cv_files) > 5:
            st.warning("Max 5 CVs. Processing first 5.")
            uploaded_cv_files = uploaded_cv_files[:5]
        if st.button("Process Uploaded CVs", key="proc_cv_btn_main"):
            cv_data_batch = []
            bert_model_for_cv = load_bert_model()
            if not bert_model_for_cv: 
                st.error("BERT model load failed for CVs."); return 
            with st.spinner("Processing CVs..."):
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
                        if cv_e is not None: st.success(f"Processed & embedded: {cv_file.name}")
                        else: st.warning(f"Failed to process/embed: {cv_file.name}")
                    except Exception as e:
                        st.error(f"Error with {cv_file.name}: {e}")
                st.session_state['uploaded_cvs_data'] = cv_data_batch
                st.success(f"CV batch processing done.")

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs:")
        for i, cv_d in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_d.get('filename', 'N/A')}"):
                st.text_area(f"Original:", cv_d.get('original_text',''), height=70, disabled=True, key=f"disp_cv_o_{i}")
                st.text_area(f"Processed:", cv_d.get('processed_text',''), height=70, disabled=True, key=f"disp_cv_p_{i}")
                st.success("Embedding OK.") if cv_d.get('embedding') is not None and cv_d.get('embedding').size > 0 else st.warning("Embedding missing.")
    return


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations for uploaded CVs based on 'processed_text' of jobs.") 
    if not st.session_state.get('uploaded_cvs_data'): 
        st.warning("Upload & process CVs first."); return
    main_data = st.session_state.get('data')
    if main_data is None or 'processed_text' not in main_data.columns:
        st.error("Job data with 'processed_text' (from 'combined_jobs') not available. Load & preprocess first."); return
    
    job_emb_for_rec, emb_src_msg_rec, job_emb_ids_for_rec = None, "", None 
    rec_choice = st.radio("Job Embeddings for Recs:", ("Standard BERT", "TSDAE"), key="rec_emb_c", horizontal=True)

    if rec_choice == "TSDAE":
        if st.session_state.get('tsdae_embeddings', np.array([])).size > 0:
            job_emb_for_rec = st.session_state['tsdae_embeddings']
            job_emb_ids_for_rec = st.session_state.get('tsdae_embedding_job_ids')
            if not job_emb_ids_for_rec or len(job_emb_ids_for_rec) != job_emb_for_rec.shape[0]: 
                st.error("TSDAE embeddings/Job ID list mismatch or missing."); return
            emb_src_msg_rec = "Using TSDAE embeddings."
        else: 
            st.warning("TSDAE embeddings unavailable."); return
    else: 
        if st.session_state.get('job_text_embeddings', np.array([])).size > 0:
            job_emb_for_rec = st.session_state['job_text_embeddings']
            job_emb_ids_for_rec = st.session_state.get('job_text_embedding_job_ids')
            if not job_emb_ids_for_rec or len(job_emb_ids_for_rec) != job_emb_for_rec.shape[0]: 
                st.error("Standard BERT embeddings/Job ID list mismatch."); return
            emb_src_msg_rec = "Using Standard BERT job embeddings."
        else: 
            st.warning("Standard BERT job embeddings unavailable."); return
    st.info(emb_src_msg_rec)

    if not job_emb_ids_for_rec:
        st.error("Job IDs for selected embeddings are missing."); return
        
    temp_df_for_align = pd.DataFrame({'Job.ID': job_emb_ids_for_rec, 'emb_order': np.arange(len(job_emb_ids_for_rec))})
    
    cols_to_fetch_for_rec = list(set(['Job.ID', 'Title'] + JOB_DETAIL_FEATURES_TO_DISPLAY + ['combined_jobs'])) 
    if 'cluster' in main_data.columns: cols_to_fetch_for_rec.append('cluster')
    
    cols_to_fetch_for_rec = [col for col in cols_to_fetch_for_rec if col in main_data.columns]
    if 'Job.ID' not in cols_to_fetch_for_rec : cols_to_fetch_for_rec.insert(0,'Job.ID')

    main_data_subset_for_rec = main_data[cols_to_fetch_for_rec].drop_duplicates(subset=['Job.ID'], keep='first')
    jobs_for_sim_df = pd.merge(temp_df_for_align, main_data_subset_for_rec, on='Job.ID', how='left').sort_values('emb_order').reset_index(drop=True)

    if len(jobs_for_sim_df) != len(job_emb_for_rec):
        st.error(f"Alignment error: `jobs_for_sim_df` ({len(jobs_for_sim_df)}) != embeddings ({len(job_emb_for_rec)})."); return

    default_details_to_show_rec_page = ['Company', 'City', 'Position'] 
    available_options_for_rec_display = [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col in jobs_for_sim_df.columns]
    if 'Job.Description' not in available_options_for_rec_display and 'Job.Description' in jobs_for_sim_df.columns: # Ensure Job.Description can be selected
        available_options_for_rec_display.append('Job.Description')
    default_details_filtered_rec_page = [col for col in default_details_to_show_rec_page if col in available_options_for_rec_display]
    
    selected_details_for_rec_display = st.multiselect(
        "Select additional job details to display in the recommendations table:",
        options=sorted(list(set(available_options_for_rec_display))), 
        default=default_details_filtered_rec_page,
        key="job_rec_detail_multiselect_page"
    )

    if st.button("Generate Recommendations", key="gen_recs_b_main"):
        st.session_state['all_recommendations_for_annotation'] = {} 
        with st.spinner("Generating recommendations..."):
            valid_cvs_rec = [cv for cv in st.session_state.get('uploaded_cvs_data', []) if cv.get('embedding') is not None and cv.get('embedding').size > 0]
            if not valid_cvs_rec: st.warning("No CVs with valid embeddings found."); return

            for cv_data_rec in valid_cvs_rec:
                cv_file_n = cv_data_rec.get('filename', 'Unknown CV')
                cv_embed = cv_data_rec['embedding']
                st.subheader(f"Recommendations for {cv_file_n}")
                cv_embed_2d = cv_embed.reshape(1, -1) if cv_embed.ndim == 1 else cv_embed
                
                if job_emb_for_rec.ndim == 1 or job_emb_for_rec.shape[0] == 0: 
                    st.error(f"Selected job embeddings invalid for {cv_file_n}."); continue 
                
                similarities_rec = cosine_similarity(cv_embed_2d, job_emb_for_rec)[0]
                temp_df_rec_with_sim = jobs_for_sim_df.copy() 
                temp_df_rec_with_sim['similarity_score'] = similarities_rec
                recommended_j_df = temp_df_rec_with_sim.sort_values(by='similarity_score', ascending=False).head(20)
                
                if not recommended_j_df.empty:
                    display_cols_on_this_page = ['Job.ID', 'Title', 'similarity_score'] + selected_details_for_rec_display
                    display_cols_on_this_page = sorted(list(set(col for col in display_cols_on_this_page if col in recommended_j_df.columns)))
                    if 'Title' in display_cols_on_this_page: display_cols_on_this_page.remove('Title'); display_cols_on_this_page.insert(0, 'Title')
                    if 'Job.ID' in display_cols_on_this_page: display_cols_on_this_page.remove('Job.ID'); display_cols_on_this_page.insert(0, 'Job.ID')
                    
                    st.dataframe(recommended_j_df[display_cols_on_this_page], use_container_width=True)
                    st.session_state['all_recommendations_for_annotation'][cv_file_n] = recommended_j_df 
                else: 
                    st.info(f"No recommendations for {cv_file_n}.")
                st.write("---") 
        st.success("Recommendation generation complete!")
    return

def annotation_page():
    st.header("Annotation of Job Recommendations")
    st.write("Annotate relevance and provide feedback for recommended jobs.")

    if not st.session_state.get('all_recommendations_for_annotation'):
        st.warning("Generate recommendations first on the 'Job Recommendation' page."); return
    
    if 'annotator_details' not in st.session_state: 
        st.session_state.annotator_details = {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS}
    if 'current_annotator_slot_for_input' not in st.session_state:
         st.session_state.current_annotator_slot_for_input = ANNOTATORS[0] if ANNOTATORS else None
    if 'annotators_saved_status' not in st.session_state:
        st.session_state.annotators_saved_status = set()

    st.subheader("🧑‍💻 Annotator Profile & Selection")
    if ANNOTATORS:
        selected_slot = st.selectbox(
            "Select Your Annotator Slot to Enter/Edit Details and Annotations:",
            options=ANNOTATORS,
            index=ANNOTATORS.index(st.session_state.current_annotator_slot_for_input) if st.session_state.current_annotator_slot_for_input in ANNOTATORS else 0,
            key="annotator_slot_selector_main"
        )
        st.session_state.current_annotator_slot_for_input = selected_slot

        if selected_slot:
            with st.expander(f"Edit Profile for {selected_slot}", expanded=True):
                name_val = st.session_state.annotator_details.get(selected_slot, {}).get('actual_name', '')
                bg_val = st.session_state.annotator_details.get(selected_slot, {}).get('profile_background', '')
                actual_name = st.text_input(f"Your Name", value=name_val, key=f"actual_name_input_{selected_slot}_main")
                profile_bg = st.text_area(f"Your Profile Background", value=bg_val, key=f"profile_bg_input_{selected_slot}_main", height=100 )
                st.session_state.annotator_details[selected_slot]['actual_name'] = actual_name
                st.session_state.annotator_details[selected_slot]['profile_background'] = profile_bg
    else:
        st.warning("No annotator slots defined."); return
    
    st.markdown("---")
    current_annotator_display_name = st.session_state.annotator_details.get(st.session_state.current_annotator_slot_for_input, {}).get('actual_name', st.session_state.current_annotator_slot_for_input)
    st.subheader(f"📝 Annotate Recommendations as {st.session_state.current_annotator_slot_for_input} ({current_annotator_display_name})")
    
    if 'collected_annotations' not in st.session_state or st.session_state.collected_annotations.empty:
        base_records_init = []
        if st.session_state.all_recommendations_for_annotation:
            for cv_fn_init, rec_df_init in st.session_state.all_recommendations_for_annotation.items():
                rec_df_unique_init = rec_df_init.drop_duplicates(subset=['Job.ID'], keep='first')
                for _, rec_row_init in rec_df_unique_init.iterrows():
                    record_init = {
                        'cv_filename': cv_fn_init, 'job_id': str(rec_row_init['Job.ID']),
                        'job_title': rec_row_init.get('Title', 'N/A'), 
                        'similarity_score': rec_row_init['similarity_score'], 
                        'cluster': rec_row_init.get('cluster', pd.NA)
                    }
                    for detail_col in JOB_DETAIL_FEATURES_TO_DISPLAY + ['combined_jobs']: 
                        record_init[detail_col] = rec_row_init.get(detail_col, '') 

                    for i_ann, slot_name_ann in enumerate(ANNOTATORS):
                        record_init[f'annotator_{i_ann+1}_slot'] = slot_name_ann
                        record_init[f'annotator_{i_ann+1}_actual_name'] = ""
                        record_init[f'annotator_{i_ann+1}_profile_background'] = ""
                        record_init[f'annotator_{i_ann+1}_relevance'] = pd.NA 
                        record_init[f'annotator_{i_ann+1}_feedback'] = ""
                    base_records_init.append(record_init)
            if base_records_init:
                st.session_state.collected_annotations = pd.DataFrame(base_records_init)
            else: 
                st.session_state.collected_annotations = pd.DataFrame() 

    relevance_options_map = {
        0: "0 (Very Irrelevant)", 1: "1 (Slightly Relevant)",
        2: "2 (Relevant)",       3: "3 (Most Relevant)"
    }
    
    current_annotator_slot = st.session_state.current_annotator_slot_for_input
    annotator_idx_for_cols = ANNOTATORS.index(current_annotator_slot) 

    with st.form(key=f"annotation_form_{current_annotator_slot}_main"):
        form_input_for_current_annotator = [] 
        expand_cv_default = len(st.session_state['all_recommendations_for_annotation']) == 1

        # Determine available detail columns for the multiselect options
        all_possible_detail_cols = list(set(JOB_DETAIL_FEATURES_TO_DISPLAY + ['Job.Description'])) # Ensure Job.Description is an option
        available_details_for_ann_display = []
        if st.session_state['all_recommendations_for_annotation']:
            first_cv_key_ann = list(st.session_state['all_recommendations_for_annotation'].keys())[0]
            if first_cv_key_ann in st.session_state['all_recommendations_for_annotation']:
                first_rec_df_ann = st.session_state['all_recommendations_for_annotation'][first_cv_key_ann]
                available_details_for_ann_display = [col for col in all_possible_detail_cols if col in first_rec_df_ann.columns]
        
        default_details_for_ann_display = [col for col in ['Company', 'Job.Description', 'Employment.Type', 'Position'] if col in available_details_for_ann_display]
        
        # Place multiselect outside the CV loop, but inside the form if its selection should be part of the submission
        # Or, if it's just for display control, it can be outside the form. Let's keep it inside for now to control view per annotator session.
        selected_details_for_annotation_display = st.multiselect(
            "Select job details to view during annotation:",
            options=sorted(list(set(available_details_for_ann_display))), 
            default=default_details_for_ann_display,
            key=f"annotation_detail_multiselect_widget_{current_annotator_slot}"
        )

        for cv_filename, recommendations_df_original in st.session_state['all_recommendations_for_annotation'].items():
            recommendations_df_unique = recommendations_df_original.drop_duplicates(subset=['Job.ID'], keep='first')
            
            with st.expander(f"Recommendations for CV: **{cv_filename}**", expanded=expand_cv_default):
                for _, job_row_ann in recommendations_df_unique.iterrows(): 
                    job_id_str_ann = str(job_row_ann['Job.ID']) 
                    st.markdown(f"**Job ID:** {job_id_str_ann} | **Title:** {job_row_ann.get('Title', 'N/A')}")
                    
                    # Display selected details
                    for detail_key in selected_details_for_annotation_display: 
                        if detail_key in job_row_ann and pd.notna(job_row_ann[detail_key]):
                            detail_value = job_row_ann[detail_key]
                            display_label = detail_key.replace('.', ' ').replace('_', ' ').title() 
                            if detail_key == "Job.Description" and isinstance(detail_value, str) and len(detail_value) > 150: 
                                st.caption(f"*{display_label}:* {detail_value[:150]}...")
                            else:
                                st.caption(f"*{display_label}:* {detail_value}")
                    
                    st.caption(f"*Similarity Score:* {job_row_ann['similarity_score']:.4f} | *Cluster:* {job_row_ann.get('cluster', 'N/A')}")
                    st.markdown("---") 
                    
                    relevance_key_ann = f"relevance_{cv_filename}_{job_id_str_ann}_{current_annotator_slot}"
                    feedback_key_ann = f"feedback_{cv_filename}_{job_id_str_ann}_{current_annotator_slot}"
                    
                    default_relevance = 0; default_feedback = ""
                    if not st.session_state.collected_annotations.empty:
                        mask = (st.session_state.collected_annotations['cv_filename'] == cv_filename) & \
                               (st.session_state.collected_annotations['job_id'] == job_id_str_ann)
                        existing_row = st.session_state.collected_annotations[mask]
                        if not existing_row.empty:
                            rel_col = f'annotator_{annotator_idx_for_cols+1}_relevance'
                            fb_col = f'annotator_{annotator_idx_for_cols+1}_feedback'
                            if rel_col in existing_row.columns:
                                val = existing_row.iloc[0].get(rel_col)
                                if pd.notna(val): default_relevance = int(val)
                            if fb_col in existing_row.columns:
                                default_feedback = str(existing_row.iloc[0].get(fb_col, ""))
                    
                    relevance_val_selected = st.radio("Relevance:", options=list(relevance_options_map.keys()), 
                                               index=default_relevance if default_relevance in relevance_options_map else 0, 
                                               key=relevance_key_ann, horizontal=True, format_func=lambda x: relevance_options_map[x])
                    feedback_val_input = st.text_area("Feedback:", value=default_feedback, key=feedback_key_ann, height=75)
                    
                    form_input_for_current_annotator.append({
                        'cv_filename': cv_filename, 'job_id': job_id_str_ann,
                        'relevance': relevance_val_selected, 'feedback': feedback_val_input
                    })
                    st.markdown("---")
        
        submitted_current_annotator = st.form_submit_button(f"Save/Update My Ratings ({current_annotator_display_name})")

    if submitted_current_annotator:
        profile_details_current = st.session_state.annotator_details.get(current_annotator_slot, {})
        actual_name_current = profile_details_current.get('actual_name', current_annotator_slot)
        profile_bg_current = profile_details_current.get('profile_background', '')
        updated_items_count = 0

        for item_ann in form_input_for_current_annotator:
            mask_update = (st.session_state.collected_annotations['cv_filename'] == item_ann['cv_filename']) & \
                          (st.session_state.collected_annotations['job_id'] == item_ann['job_id'])
            row_indices_to_update = st.session_state.collected_annotations[mask_update].index
            
            if not row_indices_to_update.empty:
                idx = row_indices_to_update[0]
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_slot'] = current_annotator_slot
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_actual_name'] = actual_name_current
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_profile_background'] = profile_bg_current
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_relevance'] = item_ann['relevance']
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_feedback'] = item_ann['feedback']
                updated_items_count +=1
            else:
                st.warning(f"Base record for CV {item_ann['cv_filename']}, Job {item_ann['job_id']} not found. Annotation not saved.")
        
        st.success(f"Annotations by {current_annotator_display_name} saved/updated for {updated_items_count} items.")
        st.session_state.annotators_saved_status.add(current_annotator_slot)

    st.markdown("---")
    st.subheader("Final Actions")
    completed_count = len(st.session_state.get('annotators_saved_status', set()))
    total_ann = len(ANNOTATORS) if ANNOTATORS else 0

    if total_ann > 0 and completed_count == total_ann:
        st.success(f"All {total_ann} annotator slots have saved their ratings! You can now download the combined data.")
        download_disabled_status = False
    else:
        st.info(f"Waiting for all annotator slots to save ratings. Completed: {completed_count}/{total_ann}.")
        st.write("Completed slots:", ", ".join(sorted(list(st.session_state.annotators_saved_status))) or "None")
        download_disabled_status = True

    if not st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        csv_export = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8') 
        st.download_button(
            label="Download All Collected Annotations as CSV", data=csv_export,
            file_name="all_job_recommendation_annotations.csv", mime="text/csv",
            key="download_all_annotations_final_main_btn", disabled=download_disabled_status
        )
        with st.expander("Show Current Collected Annotations Data (All Annotators)", expanded=False):
            st.dataframe(st.session_state['collected_annotations'], height=300)
    else:
        st.info("No annotations collected yet.")
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
    st.write("Evaluates top 20 recommendations based on human annotations.")
    
    all_recommendations = st.session_state.get('all_recommendations_for_annotation', {})
    
    st.sidebar.subheader("Evaluation Data Source")
    annotation_source = st.sidebar.radio(
        "Use annotations from:",
        ("Current Session", "Uploaded CSV File"),
        key="eval_annotation_source_selector"
    )

    anns_df = None
    if annotation_source == "Uploaded CSV File":
        uploaded_ann_file = st.sidebar.file_uploader("Upload Annotation CSV", type=['csv'], key="eval_ann_uploader")
        if uploaded_ann_file is not None:
            try:
                anns_df = pd.read_csv(uploaded_ann_file)
                required_ann_cols = ['cv_filename', 'job_id'] + [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS))]
                if not all(col in anns_df.columns for col in required_ann_cols):
                    st.error(f"Uploaded CSV is missing some required columns. Expected at least: {', '.join(required_ann_cols)}")
                    anns_df = None 
                else:
                    st.sidebar.success(f"Using uploaded annotation file: {uploaded_ann_file.name}")
                    anns_df['job_id'] = anns_df['job_id'].astype(str)
                    st.session_state['uploaded_annotation_data'] = anns_df 

            except Exception as e:
                st.error(f"Error reading or processing uploaded annotation CSV: {e}")
                anns_df = None
        elif st.session_state.get('uploaded_annotation_data') is not None: 
            anns_df = st.session_state.uploaded_annotation_data
            st.sidebar.info("Using previously uploaded annotation data.")
        else:
            st.sidebar.warning("No annotation CSV uploaded. Please upload or switch to 'Current Session'.")
    
    if anns_df is None: 
        anns_df = st.session_state.get('collected_annotations', pd.DataFrame())
        if annotation_source == "Uploaded CSV File": 
             st.sidebar.info("Falling back to annotations from current session.")
        else:
             st.sidebar.info("Using annotations collected in the current session.")


    if not all_recommendations: st.warning("No recommendations to evaluate. Run 'Job Recommendation' first."); return
    if anns_df.empty: st.warning("No annotation data available. Annotate or upload first."); return

    st.subheader("Evaluation Parameters")
    st.info("The 'Binary Relevance Threshold' converts average graded annotator scores (0-3) into binary 'relevant' (1) or 'not relevant' (0) for calculating P@20, MAP@20, MRR@20, HR@20, and Binary NDCG@20.")
    relevance_threshold_binary = st.slider("Binary Relevance Threshold", 0.0, 3.0, 1.5, 0.1, key="eval_thresh_binary_final")
    
    if st.button("Run Evaluation on Top 20 Recommendations", key="run_eval_final_btn"):
        with st.spinner("Calculating human-grounded evaluation metrics..."):
            
            per_cv_metrics_list = [] 

            relevance_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS)) if f'annotator_{i+1}_relevance' in anns_df.columns]
            if not relevance_cols: st.error("No annotator relevance columns in the annotation data."); return

            num_cvs_evaluated = 0
            for cv_filename, recommended_jobs_df in all_recommendations.items():
                if recommended_jobs_df.empty: continue
                
                recommended_jobs_df['Job.ID'] = recommended_jobs_df['Job.ID'].astype(str)
                cv_anns_subset = anns_df[anns_df['cv_filename'] == cv_filename].copy()
                if cv_anns_subset.empty: continue 
                
                num_cvs_evaluated +=1
                cv_anns_subset['job_id'] = cv_anns_subset['job_id'].astype(str)
                
                top_20_recs_df = recommended_jobs_df.head(20)
                ranked_job_ids_list = top_20_recs_df['Job.ID'].tolist()
                model_similarity_scores = top_20_recs_df['similarity_score'].tolist()

                binary_relevance_scores, graded_relevance_scores = [], []
                
                for job_id in ranked_job_ids_list:
                    job_specific_annotations = cv_anns_subset[cv_anns_subset['job_id'] == job_id]
                    avg_annotator_score = 0.0 
                    if not job_specific_annotations.empty:
                        annotator_scores_for_job = []
                        for rel_col_name in relevance_cols:
                            if rel_col_name in job_specific_annotations.columns:
                                annotator_scores_for_job.extend(pd.to_numeric(job_specific_annotations[rel_col_name], errors='coerce').dropna().tolist())
                        if annotator_scores_for_job: avg_annotator_score = np.mean(annotator_scores_for_job)
                    
                    graded_relevance_scores.append(avg_annotator_score)
                    binary_relevance_scores.append(1 if avg_annotator_score >= relevance_threshold_binary else 0)
                
                k_cutoff = 20 
                cv_p_at_20 = sum(binary_relevance_scores) / len(binary_relevance_scores) if binary_relevance_scores else 0.0
                cv_hr_at_20 = 1 if any(binary_relevance_scores) else 0
                cv_map_at_20 = _calculate_average_precision(binary_relevance_scores, k_cutoff)
                
                cv_mrr_at_20 = 0.0
                for r, is_rel in enumerate(binary_relevance_scores): 
                    if is_rel: cv_mrr_at_20 = 1.0 / (r + 1); break
                
                actual_k = len(binary_relevance_scores)
                cv_binary_ndcg_at_20 = ndcg_score([binary_relevance_scores], [model_similarity_scores[:actual_k]], k=actual_k) if actual_k == len(model_similarity_scores[:actual_k]) and actual_k > 0 else 0.0
                cv_graded_ndcg_at_20 = ndcg_score([graded_relevance_scores], [model_similarity_scores[:actual_k]], k=actual_k) if actual_k == len(graded_relevance_scores) and actual_k == len(model_similarity_scores[:actual_k]) and actual_k > 0 else 0.0

                per_cv_metrics_list.append({
                    'CV Filename': cv_filename,
                    'P@20': cv_p_at_20, 'MAP@20': cv_map_at_20, 'MRR@20': cv_mrr_at_20, 'HR@20': cv_hr_at_20,
                    'NDCG@20 (Binary)': cv_binary_ndcg_at_20, 'NDCG@20 (Graded)': cv_graded_ndcg_at_20
                })

            st.subheader("Per-CV Evaluation Metrics")
            if per_cv_metrics_list:
                per_cv_df = pd.DataFrame(per_cv_metrics_list)
                per_cv_df_display = per_cv_df.copy()
                for col in ['P@20', 'MAP@20', 'HR@20']:
                    if col in per_cv_df_display.columns:
                         per_cv_df_display[col] = (per_cv_df_display[col] * 100).round(2).astype(str) + '%'
                for col in ['MRR@20', 'NDCG@20 (Binary)', 'NDCG@20 (Graded)']:
                     if col in per_cv_df_display.columns:
                        per_cv_df_display[col] = per_cv_df_display[col].round(4)
                st.dataframe(per_cv_df_display.set_index('CV Filename'))
                
                avg_metrics_dict = {
                    'P@20': per_cv_df['P@20'].mean(),
                    'MAP@20': per_cv_df['MAP@20'].mean(),
                    'MRR@20': per_cv_df['MRR@20'].mean(),
                    'HR@20': per_cv_df['HR@20'].mean(),
                    'NDCG@20 (Binary)': per_cv_df['NDCG@20 (Binary)'].mean(),
                    'NDCG@20 (Graded)': per_cv_df['NDCG@20 (Graded)'].mean()
                }
            else:
                st.info("No CVs were evaluated (perhaps no recommendations or no matching annotations).")
                avg_metrics_dict = {key: 'N/A' for key in ['P@20', 'MAP@20', 'MRR@20', 'HR@20', 'NDCG@20 (Binary)', 'NDCG@20 (Graded)']}

            st.subheader("Average Human-Grounded Evaluation Metrics Summary")
            if num_cvs_evaluated > 0: st.write(f"Calculated based on {num_cvs_evaluated} CVs.")
            else: st.warning("No CVs with annotations found to calculate metrics."); return 

            metric_config = {
                'Precision@20': {'fmt': "{:.2%}", 'help': "Avg P@20. Proportion of top 20 relevant items (binary).", 'color': "off"},
                'MAP@20': {'fmt': "{:.2%}", 'help': "Mean Avg. Precision@20 (binary relevance, considers order).", 'color': "off"},
                'MRR@20': {'fmt': "{:.4f}", 'help': "Mean Reciprocal Rank@20 (rank of first relevant item, binary).", 'color': "normal"},
                'HR@20': {'fmt': "{:.2%}", 'help': "Hit Ratio@20: Proportion of CVs with at least one relevant item in top 20.", 'color': "normal"},
                'NDCG@20 (Binary)': {'fmt': "{:.4f}", 'help': "Avg NDCG@20 using binary relevance from threshold.", 'color': "inverse"},
                'NDCG@20 (Graded)': {'fmt': "{:.4f}", 'help': "Avg NDCG@20 using average annotator scores as graded relevance.", 'color': "inverse"}
            }
            keys_to_display = ['Precision@20', 'MAP@20', 'MRR@20', 'HR@20', 'NDCG@20 (Binary)', 'NDCG@20 (Graded)']
            
            metric_cols_r1 = st.columns(3)
            metric_cols_r2 = st.columns(3)
            
            for i, key in enumerate(keys_to_display):
                if key in avg_metrics_dict: 
                    value = avg_metrics_dict[key]
                    cfg = metric_config[key]
                    current_col = metric_cols_r1[i] if i < 3 else metric_cols_r2[i-3]
                    
                    val_str = "N/A"
                    if isinstance(value, (int, float, np.number)) and not (isinstance(value, float) and np.isnan(value)):
                        val_str = cfg['fmt'].format(value * 100 if '%' in cfg['fmt'] else value)
                    elif isinstance(value, str): val_str = value
                    
                    current_col.metric(label=f"Average {key}", value=val_str, delta_color=cfg['color'], help=cfg['help'])
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
