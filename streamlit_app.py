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
# from tqdm import tqdm # tqdm is for terminal, not needed for Streamlit progress
from sentence_transformers import SentenceTransformer, InputExample, losses 
from sentence_transformers.datasets import DenoisingAutoEncoderDataset 
from torch.utils.data import DataLoader 
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
        nltk.download('stopwords', quiet=True)
    try:
        word_tokenize("example text") # This implicitly checks if 'punkt' is available and usable
    except LookupError:
        st.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    # REMOVED the explicit check for 'punkt_tab.pickle' as it's too specific
    # and can cause issues if NLTK's internal structure changes.
    # The nltk.download('punkt') above should be sufficient.
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'

FEATURES_TO_COMBINE = [ 
    'Status', 'Title', 'Position', 'Company', 
    'City', 'State.Name', 'Industry', 'Job.Description', 
    'Employment.Type', 'Education.Required'
]
JOB_DETAIL_FEATURES = [ 
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type', 
    'Position', 'Industry', 'Education.Required', 'State.Name'
] 

N_CLUSTERS = 20 
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
default_values = {
    'data': None,
    'job_text_embeddings': None,
    'job_text_embedding_job_ids': None,
    'uploaded_cvs_data': [],
    'all_recommendations_for_annotation': {},
    'collected_annotations': pd.DataFrame(),
    'annotator_details': {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS},
    'current_annotator_slot_for_input': ANNOTATORS[0] if ANNOTATORS else None,
    'annotators_saved_status': set(),
    'model_trained_flags': {"tsdae_trained_this_session": False, "cv_job_finetuned_this_session": False},
    'bert_model_instance': None
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


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
        
        cols_to_load_set = set(['Job.ID', 'Title']) 
        cols_to_load_set.update(existing_features_to_combine)
        cols_to_load_set.update(detail_features_to_ensure) 
        
        actual_cols_to_load = [col for col in list(cols_to_load_set) if col in df_full.columns]
        df = df_full[actual_cols_to_load].copy() 

        for feature in existing_features_to_combine: 
            if feature in df.columns: 
                df[feature] = df[feature].fillna('').astype(str)
        
        df['combined_jobs'] = df[existing_features_to_combine].agg(lambda x: ' '.join(x.dropna()), axis=1)
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
    result_words_list = [] 
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0 and n > 0 : 
            idx_to_keep = np.random.choice(n) 
            keep_or_not[idx_to_keep] = True
        result_words_list = np.array(words)[keep_or_not].tolist() 
    elif method in ('b', 'c'): 
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'.")
        high_freq_indices = [i for i, w_inner in enumerate(words) if word_freq_dict.get(w_inner.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0 and num_to_remove <= len(high_freq_indices):
             to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        result_words_list = [w_inner for i, w_inner in enumerate(words) if i not in to_remove_indices]
        if not result_words_list and words: 
            result_words_list = [random.choice(words)]
        if method == 'c' and result_words_list: 
            random.shuffle(result_words_list)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result_words_list)


def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'):
    processed_results_intermediate = [] 
    if text_column_to_process not in data_df.columns:
        st.warning(f"Column '{text_column_to_process}' not found for preprocessing.")
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = pd.Series(dtype='object')
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = pd.Series(dtype='object')
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
def load_bert_model_once(model_name="all-MiniLM-L6-v2"): 
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

if st.session_state.bert_model_instance is None:
    st.session_state.bert_model_instance = load_bert_model_once()


@st.cache_data
def generate_embeddings_with_progress(_model_ref_for_cache_key, texts_list_to_embed, model_state_indicator="base"):
    model_to_use = st.session_state.bert_model_instance 
    if model_to_use is None:
        st.error("BERT model is not available for embedding generation.")
        return np.array([]) 
    if not texts_list_to_embed: 
        st.warning("Input text list for embedding is empty.")
        return np.array([])
    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts using current model state ({model_state_indicator})..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = [] 
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size] 
                batch_embeddings_np_array = model_to_use.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False) 
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
# (home_page, preprocessing_page, tsdae_page, fine_tuning_page, bert_model_page, clustering_page, upload_cv_page, job_recommendation_page, annotation_page, _calculate_average_precision, evaluation_page)
# Ensure all these functions are fully defined as in the last complete version.
# For brevity, only showing the structure and the main navigation.

def home_page():
    st.header("Home: Exploratory Data Analysis") 
    st.write("This page provides an overview of the job dataset and allows you to explore its features.") 
    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES)
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
        st.subheader('Search Word in Feature') 
        search_word = st.text_input("Enter word to search:", key="home_search_word_new_v2") 
        all_available_cols_for_search = ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE + JOB_DETAIL_FEATURES
        searchable_cols = sorted(list(set(col for col in all_available_cols_for_search if col in data_df.columns)))
        search_column = st.selectbox("Select feature to search in:", [''] + searchable_cols, key="home_search_column_new_v2") 
        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column)
                if not search_results.empty:
                    st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':") 
                    st.dataframe(search_results[display_search_cols].head(), use_container_width=True) 
                else: st.info(f"No entries found for '{search_word}' in '{search_column}'.") 
        st.subheader('Feature Information') 
        st.write('**Available Features (after processing):**', data_df.columns.tolist()) 
    else: st.error("Data could not be loaded.") 
    return

def preprocessing_page():
    st.header("Job Data Preprocessing") 
    st.write("This page performs preprocessing on the 'combined_jobs' column of the job dataset.") 
    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or 'combined_jobs' column not available. Please return to the 'Home' page to load data first.") 
        return
    data_df_to_preprocess = st.session_state['data']
    st.info("The 'combined_jobs' column will be processed to create the 'processed_text' column.") 
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before processing)"): 
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())
    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_col_prep_btn_v2"): 
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
            if valid_intermediate_steps: st.dataframe(pd.DataFrame(valid_intermediate_steps).head(), use_container_width=True)
            else: st.warning("Intermediate preprocessing steps data is not in the expected format or is empty.") 
        st.subheader("Final Preprocessed Text ('processed_text') (Preview)") 
        st.dataframe(display_data_processed[['Job.ID', 'combined_jobs', 'processed_text']].head(), use_container_width=True)
        search_word_in_processed = st.text_input("Search word in 'processed_text':", key="prep_job_proc_search_v2") 
        if search_word_in_processed:
            search_results_in_processed = display_data_processed[display_data_processed['processed_text'].astype(str).str.contains(search_word_in_processed, na=False, case=False)] 
            if not search_results_in_processed.empty:
                display_cols_search_proc = ['Job.ID', 'Title' if 'Title' in display_data_processed.columns else 'Job.ID', 'processed_text']
                st.dataframe(search_results_in_processed[display_cols_search_proc].head(), use_container_width=True)
            else: st.info(f"No results for '{search_word_in_processed}' in 'processed_text'.") 
    else: st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet. Click the button above.") 
    return

def tsdae_page(): # Placeholder
    st.header("TSDAE Fine-tuning on Job Descriptions")
    st.info("Implementation from previous version, with progress bars.")
    return

def fine_tuning_page(): # Placeholder
    st.header("Model Fine-tuning (CV-Job Matching)")
    st.info("Implementation from previous version.")
    return

def bert_model_page(): # Placeholder
    st.header("BERT Model & Embeddings (Job Descriptions)")
    st.info("Implementation from previous version.")
    return

def clustering_page(): # Placeholder
    st.header("Clustering Job Embeddings")
    st.info("Implementation from previous version.")
    return

def upload_cv_page(): # Placeholder
    st.header("Upload & Process CV(s)")
    st.info("Implementation from previous version.")
    return

def job_recommendation_page(): # Placeholder
    st.header("Job Recommendation")
    st.info("Implementation from previous version.")
    return
    
def annotation_page(): # Placeholder
    st.header("Annotation of Job Recommendations")
    st.info("Implementation from previous version.")
    return

def _calculate_average_precision(ranked_relevance_binary, k_val): # Placeholder
    if not ranked_relevance_binary: return 0.0
    ranked_relevance_binary = ranked_relevance_binary[:k_val] 
    relevant_hits, sum_precisions = 0, 0.0
    for i, is_relevant in enumerate(ranked_relevance_binary):
        if is_relevant:
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1)
    return sum_precisions / relevant_hits if relevant_hits > 0 else 0.0
    
def evaluation_page(): # Placeholder
    st.header("Model Evaluation")
    st.info("Implementation from previous version.")
    return


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", 
                "TSDAE Fine-tuning", 
                "BERT Model & Embeddings", 
                "CV-Job Fine-tuning", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio_v4") # Incremented key

if page == "Home":
    home_page()
elif page == "Preprocessing":
    preprocessing_page() 
elif page == "TSDAE Fine-tuning": 
    tsdae_page()
elif page == "BERT Model & Embeddings": 
    bert_model_page()
elif page == "CV-Job Fine-tuning": 
    fine_tuning_page()
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
