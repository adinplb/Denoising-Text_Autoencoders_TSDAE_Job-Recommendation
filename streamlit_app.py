import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os # Not strictly used in the final version but often useful
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
# from tqdm import tqdm # tqdm is for terminal progress, Streamlit has its own
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt


# --- NLTK Resource Downloads ---
@st.cache_resource
def download_nltk_resources():
    # Helper to download a resource if it's not found
    def download_if_missing(resource_name, download_name):
        try:
            nltk.data.find(resource_name)
        except LookupError:
            st.info(f"Downloading NLTK resource: {download_name}...")
            nltk.download(download_name)

    download_if_missing('corpora/stopwords', 'stopwords')
    download_if_missing('tokenizers/punkt', 'punkt')
    # punkt_tab is sometimes problematic and often not critical, so we can be more lenient
    # try:
    #     nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
    # except LookupError:
    #     st.info("Attempting to download NLTK punkt_tab resource (if available)...")
    #     try:
    #         nltk.download('punkt_tab')
    #     except Exception as e:
    #         st.warning(f"NLTK punkt_tab resource not found or download failed (usually not critical): {e}")
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
ONET_DATA_URL = 'https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv'


FEATURES_TO_COMBINE = [
    'Status', 'Title', 'Position', 'Company',
    'City', 'State.Name', 'Industry', 'Job.Description',
    'Employment.Type', 'Education.Required'
]
JOB_DETAIL_FEATURES_TO_DISPLAY = [ # Used for displaying job details in recommendations
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type',
    'Position', 'Industry', 'Education.Required', 'State.Name', 'Title'
]

N_CLUSTERS = 20 # Default number of clusters
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
# Initialize session state variables if they don't exist
default_session_state = {
    'data': None,
    'onet_data': None,
    'job_text_embeddings': None, # Standard BERT embeddings for 'processed_text'
    'job_text_embedding_job_ids': None, # Corresponding Job.IDs for job_text_embeddings
    'tsdae_embeddings': None, # TSDAE embeddings for 'final_noisy_text'
    'tsdae_embedding_job_ids': None, # Corresponding Job.IDs for tsdae_embeddings
    # 'sbert_job_embedding_onet_classified' will be a column in st.session_state['data']
    'uploaded_cvs_data': [], # List of dicts: {'filename': ..., 'original_text': ..., 'processed_text': ..., 'embedding': ...}
    'all_recommendations_for_annotation': {}, # Structure: {cv_filename: {annotator_slot: recommendations_df}}
    'collected_annotations': pd.DataFrame(), # Columns: CV_Filename, Annotator_Slot, Annotator_Name, Annotator_Background, Job_ID, Job_Title, Relevance_Score, Similarity_Score
    'annotator_details': {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS},
    'current_annotator_slot_for_input': ANNOTATORS[0] if ANNOTATORS else None, # Which annotator's details are being edited in sidebar
    'annotators_saved_status': set(), # Set of annotator_slots whose details have been saved
    'current_page': "Home" # For navigation
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading job data...')
def load_and_combine_data_from_url(url, features_to_combine_list, detail_features_to_ensure):
    try:
        df_full = pd.read_csv(url)
        # st.success('Successfully loaded data from URL!') # Success message can be verbose if called often

        if 'Job.ID' not in df_full.columns:
            st.error("Column 'Job.ID' not found in the dataset.")
            return None
        df_full['Job.ID'] = df_full['Job.ID'].astype(str) # Ensure Job.ID is string

        # Identify features present in the dataframe for combination
        existing_features_to_combine = [col for col in features_to_combine_list if col in df_full.columns]
        missing_features_for_combine = [col for col in features_to_combine_list if col not in df_full.columns]
        if missing_features_for_combine:
            st.warning(f"For 'combined_jobs', the following features were not found: {', '.join(missing_features_for_combine)}")

        # Determine all columns to load: Job.ID, Title, features for combining, and features for display
        cols_to_load_set = set(['Job.ID', 'Title']) # Ensure Title is loaded if present
        cols_to_load_set.update(existing_features_to_combine)
        cols_to_load_set.update(col for col in detail_features_to_ensure if col in df_full.columns) # Add display features if they exist
        
        actual_cols_to_load = [col for col in list(cols_to_load_set) if col in df_full.columns]
        df = df_full[actual_cols_to_load].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Fill NaN and convert to string for combination features
        for feature in existing_features_to_combine:
            if feature in df.columns: # Double check column exists
                df[feature] = df[feature].fillna('').astype(str)
        
        # Create 'combined_jobs'
        if existing_features_to_combine: # Only if there are features to combine
            df['combined_jobs'] = df[existing_features_to_combine].agg(' '.join, axis=1)
            df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
            st.success("Column 'combined_jobs' created from available features.")
        else:
            st.warning("No features found to create 'combined_jobs' column.")
            df['combined_jobs'] = "" # Create an empty column if no features
        
        return df
    except Exception as e:
        st.error(f'Error loading or combining data: {e}')
        return None

@st.cache_data(show_spinner='Loading O*NET data...')
def load_simple_csv_from_url(url, data_name="Data"):
    try:
        df = pd.read_csv(url)
        # st.success(f"Successfully loaded {data_name} from URL!")
        return df
    except Exception as e:
        st.error(f"Error loading {data_name} from {url}: {e}")
        return None

def extract_text_from_pdf(uploaded_file):
    try:
        # pdfminer.high_level.extract_text expects a file path or a binary file object
        # Streamlit's UploadedFile can be passed directly if it behaves like a binary file object
        text = pdf_extract_text(uploaded_file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF '{uploaded_file.name}': {e}")
        return None

def extract_text_from_docx(uploaded_file):
    try:
        document = DocxDocument(uploaded_file)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX '{uploaded_file.name}': {e}")
        return None

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    word_tokens = word_tokenize(text)
    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words]
    if not filtered_words:
        return "" # Return empty if all words are stopwords or non-alphanumeric
    # Stemming
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in filtered_words]
    return " ".join(stemmed_words)

def preprocess_text_for_sbert(text):
    if not isinstance(text, str):
        return ""
    # SBERT often benefits from minimal preprocessing, mainly lowercasing and stripping whitespace
    return text.lower().strip()


def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    if not isinstance(text_to_denoise, str) or not text_to_denoise.strip():
        return "" # Return empty string if input is invalid
    
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return "" # Return empty string if no words after tokenization

    result_words = []
    if method == 'a': # Random word deletion
        keep_or_not = np.random.rand(n) > del_ratio
        # Ensure at least one word is kept if the original text was not empty
        if sum(keep_or_not) == 0: 
            idx_to_keep = np.random.choice(n) # Randomly pick one word to keep
            keep_or_not[idx_to_keep] = True
        result_words = np.array(words)[keep_or_not].tolist()
    elif method in ('b', 'c'): # High-frequency word deletion (b) or deletion + shuffle (c)
        if word_freq_dict is None:
            # st.warning("word_freq_dict is required for denoising method 'b' or 'c'. Falling back to original text.")
            return text_to_denoise # Or raise ValueError
        
        # Identify indices of high-frequency words
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0:
            # Ensure num_to_remove doesn't exceed the number of available high-frequency words
            num_to_remove = min(num_to_remove, len(high_freq_indices))
            to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        
        # Ensure at least one word remains if original text had words and all were removed
        if not result_words and words:
            result_words = [random.choice(words)] 
            
        if method == 'c' and result_words: # Shuffle for method 'c'
            random.shuffle(result_words)
    else:
        st.error(f"Unknown denoising method: {method}. Using original text.")
        return text_to_denoise # Or raise ValueError

    return TreebankWordDetokenizer().detokenize(result_words)

def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'):
    # This function modifies data_df in place and also returns it.
    processed_results_intermediate_list = [] # To store dicts of intermediate steps

    if text_column_to_process not in data_df.columns:
        st.warning(f"Column '{text_column_to_process}' not found for preprocessing.")
        # Ensure 'processed_text' and 'preprocessing_steps' columns exist even if input is missing
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = ""
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = [{} for _ in range(len(data_df))]
        return data_df

    # Show spinner and progress bar for user feedback
    with st.spinner(f"Preprocessing column '{text_column_to_process}'... This may take a moment."):
        progress_bar = st.progress(0)
        status_text = st.empty() # To show text like "Processed X/Y entries"
        total_rows = len(data_df)
        
        final_processed_texts_column = [] # To store the final processed text for each row

        for i, text_content in enumerate(data_df[text_column_to_process].fillna('').astype(str)):
            intermediate_steps_dict = {'original': text_content}
            
            # 1. Symbol Removal (Punctuation and non-alphanumeric)
            temp_text = text_content.translate(str.maketrans('', '', string.punctuation))
            temp_text = re.sub(r'[^\w\s]', '', temp_text)
            intermediate_steps_dict['symbol_removed'] = temp_text
            
            # 2. Case Folding (Lowercase)
            temp_text = temp_text.lower()
            intermediate_steps_dict['case_folded'] = temp_text
            
            # 3. Tokenization
            word_tokens_list = word_tokenize(temp_text)
            intermediate_steps_dict['tokenized'] = " ".join(word_tokens_list) # Store as space-separated string
            
            # 4. Stopword Removal (on valid alphanumeric tokens)
            stop_words_set = set(stopwords.words('english'))
            # Filter for alphanumeric tokens before stopword removal and stemming for cleaner results
            valid_tokens_for_processing = [w for w in word_tokens_list if w.isalnum()]
            filtered_words_list = [w for w in valid_tokens_for_processing if w not in stop_words_set]
            intermediate_steps_dict['stopwords_removed'] = " ".join(filtered_words_list)
            
            # 5. Stemming (Porter Stemmer)
            if filtered_words_list: # Only stem if there are words left
                porter = PorterStemmer()
                stemmed_words_list = [porter.stem(w) for w in filtered_words_list]
                final_text_for_column = " ".join(stemmed_words_list)
            else: # If no words after stopword removal (e.g., text was only stopwords)
                final_text_for_column = "" 
            intermediate_steps_dict['stemmed'] = final_text_for_column
            
            # Append results
            processed_results_intermediate_list.append(intermediate_steps_dict)
            final_processed_texts_column.append(final_text_for_column)

            # Update progress bar and status text
            if total_rows > 0:
                progress_percentage = (i + 1) / total_rows
                progress_bar.progress(progress_percentage)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
        
        # Assign the new columns to the DataFrame
        data_df['processed_text'] = final_processed_texts_column
        data_df['preprocessing_steps'] = processed_results_intermediate_list
        
        st.success(f"Preprocessing of column '{text_column_to_process}' complete! 'processed_text' and 'preprocessing_steps' columns created/updated.")
        progress_bar.empty() # Clear progress bar
        status_text.empty() # Clear status text
    return data_df

@st.cache_resource # Cache the model itself
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        st.success(f"SBERT model '{model_name}' loaded successfully from cache or source.")
        return model
    except Exception as e:
        st.error(f"Error loading SBERT model '{model_name}': {e}. Ensure 'sentence-transformers' is installed and model name is correct.")
        return None

@st.cache_data # Cache the generated embeddings based on model and texts
def generate_embeddings_with_progress(_model, texts_list_to_embed): # _model is passed to make it part of cache key
    if _model is None: # Should be caught by load_bert_model, but good practice
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([]) # Return empty numpy array
    
    if not texts_list_to_embed or not any(text.strip() for text in texts_list_to_embed):
        st.warning("Input text list for embedding is empty or contains only whitespace. No embeddings generated.")
        return np.array([])

    # Filter out empty strings before sending to model, as they might cause issues or warnings
    valid_texts_to_embed = [text for text in texts_list_to_embed if text.strip()]
    if not valid_texts_to_embed: # If all texts were empty after stripping
        st.warning("All input texts were effectively empty. No embeddings generated.")
        return np.array([])

    try:
        with st.spinner(f"Generating embeddings for {len(valid_texts_to_embed)} texts... This may take a moment."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            
            embeddings_result_list = []
            total_texts_to_embed_count = len(valid_texts_to_embed)
            batch_size = 32 # Process in batches for potentially large lists

            for i in range(0, total_texts_to_embed_count, batch_size):
                batch_texts_segment = valid_texts_to_embed[i:i + batch_size]
                # show_progress_bar=False for model.encode as Streamlit has its own progress
                batch_embeddings_np_array = _model.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False)
                embeddings_result_list.extend(batch_embeddings_np_array)
                
                if total_texts_to_embed_count > 0:
                    progress_val = min(1.0, (i + len(batch_texts_segment)) / total_texts_to_embed_count)
                    embedding_progress_bar.progress(progress_val)
                    embedding_status_text.text(f"Embedded {min(i + len(batch_texts_segment), total_texts_to_embed_count)}/{total_texts_to_embed_count} texts.")
            
            st.success("Embedding generation complete!")
            embedding_progress_bar.empty()
            embedding_status_text.empty()
            return np.array(embeddings_result_list)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

@st.cache_data # Cache clustering results based on embeddings and n_clusters
def cluster_embeddings_with_progress(embeddings_to_cluster_param, n_clusters_for_algo):
    if embeddings_to_cluster_param is None or embeddings_to_cluster_param.size == 0:
        st.warning("No embeddings provided for clustering.")
        return None # Return None if no embeddings

    # Validate n_clusters_for_algo against the number of samples
    num_samples = embeddings_to_cluster_param.shape[0]
    if n_clusters_for_algo > num_samples:
        st.warning(f"Number of clusters (K={n_clusters_for_algo}) exceeds number of samples ({num_samples}). Setting K to {num_samples}.")
        n_clusters_for_algo = num_samples
    
    if n_clusters_for_algo < 1: # Should not happen if num_samples is >= 1
       st.error("Number of clusters (K) must be at least 1. Cannot cluster.")
       return None
    
    # K-Means requires at least 1 cluster. If K=1, all points go to one cluster.
    # If K=1 for multiple samples, it's trivial but valid.
    # If K > 1 but only 1 sample, KMeans will error. Adjust K to 1.
    if num_samples == 1 and n_clusters_for_algo > 1:
        st.warning(f"Only 1 sample available. Setting K=1 for clustering.")
        n_clusters_for_algo = 1
    elif num_samples > 1 and n_clusters_for_algo == 1:
         st.info("K=1: All samples will be assigned to a single cluster.")


    try:
        with st.spinner(f"Clustering {num_samples} embeddings into {n_clusters_for_algo} clusters using K-Means..."):
            # n_init='auto' is good for scikit-learn >= 0.23
            kmeans = KMeans(n_clusters=n_clusters_for_algo, random_state=42, n_init='auto')
            clusters_assigned = kmeans.fit_predict(embeddings_to_cluster_param)
            st.success(f"Clustering complete! {num_samples} items assigned to {n_clusters_for_algo} clusters.")
            return clusters_assigned
    except Exception as e:
        st.error(f"Error during K-Means clustering: {e}")
        return None

# --- SBERT O*NET Classification Function ---
@st.cache_data(show_spinner=False) # Caching the result of this expensive operation
def classify_with_sbert_onet(_df_jobs, _onet_df, sbert_model_instance): # Pass model as arg for caching
    df_jobs_classified = _df_jobs.copy() # Work on a copy
    
    # Define expected O*NET column names
    onet_title_col = 'Title'
    onet_code_col = 'O*NET-SOC Code'
    onet_desc_col = 'Description'

    # Validate O*NET DataFrame structure
    if not all(col in _onet_df.columns for col in [onet_title_col, onet_code_col, onet_desc_col]):
        st.error(f"O*NET data is missing one or more required columns: '{onet_title_col}', '{onet_code_col}', '{onet_desc_col}'. Cannot perform classification.")
        # Add error state columns to the job DataFrame
        df_jobs_classified['onet_category'] = 'Error: O*NET data columns missing'
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        df_jobs_classified['sbert_job_embedding_onet_classified'] = [None] * len(df_jobs_classified) # Store as list of Nones
        return df_jobs_classified

    st.write("Preprocessing job posting data for SBERT O*NET classification...")
    # Combine relevant text fields from job data for SBERT.
    # Use 'combined_jobs' if available and preprocessed, or create a new combination.
    if 'combined_jobs' in df_jobs_classified.columns:
         df_jobs_classified['text_for_onet_sbert'] = df_jobs_classified['combined_jobs']
    else: # Fallback: combine all object columns (excluding Job.ID)
        object_cols = df_jobs_classified.select_dtypes(include=['object']).columns
        object_cols_for_text = [col for col in object_cols if col != 'Job.ID'] # Exclude Job.ID
        df_jobs_classified['text_for_onet_sbert'] = df_jobs_classified[object_cols_for_text].fillna('').astype(str).agg(' '.join, axis=1)
    
    df_jobs_classified['processed_job_text_for_onet'] = df_jobs_classified['text_for_onet_sbert'].apply(preprocess_text_for_sbert)

    st.write("Preprocessing O*NET data for SBERT...")
    _onet_df_copy = _onet_df.copy() # Work on a copy of O*NET data
    _onet_df_copy['combined_onet_text'] = _onet_df_copy[onet_title_col].fillna('').astype(str) + ' ' + _onet_df_copy[onet_desc_col].fillna('').astype(str)
    _onet_df_copy['processed_onet_text'] = _onet_df_copy['combined_onet_text'].apply(preprocess_text_for_sbert)

    job_texts_to_embed = df_jobs_classified['processed_job_text_for_onet'].tolist()
    onet_texts_to_embed = _onet_df_copy['processed_onet_text'].tolist()

    if not job_texts_to_embed or not onet_texts_to_embed:
        st.error("No text data available for SBERT O*NET embedding after preprocessing. Check input data.")
        df_jobs_classified['onet_category'] = 'Error: No text for SBERT' # Fill error columns
        df_jobs_classified['onet_soc_code'] = 'Error'
        df_jobs_classified['onet_match_score'] = np.nan
        df_jobs_classified['sbert_job_embedding_onet_classified'] = [None] * len(df_jobs_classified)
        return df_jobs_classified

    if sbert_model_instance is None: # Should be caught by load_bert_model
        st.error("SBERT model not available for O*NET classification.")
        df_jobs_classified['onet_category'] = 'Error: SBERT model missing' # Fill error columns
        # ... (fill other error columns as above)
        return df_jobs_classified

    # Generate embeddings (using the main generate_embeddings_with_progress for UI consistency)
    st.info("Generating SBERT embeddings for job postings (for O*NET classification)...")
    job_embeddings = generate_embeddings_with_progress(sbert_model_instance, job_texts_to_embed)
    
    st.info("Generating SBERT embeddings for O*NET occupations...")
    onet_embeddings = generate_embeddings_with_progress(sbert_model_instance, onet_texts_to_embed)

    if job_embeddings.size == 0 or onet_embeddings.size == 0:
        st.error("Embedding generation failed for jobs or O*NET data. Cannot proceed with classification.")
        df_jobs_classified['onet_category'] = 'Error: Embedding failed'
        # ... (fill other error columns)
        return df_jobs_classified

    # Store the generated job embeddings in the DataFrame
    # Ensure the length matches df_jobs_classified. If generate_embeddings_with_progress filtered, this needs careful handling.
    # Assuming generate_embeddings_with_progress returns embeddings for all non-empty inputs in order.
    df_jobs_classified['sbert_job_embedding_onet_classified'] = list(job_embeddings) if job_embeddings.ndim > 1 else [[emb] for emb in job_embeddings]


    st.write("Calculating similarities and matching job postings to O*NET occupations...")
    similarity_matrix = cosine_similarity(job_embeddings, onet_embeddings)

    matched_onet_titles_list = []
    matched_onet_codes_list = []
    match_scores_list = []

    for i in range(similarity_matrix.shape[0]): # Iterate through each job posting
        best_match_onet_idx = np.argmax(similarity_matrix[i]) # Index of the best matching O*NET occupation
        best_similarity_score = similarity_matrix[i, best_match_onet_idx]
        
        matched_onet_titles_list.append(_onet_df_copy.iloc[best_match_onet_idx][onet_title_col])
        matched_onet_codes_list.append(_onet_df_copy.iloc[best_match_onet_idx][onet_code_col])
        match_scores_list.append(best_similarity_score)

    df_jobs_classified['onet_category'] = matched_onet_titles_list
    df_jobs_classified['onet_soc_code'] = matched_onet_codes_list
    df_jobs_classified['onet_match_score'] = match_scores_list
    
    # Clean up intermediate text columns to save memory, if desired
    df_jobs_classified = df_jobs_classified.drop(columns=['text_for_onet_sbert', 'processed_job_text_for_onet'], errors='ignore')
    # _onet_df_copy is local, so no need to drop from it for session state

    st.success("SBERT O*NET classification complete. Columns 'onet_category', 'onet_soc_code', 'onet_match_score', and 'sbert_job_embedding_onet_classified' added/updated.")
    return df_jobs_classified


# --- Page Functions ---
def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("This page provides an overview of the job dataset and allows basic exploration.")

    # Load data if not already in session state
    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES_TO_DISPLAY)
    
    data_df = st.session_state.get('data') # Get the potentially loaded data

    if data_df is not None and not data_df.empty:
        st.subheader('Data Preview (First 5 Rows)')
        # Select a few key columns for preview, ensure they exist
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        if 'onet_category' in data_df.columns: cols_to_preview.append('onet_category') # Show if classified
        
        existing_preview_cols = [col for col in cols_to_preview if col in data_df.columns]
        if existing_preview_cols:
            st.dataframe(data_df[existing_preview_cols].head(), use_container_width=True)
        else:
            st.info("No key columns available for preview (e.g., 'Job.ID', 'Title').")


        st.subheader('Data Summary')
        st.write(f'Number of job postings (rows): {len(data_df)}')
        st.write(f'Number of features (columns): {len(data_df.columns)}')
        
        if 'combined_jobs' in data_df.columns:
            st.subheader('Sample Content of `combined_jobs` Column')
            # Display a few samples in expanders
            for i in range(min(3, len(data_df))): # Show up to 3 samples
                title_display = data_df.iloc[i].get('Title', "N/A") # Use .get for safety
                job_id_display = data_df.iloc[i].get('Job.ID', "N/A")
                with st.expander(f"Job.ID: {job_id_display} - Title: {title_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.info("Column 'combined_jobs' has not been created or is not in the loaded data. This column is generated from other text features.")

        st.subheader('Search for a Word in a Feature')
        search_word = st.text_input("Enter word to search:", key="home_search_word_input")
        
        # Dynamically create list of searchable columns from what's available
        all_potential_search_cols = ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE + JOB_DETAIL_FEATURES_TO_DISPLAY
        if 'onet_category' in data_df.columns: all_potential_search_cols.append('onet_category')
        if 'onet_soc_code' in data_df.columns: all_potential_search_cols.append('onet_soc_code')
        # Filter to only include columns actually present in the DataFrame
        searchable_cols_present = sorted(list(set(col for col in all_potential_search_cols if col in data_df.columns)))
        
        search_column = st.selectbox(
            "Select feature to search in:", 
            options=[''] + searchable_cols_present, # Add empty option for no selection
            key="home_search_column_select"
        )

        if search_word and search_column: # If both word and column are selected
            if search_column in data_df.columns: # Ensure column exists
                # Perform case-insensitive search, handle NA values
                search_results_df = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                
                # Columns to display in search results
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column) # Add the searched column itself
                
                # Filter display_search_cols to only those existing in search_results_df
                actual_display_search_cols = [col for col in display_search_cols if col in search_results_df.columns]

                if not search_results_df.empty:
                    st.write(f"Found {len(search_results_df)} entries containing '{search_word}' in '{search_column}':")
                    st.dataframe(search_results_df[actual_display_search_cols].head(), use_container_width=True) # Show head for brevity
                else:
                    st.info(f"No entries found containing '{search_word}' in '{search_column}'.")
        
        st.subheader('Available Features (Columns)')
        st.write(data_df.columns.tolist())
    else:
        st.error("Job data could not be loaded or is empty. Please check the data source URL or your internet connection.")
        if st.button("Attempt to Reload Job Data"):
            st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES_TO_DISPLAY)
            st.experimental_rerun()
    return

def preprocessing_page():
    st.header("Job Data Preprocessing (for 'combined_jobs')")
    st.write("This page performs text preprocessing (cleaning, tokenization, stemming) on the 'combined_jobs' column, creating 'processed_text' and 'preprocessing_steps'.")

    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or the 'combined_jobs' column is not available. Please ensure data is loaded on the 'Home' page first.")
        if st.button("Go to Home Page to Load Data"):
            st.session_state.current_page = "Home" # Navigate to Home
            st.experimental_rerun()
        return
    
    data_df_to_preprocess = st.session_state['data'] # Get the DataFrame

    st.info("The 'combined_jobs' column will be processed. This involves symbol removal, case folding, tokenization, stopword removal, and stemming.")
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before preprocessing)"):
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())

    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_text_preprocessing_button"):
        # Pass a copy to the preprocessing function if it modifies inplace, or reassign result
        # preprocess_text_with_intermediate modifies and returns the df
        data_copy_for_processing = data_df_to_preprocess.copy()
        st.session_state['data'] = preprocess_text_with_intermediate(data_copy_for_processing, text_column_to_process='combined_jobs')
        # Success message is handled within preprocess_text_with_intermediate
    
    # Display results if 'processed_text' column exists (meaning preprocessing has likely run)
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.markdown("---")
        st.info("Preprocessing has been performed on 'combined_jobs'. Results are shown below.")
        display_data_after_processing = st.session_state['data']
        
        if 'preprocessing_steps' in display_data_after_processing.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run - Sample)")
            # Extract the list of dicts from the 'preprocessing_steps' column
            intermediate_steps_list_of_dicts = display_data_after_processing['preprocessing_steps'].tolist()
            # Filter out any non-dict items if any (shouldn't happen with current logic)
            valid_intermediate_steps_dicts = [s for s in intermediate_steps_list_of_dicts if isinstance(s, dict)]
            
            if valid_intermediate_steps_dicts:
                st.dataframe(pd.DataFrame(valid_intermediate_steps_dicts).head(), use_container_width=True)
            else:
                st.warning("Intermediate preprocessing steps data is not in the expected dictionary format or is empty.")
        
        st.subheader("Final Preprocessed Text ('processed_text') (Sample)")
        preview_cols_processed = ['Job.ID']
        if 'combined_jobs' in display_data_after_processing.columns: preview_cols_processed.append('combined_jobs')
        preview_cols_processed.append('processed_text') # This should exist if we are in this block
        st.dataframe(display_data_after_processing[preview_cols_processed].head(), use_container_width=True)
        
        # Allow searching in the 'processed_text' column
        search_word_in_processed_text = st.text_input("Search for a word in 'processed_text':", key="search_in_processed_text_input")
        if search_word_in_processed_text:
            results_in_processed = display_data_after_processing[
                display_data_after_processing['processed_text'].astype(str).str.contains(search_word_in_processed_text, na=False, case=False)
            ]
            if not results_in_processed.empty:
                display_cols_for_search_processed = ['Job.ID']
                if 'Title' in results_in_processed.columns: display_cols_for_search_processed.append('Title')
                display_cols_for_search_processed.append('processed_text')
                st.dataframe(results_in_processed[display_cols_for_search_processed].head(), use_container_width=True)
            else:
                st.info(f"No results found for '{search_word_in_processed_text}' in 'processed_text'.")
    else:
        st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet for the current session, or 'processed_text' column is missing. Click the button above to run it.")
    return

def tsdae_page():
    st.header("TSDAE Preprocessing (Noise Injection for 'processed_text')")
    st.write("This page applies sequential noise (Methods A, B, C) to the 'processed_text' column to create 'final_noisy_text'. Then, it generates TSDAE embeddings from this noisy text.")
    st.write("**Note:** 'processed_text' should be generated first on the 'Data Preprocessing' page.")

    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data with 'processed_text' column is not available. Please run 'Data Preprocessing' first.")
        if st.button("Go to Data Preprocessing Page"):
            st.session_state.current_page = "Data Preprocessing"
            st.experimental_rerun()
        return
    
    bert_model_instance = load_bert_model() # Load the SBERT model (cached)
    if bert_model_instance is None:
        st.error("SBERT model could not be loaded. TSDAE embedding generation will not be possible."); return

    st.subheader("TSDAE Noise Settings")
    # Sliders for noise parameters
    deletion_ratio_param = st.slider("Deletion Ratio (for noise methods)", 0.1, 0.9, 0.6, 0.05, key="tsdae_deletion_ratio_slider")
    freq_threshold_param = st.slider("High Frequency Word Threshold (for noise methods B & C)", 10, 500, 100, 10, key="tsdae_freq_threshold_slider")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="run_tsdae_noise_and_embedding_button"):
        data_for_tsdae = st.session_state['data'].copy() # Work on a copy
        
        if 'processed_text' not in data_for_tsdae.columns or data_for_tsdae['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Cannot proceed with TSDAE noise application.")
            return

        # Calculate word frequencies from 'processed_text' for methods B and C
        st.info("Calculating word frequencies from 'processed_text' for noise methods B & C...")
        all_words_from_processed = [
            word for text_entry in data_for_tsdae['processed_text'].fillna('').astype(str) 
            for word in word_tokenize(text_entry) # Tokenize each entry
        ]
        if not all_words_from_processed:
            st.warning("No words found in 'processed_text' to build frequency dictionary. Methods B & C might not be effective.")
            word_freq_dict_for_tsdae = {}
        else:
            # Create frequency dictionary (lowercase keys)
            word_freq_dict_for_tsdae = {word.lower(): all_words_from_processed.count(word.lower()) for word in set(all_words_from_processed)}
        
        # --- Apply Noise Method A (Random Deletion) ---
        st.markdown("--- \n ##### Applying Noise Method A (Random Deletion on 'processed_text')")
        noisy_text_list_A = []
        source_texts_for_A = data_for_tsdae['processed_text'].fillna('').astype(str).tolist()
        progress_bar_A = st.progress(0); status_text_A = st.empty(); total_A = len(source_texts_for_A)
        for idx, text in enumerate(source_texts_for_A):
            noisy_text_list_A.append(denoise_text(text, method='a', del_ratio=deletion_ratio_param))
            if total_A > 0: progress_bar_A.progress((idx + 1) / total_A); status_text_A.text(f"Method A: Processed {idx + 1}/{total_A}")
        data_for_tsdae['noisy_text_a'] = noisy_text_list_A
        progress_bar_A.empty(); status_text_A.empty(); st.success("Noise Method A complete.")

        # --- Apply Noise Method B (High-Frequency Word Removal on 'noisy_text_a') ---
        st.markdown("--- \n ##### Applying Noise Method B (High-Frequency Removal on 'noisy_text_a')")
        noisy_text_list_B = []
        source_texts_for_B = data_for_tsdae['noisy_text_a'].tolist() # Input is output of method A
        progress_bar_B = st.progress(0); status_text_B = st.empty(); total_B = len(source_texts_for_B)
        for idx, text in enumerate(source_texts_for_B):
            noisy_text_list_B.append(denoise_text(text, method='b', del_ratio=deletion_ratio_param, word_freq_dict=word_freq_dict_for_tsdae, freq_threshold=freq_threshold_param))
            if total_B > 0: progress_bar_B.progress((idx + 1) / total_B); status_text_B.text(f"Method B: Processed {idx + 1}/{total_B}")
        data_for_tsdae['noisy_text_b'] = noisy_text_list_B
        progress_bar_B.empty(); status_text_B.empty(); st.success("Noise Method B complete.")

        # --- Apply Noise Method C (High-Frequency Word Removal + Shuffle on 'noisy_text_b') ---
        st.markdown("--- \n ##### Applying Noise Method C (High-Frequency Removal + Shuffle on 'noisy_text_b') - Output to 'final_noisy_text'")
        final_noisy_text_list_C = []
        source_texts_for_C = data_for_tsdae['noisy_text_b'].tolist() # Input is output of method B
        progress_bar_C = st.progress(0); status_text_C = st.empty(); total_C = len(source_texts_for_C)
        for idx, text in enumerate(source_texts_for_C):
            final_noisy_text_list_C.append(denoise_text(text, method='c', del_ratio=deletion_ratio_param, word_freq_dict=word_freq_dict_for_tsdae, freq_threshold=freq_threshold_param))
            if total_C > 0: progress_bar_C.progress((idx + 1) / total_C); status_text_C.text(f"Method C: Processed {idx + 1}/{total_C}")
        data_for_tsdae['final_noisy_text'] = final_noisy_text_list_C
        progress_bar_C.empty(); status_text_C.empty(); st.success("Noise Method C complete. 'final_noisy_text' column created.")
        
        st.session_state['data'] = data_for_tsdae # Update session state with the DataFrame containing noisy text columns
        st.subheader("Preview of Noisy Text Generation Stages (Sample)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        # --- Generate TSDAE Embeddings from 'final_noisy_text' ---
        st.markdown("--- \n ##### Generating TSDAE Embeddings from 'final_noisy_text'")
        final_noisy_texts_series = st.session_state['data']['final_noisy_text'].fillna('').astype(str)
        # Filter out rows where 'final_noisy_text' is empty or only whitespace, and get corresponding Job.IDs
        non_empty_noisy_mask = final_noisy_texts_series.str.strip() != ''
        valid_final_noisy_texts_for_embedding = final_noisy_texts_series[non_empty_noisy_mask].tolist()
        job_ids_for_tsdae_embeddings_generated = st.session_state['data'].loc[non_empty_noisy_mask, 'Job.ID'].tolist()

        if not valid_final_noisy_texts_for_embedding:
            st.warning("No valid (non-empty) 'final_noisy_text' found after noise application. Cannot generate TSDAE embeddings.")
        else:
            tsdae_embeddings_array = generate_embeddings_with_progress(bert_model_instance, valid_final_noisy_texts_for_embedding)
            if tsdae_embeddings_array.size > 0:
                st.session_state['tsdae_embeddings'] = tsdae_embeddings_array
                st.session_state['tsdae_embedding_job_ids'] = job_ids_for_tsdae_embeddings_generated
                st.success(f"TSDAE embeddings generated successfully for {len(job_ids_for_tsdae_embeddings_generated)} jobs from 'final_noisy_text'!")
            else:
                st.warning("TSDAE embedding generation resulted in an empty output, though valid noisy text was provided.")
    
    # Display current TSDAE embeddings if they exist
    if 'tsdae_embeddings' in st.session_state and st.session_state.tsdae_embeddings is not None and st.session_state.tsdae_embeddings.size > 0:
        st.subheader("Current TSDAE Embeddings (from 'final_noisy_text')")
        st.write(f"Number of jobs with TSDAE embeddings: {len(st.session_state.get('tsdae_embedding_job_ids', []))}")
        st.write(f"Shape of TSDAE embeddings array: {st.session_state.tsdae_embeddings.shape}")
        st.write("Preview of first 3 embedding vectors (or fewer if less than 3):")
        st.write(st.session_state.tsdae_embeddings[:min(3, st.session_state.tsdae_embeddings.shape[0])])
    else:
        st.info("TSDAE embeddings have not been generated yet or the process yielded no embeddings.")
    return

def bert_model_page():
    st.header("Standard SBERT Embeddings (from 'processed_text')")
    st.write("This page generates standard SBERT embeddings directly from the 'processed_text' column (which should be created via 'Data Preprocessing').")
    st.write("**Note:** 'processed_text' is derived from 'combined_jobs' after cleaning, tokenization, and stemming.")

    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data with 'processed_text' column is not available. Please run 'Data Preprocessing' first.")
        if st.button("Go to Data Preprocessing Page"):
            st.session_state.current_page = "Data Preprocessing"
            st.experimental_rerun()
        return

    bert_model_instance = load_bert_model() # Load the SBERT model (cached)
    if bert_model_instance is None: 
        st.error("SBERT model could not be loaded. Embedding generation will not be possible."); return

    if st.button("Generate/Regenerate Standard Job Embeddings from 'processed_text'", key="generate_standard_job_embeddings_button"):
        data_for_bert_embeddings = st.session_state['data'] # Use the main data
        
        if 'processed_text' not in data_for_bert_embeddings.columns or data_for_bert_embeddings['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing in the loaded data. Please ensure preprocessing was run correctly.")
            return

        # Get 'processed_text' and corresponding 'Job.ID', filtering out empty texts
        processed_text_series = data_for_bert_embeddings['processed_text'].fillna('').astype(str)
        non_empty_text_mask = processed_text_series.str.strip() != ''
        
        valid_texts_for_embedding = processed_text_series[non_empty_text_mask].tolist()
        valid_job_ids_for_embeddings = data_for_bert_embeddings.loc[non_empty_text_mask, 'Job.ID'].tolist()

        if not valid_texts_for_embedding:
            st.warning("No valid (non-empty) 'processed_text' entries found to generate standard SBERT embeddings.")
        else:
            standard_job_embeddings_array = generate_embeddings_with_progress(bert_model_instance, valid_texts_for_embedding)
            if standard_job_embeddings_array.size > 0:
                st.session_state['job_text_embeddings'] = standard_job_embeddings_array
                st.session_state['job_text_embedding_job_ids'] = valid_job_ids_for_embeddings
                st.success(f"Standard SBERT job embeddings generated for {len(valid_job_ids_for_embeddings)} jobs from 'processed_text'!")
            else:
                st.warning("Standard SBERT job embedding generation resulted in an empty output, though valid text was provided.")

    # Display current standard job embeddings if they exist
    current_job_embeddings = st.session_state.get('job_text_embeddings')
    current_job_ids = st.session_state.get('job_text_embedding_job_ids')

    if current_job_embeddings is not None and current_job_embeddings.size > 0 and current_job_ids:
        st.subheader(f"Current Standard SBERT Job Embeddings ({len(current_job_ids)} jobs)")
        st.write(f"Shape of embeddings array: {current_job_embeddings.shape}")
        
        st.subheader("2D Visualization of Standard Job Embeddings (PCA)")
        if len(current_job_embeddings) >= 2: # PCA needs at least 2 samples
            try:
                with st.spinner("Generating PCA visualization..."):
                    pca_2d_instance = PCA(n_components=2, random_state=42)
                    reduced_features_2d = pca_2d_instance.fit_transform(current_job_embeddings)
                    
                    pca_plot_df = pd.DataFrame(reduced_features_2d, columns=['PC1','PC2'])
                    pca_plot_df['Job.ID'] = current_job_ids # Add Job.IDs for hover info
                    
                    # Merge with main data to get 'Title' and other hover info
                    main_data_for_hover_info = st.session_state['data']
                    # Ensure Job.ID types match for merging (both should be string)
                    pca_plot_df['Job.ID'] = pca_plot_df['Job.ID'].astype(str)
                    main_data_for_hover_info['Job.ID'] = main_data_for_hover_info['Job.ID'].astype(str)

                    hover_data_cols_to_fetch = ['Job.ID'] # Start with Job.ID
                    if 'Title' in main_data_for_hover_info.columns: hover_data_cols_to_fetch.append('Title')
                    # Use 'combined_jobs' for a richer hover text if available, else 'Job.Description'
                    hover_text_col = 'combined_jobs' if 'combined_jobs' in main_data_for_hover_info.columns else \
                                     ('Job.Description' if 'Job.Description' in main_data_for_hover_info.columns else None)
                    if hover_text_col: hover_data_cols_to_fetch.append(hover_text_col)
                    
                    # Fetch only necessary columns and drop duplicates by Job.ID before merge
                    hover_info_df = main_data_for_hover_info[main_data_for_hover_info['Job.ID'].isin(current_job_ids)][hover_data_cols_to_fetch].drop_duplicates(subset=['Job.ID'])
                    
                    pca_plot_df = pd.merge(pca_plot_df, hover_info_df, on='Job.ID', how='left')
                    
                    # Configure hover data for Plotly Express
                    hover_data_config_dict = {'Job.ID': True, 'PC1': False, 'PC2': False} # Hide PCA components from hover by default
                    if hover_text_col and hover_text_col in pca_plot_df.columns:
                        hover_data_config_dict[hover_text_col] = True # Show the description/combined text
                    
                    if not pca_plot_df.empty and 'Title' in pca_plot_df.columns:
                        fig_pca_scatter = px.scatter(
                            pca_plot_df, x='PC1', y='PC2',
                            hover_name='Title', # Use Job Title for hover name
                            hover_data=hover_data_config_dict,
                            title='2D PCA of Standard SBERT Job Embeddings (from "processed_text")'
                        )
                        st.plotly_chart(fig_pca_scatter, use_container_width=True)
                    else:
                        st.warning("PCA plot data is incomplete (e.g., 'Title' missing after merge). Cannot display plot.")
            except Exception as e_pca:
                st.error(f"Error during PCA visualization: {e_pca}")
        else:
            st.warning("Need at least 2 data points (embedded jobs) for PCA visualization.")
    else:
        st.info("Standard SBERT job embeddings (from 'processed_text') have not been generated yet or the process yielded no embeddings.")
    return

# --- SBERT O*NET Classification Page ---
def sbert_onet_classification_page():
    st.header("SBERT O*NET Classification")
    st.write("Classify job postings against O*NET standard occupations using SBERT embeddings. This process generates new columns in the main job dataset: 'onet_category', 'onet_soc_code', 'onet_match_score', and 'sbert_job_embedding_onet_classified'.")

    if st.session_state.get('data') is None:
        st.error("Job data not loaded. Please go to the 'Home' page first to load the job dataset.")
        if st.button("Go to Home Page"): st.session_state.current_page = "Home"; st.experimental_rerun()
        return

    # Load O*NET data if not already loaded
    if st.session_state.get('onet_data') is None:
        if st.button("Load O*NET Occupation Data (Required for Classification)", key="load_onet_data_button_classification"):
            st.session_state['onet_data'] = load_simple_csv_from_url(ONET_DATA_URL, "O*NET Occupations")
            if st.session_state['onet_data'] is not None:
                st.success(f"O*NET data loaded: {st.session_state['onet_data'].shape[0]} occupations found.")
                # Validate essential O*NET columns after loading
                expected_onet_cols = ['O*NET-SOC Code', 'Title', 'Description']
                if not all(col in st.session_state['onet_data'].columns for col in expected_onet_cols):
                    st.error(f"O*NET data is missing one or more required columns: {expected_onet_cols}. Classification cannot proceed. Please check the O*NET data source.")
                    st.session_state['onet_data'] = None # Invalidate O*NET data if columns are missing
            else:
                st.error("Failed to load O*NET data. Classification cannot proceed.")
    
    if st.session_state.get('onet_data') is not None:
        st.subheader("O*NET Data Preview (First 5 Rows)")
        st.dataframe(st.session_state['onet_data'].head())

        if st.button("Run SBERT O*NET Classification on Job Data", key="run_sbert_onet_classification_button"):
            sbert_model_for_onet = load_bert_model() # Uses the cached SBERT model
            # Ensure all necessary data components are available
            if sbert_model_for_onet and st.session_state.get('data') is not None and st.session_state.get('onet_data') is not None:
                with st.spinner("Classifying jobs with SBERT and O*NET... This may take some time depending on data size."):
                    # classify_with_sbert_onet returns the modified DataFrame
                    st.session_state['data'] = classify_with_sbert_onet(
                        st.session_state['data'], # Pass the main job data
                        st.session_state['onet_data'], # Pass O*NET data
                        sbert_model_for_onet # Pass the SBERT model instance
                    )
                # Success message is handled within classify_with_sbert_onet
            else:
                error_messages = []
                if not sbert_model_for_onet: error_messages.append("SBERT model is not loaded.")
                if st.session_state.get('data') is None: error_messages.append("Job data is not loaded.")
                if st.session_state.get('onet_data') is None: error_messages.append("O*NET data is not loaded.")
                st.error(f"Could not run SBERT O*NET classification. Issues: {', '.join(error_messages)}")

    # Display results if classification has run and added the 'onet_category' column
    if 'onet_category' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader("O*NET Classification Results Preview (from Job Data)")
        preview_cols_onet = ['Job.ID', 'Title', 'onet_category', 'onet_soc_code', 'onet_match_score']
        # Filter to columns that actually exist in the dataframe
        actual_preview_cols_onet = [col for col in preview_cols_onet if col in st.session_state['data'].columns]
        if actual_preview_cols_onet:
             st.dataframe(st.session_state['data'][actual_preview_cols_onet].head())

        if 'onet_match_score' in st.session_state['data'].columns:
            avg_match_score_onet = st.session_state['data']['onet_match_score'].mean()
            st.metric("Average SBERT O*NET Match Score", f"{avg_match_score_onet:.4f}" if pd.notna(avg_match_score_onet) else "N/A")

        # PCA Plot for SBERT O*NET Embeddings (using matplotlib as per original, can be switched to Plotly)
        st.subheader("🎨 Visual Canvas of O*NET Classified Jobs (PCA on SBERT Embeddings used for O*NET matching)")
        df_for_onet_plot = st.session_state['data'].copy() # Use a copy for plotting

        # Check if necessary columns for plotting are present
        if 'sbert_job_embedding_onet_classified' not in df_for_onet_plot.columns or \
           df_for_onet_plot['sbert_job_embedding_onet_classified'].isnull().all():
            st.warning("SBERT O*NET embeddings ('sbert_job_embedding_onet_classified') column not found or is empty. Cannot generate PCA plot.")
        elif 'onet_category' not in df_for_onet_plot.columns:
            st.warning("O*NET category ('onet_category') column not found. Cannot color PCA plot by category.")
        else:
            # Prepare data for plotting: drop rows with missing embeddings or categories
            plot_data_sbert_onet = df_for_onet_plot.dropna(subset=['sbert_job_embedding_onet_classified', 'onet_category']).copy()
            
            if len(plot_data_sbert_onet) < 2:
                st.warning("Not enough data points (need at least 2) with valid SBERT O*NET embeddings and categories for PCA plot.")
            else:
                try:
                    with st.spinner("Generating SBERT O*NET classification canvas visualization..."):
                        # Extract embeddings into a NumPy array
                        sbert_embeddings_for_plot_onet = np.array(plot_data_sbert_onet['sbert_job_embedding_onet_classified'].tolist())
                        
                        pca_sbert_onet_plotter = PCA(n_components=2, random_state=42)
                        reduced_sbert_features_onet = pca_sbert_onet_plotter.fit_transform(sbert_embeddings_for_plot_onet)
                        
                        # Create a DataFrame for plotting
                        pca_plot_df_sbert_onet = pd.DataFrame({
                            'pca_x': reduced_sbert_features_onet[:,0],
                            'pca_y': reduced_sbert_features_onet[:,1],
                            'category': plot_data_sbert_onet['onet_category'].astype('category'), # Convert to category for coloring
                            'title': plot_data_sbert_onet.get('Title', 'N/A') # Get title if exists
                        })
                        
                        # Matplotlib plotting
                        fig_canvas_sbert_onet, ax_canvas_sbert_onet = plt.subplots(figsize=(12, 9))
                        
                        cat_codes_sbert_onet = pca_plot_df_sbert_onet['category'].cat.codes
                        num_unique_cats_sbert_onet = len(pca_plot_df_sbert_onet['category'].cat.categories)
                        
                        # Choose colormap based on number of categories
                        cmap_name_sbert_onet = 'tab10' if num_unique_cats_sbert_onet <= 10 else \
                                               ('tab20' if num_unique_cats_sbert_onet <= 20 else 'viridis')
                        cmap_sbert_onet = plt.get_cmap(cmap_name_sbert_onet, num_unique_cats_sbert_onet if num_unique_cats_sbert_onet > 0 else 1)

                        scatter_plot = ax_canvas_sbert_onet.scatter(
                            pca_plot_df_sbert_onet['pca_x'], pca_plot_df_sbert_onet['pca_y'], 
                            c=cat_codes_sbert_onet, cmap=cmap_sbert_onet, 
                            alpha=0.7, s=50, edgecolor='k', linewidths=0.5
                        )
                        ax_canvas_sbert_onet.set_title('Job Postings by O*NET Category (SBERT Embeddings + PCA)', fontsize=15)
                        ax_canvas_sbert_onet.set_xlabel('PCA Component 1 (from SBERT O*NET embeddings)', fontsize=12)
                        ax_canvas_sbert_onet.set_ylabel('PCA Component 2 (from SBERT O*NET embeddings)', fontsize=12)
                        ax_canvas_sbert_onet.grid(True, linestyle='--', alpha=0.5)

                        # Add legend if number of categories is manageable
                        if 0 < num_unique_cats_sbert_onet <= 20:
                            legend_handles_sbert_onet = [
                                plt.Line2D([0], [0], marker='o', color='w', 
                                           label=str(cat_name)[:30], # Truncate long category names for legend
                                           markerfacecolor=cmap_sbert_onet(i / (num_unique_cats_sbert_onet -1 if num_unique_cats_sbert_onet > 1 else 1.0)), 
                                           markersize=8) 
                                for i, cat_name in enumerate(pca_plot_df_sbert_onet['category'].cat.categories)
                            ]
                            if legend_handles_sbert_onet: 
                                ax_canvas_sbert_onet.legend(handles=legend_handles_sbert_onet, title='O*NET Categories', 
                                                            bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                        elif num_unique_cats_sbert_onet > 20:
                            st.caption(f"Legend omitted for clarity due to large number of categories ({num_unique_cats_sbert_onet} found).")
                        
                        plt.tight_layout(rect=[0, 0, 0.85 if (0 < num_unique_cats_sbert_onet <=20) else 1, 1]) # Adjust layout for legend
                        st.pyplot(fig_canvas_sbert_onet)
                        
                        if st.checkbox("Show sample titles for SBERT O*NET canvas points (first 10 plotted)?", key="show_sbert_onet_canvas_sample_titles"):
                            st.dataframe(pca_plot_df_sbert_onet[['title', 'category']].head(10))
                except Exception as e_sbert_onet_plot:
                    st.error(f"Could not generate SBERT O*NET classification canvas: {e_sbert_onet_plot}")
    else:
        st.info("Load O*NET data and run the classification process to see results and visualizations.")

    return


def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("This page performs K-Means clustering on selected job embeddings (Standard SBERT, TSDAE, or SBERT O*NET Classified) and visualizes the clusters using PCA.")

    if st.session_state.get('data') is None:
        st.error("Job data not loaded. Please go to Home page first to load data."); return

    embeddings_to_cluster_array, job_ids_for_clustering, source_name_for_clustering_ui = None, None, ""
    
    # Determine available embedding types for clustering
    clustering_embedding_options = []
    if st.session_state.get('job_text_embeddings', np.array([])).size > 0 and st.session_state.get('job_text_embedding_job_ids'):
        clustering_embedding_options.append("Standard SBERT (from 'processed_text')")
    if st.session_state.get('tsdae_embeddings', np.array([])).size > 0 and st.session_state.get('tsdae_embedding_job_ids'):
        clustering_embedding_options.append("TSDAE (from 'final_noisy_text')")
    
    # Check for SBERT O*NET classified embeddings
    sbert_onet_embeddings_present = False
    if 'sbert_job_embedding_onet_classified' in st.session_state.get('data', pd.DataFrame()).columns:
        # Check if the column contains actual non-None embeddings
        sbert_onet_emb_series = st.session_state['data']['sbert_job_embedding_onet_classified'].dropna()
        if not sbert_onet_emb_series.empty and isinstance(sbert_onet_emb_series.iloc[0], np.ndarray):
            sbert_onet_embeddings_present = True
            clustering_embedding_options.append("SBERT O*NET Classified")

    if not clustering_embedding_options:
        st.warning("No embeddings are currently available for clustering. Please generate embeddings on the 'Standard SBERT Embeddings', 'TSDAE Embeddings', or 'SBERT O*NET Classification' pages first.")
        return

    chosen_embedding_type = st.radio(
        "Select Embeddings for Clustering:", 
        options=clustering_embedding_options, 
        key="clustering_embedding_type_radio", 
        horizontal=True
    )

    # Load the selected embeddings and corresponding Job IDs
    if chosen_embedding_type == "Standard SBERT (from 'processed_text')":
        embeddings_to_cluster_array = st.session_state['job_text_embeddings']
        job_ids_for_clustering = st.session_state.get('job_text_embedding_job_ids')
        source_name_for_clustering_ui = "Standard SBERT Embeddings"
    elif chosen_embedding_type == "TSDAE (from 'final_noisy_text')":
        embeddings_to_cluster_array = st.session_state['tsdae_embeddings']
        job_ids_for_clustering = st.session_state.get('tsdae_embedding_job_ids')
        source_name_for_clustering_ui = "TSDAE Embeddings"
    elif chosen_embedding_type == "SBERT O*NET Classified":
        if sbert_onet_embeddings_present and 'Job.ID' in st.session_state['data'].columns:
            # Extract embeddings and corresponding Job.IDs, ensuring alignment
            valid_sbert_onet_mask = st.session_state['data']['sbert_job_embedding_onet_classified'].notna()
            sbert_onet_emb_series_for_cluster = st.session_state['data'].loc[valid_sbert_onet_mask, 'sbert_job_embedding_onet_classified']
            if not sbert_onet_emb_series_for_cluster.empty:
                embeddings_to_cluster_array = np.array(sbert_onet_emb_series_for_cluster.tolist())
                job_ids_for_clustering = st.session_state['data'].loc[valid_sbert_onet_mask, 'Job.ID'].tolist()
                source_name_for_clustering_ui = "SBERT O*NET Classified Embeddings"
                if len(job_ids_for_clustering) != embeddings_to_cluster_array.shape[0]:
                    st.error("Mismatch in SBERT O*NET classified embeddings count and Job IDs count. Cannot proceed with clustering."); return
            else:
                st.warning("No valid SBERT O*NET classified embeddings found in the data for clustering."); return
        else:
            st.warning("SBERT O*NET classified embeddings are not available or Job.ID column is missing."); return

    if embeddings_to_cluster_array is not None and job_ids_for_clustering:
        st.info(f"Using: {source_name_for_clustering_ui} ({len(job_ids_for_clustering)} items available for clustering)")
        
        max_k_value_slider = embeddings_to_cluster_array.shape[0]
        if max_k_value_slider < 2: 
            st.error("Need at least 2 embedded items to perform meaningful clustering."); return
        
        num_clusters_for_kmeans = st.slider(
            "Number of Clusters (K) for K-Means:", 
            min_value=2, # K-Means usually for K>=2
            max_value=min(50, max_k_value_slider), # Cap max K for slider usability
            value=min(N_CLUSTERS, max_k_value_slider), # Default K
            key="kmeans_k_slider_main"
        )
        
        if st.button(f"Run K-Means Clustering (K={num_clusters_for_kmeans}) on {source_name_for_clustering_ui}", key="run_kmeans_clustering_button"):
            assigned_cluster_labels = cluster_embeddings_with_progress(embeddings_to_cluster_array, num_clusters_for_kmeans)
            if assigned_cluster_labels is not None:
                if len(job_ids_for_clustering) == len(assigned_cluster_labels):
                    # Create a DataFrame with Job.ID and cluster label
                    cluster_info_temp_df = pd.DataFrame({'Job.ID': job_ids_for_clustering, 'cluster_temp_col': assigned_cluster_labels})
                    
                    # Merge this cluster information back into the main session_state['data']
                    data_df_with_new_clusters = st.session_state['data'].copy()
                    # Ensure Job.ID types are consistent for merging (string)
                    data_df_with_new_clusters['Job.ID'] = data_df_with_new_clusters['Job.ID'].astype(str)
                    cluster_info_temp_df['Job.ID'] = cluster_info_temp_df['Job.ID'].astype(str)

                    # Remove old 'cluster' column if it exists to avoid conflicts
                    if 'cluster' in data_df_with_new_clusters.columns:
                        data_df_with_new_clusters = data_df_with_new_clusters.drop(columns=['cluster'])
                    
                    # Perform merge. Use left merge to keep all jobs in main data, adding clusters where available.
                    # Drop duplicates from cluster_info_temp_df just in case, though Job.IDs should be unique from embedding source.
                    cluster_info_temp_df = cluster_info_temp_df.drop_duplicates(subset=['Job.ID'], keep='first')
                    st.session_state['data'] = pd.merge(data_df_with_new_clusters, cluster_info_temp_df, on='Job.ID', how='left')
                    
                    # Rename the temporary cluster column to 'cluster'
                    if 'cluster_temp_col' in st.session_state['data'].columns:
                        st.session_state['data'].rename(columns={'cluster_temp_col': 'cluster'}, inplace=True)
                    st.success(f"K-Means clustering complete. 'cluster' column updated in the main dataset for {len(job_ids_for_clustering)} jobs based on {source_name_for_clustering_ui}.")
                else:
                    st.error("Mismatch between the number of Job IDs and the number of generated cluster labels. Cannot merge cluster information.")
            else:
                st.error("K-Means clustering algorithm failed to return labels.")

    # Visualization of clusters if 'cluster' column exists and matches current embedding source
    current_main_data = st.session_state.get('data', pd.DataFrame())
    if 'cluster' in current_main_data.columns and \
      embeddings_to_cluster_array is not None and job_ids_for_clustering is not None and \
      len(job_ids_for_clustering) == embeddings_to_cluster_array.shape[0]: # Ensure alignment

        st.subheader(f"2D Visualization of Clustered {source_name_for_clustering_ui} (PCA)")
        if embeddings_to_cluster_array.shape[0] >= 2: # PCA needs at least 2 samples
            try:
                with st.spinner("Generating PCA visualization for clusters..."):
                    pca_for_cluster_plot = PCA(n_components=2, random_state=42)
                    reduced_embeddings_for_plot = pca_for_cluster_plot.fit_transform(embeddings_to_cluster_array)
                    
                    # Create DataFrame for plotting PCA results
                    plot_df_for_clusters = pd.DataFrame(reduced_embeddings_for_plot, columns=['PC1', 'PC2'])
                    plot_df_for_clusters['Job.ID'] = job_ids_for_clustering # Add Job.IDs
                    
                    # Merge with relevant columns from current_main_data for hover info
                    current_main_data['Job.ID'] = current_main_data['Job.ID'].astype(str) # Ensure type consistency
                    plot_df_for_clusters['Job.ID'] = plot_df_for_clusters['Job.ID'].astype(str)

                    # Select columns for merging: Job.ID, Title, cluster, and optionally text for hover
                    cols_for_plot_merge = ['Job.ID', 'cluster'] # Cluster is essential
                    if 'Title' in current_main_data.columns: cols_for_plot_merge.append('Title')
                    hover_text_col_cluster = 'combined_jobs' if 'combined_jobs' in current_main_data.columns else \
                                             ('Job.Description' if 'Job.Description' in current_main_data.columns else None)
                    if hover_text_col_cluster: cols_for_plot_merge.append(hover_text_col_cluster)
                    if 'onet_category' in current_main_data.columns: cols_for_plot_merge.append('onet_category') # Add O*NET if available

                    # Fetch data for merge, ensuring only relevant Job.IDs and unique entries
                    data_to_merge_for_plot = current_main_data[current_main_data['Job.ID'].isin(job_ids_for_clustering)]
                    actual_cols_for_plot_merge = [col for col in cols_for_plot_merge if col in data_to_merge_for_plot.columns]
                    data_to_merge_for_plot = data_to_merge_for_plot[actual_cols_for_plot_merge].drop_duplicates(subset=['Job.ID'])
                    
                    plot_df_for_clusters = pd.merge(plot_df_for_clusters, data_to_merge_for_plot, on='Job.ID', how='left')
                    
                    if 'cluster' in plot_df_for_clusters.columns and not plot_df_for_clusters['cluster'].isnull().all():
                        plot_df_for_clusters['cluster'] = plot_df_for_clusters['cluster'].astype('category') # For Plotly color mapping
                        
                        hover_data_config_cluster_plot = {'Job.ID': True, 'cluster': True, 'PC1': False, 'PC2': False}
                        if hover_text_col_cluster and hover_text_col_cluster in plot_df_for_clusters.columns:
                            hover_data_config_cluster_plot[hover_text_col_cluster] = True
                        if 'onet_category' in plot_df_for_clusters.columns:
                            hover_data_config_cluster_plot['onet_category'] = True
                        
                        if not plot_df_for_clusters.empty and 'Title' in plot_df_for_clusters.columns:
                            fig_cluster_pca_plot = px.scatter(
                                plot_df_for_clusters, x='PC1', y='PC2', color='cluster',
                                hover_name='Title', hover_data=hover_data_config_cluster_plot,
                                title=f'2D PCA of Clustered {source_name_for_clustering_ui}'
                            )
                            st.plotly_chart(fig_cluster_pca_plot, use_container_width=True)
                        else:
                            st.warning("Could not generate cluster visualization. Required data for plotting (e.g., 'Title', valid 'cluster' info) is missing after merge.")
                    else:
                        st.warning("Cluster information ('cluster' column) is missing or all NaN in the plot DataFrame. Cannot visualize clusters.")
            except Exception as e_pca_cluster_plot:
                st.error(f"Error during PCA visualization of clusters: {e_pca_cluster_plot}")
        else:
            st.warning("Not enough data points (need at least 2) for PCA visualization of clusters.")
            
    elif 'cluster' in current_main_data.columns: # If 'cluster' column exists but conditions for plotting not met
        st.info("Cluster information ('cluster' column) is present in the main dataset. To re-visualize with current settings, please select an embedding type and run K-Means clustering again.")
    else: # No 'cluster' column
        st.info("No cluster information available to visualize. Please select an embedding type and run K-Means clustering first.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs in PDF or DOCX format (maximum 5 files at a time). Each CV will be processed to extract text, then preprocessed, and finally embedded using the standard SBERT model.")

    # File uploader allows multiple files
    uploaded_cv_files_list = st.file_uploader(
        "Choose CV files to upload:", 
        type=["pdf", "docx"], 
        accept_multiple_files=True, 
        key="cv_file_uploader_widget"
    )

    if uploaded_cv_files_list:
        if len(uploaded_cv_files_list) > 5:
            st.warning(f"You uploaded {len(uploaded_cv_files_list)} files. Processing only the first 5 CVs as per limit.")
            uploaded_cv_files_list = uploaded_cv_files_list[:5] # Limit to first 5
        
        st.write(f"Number of CVs to process: {len(uploaded_cv_files_list)}")
        for cv_file_item in uploaded_cv_files_list:
            st.write(f"- {cv_file_item.name}")

        if st.button("Process Uploaded CVs", key="process_uploaded_cvs_button"):
            processed_cv_data_batch_list = [] # To store dicts for each processed CV
            sbert_model_for_cv_processing = load_bert_model() # Load the standard SBERT model (cached)
            
            if not sbert_model_for_cv_processing:
                st.error("SBERT model failed to load. Cannot process CVs for embedding."); return

            with st.spinner("Processing uploaded CVs... This includes text extraction, preprocessing, and embedding."):
                for i, cv_file_to_process in enumerate(uploaded_cv_files_list):
                    st.markdown(f"--- \n Processing CV {i+1}: **{cv_file_to_process.name}**")
                    original_text_from_cv, processed_text_for_cv, cv_embedding_vector = "", "", None
                    
                    try:
                        file_extension = cv_file_to_process.name.split(".")[-1].lower()
                        if file_extension == "pdf":
                            original_text_from_cv = extract_text_from_pdf(cv_file_to_process)
                        elif file_extension == "docx":
                            original_text_from_cv = extract_text_from_docx(cv_file_to_process)
                        else:
                            st.warning(f"Unsupported file type for '{cv_file_to_process.name}'. Skipping.")
                            continue # Skip to next file

                        if original_text_from_cv and original_text_from_cv.strip():
                            # Preprocess the extracted text (standard preprocessing)
                            processed_text_for_cv = preprocess_text(original_text_from_cv)
                            
                            if processed_text_for_cv and processed_text_for_cv.strip():
                                # Generate embedding for the processed CV text
                                # generate_embeddings_with_progress expects a list of texts
                                cv_embedding_array = generate_embeddings_with_progress(sbert_model_for_cv_processing, [processed_text_for_cv])
                                if cv_embedding_array is not None and cv_embedding_array.size > 0:
                                    cv_embedding_vector = cv_embedding_array[0] # Get the first (and only) embedding
                            else:
                                st.warning(f"Processed text for '{cv_file_to_process.name}' is empty. No embedding generated.")
                        else:
                            st.warning(f"Extracted text from '{cv_file_to_process.name}' is empty. Cannot process further.")
                    
                    except Exception as e_cv_proc:
                        st.error(f"An error occurred while processing CV '{cv_file_to_process.name}': {e_cv_proc}")
                    
                    finally: # Store results even if some steps failed (embedding might be None)
                        processed_cv_data_batch_list.append({
                            'filename': cv_file_to_process.name,
                            'original_text': original_text_from_cv or "", # Ensure not None
                            'processed_text': processed_text_for_cv or "", # Ensure not None
                            'embedding': cv_embedding_vector # This can be None
                        })
                        if cv_embedding_vector is not None:
                            st.success(f"Successfully processed and embedded: {cv_file_to_process.name}")
                        else:
                            st.error(f"Failed to generate embedding for: {cv_file_to_process.name} (original/processed text might be empty or error occurred).")
            
            st.session_state['uploaded_cvs_data'] = processed_cv_data_batch_list # Store all processed CV data
            st.success(f"CV batch processing complete. {len(processed_cv_data_batch_list)} CVs attempted.")

    # Display currently stored/processed CVs
    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored Processed CVs:")
        for i, cv_data_item in enumerate(st.session_state['uploaded_cvs_data']):
            cv_filename_display = cv_data_item.get('filename', f'CV {i+1} (Unknown Filename)')
            with st.expander(f"Details for CV: {cv_filename_display}"):
                st.text_area(f"Original Extracted Text:", value=cv_data_item.get('original_text','(No text extracted or available)'), height=100, disabled=True, key=f"display_cv_original_text_{i}")
                st.text_area(f"Preprocessed Text:", value=cv_data_item.get('processed_text','(No text processed or available)'), height=100, disabled=True, key=f"display_cv_processed_text_{i}")
                if cv_data_item.get('embedding') is not None and cv_data_item.get('embedding').size > 0:
                    st.success("Embedding generated and stored for this CV.")
                    st.write(f"Embedding shape: {cv_data_item.get('embedding').shape}")
                else:
                    st.warning("Embedding is missing or not generated for this CV.")
    else:
        st.info("No CVs have been uploaded and processed yet in this session.")
    return


def job_recommendation_page():
    st.header("Job Recommendation for Uploaded CVs")
    st.write("Generates job recommendations for each processed CV by comparing its embedding with job embeddings from the selected source.")

    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("No CVs have been uploaded and processed. Please go to the 'Upload & Process CVs' page first.")
        if st.button("Go to Upload CVs Page"): st.session_state.current_page = "Upload & Process CVs"; st.experimental_rerun()
        return
    
    main_job_data = st.session_state.get('data')
    if main_job_data is None or main_job_data.empty:
        st.error("Main job dataset is not available or empty. Please load data on the 'Home' page and ensure it's processed if needed.")
        if st.button("Go to Home Page"): st.session_state.current_page = "Home"; st.experimental_rerun()
        return
    
    job_embeddings_for_recommendation, source_message_for_recommendation_ui, job_embedding_ids_for_recommendation = None, "", None
    
    # Determine available job embedding types for recommendation
    recommendation_embedding_options = []
    if st.session_state.get('job_text_embeddings', np.array([])).size > 0 and st.session_state.get('job_text_embedding_job_ids'):
        recommendation_embedding_options.append("Standard SBERT (from 'processed_text')")
    if st.session_state.get('tsdae_embeddings', np.array([])).size > 0 and st.session_state.get('tsdae_embedding_job_ids'):
        recommendation_embedding_options.append("TSDAE (from 'final_noisy_text')")
    
    sbert_onet_rec_emb_available = False
    if 'sbert_job_embedding_onet_classified' in main_job_data.columns:
        sbert_onet_emb_series_rec = main_job_data['sbert_job_embedding_onet_classified'].dropna()
        if not sbert_onet_emb_series_rec.empty and isinstance(sbert_onet_emb_series_rec.iloc[0], np.ndarray):
            sbert_onet_rec_emb_available = True
            recommendation_embedding_options.append("SBERT O*NET Classified")

    if not recommendation_embedding_options:
        st.warning("No job embeddings are available for making recommendations. Please generate them on the respective embedding pages.")
        return

    chosen_job_embedding_type_for_rec = st.radio(
        "Select Job Embeddings to Use for Recommendations:", 
        options=recommendation_embedding_options, 
        key="recommendation_job_embedding_type_radio", 
        horizontal=True
    )

    # Load selected job embeddings and their IDs
    if chosen_job_embedding_type_for_rec == "Standard SBERT (from 'processed_text')":
        job_embeddings_for_recommendation = st.session_state['job_text_embeddings']
        job_embedding_ids_for_recommendation = st.session_state.get('job_text_embedding_job_ids')
        source_message_for_recommendation_ui = "Using Standard SBERT job embeddings (derived from 'processed_text')."
    elif chosen_job_embedding_type_for_rec == "TSDAE (from 'final_noisy_text')":
        job_embeddings_for_recommendation = st.session_state['tsdae_embeddings']
        job_embedding_ids_for_recommendation = st.session_state.get('tsdae_embedding_job_ids')
        source_message_for_recommendation_ui = "Using TSDAE job embeddings (derived from 'final_noisy_text')."
    elif chosen_job_embedding_type_for_rec == "SBERT O*NET Classified":
        if sbert_onet_rec_emb_available and 'Job.ID' in main_job_data.columns:
            valid_sbert_onet_mask_rec = main_job_data['sbert_job_embedding_onet_classified'].notna()
            sbert_onet_emb_series_for_rec = main_job_data.loc[valid_sbert_onet_mask_rec, 'sbert_job_embedding_onet_classified']
            if not sbert_onet_emb_series_for_rec.empty:
                job_embeddings_for_recommendation = np.array(sbert_onet_emb_series_for_rec.tolist())
                job_embedding_ids_for_recommendation = main_job_data.loc[valid_sbert_onet_mask_rec, 'Job.ID'].tolist()
                source_message_for_recommendation_ui = "Using SBERT O*NET Classified job embeddings."
                if len(job_embedding_ids_for_recommendation) != job_embeddings_for_recommendation.shape[0]:
                    st.error("Mismatch in SBERT O*NET classified embeddings and Job IDs for recommendations."); return
            else: st.warning("No valid SBERT O*NET classified embeddings found for recommendations."); return
        else: st.warning("SBERT O*NET classified embeddings not available or Job.ID column missing for recommendations."); return
    
    st.info(source_message_for_recommendation_ui)

    if job_embeddings_for_recommendation is None or not job_embedding_ids_for_recommendation:
        st.error("Selected job embeddings or their corresponding Job IDs are missing or invalid. Cannot proceed with recommendations."); return
        
    # --- Align job embeddings with job details from main_job_data ---
    # This ensures that the job details displayed correspond correctly to the embeddings used.
    # Create a DataFrame from job_embedding_ids_for_recommendation to preserve their order
    job_ids_df_for_alignment = pd.DataFrame({
        'Job.ID': job_embedding_ids_for_recommendation, 
        'emb_order': np.arange(len(job_embedding_ids_for_recommendation)) # Original order of embeddings
    })
    job_ids_df_for_alignment['Job.ID'] = job_ids_df_for_alignment['Job.ID'].astype(str) # Ensure string type for merge

    # Columns to fetch from main_job_data for displaying recommendations
    cols_to_fetch_for_display = list(set(['Job.ID', 'Title'] + JOB_DETAIL_FEATURES_TO_DISPLAY)) # JOB_DETAIL_FEATURES_TO_DISPLAY includes Title
    if 'cluster' in main_job_data.columns: cols_to_fetch_for_display.append('cluster')
    if 'onet_category' in main_job_data.columns: cols_to_fetch_for_display.append('onet_category')
    if 'onet_soc_code' in main_job_data.columns: cols_to_fetch_for_display.append('onet_soc_code')
    
    # Ensure all columns to fetch actually exist in main_job_data
    actual_cols_to_fetch_for_display = [col for col in cols_to_fetch_for_display if col in main_job_data.columns]
    
    main_job_data_copy_for_rec_alignment = main_job_data.copy()
    main_job_data_copy_for_rec_alignment['Job.ID'] = main_job_data_copy_for_rec_alignment['Job.ID'].astype(str) # Ensure string type
    
    # Merge to get details for jobs that have embeddings, preserving embedding order
    aligned_job_details_for_rec_df = pd.merge(
        job_ids_df_for_alignment, 
        main_job_data_copy_for_rec_alignment[actual_cols_to_fetch_for_display].drop_duplicates(subset=['Job.ID']), # Avoid duplicate Job.IDs from main data
        on='Job.ID', 
        how='left' # Keep all jobs that have embeddings
    )
    # Sort by original embedding order and reset index
    aligned_job_details_for_rec_df = aligned_job_details_for_rec_df.sort_values(by='emb_order').reset_index(drop=True)

    # Critical check: if merge resulted in fewer rows than embeddings, some Job.IDs were not found in main_job_data
    if len(aligned_job_details_for_rec_df) != len(job_embedding_ids_for_recommendation) or \
       aligned_job_details_for_rec_df['Job.ID'].isnull().any(): # Also check if any Job.ID became NaN after merge (should not happen with left merge on Job.ID)
        st.warning(f"Warning: Could not find details for all {len(job_embedding_ids_for_recommendation)} job embeddings in the main dataset. "
                   f"Found details for {len(aligned_job_details_for_rec_df.dropna(subset=['Title']))} jobs (based on Title presence). " # Example check
                   "Recommendations will be based on the subset of jobs for which details could be aligned.")
        
        # Filter embeddings and their IDs to only those successfully aligned
        job_ids_successfully_aligned = aligned_job_details_for_rec_df.dropna(subset=['Job.ID'])['Job.ID'].tolist() # Get Job.IDs that were found
        
        # Find indices in the original job_embedding_ids_for_recommendation that match the successfully aligned ones
        indices_to_keep_after_alignment = [
            i for i, job_id_orig in enumerate(job_embedding_ids_for_recommendation) 
            if job_id_orig in job_ids_successfully_aligned
        ]
        
        if not indices_to_keep_after_alignment:
            st.error("No job embeddings could be matched with job details after alignment. Cannot proceed with recommendations.")
            return
            
        job_embeddings_for_recommendation = job_embeddings_for_recommendation[indices_to_keep_after_alignment]
        job_embedding_ids_for_recommendation = [job_embedding_ids_for_recommendation[i] for i in indices_to_keep_after_alignment]
        
        # Re-create aligned_job_details_for_rec_df with the now perfectly matched subset
        job_ids_df_for_alignment = pd.DataFrame({
            'Job.ID': job_embedding_ids_for_recommendation, 
            'emb_order': np.arange(len(job_embedding_ids_for_recommendation))
        })
        job_ids_df_for_alignment['Job.ID'] = job_ids_df_for_alignment['Job.ID'].astype(str)
        aligned_job_details_for_rec_df = pd.merge(
            job_ids_df_for_alignment, 
            main_job_data_copy_for_rec_alignment[actual_cols_to_fetch_for_display].drop_duplicates(subset=['Job.ID']),
            on='Job.ID', how='left' 
        ).sort_values(by='emb_order').reset_index(drop=True)

        if len(aligned_job_details_for_rec_df) != job_embeddings_for_recommendation.shape[0]:
             st.error("Critical error: Job details and embeddings count mismatch even after re-alignment. Please check data integrity.")
             return

    num_recommendations_per_cv = st.slider("Number of recommendations per CV:", 1, 20, 5, key="num_recommendations_slider")

    # Initialize/ensure structure for storing recommendations for annotation
    if 'all_recommendations_for_annotation' not in st.session_state:
        st.session_state.all_recommendations_for_annotation = {} # {cv_filename: {annotator_slot: recommendations_df}}

    st.markdown("---")
    for cv_data_item_rec in st.session_state['uploaded_cvs_data']:
        cv_filename_rec = cv_data_item_rec.get('filename', 'Unknown CV')
        cv_embedding_vector_rec = cv_data_item_rec.get('embedding')

        st.subheader(f"Recommendations for CV: {cv_filename_rec}")
        if cv_embedding_vector_rec is not None and cv_embedding_vector_rec.size > 0:
            if cv_embedding_vector_rec.ndim == 1: # Ensure CV embedding is 2D for cosine_similarity
                cv_embedding_vector_rec = cv_embedding_vector_rec.reshape(1, -1) 

            # Calculate cosine similarities between CV embedding and all job embeddings
            similarities_array = cosine_similarity(cv_embedding_vector_rec, job_embeddings_for_recommendation)[0]
            # Get indices of top N similar jobs
            top_n_indices = np.argsort(similarities_array)[::-1][:num_recommendations_per_cv]
            
            recommendations_for_this_cv_list = []
            for rank, job_emb_idx in enumerate(top_n_indices):
                # job_emb_idx is the index in job_embeddings_for_recommendation and aligned_job_details_for_rec_df
                job_detail_series_rec = aligned_job_details_for_rec_df.iloc[job_emb_idx]
                
                # Construct recommendation data dictionary
                recommendation_entry = {
                    'Rank': rank + 1,
                    'Job.ID': job_detail_series_rec['Job.ID'],
                    'Similarity': similarities_array[job_emb_idx]
                }
                # Add other display features from JOB_DETAIL_FEATURES_TO_DISPLAY and special columns
                for feature_key in JOB_DETAIL_FEATURES_TO_DISPLAY + ['cluster', 'onet_category', 'onet_soc_code']:
                    if feature_key in job_detail_series_rec and pd.notna(job_detail_series_rec[feature_key]):
                         # Standardize display keys (e.g., 'onet_category' -> 'O*NET Category')
                        display_key = feature_key.replace('_', ' ').title() if 'onet' in feature_key else feature_key.title()
                        if feature_key == 'Job.Description': display_key = 'Job Description' # Keep as is
                        recommendation_entry[display_key] = job_detail_series_rec[feature_key]
                
                recommendations_for_this_cv_list.append(recommendation_entry)

            if recommendations_for_this_cv_list:
                recommendations_df_for_cv = pd.DataFrame(recommendations_for_this_cv_list)
                
                # Store these recommendations for annotation, keyed by CV filename and then by annotator slot
                if cv_filename_rec not in st.session_state.all_recommendations_for_annotation:
                    st.session_state.all_recommendations_for_annotation[cv_filename_rec] = {}
                for annotator_slot_key in ANNOTATORS: # Store for all potential annotator slots
                     st.session_state.all_recommendations_for_annotation[cv_filename_rec][annotator_slot_key] = recommendations_df_for_cv.copy()

                st.dataframe(recommendations_df_for_cv, use_container_width=True)
                # Display details in expanders
                for _, rec_row in recommendations_df_for_cv.iterrows():
                    expander_title = f"{rec_row['Rank']}. {rec_row.get('Title', 'N/A')} (Similarity: {rec_row['Similarity']:.4f})"
                    with st.expander(expander_title):
                        for col_name, col_value in rec_row.items():
                            if col_name not in ['Rank', 'Similarity', 'Title']: # Avoid re-displaying these
                                if col_name == 'Job Description':
                                    st.markdown(f"**{col_name}:**")
                                    st.caption(f"{str(col_value)[:500]}...") # Show snippet of description
                                else:
                                    st.write(f"**{col_name}:** {col_value}")
            else:
                st.info(f"No recommendations could be generated for CV '{cv_filename_rec}' (e.g., no similar jobs found).")
        else:
            st.warning(f"CV '{cv_filename_rec}' does not have a valid embedding. Cannot generate recommendations.")
        st.markdown("---") # Separator between CVs

    st.success("Job recommendation process complete for all uploaded and processed CVs.")
    if st.session_state.all_recommendations_for_annotation:
        st.info("Recommendations have been generated and stored. You can now proceed to the 'Annotation Input' page to rate them.")
    return


def annotation_input_page():
    st.header("Annotation Input: Rate Job Recommendation Relevance")
    st.write("Select your annotator role, choose a CV, and rate the relevance of each recommended job on a scale of 1 (Not Relevant) to 5 (Highly Relevant).")

    if not st.session_state.get('all_recommendations_for_annotation'):
        st.warning("No recommendations are available to annotate. Please generate recommendations on the 'Job Recommendation' page first.")
        if st.button("Go to Job Recommendation Page"): st.session_state.current_page = "Job Recommendation"; st.experimental_rerun()
        return
    
    if not ANNOTATORS: # Should not happen if ANNOTATORS constant is defined
        st.error("No annotator roles (ANNOTATORS list) are defined in the system. Cannot proceed with annotation."); return

    # --- Annotator Details Input (in Sidebar) ---
    st.sidebar.subheader("Annotator Setup (Current User)")
    st.sidebar.caption("Enter your details. This will be associated with your annotations.")
    
    # Dropdown to select which annotator slot's details to input/edit for the current user session
    st.session_state.current_annotator_slot_for_input = st.sidebar.selectbox(
        "Select Your Annotator Slot to Define/Edit Details:",
        options=ANNOTATORS,
        index=ANNOTATORS.index(st.session_state.current_annotator_slot_for_input) if st.session_state.current_annotator_slot_for_input in ANNOTATORS else 0,
        key="annotator_slot_selector_for_detail_input"
    )
    current_slot_for_detail_entry = st.session_state.current_annotator_slot_for_input

    st.sidebar.markdown(f"**Defining details for role: {current_slot_for_detail_entry}**")

    # Check if details for this slot have been saved by the current user/session
    if current_slot_for_detail_entry in st.session_state.annotators_saved_status:
        st.sidebar.success(f"Details for {current_slot_for_detail_entry} are saved for this session.")
        st.sidebar.write(f"Name: {st.session_state.annotator_details[current_slot_for_detail_entry]['actual_name']}")
        st.sidebar.write(f"Background: {st.session_state.annotator_details[current_slot_for_detail_entry]['profile_background']}")
        if st.sidebar.button(f"Edit Details for {current_slot_for_detail_entry}", key=f"edit_annotator_details_button_{current_slot_for_detail_entry}"):
            st.session_state.annotators_saved_status.remove(current_slot_for_detail_entry) # Allow re-entry
            st.experimental_rerun() # Rerun to show input fields again
    else:
        # Input fields for the selected annotator slot's details
        annotator_name_input = st.sidebar.text_input(
            f"Your Name (for role {current_slot_for_detail_entry}):",
            value=st.session_state.annotator_details.get(current_slot_for_detail_entry, {}).get('actual_name', ''),
            key=f"annotator_name_input_{current_slot_for_detail_entry}"
        )
        annotator_background_input = st.sidebar.text_area(
            f"Your Profile/Background (for role {current_slot_for_detail_entry}):",
            value=st.session_state.annotator_details.get(current_slot_for_detail_entry, {}).get('profile_background', ''),
            key=f"annotator_background_input_{current_slot_for_detail_entry}",
            height=100
        )
        if st.sidebar.button(f"Save My Details for {current_slot_for_detail_entry}", key=f"save_annotator_details_button_{current_slot_for_detail_entry}"):
            if annotator_name_input.strip() and annotator_background_input.strip():
                st.session_state.annotator_details[current_slot_for_detail_entry]['actual_name'] = annotator_name_input.strip()
                st.session_state.annotator_details[current_slot_for_detail_entry]['profile_background'] = annotator_background_input.strip()
                st.session_state.annotators_saved_status.add(current_slot_for_detail_entry) # Mark as saved for this session
                st.sidebar.success(f"Details saved for {current_slot_for_detail_entry}!")
                st.experimental_rerun() # Rerun to reflect saved status
            else:
                st.sidebar.error("Please provide both your name and background.")
    
    st.markdown("---") # Main page separator
    
    # --- Annotation Section (Main Page) ---
    # Annotator selects their designated slot for performing the actual annotations
    active_annotator_role_for_rating = st.selectbox(
        "Select Your Annotator Role for Rating Recommendations:",
        options=ANNOTATORS,
        key="active_annotator_role_selector_rating"
    )

    # Annotation can only proceed if details for the selected active role are saved
    if active_annotator_role_for_rating not in st.session_state.annotators_saved_status:
        st.warning(f"Details for your selected annotator role '{active_annotator_role_for_rating}' are not yet saved. "
                   "Please save them using the 'Annotator Setup' in the sidebar before proceeding with rating.")
        return 

    annotator_actual_name_for_rating = st.session_state.annotator_details[active_annotator_role_for_rating]['actual_name']
    annotator_background_for_rating = st.session_state.annotator_details[active_annotator_role_for_rating]['profile_background']
    st.success(f"You are annotating as: **{annotator_actual_name_for_rating}** (Role: {active_annotator_role_for_rating})")

    cv_filenames_with_recommendations = list(st.session_state.all_recommendations_for_annotation.keys())
    if not cv_filenames_with_recommendations:
        st.info("No CVs with recommendations found in the system currently.")
        return

    selected_cv_filename_for_annotation = st.selectbox(
        "Select CV to Annotate Recommendations For:",
        options=cv_filenames_with_recommendations,
        key="cv_filename_selector_for_annotation"
    )

    if selected_cv_filename_for_annotation:
        # Retrieve recommendations for the selected CV (recommendations are same for all annotator slots per CV)
        # We use the first annotator slot's data as representative, as they should be identical.
        # Or, ensure that all_recommendations_for_annotation[cv_filename] directly holds the df if not per-slot.
        # Current structure: all_recommendations_for_annotation[cv_filename][annotator_slot] = recommendations_df
        # So, we can pick any slot's data for the CV, e.g., ANNOTATORS[0]
        
        recommendations_to_annotate_df = None
        if selected_cv_filename_for_annotation in st.session_state.all_recommendations_for_annotation and \
           st.session_state.all_recommendations_for_annotation[selected_cv_filename_for_annotation]:
            # Get the recommendations; they are stored per CV, duplicated per annotator slot initially.
            # So, picking the first annotator slot's recs for this CV is fine.
            first_annotator_slot_key = next(iter(st.session_state.all_recommendations_for_annotation[selected_cv_filename_for_annotation]))
            recommendations_to_annotate_df = st.session_state.all_recommendations_for_annotation[selected_cv_filename_for_annotation][first_annotator_slot_key]

        if recommendations_to_annotate_df is None or recommendations_to_annotate_df.empty:
            st.info(f"No recommendations found for CV '{selected_cv_filename_for_annotation}'. This should not happen if CV is in selector.")
            return

        st.subheader(f"Annotating recommendations for CV: {selected_cv_filename_for_annotation}")
        
        # Check for existing annotations for this CV by this annotator role to pre-fill ratings or indicate completion
        existing_annotations_main_df = st.session_state.collected_annotations
        previous_annotations_for_this_cv_annotator_role = pd.DataFrame() # Initialize as empty
        if not existing_annotations_main_df.empty:
             previous_annotations_for_this_cv_annotator_role = existing_annotations_main_df[
                (existing_annotations_main_df['CV_Filename'] == selected_cv_filename_for_annotation) &
                (existing_annotations_main_df['Annotator_Slot'] == active_annotator_role_for_rating) # Match by Annotator_Slot
            ]

        # Use a form for submitting all ratings for a CV at once
        with st.form(key=f"annotation_rating_form_{selected_cv_filename_for_annotation}_{active_annotator_role_for_rating}"):
            ratings_data_for_submission = [] # Store dicts of {job_id: rating, ...} for this form submission
            
            for index, rec_row_to_annotate in recommendations_to_annotate_df.iterrows():
                job_id_annotate = rec_row_to_annotate['Job.ID']
                job_title_annotate = rec_row_to_annotate.get('Title', 'N/A')
                
                st.markdown(f"---") # Separator for each job
                st.markdown(f"**Recommended Job Title:** {job_title_annotate} (Job ID: {job_id_annotate})")
                # Display some key details of the job for context
                if 'Company' in rec_row_to_annotate: st.write(f"Company: {rec_row_to_annotate['Company']}")
                if 'Job Description' in rec_row_to_annotate: 
                    st.caption(f"Description Snippet: {str(rec_row_to_annotate['Job Description'])[:200]}...")

                # Check if this job was previously rated by this annotator for this CV
                default_rating_value = 3 # Default slider value
                if not previous_annotations_for_this_cv_annotator_role.empty:
                    matched_previous_rating_series = previous_annotations_for_this_cv_annotator_role[
                        previous_annotations_for_this_cv_annotator_role['Job_ID'] == job_id_annotate
                    ]
                    if not matched_previous_rating_series.empty:
                        default_rating_value = int(matched_previous_rating_series['Relevance_Score'].iloc[0])
                
                relevance_score_input = st.slider(
                    label=f"Your Relevance Rating for '{job_title_annotate.strip()[:30]}...' (Job ID: {job_id_annotate})",
                    min_value=1, max_value=5, value=default_rating_value,
                    help="1=Not Relevant, 2=Slightly Relevant, 3=Moderately Relevant, 4=Relevant, 5=Highly Relevant",
                    key=f"relevance_score_slider_{selected_cv_filename_for_annotation}_{active_annotator_role_for_rating}_{job_id_annotate}"
                )
                ratings_data_for_submission.append({
                    'CV_Filename': selected_cv_filename_for_annotation,
                    'Annotator_Slot': active_annotator_role_for_rating, 
                    'Annotator_Name': annotator_actual_name_for_rating,
                    'Annotator_Background': annotator_background_for_rating,
                    'Job_ID': job_id_annotate,
                    'Job_Title': job_title_annotate, # Store title for easier review later
                    'Relevance_Score': relevance_score_input,
                    'Similarity_Score': rec_row_to_annotate['Similarity'] # Store original similarity for evaluation
                })

            submitted_all_annotations_for_cv = st.form_submit_button("Submit All Ratings for This CV")

            if submitted_all_annotations_for_cv:
                newly_submitted_annotations_df = pd.DataFrame(ratings_data_for_submission)
                
                # Remove any previous annotations for this CV by this annotator role before adding the new/updated ones
                # This ensures that submitting again updates the ratings rather than creating duplicates.
                st.session_state.collected_annotations = st.session_state.collected_annotations[
                    ~((st.session_state.collected_annotations['CV_Filename'] == selected_cv_filename_for_annotation) &
                      (st.session_state.collected_annotations['Annotator_Slot'] == active_annotator_role_for_rating))
                ]
                
                # Concatenate the new annotations
                st.session_state.collected_annotations = pd.concat(
                    [st.session_state.collected_annotations, newly_submitted_annotations_df],
                    ignore_index=True # Reset index for the combined DataFrame
                )
                st.success(f"Your ratings for CV '{selected_cv_filename_for_annotation}' (as {active_annotator_role_for_rating}) have been submitted successfully!")
                st.info("You can now select another CV or annotator role, or proceed to the 'Evaluation' page to see aggregated results.")
                # Optionally, could clear ratings_data_for_submission or disable form elements after submission if needed for UX.
    
    # Display a preview of all collected annotations so far (e.g., tail)
    if not st.session_state.collected_annotations.empty:
        st.subheader("Current Batch of All Collected Annotations (Last 5 Entries)")
        st.dataframe(st.session_state.collected_annotations.tail())
    return

def evaluation_page():
    st.header("Evaluation of Annotations")
    st.write("This page reviews all collected annotations, displays summary statistics, and provides example evaluation metrics like NDCG.")

    all_collected_annotations_df = st.session_state.get('collected_annotations', pd.DataFrame())

    if all_collected_annotations_df.empty:
        st.warning("No annotations have been collected yet. Please go to the 'Annotation Input' page to rate recommendations.")
        if st.button("Go to Annotation Input Page"): st.session_state.current_page = "Annotation Input"; st.experimental_rerun()
        return

    st.subheader("All Collected Annotations Data")
    st.dataframe(all_collected_annotations_df, use_container_width=True)

    # --- Display Annotator Details Used ---
    st.subheader("Annotator Details (from Saved Profiles)")
    annotator_profiles_info_list = []
    # Iterate through annotator_details, show only those whose details were saved (i.e., name is present)
    for slot_key, details_dict in st.session_state.annotator_details.items():
        if details_dict.get('actual_name', '').strip(): # Check if actual_name is not empty
            annotator_profiles_info_list.append({
                "Annotator Role (Slot)": slot_key,
                "Name": details_dict['actual_name'],
                "Profile/Background": details_dict['profile_background']
            })
    if annotator_profiles_info_list:
        st.table(pd.DataFrame(annotator_profiles_info_list))
    else:
        st.info("No annotator profile details have been saved yet in the 'Annotation Input' page sidebar.")

    st.subheader("Annotation Statistics")
    # Average relevance score per CV
    avg_relevance_per_cv_df = all_collected_annotations_df.groupby('CV_Filename')['Relevance_Score'].mean().reset_index()
    avg_relevance_per_cv_df.rename(columns={'Relevance_Score': 'Average Relevance Score'}, inplace=True)
    st.write("Average Relevance Score per CV (across all annotators):")
    st.dataframe(avg_relevance_per_cv_df)

    # Average relevance score and count per Annotator Role (Slot)
    avg_relevance_per_annotator_role_df = all_collected_annotations_df.groupby(['Annotator_Slot', 'Annotator_Name'])['Relevance_Score'].agg(['mean', 'count']).reset_index()
    avg_relevance_per_annotator_role_df.rename(columns={'mean': 'Average Relevance Score', 'count': 'Number of Annotations'}, inplace=True)
    st.write("Average Relevance Score and Annotation Count per Annotator Role:")
    st.dataframe(avg_relevance_per_annotator_role_df)

    # Distribution of all Relevance Scores
    st.write("Overall Distribution of Relevance Scores (All Annotations):")
    if 'Relevance_Score' in all_collected_annotations_df.columns:
        fig_relevance_score_distribution = px.histogram(
            all_collected_annotations_df, x='Relevance_Score', nbins=5, # Bins for 1-5 scale
            title="Overall Relevance Score Distribution"
        )
        st.plotly_chart(fig_relevance_score_distribution, use_container_width=True)
    else:
        st.warning("Column 'Relevance_Score' not found in annotations data.")


    # --- NDCG Calculation Example ---
    # NDCG compares the system's ranking (based on similarity scores) against the ideal ranking (annotator's relevance scores).
    st.subheader("NDCG Scores (Normalized Discounted Cumulative Gain - Example per CV & Annotator)")
    st.caption("NDCG measures ranking quality. Higher is better. Calculated for items rated by each annotator for each CV.")
    
    ndcg_scores_results_list = []
    # Group by CV and Annotator Slot to calculate NDCG for each set of annotations
    for group_keys, group_data_df in all_collected_annotations_df.groupby(['CV_Filename', 'Annotator_Slot']):
        cv_filename_ndcg, annotator_slot_ndcg = group_keys
        
        # True relevance scores from the annotator for this group
        true_relevance_scores_ndcg = group_data_df['Relevance_Score'].values.reshape(1, -1) # Needs to be 2D for ndcg_score
        
        # Predicted scores (system's similarity scores for these recommended items)
        # The items were originally recommended (and thus ranked) by their similarity.
        predicted_similarity_scores_ndcg = group_data_df['Similarity_Score'].values.reshape(1, -1)

        if len(true_relevance_scores_ndcg[0]) > 1: # NDCG requires at least 2 items for meaningful calculation
            try:
                # k for NDCG@k: use number of items rated, or a fixed smaller k (e.g., 5 or 10)
                k_for_ndcg = min(len(true_relevance_scores_ndcg[0]), 10) # Example: NDCG@10 or less
                ndcg_value_calculated = ndcg_score(true_relevance_scores_ndcg, predicted_similarity_scores_ndcg, k=k_for_ndcg)
                ndcg_scores_results_list.append({
                    'CV_Filename': cv_filename_ndcg,
                    'Annotator_Slot': annotator_slot_ndcg,
                    f'NDCG@{k_for_ndcg}': ndcg_value_calculated,
                    'Number_of_Items_Rated': len(true_relevance_scores_ndcg[0])
                })
            except Exception as e_ndcg:
                st.warning(f"Could not calculate NDCG for CV '{cv_filename_ndcg}' by Annotator '{annotator_slot_ndcg}': {e_ndcg}")
                ndcg_scores_results_list.append({ # Log failure for this group
                    'CV_Filename': cv_filename_ndcg, 'Annotator_Slot': annotator_slot_ndcg, 
                    f'NDCG@N/A': "Error", 'Number_of_Items_Rated': len(true_relevance_scores_ndcg[0])
                })
        else: # Not enough items for NDCG
            ndcg_scores_results_list.append({
                'CV_Filename': cv_filename_ndcg, 'Annotator_Slot': annotator_slot_ndcg,
                f'NDCG@N/A': "N/A (Too few items)", 'Number_of_Items_Rated': len(true_relevance_scores_ndcg[0])
            })

    if ndcg_scores_results_list:
        ndcg_results_df = pd.DataFrame(ndcg_scores_results_list)
        st.write("Calculated NDCG Scores:")
        st.dataframe(ndcg_results_df)
        
        # Calculate and display average NDCG (find the relevant NDCG column dynamically)
        ndcg_value_column_name = next((col for col in ndcg_results_df.columns if 'NDCG@' in col and col != 'NDCG@N/A'), None)
        if ndcg_value_column_name:
            # Convert NDCG column to numeric, coercing errors (like "Error" string) to NaN
            ndcg_results_df[ndcg_value_column_name] = pd.to_numeric(ndcg_results_df[ndcg_value_column_name], errors='coerce')
            average_ndcg_score = ndcg_results_df[ndcg_value_column_name].mean() # .mean() ignores NaNs by default
            st.metric(f"Average {ndcg_value_column_name} (across valid calculations)", 
                      f"{average_ndcg_score:.4f}" if pd.notna(average_ndcg_score) else "N/A (No valid scores to average)")
        else:
            st.info("No valid NDCG score columns found for averaging (e.g., all calculations resulted in errors or too few items).")
    else:
        st.info("No NDCG scores were calculated (e.g., due to insufficient items per group, or errors during calculation).")

    # --- Download Annotations ---
    st.subheader("Download All Collected Annotations")
    # Convert DataFrame to CSV string
    csv_export_data = all_collected_annotations_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Annotations as CSV File",
        data=csv_export_data,
        file_name='collected_job_recommendation_annotations.csv',
        mime='text/csv',
        key='download_all_annotations_csv_button'
    )
    return


# --- Main App Structure ---
def main():
    st.set_page_config(page_title="Job Recommendation System", layout="wide", initial_sidebar_state="expanded")
    
    st.sidebar.title("📄 Navigation Menu")
    
    # Define pages and their corresponding functions
    # Order here dictates sidebar button order
    PAGES = {
        "🏠 Home": home_page,
        "⚙️ Data Preprocessing": preprocessing_page,
        "🌀 TSDAE Embeddings": tsdae_page,
        "🤖 Standard SBERT Embeddings": bert_model_page,
        "🔗 SBERT O*NET Classification": sbert_onet_classification_page,
        "🧩 Clustering": clustering_page,
        "📤 Upload & Process CVs": upload_cv_page,
        "💡 Job Recommendation": job_recommendation_page,
        "✍️ Annotation Input": annotation_input_page,
        "📊 Evaluation": evaluation_page
    }

    # Create buttons in the sidebar for navigation
    # st.session_state.current_page is initialized in default_session_state
    for page_display_name, page_func_name in PAGES.items():
        if st.sidebar.button(page_display_name, key=f"nav_button_{page_display_name}"):
            st.session_state.current_page = page_display_name # Store the display name
            st.experimental_rerun() # Force rerun for page change

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Currently Viewing: {st.session_state.current_page}**")

    # Call the function for the current page
    page_function_to_call = PAGES.get(st.session_state.current_page)
    if page_function_to_call:
        page_function_to_call()
    else: # Should not happen if current_page is always a key in PAGES
        st.error("Error: Selected page not found in the application structure.")
        st.session_state.current_page = "🏠 Home" # Default to Home on error
        st.experimental_rerun()


if __name__ == '__main__':
    main()
