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
    # Removed punkt_tab download as it's often problematic and not strictly necessary for basic tokenization.
    # If specific advanced tokenization features relying on it are needed, it can be re-added with more robust error handling.
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
# URL for your O*NET Occupation Data with categories
ONET_DATA_URL = 'https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv'


FEATURES_TO_COMBINE = [
    'Status', 'Title', 'Position', 'Company',
    'City', 'State.Name', 'Industry', 'Job.Description',
    'Employment.Type', 'Education.Required'
]
# Add 'category' and ONET specific fields to JOB_DETAIL_FEATURES_TO_DISPLAY
JOB_DETAIL_FEATURES_TO_DISPLAY = [
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type',
    'Position', 'Industry', 'Education.Required', 'State.Name', 'category',
    'onet_soc_code', 'onet_match_score'
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
        df_full['Job.ID'] = df_full['Job.ID'].astype(str) # Ensure Job.ID is string for consistency

        existing_features_to_combine = [col for col in features_to_combine_list if col in df_full.columns]
        missing_features_for_combine = [col for col in features_to_combine_list if col not in df_full.columns]
        if missing_features_for_combine:
            st.warning(f"The following features intended for combination were not found: {', '.join(missing_features_for_combine)}")

        cols_to_load_set = set(['Job.ID', 'Title']) # Ensure Title is loaded for merging categories
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

@st.cache_data(show_spinner='Loading and merging category data...')
def load_and_merge_categories(_main_df, category_data_url):
    """
    Loads job category data from a URL and merges it into the main DataFrame.
    """
    if _main_df is None:
        st.error("Main data is not loaded. Cannot merge categories.")
        return None
    if 'Title' not in _main_df.columns: # Title is crucial for the merge strategy
        st.error("Main data requires a 'Title' column for merging categories.")
        return _main_df

    try:
        df_categories = pd.read_csv(category_data_url)
        st.success(f"Successfully loaded category data from URL.")

        # Columns to merge from the category data file
        category_cols_to_merge = ['Title', 'category', 'onet_soc_code', 'onet_match_score']
        actual_category_cols_to_merge = [col for col in category_cols_to_merge if col in df_categories.columns]

        missing_cat_cols = set(category_cols_to_merge) - set(actual_category_cols_to_merge)
        if missing_cat_cols:
            st.warning(f"Category data from URL is missing some expected columns: {', '.join(missing_cat_cols)}. Proceeding with available ones: {', '.join(actual_category_cols_to_merge)}")

        if 'Title' not in actual_category_cols_to_merge:
            st.error("Category data from URL must contain a 'Title' column for merging.")
            return _main_df # Return original if 'Title' is missing in category file

        df_categories_subset = df_categories[actual_category_cols_to_merge].copy()
        df_categories_subset['Title'] = df_categories_subset['Title'].astype(str).str.strip()

        # Handle potential duplicate titles in category data by keeping the first entry
        df_categories_subset.drop_duplicates(subset=['Title'], keep='first', inplace=True)

        # Ensure 'Title' in main_df is also string and stripped for accurate merging
        _main_df['Title'] = _main_df['Title'].astype(str).str.strip()

        merged_df = pd.merge(_main_df, df_categories_subset, on='Title', how='left')

        # Fill NaN for category-specific columns that might result from the left merge
        if 'category' in merged_df.columns:
            merged_df['category'] = merged_df['category'].fillna('Uncategorized')
        else: # If 'category' column itself was missing from category file
            merged_df['category'] = 'Uncategorized'

        if 'onet_soc_code' in merged_df.columns:
            merged_df['onet_soc_code'] = merged_df['onet_soc_code'].fillna('N/A')
        elif 'onet_soc_code' in category_cols_to_merge: # If it was expected but not found
             merged_df['onet_soc_code'] = 'N/A'


        if 'onet_match_score' in merged_df.columns:
            merged_df['onet_match_score'] = merged_df['onet_match_score'].fillna(0.0)
        elif 'onet_match_score' in category_cols_to_merge: # If it was expected but not found
            merged_df['onet_match_score'] = 0.0


        st.success("Category data merged successfully.")
        if 'category' in merged_df.columns:
            st.write("Sample of categories after merge (first 5 rows):")
            st.dataframe(merged_df[['Job.ID', 'Title', 'category']].head())
        return merged_df

    except Exception as e:
        st.error(f"Error loading or merging category data from URL: {e}")
        return _main_df # Return original df on error

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
    text = re.sub(r'[^\w\s]', '', text) # Remove non-alphanumeric characters except whitespace
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    # Filter out non-alphanumeric tokens and stopwords
    filtered_words = [w for w in word_tokens if w.isalnum() and w not in stop_words]
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
    if method == 'a': # Random deletion
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0 and n > 0 : # Ensure at least one word is kept if input was not empty
            idx_to_keep = np.random.choice(n)
            keep_or_not[idx_to_keep] = True
        result_words = np.array(words)[keep_or_not].tolist()
    elif method in ('b', 'c'): # High-frequency word removal (b) or removal + shuffle (c)
        if word_freq_dict is None:
            # Fallback if word_freq_dict is not provided, though it's expected
            st.warning("word_freq_dict not provided for denoising method 'b' or 'c'. Performing random deletion instead.")
            return denoise_text(text_to_denoise, method='a', del_ratio=del_ratio)

        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0 and num_to_remove <= len(high_freq_indices):
                 to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result_words and words: # Ensure at least one word if original was not empty
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
            # 1. Remove punctuation and special symbols (more comprehensive)
            symbol_removed = text_content.translate(str.maketrans('', '', string.punctuation))
            symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed) # Keep alphanumeric and whitespace
            intermediate['symbol_removed'] = symbol_removed
            # 2. Case folding
            case_folded = symbol_removed.lower()
            intermediate['case_folded'] = case_folded
            # 3. Tokenization
            word_tokens_temp = word_tokenize(case_folded)
            intermediate['tokenized'] = " ".join(word_tokens_temp) # Store as string for display
            # 4. Stopword removal (on alphanumeric tokens)
            stop_words_temp = set(stopwords.words('english'))
            valid_tokens_for_stop_stem = [w for w in word_tokens_temp if w.isalnum()] # Process only alphanumeric tokens
            filtered_temp = [w for w in valid_tokens_for_stop_stem if w not in stop_words_temp]
            intermediate['stopwords_removed'] = " ".join(filtered_temp)
            # 5. Stemming (on filtered, alphanumeric tokens)
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
        st.success(f"BERT model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

@st.cache_data # Changed to st.cache_data as embeddings depend on the input texts
def generate_embeddings_with_progress(_model, texts_list_to_embed): # _model is passed as arg
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([])
    if not texts_list_to_embed: # Check if the list itself is empty
        st.warning("Input text list for embedding is empty.")
        return np.array([])
    # Further check if all texts in the list are empty strings
    if all(not text.strip() for text in texts_list_to_embed):
        st.warning("All texts in the input list are empty or whitespace. No embeddings will be generated.")
        return np.array([])

    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = []
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 # Recommended batch size for sentence-transformers
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size]
                # Ensure no empty strings are passed to encode, though SentenceTransformer might handle it
                valid_batch_texts = [text for text in batch_texts_segment if text.strip()]
                if not valid_batch_texts: # if all texts in batch are empty
                    # Add zero vectors or handle as appropriate for your model/downstream tasks
                    # For now, we'll skip embedding for entirely empty batches if that occurs,
                    # or add zero vectors of the correct dimensionality if known.
                    # Assuming model outputs 384 for all-MiniLM-L6-v2
                    # This part might need adjustment based on how empty texts should be represented.
                    # For simplicity, if a batch is entirely empty strings after filtering,
                    # we might skip it or add placeholder embeddings.
                    # However, the initial checks should prevent fully empty lists.
                    # If an individual text within a batch is empty, SentenceTransformer usually handles it.
                    pass # Let SentenceTransformer handle potentially empty strings within a valid batch

                if valid_batch_texts: # Proceed if there are valid texts in the batch
                    batch_embeddings_np_array = _model.encode(valid_batch_texts, convert_to_tensor=False, show_progress_bar=False)
                    embeddings_result_list.extend(batch_embeddings_np_array)

                if total_texts_to_embed > 0:
                    progress_val = min(1.0, (i + len(batch_texts_segment)) / total_texts_to_embed)
                    embedding_progress_bar.progress(progress_val)
                    embedding_status_text.text(f"Embedded {min(i + len(batch_texts_segment), total_texts_to_embed)}/{total_texts_to_embed} texts.")
            st.success("Embedding generation complete!")
            embedding_progress_bar.empty()
            embedding_status_text.empty()
            if not embeddings_result_list: # If somehow the list is empty (e.g. all input texts were empty)
                return np.array([])
            return np.array(embeddings_result_list)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])


@st.cache_data # Changed to st.cache_data as clusters depend on embeddings and K
def cluster_embeddings_with_progress(embeddings_to_cluster_param, n_clusters_for_algo): # Renamed params
    if embeddings_to_cluster_param is None or embeddings_to_cluster_param.size == 0:
        st.warning("No embeddings provided for clustering.")
        return None

    # Adjust n_clusters if it's problematic
    n_samples = embeddings_to_cluster_param.shape[0]
    if n_clusters_for_algo > n_samples:
        st.warning(f"Number of clusters ({n_clusters_for_algo}) is greater than the number of samples ({n_samples}). Adjusting K to {n_samples}.")
        n_clusters_for_algo = n_samples
    if n_clusters_for_algo < 1: # Should not happen with slider starting at 2, but good check
        st.error("Number of clusters must be at least 1.")
        return None
    if n_samples == 1 and n_clusters_for_algo > 1: # Special case for single sample
        st.warning("Only 1 sample available. Setting K=1.")
        n_clusters_for_algo = 1
    elif n_clusters_for_algo == 1 and n_samples > 1: # K=1 for multiple samples is trivial
        st.warning("K=1 for multiple samples means all samples in one cluster. This is trivial but allowed.")
        # Or you could force K=2: n_clusters_for_algo = 2; st.warning("K=1 for >1 samples. Setting K=2 for meaningful clustering.")

    try:
        with st.spinner(f"Clustering {embeddings_to_cluster_param.shape[0]} embeddings into {n_clusters_for_algo} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters_for_algo, random_state=42, n_init='auto')
            clusters_assigned = kmeans.fit_predict(embeddings_to_cluster_param)
            st.success(f"Clustering complete! Assigned {len(clusters_assigned)} items to clusters.")
            return clusters_assigned
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

def _calculate_average_precision(ranked_relevance_binary, k_val):
    if not ranked_relevance_binary: return 0.0
    ranked_relevance_binary = ranked_relevance_binary[:k_val] # Consider only top k
    relevant_hits, sum_precisions = 0, 0.0
    for i, is_relevant in enumerate(ranked_relevance_binary):
        if is_relevant: # is_relevant should be 0 or 1
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1) # Precision at this rank
    return sum_precisions / relevant_hits if relevant_hits > 0 else 0.0


# --- Page Functions ---
def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("This page provides an overview of the job dataset, merges it with O*NET category data, and allows you to explore its features.")

    if st.session_state.get('data') is None:
        # Load base data
        base_data = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES_TO_DISPLAY)
        if base_data is not None:
            # Load and merge category data using ONET_DATA_URL
            st.session_state['data'] = load_and_merge_categories(base_data, ONET_DATA_URL)
        else:
            st.session_state['data'] = None # Explicitly set to None if base_data loading failed

    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Data Preview (including `combined_jobs` and `category`)')
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        if 'category' in data_df.columns: cols_to_preview.append('category')
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Data Summary')
        st.write(f'Number of rows: {len(data_df)}')
        st.write(f'Number of columns: {len(data_df.columns)}')

        if 'combined_jobs' in data_df.columns:
            st.subheader('Sample Content of `combined_jobs` Column')
            for i in range(min(3, len(data_df))): # Display first 3 samples
                title_display = data_df.iloc[i].get('Title', "N/A")
                category_display = f" (Category: {data_df.iloc[i].get('category', 'N/A')})" if 'category' in data_df.columns else ""
                job_id_display = data_df.iloc[i].get('Job.ID', 'N/A')
                with st.expander(f"Job.ID: {job_id_display} - {title_display}{category_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Column 'combined_jobs' has not been created or is not in the data.")

        st.subheader('Search Word in Feature')
        search_word = st.text_input("Enter word to search:", key="home_search_word_new")

        # Dynamically create list of searchable columns
        all_available_cols_for_search = ['Job.ID', 'Title', 'combined_jobs'] + \
                                        [col for col in FEATURES_TO_COMBINE if col in data_df.columns] + \
                                        [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col in data_df.columns]
        searchable_cols = sorted(list(set(col for col in all_available_cols_for_search if col in data_df.columns)))

        search_column = st.selectbox("Select feature to search in:", [''] + searchable_cols, key="home_search_column_new")

        if search_word and search_column:
            if search_column in data_df.columns:
                try:
                    search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                    display_search_cols = ['Job.ID']
                    if 'Title' in data_df.columns: display_search_cols.append('Title')
                    if search_column not in display_search_cols: display_search_cols.append(search_column)
                    if 'category' in data_df.columns and 'category' not in display_search_cols: display_search_cols.append('category')

                    # Ensure all display_search_cols actually exist in search_results
                    display_search_cols = [col for col in display_search_cols if col in search_results.columns]

                    if not search_results.empty:
                        st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':")
                        st.dataframe(search_results[display_search_cols].head(), use_container_width=True)
                    else:
                        st.info(f"No entries found for '{search_word}' in '{search_column}'.")
                except Exception as e:
                    st.error(f"Error during search: {e}")
            else:
                st.warning(f"Selected search column '{search_column}' not found in the data.")


        st.subheader('Feature Information')
        st.write('**Available Features (after processing and merge):**', data_df.columns.tolist())
        if 'category' in data_df.columns:
            st.write("**Category Distribution (Top 10):**")
            try:
                st.dataframe(data_df['category'].value_counts().nlargest(10).reset_index().rename(columns={'index':'Category', 'category':'Count'}))
            except Exception as e:
                st.warning(f"Could not display category distribution: {e}")


    else:
        st.error("Data could not be loaded. Please check the data source URLs or your connection.")
        st.info(f"Main data URL: {DATA_URL}")
        st.info(f"Category data URL: {ONET_DATA_URL}")
    return

def preprocessing_page():
    st.header("Job Data Preprocessing")
    st.write("This page performs preprocessing on the 'combined_jobs' column of the job dataset.")

    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or 'combined_jobs' column not available. Please return to the 'Home' page to load data first.")
        if st.button("Return to Home to Load Data"):
            st.info("Please select 'Home' from the sidebar navigation.") # User needs to manually navigate
        return

    data_df_to_preprocess = st.session_state['data']

    st.info("The 'combined_jobs' column will be processed to create the 'processed_text' column.")
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before processing)"):
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())

    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_col_prep_btn"):
        with st.spinner("Preprocessing 'combined_jobs'..."):
            data_copy = data_df_to_preprocess.copy() # Work on a copy
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy, text_column_to_process='combined_jobs')
        st.success("Preprocessing of 'combined_jobs' column complete! 'processed_text' column has been created/updated in session state.")

    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Preprocessing has been performed on 'combined_jobs'.")
        display_data_processed = st.session_state['data']

        if 'preprocessing_steps' in display_data_processed.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)")
            # Ensure 'preprocessing_steps' contains list of dicts
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
    elif 'combined_jobs' in data_df_to_preprocess.columns: # combined_jobs exists but processed_text doesn't
        st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet. Click the button above.")
    return


def tsdae_page():
    st.header("TSDAE (Noise Injection & Embedding for Job Text)")
    st.write("Applies sequential noise and generates TSDAE embeddings for 'processed_text' (derived from 'combined_jobs').")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed (from 'combined_jobs') first. Visit 'Preprocessing' page.")
        return

    bert_model = load_bert_model() # Load the model once
    if bert_model is None:
        st.error("BERT model could not be loaded for TSDAE page. Cannot proceed."); return

    st.subheader("TSDAE Settings")
    deletion_ratio = st.slider("Deletion Ratio (for all noise methods)", 0.1, 0.9, 0.6, 0.05, key="tsdae_del_ratio_main")
    freq_threshold = st.slider("High Frequency Threshold (for methods B & C)", 10, 500, 100, 10, key="tsdae_freq_thresh_main")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="tsdae_run_button_main"):
        data_tsdae_local = st.session_state['data'].copy() # Work on a copy
        if 'processed_text' not in data_tsdae_local.columns or data_tsdae_local['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Cannot proceed with TSDAE.")
            return

        # Calculate word frequencies from 'processed_text' for methods B and C
        all_words_for_freq = [word for text_content in data_tsdae_local['processed_text'].fillna('').astype(str)
                              for word in word_tokenize(str(text_content).lower()) if word.isalnum()]
        word_freq_dict_tsdae = pd.Series(all_words_for_freq).value_counts().to_dict()
        if not word_freq_dict_tsdae:
            st.warning("Word frequency dictionary for TSDAE is empty (all processed texts might be empty or non-alphanumeric). Methods B & C might behave like random deletion.")


        st.markdown("---")
        st.markdown("##### Applying Noise Method A (Random Deletion)")
        noisy_text_stage_a = []
        source_texts_a = data_tsdae_local['processed_text'].fillna('').astype(str).tolist()
        total_items_a = len(source_texts_a)
        progress_bar_a = st.progress(0); status_text_a = st.empty()
        for idx, text_content in enumerate(source_texts_a):
            noisy_text_stage_a.append(denoise_text(text_content, method='a', del_ratio=deletion_ratio))
            if total_items_a > 0: progress_bar_a.progress((idx + 1) / total_items_a); status_text_a.text(f"Method A: Processed {idx + 1}/{total_items_a}")
        data_tsdae_local['noisy_text_a'] = noisy_text_stage_a
        progress_bar_a.empty(); status_text_a.empty(); st.success("Method A noise application complete.")

        st.markdown("---")
        st.markdown("##### Applying Noise Method B (High-Frequency Word Removal)")
        noisy_text_stage_b = []
        source_texts_b = data_tsdae_local['noisy_text_a'].tolist() # Noise applied sequentially
        total_items_b = len(source_texts_b)
        progress_bar_b = st.progress(0); status_text_b = st.empty()
        for idx, text_content in enumerate(source_texts_b):
            noisy_text_stage_b.append(denoise_text(text_content, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict_tsdae, freq_threshold=freq_threshold))
            if total_items_b > 0: progress_bar_b.progress((idx + 1) / total_items_b); status_text_b.text(f"Method B: Processed {idx + 1}/{total_items_b}")
        data_tsdae_local['noisy_text_b'] = noisy_text_stage_b
        progress_bar_b.empty(); status_text_b.empty(); st.success("Method B noise application complete.")

        st.markdown("---")
        st.markdown("##### Applying Noise Method C (High-Frequency Word Removal + Shuffle)")
        final_noisy_texts_list = []
        source_texts_c = data_tsdae_local['noisy_text_b'].tolist() # Noise applied sequentially
        total_items_c = len(source_texts_c)
        progress_bar_c = st.progress(0); status_text_c = st.empty()
        for idx, text_content in enumerate(source_texts_c):
            final_noisy_texts_list.append(denoise_text(text_content, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict_tsdae, freq_threshold=freq_threshold))
            if total_items_c > 0: progress_bar_c.progress((idx + 1) / total_items_c); status_text_c.text(f"Method C: Processed {idx + 1}/{total_items_c}")
        data_tsdae_local['final_noisy_text'] = final_noisy_texts_list
        progress_bar_c.empty(); status_text_c.empty(); st.success("Method C noise application complete.")

        st.session_state['data'] = data_tsdae_local # Update session state with noise columns
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        # Generate embeddings from 'final_noisy_text'
        final_noisy_texts_series = st.session_state['data']['final_noisy_text'].fillna('').astype(str)
        non_empty_mask_tsdae = final_noisy_texts_series.str.strip() != '' # Mask for non-empty texts
        valid_final_noisy_texts = final_noisy_texts_series[non_empty_mask_tsdae].tolist()
        job_ids_for_tsdae_embeddings = st.session_state['data'].loc[non_empty_mask_tsdae, 'Job.ID'].tolist()

        if not valid_final_noisy_texts:
            st.warning("No valid (non-empty) 'final_noisy_text' found to generate TSDAE embeddings.")
            st.session_state['tsdae_embeddings'] = None # Clear previous if any
            st.session_state['tsdae_embedding_job_ids'] = None
        else:
            tsdae_embeddings_generated = generate_embeddings_with_progress(bert_model, valid_final_noisy_texts)
            if tsdae_embeddings_generated is not None and tsdae_embeddings_generated.size > 0:
                st.session_state['tsdae_embeddings'] = tsdae_embeddings_generated
                st.session_state['tsdae_embedding_job_ids'] = job_ids_for_tsdae_embeddings
                st.success(f"TSDAE embeddings generated for {len(job_ids_for_tsdae_embeddings)} jobs!")
            else:
                st.warning("TSDAE embedding generation resulted in an empty or invalid output.")
                st.session_state['tsdae_embeddings'] = None
                st.session_state['tsdae_embedding_job_ids'] = None

    if 'tsdae_embeddings' in st.session_state and st.session_state.tsdae_embeddings is not None and st.session_state.tsdae_embeddings.size > 0:
        st.subheader("Current TSDAE Embeddings (Preview)")
        st.write(f"Shape: {st.session_state.tsdae_embeddings.shape}")
        st.write(f"Associated Job IDs: {len(st.session_state.get('tsdae_embedding_job_ids', []))}")
        st.dataframe(pd.DataFrame(st.session_state.tsdae_embeddings).head())
    else:
        st.info("TSDAE embeddings have not been generated yet or the last attempt was unsuccessful.")
    return

def bert_model_page():
    st.header("Standard BERT Embeddings (Job Descriptions)")
    st.write("Generates standard BERT embeddings from 'processed_text' (derived from 'combined_jobs').")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed. Visit 'Preprocessing' or 'Home' page.")
        return

    bert_model = load_bert_model() # Load the model once
    if bert_model is None:
        st.error("BERT model could not be loaded. Cannot proceed."); return

    if st.button("Generate/Regenerate Standard Job Embeddings", key="gen_std_emb_btn"):
        data_bert = st.session_state['data'] # Use current data from session state
        if 'processed_text' not in data_bert.columns or data_bert['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Please run preprocessing first.")
            return

        proc_series = data_bert['processed_text'].fillna('').astype(str)
        mask = proc_series.str.strip() != '' # Ensure we only embed non-empty texts
        valid_texts = proc_series[mask].tolist()
        valid_job_ids = data_bert.loc[mask, 'Job.ID'].tolist() # Get corresponding Job.IDs

        if not valid_texts:
            st.warning("No valid (non-empty) processed job texts found for embedding.")
            st.session_state['job_text_embeddings'] = None # Clear previous if any
            st.session_state['job_text_embedding_job_ids'] = None
        else:
            embeddings_generated = generate_embeddings_with_progress(bert_model, valid_texts)
            if embeddings_generated is not None and embeddings_generated.size > 0:
                st.session_state['job_text_embeddings'] = embeddings_generated
                st.session_state['job_text_embedding_job_ids'] = valid_job_ids
                st.success(f"Standard job embeddings generated for {len(valid_job_ids)} jobs!")
            else:
                st.warning("Standard job embedding generation resulted in an empty or invalid output.")
                st.session_state['job_text_embeddings'] = None
                st.session_state['job_text_embedding_job_ids'] = None


    job_emb = st.session_state.get('job_text_embeddings')
    job_ids_for_emb = st.session_state.get('job_text_embedding_job_ids') # Use the correct variable name

    if job_emb is not None and job_emb.size > 0 and job_ids_for_emb:
        st.subheader(f"Current Standard Job Embeddings ({len(job_ids_for_emb)} jobs)")
        st.write(f"Shape: {job_emb.shape}")
        st.subheader("2D Visualization (PCA)")

        if len(job_emb) >= 2: # PCA needs at least 2 samples
            try:
                pca_2d_model = PCA(n_components=2) # Create PCA model
                pca_transformed_embeddings = pca_2d_model.fit_transform(job_emb)
                plot_pca_df = pd.DataFrame(pca_transformed_embeddings, columns=['PC1','PC2'])
                plot_pca_df['Job.ID'] = job_ids_for_emb # Add Job.IDs corresponding to the embeddings

                # Merge with main data to get 'Title', 'category', and 'combined_jobs' for hover
                main_data_for_hover = st.session_state['data']
                # Ensure Job.ID types match for merging if necessary, though both should be strings
                main_data_for_hover['Job.ID'] = main_data_for_hover['Job.ID'].astype(str)
                plot_pca_df['Job.ID'] = plot_pca_df['Job.ID'].astype(str)


                cols_for_hover_info = ['Job.ID', 'Title', 'category']
                # Determine which description column to use
                desc_col_options = ['combined_jobs', 'Job.Description']
                description_col_for_hover = next((col for col in desc_col_options if col in main_data_for_hover.columns), None)
                if description_col_for_hover:
                    cols_for_hover_info.append(description_col_for_hover)

                # Filter main_data_for_hover to only include necessary columns and rows matching job_ids_for_emb
                hover_info_source_df = main_data_for_hover[main_data_for_hover['Job.ID'].isin(job_ids_for_emb)][cols_for_hover_info].copy()
                hover_info_source_df.drop_duplicates(subset=['Job.ID'], keep='first', inplace=True) # Ensure unique Job.IDs

                # Merge PCA results with hover information
                plot_pca_df = pd.merge(plot_pca_df, hover_info_source_df, on='Job.ID', how='left')

                # Handle cases where 'category' might be missing or NaN after merge
                if 'category' not in plot_pca_df.columns:
                    plot_pca_df['category'] = 'Uncategorized' # Add default if column doesn't exist
                plot_pca_df['category'] = plot_pca_df['category'].fillna('Uncategorized')


                hover_data_config = {'Job.ID': True, 'PC1': False, 'PC2': False}
                if 'Title' in plot_pca_df.columns: hover_data_config['Title'] = True
                if description_col_for_hover and description_col_for_hover in plot_pca_df.columns:
                    hover_data_config[description_col_for_hover] = True
                if 'category' in plot_pca_df.columns:
                    hover_data_config['category'] = True # Show category on hover


                if not plot_pca_df.empty and 'Title' in plot_pca_df.columns:
                    fig_pca = px.scatter(plot_pca_df, x='PC1', y='PC2',
                                         color='category', # Use 'category' for color
                                         hover_name='Title', # Use 'Title' for hover name
                                         hover_data=hover_data_config,
                                         title='2D PCA of Standard Job Embeddings (Colored by Category)',
                                         color_discrete_map={'Uncategorized': 'lightgrey'}) # Optional: specific color for uncategorized
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    st.warning("PCA plot data is incomplete (e.g., 'Title' missing after merge). Cannot generate plot.")
            except Exception as e:
                st.error(f"Error during PCA Visualization: {e}")
                st.error("This might be due to issues in merging PCA data with job details. Check Job.ID consistency and data availability.")
        else:
            st.warning("Need at least 2 data points (embedded jobs) for PCA visualization.")
    elif job_emb is not None and job_emb.size == 0 : # Embeddings generated but were empty
         st.warning("Standard job embeddings were generated but resulted in an empty set (e.g., all input texts were empty).")
    else: # Embeddings not generated yet
        st.info("Standard job embeddings have not been generated yet. Click the button above.")
    return

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings generated from 'processed_text' (either TSDAE or Standard BERT).")
    if st.session_state.get('data') is None:
        st.error("Job data not loaded. Please go to Home page first."); return

    emb_to_cluster, job_ids_clust, src_name_clust = None, None, ""
    # Radio button to choose which embeddings to use for clustering
    choice = st.radio("Select Embeddings for Clustering:",
                      ("Standard BERT Job Embeddings", "TSDAE Embeddings"),
                      key="clust_emb_choice_main", horizontal=True)

    if choice == "TSDAE Embeddings":
        if st.session_state.get('tsdae_embeddings') is not None and st.session_state.tsdae_embeddings.size > 0:
            emb_to_cluster = st.session_state['tsdae_embeddings']
            job_ids_clust = st.session_state.get('tsdae_embedding_job_ids')
            src_name_clust = "TSDAE Embeddings"
            if not job_ids_clust or len(job_ids_clust) != emb_to_cluster.shape[0]:
                st.error("TSDAE Job IDs are missing or do not match embedding dimensions. Please regenerate TSDAE embeddings."); return
        else:
            st.warning("TSDAE embeddings are unavailable. Please generate them on the 'TSDAE (Noise Injection)' page."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings') is not None and st.session_state.job_text_embeddings.size > 0:
            emb_to_cluster = st.session_state['job_text_embeddings']
            job_ids_clust = st.session_state.get('job_text_embedding_job_ids')
            src_name_clust = "Standard BERT Job Embeddings"
            if not job_ids_clust or len(job_ids_clust) != emb_to_cluster.shape[0]:
                st.error("Standard BERT Job IDs are missing or do not match embedding dimensions. Please regenerate them on the 'BERT Model' page."); return
        else:
            st.warning("Standard BERT job embeddings are unavailable. Please generate them on the 'BERT Model' page."); return

    st.info(f"Using: {src_name_clust} ({len(job_ids_clust) if job_ids_clust else 0} items for potential clustering)")

    if emb_to_cluster is not None and job_ids_clust:
        max_k_val = emb_to_cluster.shape[0]
        if max_k_val < 2:
            st.error("Need at least 2 embedded items to perform clustering."); return

        # Slider for number of clusters
        num_clusters_input = st.slider("Number of Clusters (K)",
                                       min_value=2,
                                       max_value=min(50, max_k_val), # Cap K at 50 or num_samples
                                       value=min(N_CLUSTERS, max_k_val), # Default K
                                       key="k_slider_cluster_main")

        if st.button(f"Run K-Means (K={num_clusters_input}) on {src_name_clust}", key="run_kmeans_button_main"):
            cluster_labels = cluster_embeddings_with_progress(emb_to_cluster, num_clusters_input)
            if cluster_labels is not None:
                if len(job_ids_clust) == len(cluster_labels):
                    # Create a DataFrame with Job.ID and cluster label
                    cluster_info_df = pd.DataFrame({'Job.ID': job_ids_clust, 'cluster_temp': cluster_labels})
                    cluster_info_df['Job.ID'] = cluster_info_df['Job.ID'].astype(str)


                    data_df_with_clusters = st.session_state['data'].copy()
                    data_df_with_clusters['Job.ID'] = data_df_with_clusters['Job.ID'].astype(str)


                    # Remove old 'cluster' column if it exists, before merging new one
                    if 'cluster' in data_df_with_clusters.columns:
                        data_df_with_clusters = data_df_with_clusters.drop(columns=['cluster'])

                    # Merge new cluster information
                    # Use a suffix for the merge if 'cluster' might already exist from a different source
                    st.session_state['data'] = pd.merge(data_df_with_clusters, cluster_info_df, on='Job.ID', how='left')
                    # Rename 'cluster_temp' to 'cluster'
                    if 'cluster_temp' in st.session_state['data'].columns:
                        st.session_state['data'].rename(columns={'cluster_temp': 'cluster'}, inplace=True)
                        # Fill NaN clusters that might arise if some Job.IDs in main data weren't in job_ids_clust
                        st.session_state['data']['cluster'] = st.session_state['data']['cluster'].fillna(-1).astype(int) # -1 for unclustered/missing

                    st.success(f"'cluster' column updated in the main dataset for {len(job_ids_clust)} jobs based on {src_name_clust}.")
                else:
                    st.error("Mismatch between number of Job IDs and generated cluster labels. Cannot merge.")
            else:
                st.error("Clustering algorithm failed to return labels.")

        # Visualization of clustered embeddings
        if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns and \
           emb_to_cluster is not None and job_ids_clust is not None and \
           len(job_ids_clust) == emb_to_cluster.shape[0]: # Check if embeddings and IDs used for clustering are still valid

            st.subheader(f"2D Visualization of Clustered {src_name_clust} (PCA)")
            if emb_to_cluster.shape[0] >= 2: # Need at least 2 samples for PCA
                try:
                    pca_cluster_model = PCA(n_components=2)
                    reduced_embeddings_for_plot = pca_cluster_model.fit_transform(emb_to_cluster)
                    plot_df_cluster = pd.DataFrame(reduced_embeddings_for_plot, columns=['PC1', 'PC2'])
                    plot_df_cluster['Job.ID'] = job_ids_clust # Job IDs corresponding to the clustered embeddings
                    plot_df_cluster['Job.ID'] = plot_df_cluster['Job.ID'].astype(str)


                    # Merge with main data to get Title, actual cluster labels, and category
                    data_for_plot_merge = st.session_state['data'][st.session_state['data']['Job.ID'].isin(job_ids_clust)].copy()
                    data_for_plot_merge['Job.ID'] = data_for_plot_merge['Job.ID'].astype(str)


                    cols_for_merge = ['Job.ID', 'Title', 'cluster'] # 'cluster' is from the main data now
                    if 'category' in data_for_plot_merge.columns: cols_for_merge.append('category')
                    text_col_for_hover = 'combined_jobs' if 'combined_jobs' in data_for_plot_merge.columns else 'Job.Description'
                    if text_col_for_hover not in cols_for_merge and text_col_for_hover in data_for_plot_merge.columns:
                        cols_for_merge.append(text_col_for_hover)

                    # Ensure only existing columns are selected
                    cols_for_merge = [col for col in cols_for_merge if col in data_for_plot_merge.columns]
                    data_for_plot_merge = data_for_plot_merge[cols_for_merge].drop_duplicates(subset=['Job.ID'])


                    plot_df_cluster = pd.merge(plot_df_cluster, data_for_plot_merge, on='Job.ID', how='left')
                    plot_df_cluster['cluster'] = plot_df_cluster['cluster'].fillna(-1).astype(str) # Treat -1 as a string for discrete colors

                    if 'category' not in plot_df_cluster.columns: plot_df_cluster['category'] = 'Uncategorized'
                    plot_df_cluster['category'] = plot_df_cluster['category'].fillna('Uncategorized')


                    if not plot_df_cluster.empty and 'Title' in plot_df_cluster.columns and 'cluster' in plot_df_cluster.columns:
                        hover_data_plot = {'Job.ID': True, 'cluster': True, 'PC1': False, 'PC2': False, 'category':True}
                        if text_col_for_hover in plot_df_cluster.columns:
                            hover_data_plot[text_col_for_hover] = True

                        fig_cluster_pca = px.scatter(
                            plot_df_cluster, x='PC1', y='PC2', color='cluster',
                            hover_name='Title', hover_data=hover_data_plot,
                            title=f'2D PCA of Clustered {src_name_clust} (Color by K-Means Cluster)',
                            color_discrete_sequence=px.colors.qualitative.Plotly # Use a qualitative color scheme
                        )
                        st.plotly_chart(fig_cluster_pca, use_container_width=True)
                    else:
                        st.warning("Could not generate cluster visualization. Required data (Title, cluster) missing in plot DataFrame.")
                except Exception as e_pca_plot:
                    st.error(f"Error during PCA visualization of clusters: {e_pca_plot}")
            else:
                st.warning("Not enough data points (embedded items) for PCA visualization.")
        elif 'cluster' in st.session_state.get('data', pd.DataFrame()).columns:
            st.info("Cluster information is present in the main dataset. To re-visualize with current embeddings, run clustering again.")
        else:
            st.info("No cluster information to visualize. Run K-Means clustering first.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5 per batch). Processed CVs are stored in session.")
    uploaded_cv_files = st.file_uploader("Choose CV files:", type=["pdf","docx"], accept_multiple_files=True, key="cv_upload_widget_main")

    if uploaded_cv_files:
        if len(uploaded_cv_files) > 5:
            st.warning("You have selected more than 5 CVs. Only the first 5 will be processed in this batch.")
            uploaded_cv_files = uploaded_cv_files[:5]

        if st.button("Process Uploaded CVs", key="proc_cv_btn_main"):
            cv_data_batch = [] # Store this batch's processed CVs
            bert_model_for_cv = load_bert_model() # Load model if not already loaded
            if not bert_model_for_cv:
                st.error("BERT model failed to load. Cannot process CVs."); return

            with st.spinner("Processing CVs..."):
                for i, cv_file in enumerate(uploaded_cv_files):
                    original_text, processed_text, cv_embedding = "", "", None
                    try:
                        file_ext = cv_file.name.split(".")[-1].lower()
                        if file_ext == "pdf":
                            original_text = extract_text_from_pdf(cv_file)
                        elif file_ext == "docx":
                            original_text = extract_text_from_docx(cv_file)

                        if original_text and original_text.strip():
                            processed_text = preprocess_text(original_text)
                            if processed_text and processed_text.strip():
                                # generate_embeddings_with_progress expects a list of texts
                                embedding_array = generate_embeddings_with_progress(bert_model_for_cv, [processed_text])
                                if embedding_array is not None and embedding_array.size > 0:
                                    cv_embedding = embedding_array[0] # Get the single embedding
                        else: # Original text was empty
                            st.warning(f"CV '{cv_file.name}' appears to be empty or text extraction failed.")


                        cv_data_batch.append({
                            'filename': cv_file.name,
                            'original_text': original_text or "", # Ensure not None
                            'processed_text': processed_text or "", # Ensure not None
                            'embedding': cv_embedding # Can be None if processing failed
                        })

                        if cv_embedding is not None:
                            st.success(f"Successfully processed and embedded: {cv_file.name}")
                        else:
                            st.warning(f"Failed to generate embedding for: {cv_file.name} (text might be empty after preprocessing).")
                    except Exception as e:
                        st.error(f"Error processing CV {cv_file.name}: {e}")
                        cv_data_batch.append({ # Add entry even on error to acknowledge attempt
                            'filename': cv_file.name, 'original_text':original_text or "",
                            'processed_text':processed_text or "", 'embedding':None
                        })

            # Append new batch to existing CVs in session state
            st.session_state['uploaded_cvs_data'].extend(cv_data_batch)
            # Deduplicate based on filename, keeping the latest processed version
            if st.session_state['uploaded_cvs_data']:
                temp_df = pd.DataFrame(st.session_state['uploaded_cvs_data'])
                temp_df.drop_duplicates(subset=['filename'], keep='last', inplace=True)
                st.session_state['uploaded_cvs_data'] = temp_df.to_dict('records')

            st.success(f"CV batch processing complete. {len(st.session_state['uploaded_cvs_data'])} CV(s) now in session.")

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader(f"Stored CVs in Session ({len(st.session_state['uploaded_cvs_data'])}):")
        for i, cv_d in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_d.get('filename', 'N/A')}"):
                st.text_area(f"Original Text:", value=cv_d.get('original_text',''), height=70, disabled=True, key=f"disp_cv_o_{i}_{cv_d.get('filename')}")
                st.text_area(f"Processed Text:", value=cv_d.get('processed_text',''), height=70, disabled=True, key=f"disp_cv_p_{i}_{cv_d.get('filename')}")
                if cv_d.get('embedding') is not None and cv_d.get('embedding').size > 0:
                    st.success("Embedding generated and stored.")
                else:
                    st.warning("Embedding is missing or invalid.")
        if st.button("Clear All Stored CVs from Session", key="clear_cvs_btn"):
            st.session_state['uploaded_cvs_data'] = []
            st.session_state['all_recommendations_for_annotation'] = {} # Also clear recommendations tied to old CVs
            st.info("All stored CVs and their recommendations have been cleared from the session.")
            st.experimental_rerun()

    else:
        st.info("No CVs uploaded and processed in the current session yet.")
    return


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations for uploaded CVs based on similarity with job embeddings.")

    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("No CVs have been uploaded and processed. Please go to the 'Upload CV' page first."); return

    main_data = st.session_state.get('data')
    if main_data is None or 'processed_text' not in main_data.columns: # Check for processed_text in jobs
        st.error("Job data with 'processed_text' (from 'combined_jobs') is not available. Please load and preprocess job data on the 'Home' or 'Preprocessing' pages."); return

    job_emb_for_rec, emb_src_msg_rec, job_emb_ids_for_rec = None, "", None
    rec_choice = st.radio("Choose Job Embeddings for Recommendation:",
                          ("Standard BERT Job Embeddings", "TSDAE Embeddings"),
                          key="rec_emb_choice_main_page", horizontal=True)

    if rec_choice == "TSDAE Embeddings":
        if st.session_state.get('tsdae_embeddings') is not None and st.session_state.tsdae_embeddings.size > 0:
            job_emb_for_rec = st.session_state['tsdae_embeddings']
            job_emb_ids_for_rec = st.session_state.get('tsdae_embedding_job_ids')
            if not job_emb_ids_for_rec or len(job_emb_ids_for_rec) != job_emb_for_rec.shape[0]:
                st.error("TSDAE Job IDs mismatch or missing. Please regenerate TSDAE embeddings."); return
            emb_src_msg_rec = "Using TSDAE embeddings for job matching."
        else:
            st.warning("TSDAE embeddings are unavailable. Please generate them first."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings') is not None and st.session_state.job_text_embeddings.size > 0:
            job_emb_for_rec = st.session_state['job_text_embeddings']
            job_emb_ids_for_rec = st.session_state.get('job_text_embedding_job_ids')
            if not job_emb_ids_for_rec or len(job_emb_ids_for_rec) != job_emb_for_rec.shape[0]:
                st.error("Standard BERT Job IDs mismatch or missing. Please regenerate them."); return
            emb_src_msg_rec = "Using Standard BERT job embeddings for job matching."
        else:
            st.warning("Standard BERT job embeddings are unavailable. Please generate them first."); return
    st.info(emb_src_msg_rec)


    # Prepare job data for similarity calculation, ensuring alignment with embeddings
    # temp_df_for_align creates an ordered list of Job.IDs based on the embedding order
    temp_df_for_align = pd.DataFrame({'Job.ID': job_emb_ids_for_rec, 'emb_order': np.arange(len(job_emb_ids_for_rec))})
    temp_df_for_align['Job.ID'] = temp_df_for_align['Job.ID'].astype(str)


    # Select columns to fetch for recommendation display
    cols_to_fetch_for_rec = ['Job.ID', 'Title'] + \
                            [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col in main_data.columns] + \
                            (['combined_jobs'] if 'combined_jobs' in main_data.columns else []) + \
                            (['cluster'] if 'cluster' in main_data.columns else [])
    cols_to_fetch_for_rec = sorted(list(set(cols_to_fetch_for_rec))) # Unique and sorted

    main_data_subset_for_rec = main_data[main_data['Job.ID'].isin(job_emb_ids_for_rec)][cols_to_fetch_for_rec].copy()
    main_data_subset_for_rec['Job.ID'] = main_data_subset_for_rec['Job.ID'].astype(str)
    main_data_subset_for_rec.drop_duplicates(subset=['Job.ID'], keep='first', inplace=True)


    # Merge to align job details with the order of embeddings
    jobs_for_sim_df = pd.merge(temp_df_for_align, main_data_subset_for_rec, on='Job.ID', how='left')
    jobs_for_sim_df.sort_values('emb_order', inplace=True) # Critical: ensure order matches embeddings
    jobs_for_sim_df.reset_index(drop=True, inplace=True)


    if len(jobs_for_sim_df) != len(job_emb_for_rec):
        st.error(f"Alignment error: Number of jobs for similarity ({len(jobs_for_sim_df)}) does not match number of embeddings ({len(job_emb_for_rec)}). This may be due to missing Job.IDs in the main data that were present in the embedding ID list. Please check data integrity."); return

    # Multiselect for choosing which details to display in the recommendation table
    default_details_to_show_rec_page = ['Company', 'City', 'Position', 'category'] # Added category
    options_for_rec_display = [col for col in JOB_DETAIL_FEATURES_TO_DISPLAY if col in jobs_for_sim_df.columns]
    if 'Job.Description' not in options_for_rec_display and 'Job.Description' in jobs_for_sim_df.columns:
        options_for_rec_display.append('Job.Description') # Ensure Job.Description is an option

    default_details_filtered_rec_page = [col for col in default_details_to_show_rec_page if col in options_for_rec_display]

    selected_details_for_rec_display = st.multiselect(
        "Select additional job details to display in the recommendations table:",
        options=sorted(list(set(options_for_rec_display))),
        default=default_details_filtered_rec_page,
        key="job_rec_detail_multiselect_page"
    )
    num_recommendations = st.slider("Number of recommendations per CV (Top N):", 5, 50, 20, 5, key="num_recs_slider")


    if st.button("Generate Recommendations", key="gen_recs_b_main"):
        st.session_state['all_recommendations_for_annotation'] = {} # Clear previous recommendations
        with st.spinner("Generating recommendations..."):
            valid_cvs_for_rec = [cv for cv in st.session_state.get('uploaded_cvs_data', [])
                                 if cv.get('embedding') is not None and cv.get('embedding').size > 0]
            if not valid_cvs_for_rec:
                st.warning("No CVs with valid embeddings found. Please upload and process CVs first."); return

            for cv_data_rec in valid_cvs_for_rec:
                cv_file_n = cv_data_rec.get('filename', 'Unknown CV')
                cv_embed = cv_data_rec['embedding'] # This is a 1D array
                st.subheader(f"Top {num_recommendations} Recommendations for CV: {cv_file_n}")

                # Reshape CV embedding to 2D for cosine_similarity: (1, embedding_dim)
                cv_embed_2d = cv_embed.reshape(1, -1)

                if job_emb_for_rec.ndim == 1 or job_emb_for_rec.shape[0] == 0:
                    st.error(f"Selected job embeddings are invalid or empty for CV '{cv_file_n}'. Cannot calculate similarity."); continue

                try:
                    similarities_rec = cosine_similarity(cv_embed_2d, job_emb_for_rec)[0] # Get 1D array of similarities
                except ValueError as ve:
                    st.error(f"Error calculating similarity for {cv_file_n}: {ve}. Check embedding dimensions.")
                    st.error(f"CV embedding shape: {cv_embed_2d.shape}, Job embeddings shape: {job_emb_for_rec.shape}")
                    continue


                temp_df_rec_with_sim = jobs_for_sim_df.copy() # Use the aligned jobs_for_sim_df
                temp_df_rec_with_sim['similarity_score'] = similarities_rec
                recommended_j_df = temp_df_rec_with_sim.sort_values(by='similarity_score', ascending=False).head(num_recommendations)

                if not recommended_j_df.empty:
                    display_cols_on_this_page = ['Job.ID', 'Title', 'similarity_score'] + selected_details_for_rec_display
                    # Ensure columns exist and are unique, preserving order somewhat
                    final_display_cols = []
                    for col in ['Job.ID', 'Title', 'similarity_score']: # Prioritize these
                        if col in recommended_j_df.columns and col not in final_display_cols: final_display_cols.append(col)
                    for col in selected_details_for_rec_display: # Add selected details
                        if col in recommended_j_df.columns and col not in final_display_cols: final_display_cols.append(col)
                    # Add category if not already there and exists
                    if 'category' in recommended_j_df.columns and 'category' not in final_display_cols: final_display_cols.append('category')


                    st.dataframe(recommended_j_df[final_display_cols], use_container_width=True)
                    # Store the full recommended_j_df (with all columns from jobs_for_sim_df + similarity) for annotation
                    st.session_state['all_recommendations_for_annotation'][cv_file_n] = recommended_j_df
                else:
                    st.info(f"No recommendations found for CV: {cv_file_n}.")
                st.write("---")
        st.success("Recommendation generation complete!")
    elif st.session_state.get('all_recommendations_for_annotation'): # If recommendations already exist
        st.info("Recommendations have already been generated. To re-generate with different settings or for new CVs, click the button above.")
        # Optionally, display existing recommendations here if needed
    return

def annotation_page():
    st.header("Annotation of Job Recommendations")
    st.write("Annotate relevance and provide feedback for recommended jobs.")

    if not st.session_state.get('all_recommendations_for_annotation'):
        st.warning("No recommendations have been generated yet. Please go to the 'Job Recommendation' page first."); return

    # Initialize session state variables if they don't exist (defensive)
    if 'annotator_details' not in st.session_state:
        st.session_state.annotator_details = {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS}
    if 'current_annotator_slot_for_input' not in st.session_state:
        st.session_state.current_annotator_slot_for_input = ANNOTATORS[0] if ANNOTATORS else None
    if 'annotators_saved_status' not in st.session_state:
        st.session_state.annotators_saved_status = set()
    if 'collected_annotations' not in st.session_state:
        st.session_state.collected_annotations = pd.DataFrame()


    # --- Section for Uploading Pre-filled Annotations ---
    st.sidebar.subheader("Load Annotations from CSV")
    uploaded_annotations_file = st.sidebar.file_uploader("Upload completed annotation CSV", type="csv", key="annotation_csv_uploader")

    if uploaded_annotations_file is not None:
        if st.sidebar.button("Load Annotations from Uploaded CSV", key="load_uploaded_annotations_btn"):
            try:
                uploaded_df = pd.read_csv(uploaded_annotations_file)
                # Basic validation (can be expanded based on expected columns)
                # For now, we assume it's compatible or user knows the format.
                # A more robust check would verify essential columns like 'cv_filename', 'job_id', and relevance scores.
                st.session_state.collected_annotations = uploaded_df
                # Try to infer saved annotators if possible, or reset
                # For simplicity, we'll assume an uploaded CSV means all annotators in it are "saved"
                # This part could be more sophisticated by checking which annotator columns have data.
                st.session_state.annotators_saved_status = set(ANNOTATORS) # Tentative: assumes uploaded CSV is complete
                st.success(f"Successfully loaded annotations from '{uploaded_annotations_file.name}'. Annotator statuses reset based on upload.")
                st.experimental_rerun() # Rerun to reflect loaded data in the form
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")
    st.sidebar.markdown("---")


    # --- Annotator Profile & Slot Selection ---
    st.subheader("🧑‍💻 Annotator Profile & Current Slot")
    if not ANNOTATORS:
        st.error("No annotator slots defined in ANNOTATORS constant. Annotation page cannot function."); return

    selected_slot = st.selectbox(
        "Select Your Annotator Slot to Edit Profile and Enter Annotations:",
        options=ANNOTATORS,
        index=ANNOTATORS.index(st.session_state.current_annotator_slot_for_input) if st.session_state.current_annotator_slot_for_input in ANNOTATORS else 0,
        key="annotator_slot_selector_main_page"
    )
    st.session_state.current_annotator_slot_for_input = selected_slot # Update current slot

    # Edit profile for the selected slot
    with st.expander(f"Edit Profile for {selected_slot}", expanded=True):
        current_name = st.session_state.annotator_details.get(selected_slot, {}).get('actual_name', selected_slot)
        current_profile_bg = st.session_state.annotator_details.get(selected_slot, {}).get('profile_background', '')

        actual_name = st.text_input(f"Your Name (for {selected_slot}):", value=current_name, key=f"actual_name_input_{selected_slot}_page")
        profile_bg = st.text_area(f"Your Profile Background (for {selected_slot}):", value=current_profile_bg, key=f"profile_bg_input_{selected_slot}_page", height=100)

        # Update session state for annotator details as inputs change (Streamlit handles this on interaction)
        st.session_state.annotator_details[selected_slot]['actual_name'] = actual_name
        st.session_state.annotator_details[selected_slot]['profile_background'] = profile_bg

    st.markdown("---")
    current_annotator_display_name = st.session_state.annotator_details.get(st.session_state.current_annotator_slot_for_input, {}).get('actual_name', st.session_state.current_annotator_slot_for_input)
    st.subheader(f"📝 Annotate Recommendations as: {st.session_state.current_annotator_slot_for_input} ({current_annotator_display_name})")

    # Initialize or ensure structure of collected_annotations DataFrame if it's empty but recommendations exist
    if st.session_state.collected_annotations.empty and st.session_state.all_recommendations_for_annotation:
        base_records_init = []
        for cv_fn_init, rec_df_init in st.session_state.all_recommendations_for_annotation.items():
            rec_df_unique_init = rec_df_init.drop_duplicates(subset=['Job.ID'], keep='first')
            for _, rec_row_init in rec_df_unique_init.iterrows():
                record_init = {
                    'cv_filename': cv_fn_init,
                    'job_id': str(rec_row_init['Job.ID']), # Ensure Job.ID is string
                    'job_title': rec_row_init.get('Title', 'N/A'),
                    'similarity_score': rec_row_init.get('similarity_score', 0.0),
                    'cluster': rec_row_init.get('cluster', pd.NA) # Use pd.NA for missing cluster
                }
                # Add other job details that were part of the recommendation
                for detail_col in JOB_DETAIL_FEATURES_TO_DISPLAY + ['combined_jobs', 'category']: # Added category
                    if detail_col in rec_row_init:
                         record_init[detail_col] = rec_row_init.get(detail_col, '')

                # Initialize columns for each annotator
                for i_ann, slot_name_ann in enumerate(ANNOTATORS):
                    record_init[f'annotator_{i_ann+1}_slot'] = slot_name_ann
                    record_init[f'annotator_{i_ann+1}_actual_name'] = ""
                    record_init[f'annotator_{i_ann+1}_profile_background'] = ""
                    record_init[f'annotator_{i_ann+1}_relevance'] = pd.NA # Use pd.NA for missing relevance
                    record_init[f'annotator_{i_ann+1}_feedback'] = ""
                base_records_init.append(record_init)
        if base_records_init:
            st.session_state.collected_annotations = pd.DataFrame(base_records_init)
        # If still empty (no recommendations), it remains an empty DataFrame

    relevance_options_map = {
        0: "0 (Very Irrelevant)", 1: "1 (Slightly Relevant)",
        2: "2 (Relevant)", 3: "3 (Most Relevant)"
    }
    current_annotator_slot = st.session_state.current_annotator_slot_for_input
    annotator_idx_for_cols = ANNOTATORS.index(current_annotator_slot) # 0-indexed

    with st.form(key=f"annotation_form_slot_{current_annotator_slot}"):
        form_input_for_current_annotator = []
        expand_cv_default = len(st.session_state['all_recommendations_for_annotation']) == 1

        # Determine available details for multiselect from the first recommendation DF
        available_details_for_ann_multiselect = []
        if st.session_state['all_recommendations_for_annotation']:
            first_cv_key = list(st.session_state['all_recommendations_for_annotation'].keys())[0]
            if first_cv_key in st.session_state['all_recommendations_for_annotation']:
                first_rec_df = st.session_state['all_recommendations_for_annotation'][first_cv_key]
                all_possible_details = list(set(JOB_DETAIL_FEATURES_TO_DISPLAY + ['Job.Description', 'category', 'combined_jobs'])) # Added category
                available_details_for_ann_multiselect = [col for col in all_possible_details if col in first_rec_df.columns]

        default_details = [col for col in ['Company', 'Job.Description', 'Employment.Type', 'category'] if col in available_details_for_ann_multiselect] # Added category

        selected_details_to_display_ann = st.multiselect(
            "Select job details to view during annotation:",
            options=sorted(list(set(available_details_for_ann_multiselect))),
            default=default_details,
            key=f"detail_multiselect_ann_{current_annotator_slot}"
        )

        for cv_filename, recommendations_df_original in st.session_state['all_recommendations_for_annotation'].items():
            recommendations_df_unique = recommendations_df_original.drop_duplicates(subset=['Job.ID'], keep='first')

            with st.expander(f"Recommendations for CV: **{cv_filename}**", expanded=expand_cv_default):
                for _, job_row_ann in recommendations_df_unique.iterrows():
                    job_id_str_ann = str(job_row_ann['Job.ID'])
                    st.markdown(f"**Job ID:** {job_id_str_ann} | **Title:** {job_row_ann.get('Title', 'N/A')}")

                    for detail_key in selected_details_to_display_ann:
                        if detail_key in job_row_ann and pd.notna(job_row_ann[detail_key]):
                            detail_value = job_row_ann[detail_key]
                            display_label = detail_key.replace('.', ' ').replace('_', ' ').title()
                            # Truncate long descriptions
                            if isinstance(detail_value, str) and len(detail_value) > 200:
                                st.caption(f"*{display_label}:* {detail_value[:200]}...")
                            else:
                                st.caption(f"*{display_label}:* {detail_value}")

                    st.caption(f"*Similarity Score:* {job_row_ann.get('similarity_score', 0.0):.4f} | *Original Cluster (if any):* {job_row_ann.get('cluster', 'N/A')} | *Category:* {job_row_ann.get('category', 'N/A')}")
                    st.markdown("---")

                    relevance_key_ann = f"relevance_{cv_filename}_{job_id_str_ann}_{current_annotator_slot}"
                    feedback_key_ann = f"feedback_{cv_filename}_{job_id_str_ann}_{current_annotator_slot}"

                    # Get default values from collected_annotations DataFrame
                    default_relevance = 0 # Default to "Very Irrelevant"
                    default_feedback = ""
                    if not st.session_state.collected_annotations.empty:
                        mask = (st.session_state.collected_annotations['cv_filename'] == cv_filename) & \
                               (st.session_state.collected_annotations['job_id'] == job_id_str_ann)
                        existing_row_df = st.session_state.collected_annotations[mask]

                        if not existing_row_df.empty:
                            existing_row = existing_row_df.iloc[0]
                            rel_col = f'annotator_{annotator_idx_for_cols+1}_relevance'
                            fb_col = f'annotator_{annotator_idx_for_cols+1}_feedback'
                            if rel_col in existing_row and pd.notna(existing_row[rel_col]):
                                try:
                                    default_relevance = int(existing_row[rel_col])
                                    if default_relevance not in relevance_options_map: # Ensure it's a valid key
                                        default_relevance = 0
                                except ValueError:
                                    default_relevance = 0 # Fallback if conversion fails
                            if fb_col in existing_row and pd.notna(existing_row[fb_col]):
                                default_feedback = str(existing_row[fb_col])


                    relevance_val_selected = st.radio("Relevance:", options=list(relevance_options_map.keys()),
                                                      index=default_relevance, # Index must be valid
                                                      key=relevance_key_ann, horizontal=True,
                                                      format_func=lambda x: relevance_options_map[x])
                    feedback_val_input = st.text_area("Feedback (optional):", value=default_feedback, key=feedback_key_ann, height=75)

                    form_input_for_current_annotator.append({
                        'cv_filename': cv_filename, 'job_id': job_id_str_ann,
                        'relevance': relevance_val_selected, 'feedback': feedback_val_input,
                        # Include all other details from job_row_ann for potential reconstruction if needed
                        **job_row_ann.to_dict()
                    })
                    st.markdown("---")

        submitted_current_annotator = st.form_submit_button(f"Save/Update My Ratings ({current_annotator_display_name})")

    if submitted_current_annotator:
        profile_details_current = st.session_state.annotator_details.get(current_annotator_slot, {})
        actual_name_current = profile_details_current.get('actual_name', current_annotator_slot)
        profile_bg_current = profile_details_current.get('profile_background', '')
        updated_items_count = 0

        # Ensure collected_annotations has the necessary annotator columns before trying to update
        # This is crucial if collected_annotations was loaded from a CSV that didn't have all annotator columns
        for i_init, slot_init_name in enumerate(ANNOTATORS):
            for col_suffix in ['slot', 'actual_name', 'profile_background', 'relevance', 'feedback']:
                col_name_init = f'annotator_{i_init+1}_{col_suffix}'
                if col_name_init not in st.session_state.collected_annotations.columns:
                    if 'relevance' in col_suffix :
                        st.session_state.collected_annotations[col_name_init] = pd.NA # Use pd.NA
                    else:
                        st.session_state.collected_annotations[col_name_init] = ""


        for item_ann_data in form_input_for_current_annotator:
            mask_update = (st.session_state.collected_annotations['cv_filename'] == item_ann_data['cv_filename']) & \
                          (st.session_state.collected_annotations['job_id'] == item_ann_data['job_id'])
            row_indices_to_update = st.session_state.collected_annotations[mask_update].index

            if not row_indices_to_update.empty:
                idx = row_indices_to_update[0] # Update the first match
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_slot'] = current_annotator_slot
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_actual_name'] = actual_name_current
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_profile_background'] = profile_bg_current
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_relevance'] = item_ann_data['relevance']
                st.session_state.collected_annotations.loc[idx, f'annotator_{annotator_idx_for_cols+1}_feedback'] = item_ann_data['feedback']
                updated_items_count +=1
            else: # If the record doesn't exist, create it (e.g., if recommendations changed or CSV was partial)
                new_record = {
                    'cv_filename': item_ann_data['cv_filename'],
                    'job_id': item_ann_data['job_id'],
                    'job_title': item_ann_data.get('Title', 'N/A'),
                    'similarity_score': item_ann_data.get('similarity_score', 0.0),
                    'cluster': item_ann_data.get('cluster', pd.NA)
                }
                # Add other job details
                for detail_col_new in JOB_DETAIL_FEATURES_TO_DISPLAY + ['combined_jobs', 'category']:
                     if detail_col_new in item_ann_data:
                         new_record[detail_col_new] = item_ann_data.get(detail_col_new, '')


                for i_ann_new, slot_name_ann_new in enumerate(ANNOTATORS):
                    new_record[f'annotator_{i_ann_new+1}_slot'] = slot_name_ann_new
                    if i_ann_new == annotator_idx_for_cols: # Current annotator's data
                        new_record[f'annotator_{i_ann_new+1}_actual_name'] = actual_name_current
                        new_record[f'annotator_{i_ann_new+1}_profile_background'] = profile_bg_current
                        new_record[f'annotator_{i_ann_new+1}_relevance'] = item_ann_data['relevance']
                        new_record[f'annotator_{i_ann_new+1}_feedback'] = item_ann_data['feedback']
                    else: # Other annotators - initialize empty for this new record
                        new_record[f'annotator_{i_ann_new+1}_actual_name'] = ""
                        new_record[f'annotator_{i_ann_new+1}_profile_background'] = ""
                        new_record[f'annotator_{i_ann_new+1}_relevance'] = pd.NA
                        new_record[f'annotator_{i_ann_new+1}_feedback'] = ""
                st.session_state.collected_annotations = pd.concat([st.session_state.collected_annotations, pd.DataFrame([new_record])], ignore_index=True)
                updated_items_count +=1
                st.info(f"New annotation record created for CV {item_ann_data['cv_filename']}, Job {item_ann_data['job_id']}.")


        if updated_items_count > 0:
            st.success(f"Annotations by {current_annotator_display_name} saved/updated for {updated_items_count} items.")
            st.session_state.annotators_saved_status.add(current_annotator_slot)
            st.experimental_rerun()
        elif form_input_for_current_annotator :
            st.warning("No existing records were updated, and no new records were created. This might happen if the form was empty or data matching failed unexpectedly.")
        else:
            st.info("No annotation entries were submitted in the form (e.g., no recommendations were displayed).")


    st.markdown("---")
    st.subheader("Final Actions & Download")
    completed_count = len(st.session_state.get('annotators_saved_status', set()))
    total_ann = len(ANNOTATORS) if ANNOTATORS else 0

    if total_ann > 0:
        if completed_count == total_ann:
            st.success(f"All {total_ann} annotator slots have saved their ratings! You can now download the combined data.")
            download_disabled_status = False
        else:
            st.info(f"Waiting for all annotator slots to save ratings. Completed: {completed_count}/{total_ann}.")
            st.write("Completed slots:", ", ".join(sorted(list(st.session_state.annotators_saved_status))) or "None")
            download_disabled_status = True
    else: # No annotators defined
        st.warning("No annotators defined. Download button will be enabled by default if data exists.")
        download_disabled_status = st.session_state.get('collected_annotations', pd.DataFrame()).empty


    if not st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        try:
            csv_export = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Collected Annotations as CSV", data=csv_export,
                file_name="all_job_recommendation_annotations.csv", mime="text/csv",
                key="download_all_annotations_final_main_btn", disabled=download_disabled_status
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")

        with st.expander("Show Current Collected Annotations Data (All Annotators)", expanded=False):
            st.dataframe(st.session_state['collected_annotations'], height=300)
    else:
        st.info("No annotations collected yet, or annotations from CSV not loaded/empty.")
    return


def evaluation_page():
    st.header("Model Evaluation")
    st.write("Evaluates top N recommendations based on human annotations.")

    all_recommendations_eval = st.session_state.get('all_recommendations_for_annotation', {})
    anns_df_eval = st.session_state.get('collected_annotations', pd.DataFrame())

    if not all_recommendations_eval:
        st.warning("No recommendations available to evaluate. Please run 'Job Recommendation' first."); return
    if anns_df_eval.empty:
        st.warning("No annotation data available. Please use the 'Annotation' page to input or upload annotations."); return

    st.info("This evaluation uses the annotation data currently loaded/entered in the 'Annotation' page.")
    st.subheader("Evaluation Parameters")
    st.info("The 'Binary Relevance Threshold' converts average graded annotator scores (0-3) into binary 'relevant' (1) or 'not relevant' (0) for calculating binary metrics.")
    relevance_threshold_binary = st.slider("Binary Relevance Threshold (Avg Annotator Score >= Threshold is Relevant)", 0.0, 3.0, 1.5, 0.1, key="eval_thresh_binary_final")
    eval_k_cutoff = st.slider("Evaluate Top K recommendations:", 5, 50, 20, 5, key="eval_k_slider")


    if st.button(f"Run Evaluation on Top {eval_k_cutoff} Recommendations", key="run_eval_final_btn"):
        with st.spinner("Calculating human-grounded evaluation metrics..."):
            per_cv_metrics_list = []
            # Identify relevance columns from actual annotators who have data
            relevance_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS))
                              if f'annotator_{i+1}_relevance' in anns_df_eval.columns and
                              anns_df_eval[f'annotator_{i+1}_relevance'].notna().any()] # Only consider if col has some non-NA values

            if not relevance_cols:
                st.error("No annotator relevance columns with data found in the annotation DataFrame. Cannot perform evaluation."); return

            num_cvs_evaluated = 0
            for cv_filename, recommended_jobs_df_orig in all_recommendations_eval.items():
                if recommended_jobs_df_orig.empty: continue

                # Ensure Job.ID is string for merging
                recommended_jobs_df = recommended_jobs_df_orig.copy()
                recommended_jobs_df['Job.ID'] = recommended_jobs_df['Job.ID'].astype(str)

                cv_anns_subset = anns_df_eval[anns_df_eval['cv_filename'] == cv_filename].copy()
                if cv_anns_subset.empty:
                    st.warning(f"No annotations found for CV: {cv_filename}. Skipping evaluation for this CV.")
                    continue
                cv_anns_subset['job_id'] = cv_anns_subset['job_id'].astype(str)

                num_cvs_evaluated +=1
                top_k_recs_df = recommended_jobs_df.head(eval_k_cutoff) # Consider top K
                ranked_job_ids_list = top_k_recs_df['Job.ID'].tolist()
                # Model's scores for these top K items (for NDCG)
                model_similarity_scores_for_k = top_k_recs_df['similarity_score'].tolist()


                binary_relevance_scores_for_k, graded_relevance_scores_for_k = [], []

                for job_id_eval in ranked_job_ids_list:
                    job_specific_annotations = cv_anns_subset[cv_anns_subset['job_id'] == job_id_eval]
                    avg_annotator_score_for_job = 0.0 # Default if no annotations for this specific job
                    if not job_specific_annotations.empty:
                        annotator_scores_for_this_job = []
                        for rel_col_name in relevance_cols: # Use only relevant_cols with data
                            # Get scores, convert to numeric, drop NaNs
                            scores_from_col = pd.to_numeric(job_specific_annotations[rel_col_name], errors='coerce').dropna().tolist()
                            annotator_scores_for_this_job.extend(scores_from_col)

                        if annotator_scores_for_this_job: # If any valid scores were found
                            avg_annotator_score_for_job = np.mean(annotator_scores_for_this_job)

                    graded_relevance_scores_for_k.append(avg_annotator_score_for_job)
                    binary_relevance_scores_for_k.append(1 if avg_annotator_score_for_job >= relevance_threshold_binary else 0)

                # Calculate metrics for this CV at K
                actual_len_ranked = len(binary_relevance_scores_for_k) # Should be eval_k_cutoff or less if fewer recs
                if actual_len_ranked == 0: continue # Skip if no items to evaluate

                cv_p_at_k = sum(binary_relevance_scores_for_k) / actual_len_ranked
                cv_hr_at_k = 1 if any(binary_relevance_scores_for_k) else 0
                cv_map_at_k = _calculate_average_precision(binary_relevance_scores_for_k, actual_len_ranked)

                cv_mrr_at_k = 0.0
                for r_idx, is_rel_mrr in enumerate(binary_relevance_scores_for_k):
                    if is_rel_mrr: cv_mrr_at_k = 1.0 / (r_idx + 1); break

                # NDCG requires scores to be in list of lists format: [[scores]]
                # Ensure model_similarity_scores_for_k also matches length of relevance scores
                # This might happen if recommendations were fewer than K
                model_sim_scores_ndcg = [model_similarity_scores_for_k[:actual_len_ranked]]

                cv_binary_ndcg_at_k = ndcg_score([binary_relevance_scores_for_k], model_sim_scores_ndcg, k=actual_len_ranked) if actual_len_ranked > 0 else 0.0
                cv_graded_ndcg_at_k = ndcg_score([graded_relevance_scores_for_k], model_sim_scores_ndcg, k=actual_len_ranked) if actual_len_ranked > 0 else 0.0


                per_cv_metrics_list.append({
                    'CV Filename': cv_filename,
                    f'P@{eval_k_cutoff}': cv_p_at_k, f'MAP@{eval_k_cutoff}': cv_map_at_k,
                    f'MRR@{eval_k_cutoff}': cv_mrr_at_k, f'HR@{eval_k_cutoff}': cv_hr_at_k,
                    f'NDCG@{eval_k_cutoff} (Binary)': cv_binary_ndcg_at_k,
                    f'NDCG@{eval_k_cutoff} (Graded)': cv_graded_ndcg_at_k
                })

            st.subheader(f"Per-CV Evaluation Metrics (Top {eval_k_cutoff})")
            if per_cv_metrics_list:
                per_cv_df = pd.DataFrame(per_cv_metrics_list)
                per_cv_df_display = per_cv_df.copy()
                # Format for display
                for col_fmt in [f'P@{eval_k_cutoff}', f'MAP@{eval_k_cutoff}', f'HR@{eval_k_cutoff}']:
                    if col_fmt in per_cv_df_display.columns:
                        per_cv_df_display[col_fmt] = (per_cv_df_display[col_fmt] * 100).round(2).astype(str) + '%'
                for col_fmt_round in [f'MRR@{eval_k_cutoff}', f'NDCG@{eval_k_cutoff} (Binary)', f'NDCG@{eval_k_cutoff} (Graded)']:
                     if col_fmt_round in per_cv_df_display.columns:
                        per_cv_df_display[col_fmt_round] = per_cv_df_display[col_fmt_round].round(4)
                st.dataframe(per_cv_df_display.set_index('CV Filename'), use_container_width=True)

                # Calculate and display average metrics
                avg_metrics_dict = {
                    f'P@{eval_k_cutoff}': per_cv_df[f'P@{eval_k_cutoff}'].mean(),
                    f'MAP@{eval_k_cutoff}': per_cv_df[f'MAP@{eval_k_cutoff}'].mean(),
                    f'MRR@{eval_k_cutoff}': per_cv_df[f'MRR@{eval_k_cutoff}'].mean(),
                    f'HR@{eval_k_cutoff}': per_cv_df[f'HR@{eval_k_cutoff}'].mean(),
                    f'NDCG@{eval_k_cutoff} (Binary)': per_cv_df[f'NDCG@{eval_k_cutoff} (Binary)'].mean(),
                    f'NDCG@{eval_k_cutoff} (Graded)': per_cv_df[f'NDCG@{eval_k_cutoff} (Graded)'].mean()
                }
            else: # No CVs were evaluated (e.g., no annotations matched)
                st.info("No CVs were evaluated (e.g., no matching annotations found or all CVs skipped).")
                avg_metrics_dict = {key: 'N/A' for key in [f'P@{eval_k_cutoff}', f'MAP@{eval_k_cutoff}', f'MRR@{eval_k_cutoff}', f'HR@{eval_k_cutoff}', f'NDCG@{eval_k_cutoff} (Binary)', f'NDCG@{eval_k_cutoff} (Graded)']}


            st.subheader(f"Average Human-Grounded Evaluation Metrics Summary (Top {eval_k_cutoff})")
            if num_cvs_evaluated > 0:
                st.write(f"Calculated based on {num_cvs_evaluated} CVs with annotations.")
            else:
                st.warning("No CVs with annotations were found to calculate average metrics."); return

            metric_config = {
                f'Precision@{eval_k_cutoff}': {'key_map': f'P@{eval_k_cutoff}', 'fmt': "{:.2%}", 'help': f"Avg P@{eval_k_cutoff}. Proportion of top {eval_k_cutoff} relevant items (binary).", 'color': "off"},
                f'MAP@{eval_k_cutoff}': {'key_map': f'MAP@{eval_k_cutoff}', 'fmt': "{:.2%}", 'help': f"Mean Avg. Precision@{eval_k_cutoff} (binary relevance, considers order).", 'color': "off"},
                f'MRR@{eval_k_cutoff}': {'key_map': f'MRR@{eval_k_cutoff}', 'fmt': "{:.4f}", 'help': f"Mean Reciprocal Rank@{eval_k_cutoff} (rank of first relevant item, binary).", 'color': "normal"},
                f'HR@{eval_k_cutoff}': {'key_map': f'HR@{eval_k_cutoff}', 'fmt': "{:.2%}", 'help': f"Hit Ratio@{eval_k_cutoff}: Proportion of CVs with at least one relevant item in top {eval_k_cutoff}.", 'color': "normal"},
                f'NDCG@{eval_k_cutoff} (Binary)': {'key_map': f'NDCG@{eval_k_cutoff} (Binary)', 'fmt': "{:.4f}", 'help': f"Avg NDCG@{eval_k_cutoff} using binary relevance from threshold.", 'color': "inverse"},
                f'NDCG@{eval_k_cutoff} (Graded)': {'key_map': f'NDCG@{eval_k_cutoff} (Graded)', 'fmt': "{:.4f}", 'help': f"Avg NDCG@{eval_k_cutoff} using average annotator scores as graded relevance.", 'color': "inverse"}
            }
            keys_to_display_eval = list(metric_config.keys())

            # Display metrics in columns
            num_metrics_cols = 3
            metric_cols_display = st.columns(num_metrics_cols)
            for i, display_key in enumerate(keys_to_display_eval):
                cfg = metric_config[display_key]
                actual_metric_key = cfg['key_map'] # The key used in avg_metrics_dict
                value = avg_metrics_dict.get(actual_metric_key, "N/A") # Get value using actual key

                current_col_display = metric_cols_display[i % num_metrics_cols]

                val_str_display = "N/A"
                if isinstance(value, (int, float, np.number)) and not (isinstance(value, float) and np.isnan(value)):
                    val_str_display = cfg['fmt'].format(value * 100 if '%' in cfg['fmt'] else value)
                elif isinstance(value, str): # If it was already 'N/A'
                    val_str_display = value

                current_col_display.metric(label=f"Average {display_key}", value=val_str_display, delta_color=cfg['color'], help=cfg['help'])
    return

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Job Recommender Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model",
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
# Use a unique key for the main navigation radio to avoid conflicts if 'page' is used elsewhere
page_selection = st.sidebar.radio("Go to", page_options, key="main_nav_radio_selector")

if page_selection == "Home":
    home_page()
elif page_selection == "Preprocessing":
    preprocessing_page()
elif page_selection == "TSDAE (Noise Injection)":
    tsdae_page()
elif page_selection == "BERT Model":
    bert_model_page()
elif page_selection == "Clustering Job2Vec":
    clustering_page()
elif page_selection == "Upload CV":
    upload_cv_page()
elif page_selection == "Job Recommendation":
    job_recommendation_page()
elif page_selection == "Annotation":
    annotation_page()
elif page_selection == "Evaluation":
    evaluation_page()
