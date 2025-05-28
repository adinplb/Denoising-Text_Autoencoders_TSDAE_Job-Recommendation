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
from tqdm import tqdm # Used for local progress bar simulation, not directly visible in Streamlit's st.progress
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator # Added for evaluation
from sklearn.cluster import KMeans
# from sklearn.preprocessing import normalize # Not used directly, can be removed if not needed
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
        nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle') # This specific check might be too granular
    except LookupError:
        st.info("Downloading NLTK punkt_tab resource (if available)...")
        try:
            nltk.download('punkt_tab')
        except Exception:
            st.warning("NLTK punkt_tab resource not found or download failed, usually not critical.")
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv'
RELEVANT_FEATURES = ['Job.ID', 'text', 'Title']
N_CLUSTERS = 20 # Default number of clusters for KMeans
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
# This helps share data between pages without re-running heavy computations
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None
if 'job_clusters' not in st.session_state:
    st.session_state['job_clusters'] = None
# cv_text and cv_embedding are for single CV context, less used with multi-CV approach
# if 'cv_text' not in st.session_state:
#     st.session_state['cv_text'] = ""
# if 'cv_embedding' not in st.session_state:
#     st.session_state['cv_embedding'] = None
if 'tsdae_embeddings' not in st.session_state:
    st.session_state['tsdae_embeddings'] = None
if 'uploaded_cvs_data' not in st.session_state:
    # MODIFIED: added 'processed_text' field
    st.session_state['uploaded_cvs_data'] = [] # Stores list of {'filename', 'original_text', 'processed_text', 'embedding'}
if 'all_recommendations_for_annotation' not in st.session_state:
    st.session_state['all_recommendations_for_annotation'] = {} # Format: {cv_filename: DataFrame of top 20 recs}
if 'collected_annotations' not in st.session_state:
    st.session_state['collected_annotations'] = pd.DataFrame()


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading data...')
def load_data_from_url(url):
    """Loads job data from a given URL and selects relevant features."""
    try:
        df = pd.read_csv(url)
        st.success('Successfully loaded data!')
        # Ensure Job.ID is string for consistency if it's used as dict key later
        if 'Job.ID' in df.columns:
            df['Job.ID'] = df['Job.ID'].astype(str)
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
        document = DocxDocument(uploaded_file)
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
    if not isinstance(text, str): # Ensure input is a string
        return ""
    # Symbol Removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    # Case Folding
    text = text.lower()
    # Stopwords Removal and Tokenization
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words and w.isalnum()] # Keep alphanumeric
    # Stemming
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in filtered_words]
    return " ".join(stemmed_words)

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
            for i, text_content in enumerate(data_df['text'].fillna('')):
                intermediate = {}
                if isinstance(text_content, str):
                    intermediate['original'] = text_content
                    # Reuse preprocess_text for the final step to ensure consistency
                    # and capture intermediate steps manually for display
                    symbol_removed = text_content.translate(str.maketrans('', '', string.punctuation))
                    symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed)
                    intermediate['symbol_removed'] = symbol_removed
                    
                    case_folded = symbol_removed.lower()
                    intermediate['case_folded'] = case_folded
                    
                    word_tokens_temp = word_tokenize(case_folded)
                    intermediate['tokenized'] = " ".join(word_tokens_temp)
                    
                    stop_words_temp = set(stopwords.words('english'))
                    filtered_temp = [w for w in word_tokens_temp if w not in stop_words_temp and w.isalnum()]
                    intermediate['stopwords_removed'] = " ".join(filtered_temp)
                    
                    porter_temp = PorterStemmer()
                    stemmed_temp = [porter_temp.stem(w) for w in filtered_temp]
                    intermediate['stemmed'] = " ".join(stemmed_temp)
                    
                    processed_results.append(intermediate)
                else: # Handle non-string case (e.g. NaN that wasn't caught by fillna(''))
                    processed_results.append({
                        'original': str(text_content), 'symbol_removed': '', 'case_folded': '',
                        'tokenized': '', 'stopwords_removed': '', 'stemmed': ''
                    })
                if total_rows > 0:
                    progress_bar.progress((i + 1) / total_rows)
                    status_text.text(f"Processed {i + 1}/{total_rows} entries.")
            
            # Apply the consistent preprocess_text function for the final 'processed_text' column
            data_df['processed_text'] = data_df['text'].fillna('').apply(preprocess_text)
            data_df['preprocessing_steps'] = processed_results # Store intermediate steps for display

            st.success("Preprocessing of 'text' column complete!")
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("The 'text' column was not found in the dataset.")
    return data_df

# --- Text Denoising Function ---
def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    """
    Applies noise to text based on specified method for TSDAE.
    Methods: 'a' (random deletion), 'b' (high-frequency word removal),
    'c' (high-frequency word removal + shuffle).
    """
    if not isinstance(text_to_denoise, str): # Ensure input is string
        return ""
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return "" 
    
    result_words = [] 
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0: 
            if n > 0: # Ensure words list is not empty
                 idx_to_keep = np.random.choice(n) 
                 keep_or_not[idx_to_keep] = True
            else: # Should not happen if n > 0 check passed, but defensive
                return "" 
        result_words = np.array(words)[keep_or_not].tolist() 
    elif method in ('b', 'c'):
        if word_freq_dict is None:
            # st.warning("word_freq_dict is missing for denoising method 'b' or 'c'. Returning original text.")
            # return text_to_denoise # Or raise error, depending on desired strictness
            raise ValueError("word_freq_dict is required for method 'b' or 'c'.")

        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0 and num_to_remove <= len(high_freq_indices) :
             to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result_words and words: 
            result_words = [random.choice(words)]
        
        if method == 'c':
            random.shuffle(result_words)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result_words)


# --- Embedding Generation Functions ---
@st.cache_resource
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    """Loads the SentenceTransformer model (cached)."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        # st.stop() # Consider stopping execution if model is critical and fails to load
        return None


@st.cache_data
def generate_embeddings_with_progress(_model, texts_list):
    """
    Generates embeddings for a list of texts using the provided model (cached).
    Includes a Streamlit spinner and a progress bar.
    _model argument is prefixed with underscore to prevent Streamlit hashing errors.
    """
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([])
    if not texts_list or not any(texts_list): # Check if list is empty or contains only empty strings
        st.warning("Input text list is empty or contains no actual text. Skipping embedding generation.")
        return np.array([])
        
    try:
        with st.spinner("Generating embeddings... This can take a few minutes."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            
            # Filter out empty or whitespace-only strings before sending to model
            valid_texts = [text for text in texts_list if text and text.strip()]
            if not valid_texts:
                st.warning("No valid text found after filtering. Skipping embedding generation.")
                embedding_progress_bar.empty()
                embedding_status_text.empty()
                return np.array([])

            embeddings_list = []
            total_texts = len(valid_texts)
            batch_size = 32 
            for i in range(0, total_texts, batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                batch_embeddings_np = _model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False) # No tensor, no internal bar
                embeddings_list.extend(batch_embeddings_np) 
                
                if total_texts > 0:
                    progress_val = (i + len(batch_texts)) / total_texts
                    embedding_progress_bar.progress(progress_val)
                    embedding_status_text.text(f"Generated embeddings for {i + len(batch_texts)}/{total_texts} entries.")
            
            st.success("Embedding generation complete!")
            embedding_progress_bar.empty()
            embedding_status_text.empty()
            return np.array(embeddings_list)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

@st.cache_data
def cluster_embeddings_with_progress(embeddings_param, n_clusters_param):
    """Clusters embeddings using KMeans and displays a progress bar."""
    if embeddings_param is None or embeddings_param.size == 0:
        st.warning("No embeddings to cluster.")
        return None
    if n_clusters_param > embeddings_param.shape[0]:
        st.warning(f"Number of clusters ({n_clusters_param}) cannot exceed number of samples ({embeddings_param.shape[0]}). Adjusting K.")
        n_clusters_param = embeddings_param.shape[0] # Adjust K
        if n_clusters_param < 1: # Should not happen if embeddings_param.size > 0
             st.error("Not enough samples to cluster.")
             return None


    try:
        with st.spinner(f"Clustering embeddings into {n_clusters_param} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters_param, random_state=42, n_init='auto')
            clusters_result = kmeans.fit_predict(embeddings_param)
            st.success(f"Clustering complete!")
            return clusters_result
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

# --- Page Functions ---

def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("This page provides an overview of the job dataset and allows you to explore its features.")

    if st.session_state.get('data') is None: # Use .get for safety
        st.session_state['data'] = load_data_from_url(DATA_URL)

    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Data Preview')
        st.dataframe(data_df.head(), use_container_width=True)

        st.subheader('Data Summary')
        st.write(f'Number of rows: {len(data_df)}')
        st.write(f'Number of columns: {len(data_df.columns)}')

        st.subheader('Search for a Word in a Feature')
        search_word = st.text_input("Enter a word to search:", key="home_search_word")
        column_options = [''] + [str(col) for col in data_df.columns.tolist()]
        search_column = st.selectbox("Select the feature to search in:", column_options, key="home_search_column")

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                if not search_results.empty:
                    st.subheader(f"Search results for '{search_word}' in '{search_column}':")
                    st.dataframe(search_results, use_container_width=True)
                    st.write(f"Found {len(search_results)} matching entries in '{search_column}'.")
                else:
                    st.info(f"No entries found in '{search_column}' containing '{search_word}'.")
            # Removed else: st.error(...) as selectbox ensures column is valid if not empty string

        st.subheader('Feature Information')
        feature_list = data_df.columns.tolist()
        st.write(f'Total Features: **{len(feature_list)}**')
        st.write('**Features:**')
        st.code(str(feature_list))

        st.subheader('Explore Feature Details')
        selected_feature = st.selectbox('Select a Feature to see details:', [''] + feature_list, key="home_feature_select")
        if selected_feature:
            st.write(f'**Feature:** `{selected_feature}`')
            st.write(f'**Data Type:** `{data_df[selected_feature].dtype}`')
            st.write(f'**Number of Unique Values:** `{data_df[selected_feature].nunique()}`')
            st.write('**Sample Unique Values (first 20):**')
            unique_values = data_df[selected_feature].unique()
            st.write(unique_values[:20])
            if len(unique_values) > 20:
                st.caption(f'(Showing first 20 of {len(unique_values)} unique values)')


            if pd.api.types.is_numeric_dtype(data_df[selected_feature]):
                st.subheader(f'Descriptive Statistics for `{selected_feature}`')
                st.write(data_df[selected_feature].describe())
                try: 
                    fig = px.histogram(data_df, x=selected_feature, title=f'Distribution of {selected_feature}')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate histogram for {selected_feature}: {e}")
            elif pd.api.types.is_string_dtype(data_df[selected_feature]) or pd.api.types.is_object_dtype(data_df[selected_feature]):
                st.subheader(f'Value Counts for `{selected_feature}` (Top 20)')
                st.write(data_df[selected_feature].value_counts().nlargest(20)) # Use nlargest for clarity
            # else: # No need for this else, covered by above
            #     st.info('No specific descriptive statistics or value counts for this data type.')
    else:
        st.error("Data could not be loaded. Please check the data source or network connection.")


def preprocessing_page():
    st.header("Preprocessing")
    st.write("This page performs text preprocessing on the 'text' column, showing intermediate steps.")

    if st.session_state.get('data') is None:
        st.info("Data not loaded. Attempting to load now...")
        st.session_state['data'] = load_data_from_url(DATA_URL)
        if st.session_state.get('data') is None:
            st.error("Failed to load data. Please go to Home page and try again.")
            return

    data_df = st.session_state['data']

    # Button to trigger preprocessing
    if st.button("Run Preprocessing", key="run_preprocessing_button"):
        with st.spinner("Preprocessing data..."):
            st.session_state['data'] = preprocess_text_with_intermediate(data_df.copy()) # Work on a copy then assign
        st.success("Preprocessing complete!")
    
    # Display results if preprocessing has been done
    if 'processed_text' in st.session_state['data'].columns:
        st.info("Preprocessing has been performed.")
        if 'preprocessing_steps' in st.session_state['data'].columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)")
            valid_steps = [step for step in st.session_state['data']['preprocessing_steps'] if isinstance(step, dict)]
            if valid_steps:
                 display_df = pd.DataFrame(valid_steps)
                 st.dataframe(display_df.head(), use_container_width=True)
            else:
                 st.warning("Preprocessing steps data is not in the expected format or empty.")

        st.subheader("Final Preprocessed Text (Preview from last run)")
        st.dataframe(st.session_state['data'][['text', 'processed_text']].head(), use_container_width=True)

        st.subheader('Search for a Word in Preprocessed Text')
        search_word_preprocessed = st.text_input("Enter a word to search in 'processed_text':", key="prep_search_word")
        if search_word_preprocessed:
            search_results_preprocessed = st.session_state['data'][st.session_state['data']['processed_text'].astype(str).str.contains(search_word_preprocessed, case=False, na=False)]
            if not search_results_preprocessed.empty:
                st.subheader(f"Search results for '{search_word_preprocessed}' in 'processed_text':")
                st.dataframe(search_results_preprocessed[['Job.ID', 'Title', 'processed_text']], use_container_width=True)
                st.write(f"Found {len(search_results_preprocessed)} matching entries.")
            else:
                st.info(f"No entries found in 'processed_text' containing '{search_word_preprocessed}'.")
    else:
        st.info("Data loaded, but preprocessing has not been run yet. Click the button above.")


def tsdae_page():
    st.header("TSDAE (Sequential Noise Injection)")
    st.write("This page applies noise to the preprocessed text using methods 'a', 'b', and 'c' sequentially and generates embeddings.")

    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page and running it.")
        return

    data_df = st.session_state['data'] # Make a copy to modify locally if needed, then update session_state
    bert_model = load_bert_model()

    if bert_model is None: # Check if model loaded successfully
        st.error("BERT model could not be loaded. Cannot proceed with TSDAE.")
        return

    st.subheader("TSDAE Settings")
    deletion_ratio = st.slider("Deletion Ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.1, key="tsdae_del_ratio")
    freq_threshold = st.slider("High Frequency Threshold (for methods 'b' & 'c')", min_value=10, max_value=500, value=100, step=10, key="tsdae_freq_thresh")

    if st.button("Apply Sequential Noise and Generate Embeddings", key="tsdae_apply_noise_button"):
        current_data_df = st.session_state['data'].copy() # Work on a copy

        # Compute word_freq_dict from current processed_text
        all_words = []
        for text_content in current_data_df['processed_text'].fillna('').tolist():
            all_words.extend(word_tokenize(str(text_content))) 
        word_freq_dict = {word.lower(): all_words.count(word.lower()) for word in set(all_words)}
        if not word_freq_dict:
            st.warning("Word frequency dictionary is empty. Methods 'b' and 'c' might not work as expected.")
            # Allow proceeding, denoise_text handles None word_freq_dict by raising error for b,c

        # Method A
        with st.spinner("Applying Random Deletion (Method 'a')..."):
            current_data_df['noisy_text_a'] = current_data_df['processed_text'].fillna('').astype(str).apply(
                lambda x: denoise_text(x, method='a', del_ratio=deletion_ratio)
            )
        st.success("Method 'a' noise application complete.")

        # Method B
        with st.spinner("Applying High-Frequency Word Removal (Method 'b')..."):
            current_data_df['noisy_text_b'] = current_data_df['noisy_text_a'].astype(str).apply(
                lambda x: denoise_text(x, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold)
            )
        st.success("Method 'b' noise application complete.")

        # Method C
        with st.spinner("Applying High-Frequency Word Removal + Shuffle (Method 'c')..."):
            current_data_df['final_noisy_text'] = current_data_df['noisy_text_b'].astype(str).apply(
                lambda x: denoise_text(x, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold)
            )
        st.success("Method 'c' noise application complete.")
        
        st.session_state['data'] = current_data_df # Update session state with noisy texts

        st.subheader("Sequentially Noisy Text (Preview)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), use_container_width=True)

        st.subheader("Generating Embeddings for Final Noisy Text (TSDAE Embeddings)")
        if 'final_noisy_text' in st.session_state['data'].columns:
            texts_for_tsdae_embedding = st.session_state['data']['final_noisy_text'].astype(str).tolist()
            final_noisy_embeddings = generate_embeddings_with_progress(bert_model, texts_for_tsdae_embedding)
            if final_noisy_embeddings.size > 0:
                st.session_state['tsdae_embeddings'] = final_noisy_embeddings
                st.success("TSDAE embeddings generated successfully!")
                st.write("Shape of TSDAE embeddings:", final_noisy_embeddings.shape)
                st.write("Preview (first 3):", final_noisy_embeddings[:3])
            else:
                st.warning("Failed to generate TSDAE embeddings for the final noisy text (result was empty).")
        else:
            st.error("'final_noisy_text' column not found. Cannot generate TSDAE embeddings.")
    
    # Display existing TSDAE embeddings if already generated
    if 'tsdae_embeddings' in st.session_state and st.session_state['tsdae_embeddings'] is not None:
        st.subheader("Existing TSDAE Embeddings (Preview)")
        st.write("Shape:", st.session_state['tsdae_embeddings'].shape)
        st.write(st.session_state['tsdae_embeddings'][:3])
    if 'final_noisy_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader("Current Noisy Text Columns (Preview from last run)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), use_container_width=True)


def bert_model_page():
    st.header("BERT Model: Job Description Embedding Generation & Visualization")
    st.write("This page generates standard BERT embeddings from the preprocessed job descriptions and visualizes them.")

    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page and running it.")
        return

    data_df = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is None:
        st.error("BERT model could not be loaded. Cannot proceed.")
        return

    if st.button("Generate/Regenerate Job Description Embeddings", key="generate_job_embeddings_button"):
        if 'processed_text' in data_df.columns and not data_df['processed_text'].empty:
            texts_to_embed = data_df['processed_text'].fillna('').astype(str).tolist()
            with st.spinner("Generating job description embeddings..."):
                 st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, texts_to_embed)
            if st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
                st.success("Job description embeddings generated successfully!")
            else:
                st.warning("Job description embedding generation resulted in empty output.")
        else:
            st.warning("No text found in 'processed_text' column or column is missing. Cannot generate embeddings.")

    job_text_embeddings = st.session_state.get('job_text_embeddings')
    if job_text_embeddings is not None and job_text_embeddings.size > 0:
        st.subheader("Job Description Embeddings (Matrix Preview)")
        st.write("Shape of the embedding matrix:", job_text_embeddings.shape)
        st.write("Preview of the first 3 embeddings:")
        st.write(job_text_embeddings[:3])
        st.info(f"Each processed job description is represented by a vector of {job_text_embeddings.shape[1]} dimensions.")

        st.subheader("2D Visualization of Job Description Embeddings (PCA)")
        if len(job_text_embeddings) >= 2: 
            try:
                pca = PCA(n_components=2)
                reduced_embeddings_2d = pca.fit_transform(job_text_embeddings)

                plot_df = pd.DataFrame(reduced_embeddings_2d, columns=['PC1', 'PC2'])
                num_embeddings = len(plot_df)
                # Ensure data_df has enough rows and the columns exist
                plot_df['title'] = data_df['Title'].tolist()[:num_embeddings] if 'Title' in data_df.columns else [f"Title {i}" for i in range(num_embeddings)]
                plot_df['description'] = data_df['text'].tolist()[:num_embeddings] if 'text' in data_df.columns else [f"Desc {i}" for i in range(num_embeddings)]

                fig = px.scatter(plot_df, x='PC1', y='PC2',
                                 hover_name='title', 
                                 hover_data={'description': True, 'PC1': False, 'PC2': False},
                                 title='2D Visualization of Job Embeddings (PCA)',
                                 width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error during PCA visualization: {e}")
        else:
            st.warning("Not enough data points (need at least 2) to perform PCA.")
    else:
        st.info("Job description embeddings have not been generated yet. Click the button above.")


def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("This page clusters the generated job embeddings (either TSDAE or standard BERT) using K-Means.")

    embeddings_to_cluster = None
    embedding_source_name = ""
    data_df = st.session_state.get('data')

    if data_df is None:
        st.error("Job data not loaded. Cannot proceed with clustering.")
        return

    # Determine which embeddings to use
    use_tsdae_key = "use_tsdae_for_clustering_toggle"
    if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
        use_tsdae = st.toggle("Use TSDAE embeddings for clustering (if available)", value=True, key=use_tsdae_key)
        if use_tsdae:
            embeddings_to_cluster = st.session_state['tsdae_embeddings']
            embedding_source_name = "TSDAE Embeddings"
        elif st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0 : # Fallback if toggle is off
            embeddings_to_cluster = st.session_state['job_text_embeddings']
            embedding_source_name = "Standard BERT Job Embeddings"
        else: # TSDAE available, toggle off, but no standard BERT embeddings
            st.warning("TSDAE embeddings available but deselected. Standard BERT job embeddings are not generated. Please generate them on the 'BERT Model' page or select TSDAE.")
            return
    elif st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
        embeddings_to_cluster = st.session_state['job_text_embeddings']
        embedding_source_name = "Standard BERT Job Embeddings"
        st.info("Using Standard BERT Job Embeddings. TSDAE embeddings not found.")
    else:
        st.warning("No embeddings (neither TSDAE nor standard BERT) available for clustering. Please generate them on the 'TSDAE' or 'BERT Model' pages.")
        return
    
    st.info(f"Selected embeddings for clustering: {embedding_source_name}")

    if embeddings_to_cluster is not None and embeddings_to_cluster.size > 0:
        st.subheader(f"Clustering Settings ({embedding_source_name})")
        
        max_k_value = embeddings_to_cluster.shape[0] # K cannot be more than num_samples
        if max_k_value < 2: 
            st.error("Not enough samples in the selected embeddings to perform clustering (need at least 2).")
            return

        num_clusters_input = st.slider("Number of Clusters (K)", 
                                       min_value=2, 
                                       max_value=min(50, max_k_value), 
                                       value=min(N_CLUSTERS, max_k_value), 
                                       key="clustering_k_slider")

        if st.button(f"Perform K-Means Clustering with K={num_clusters_input}", key="perform_clustering_button"):
            clusters = cluster_embeddings_with_progress(embeddings_to_cluster, num_clusters_input)
            if clusters is not None:
                st.session_state['job_clusters'] = clusters # Store raw cluster labels
                # Add/update 'cluster' column in the main data DataFrame
                if len(data_df) == len(clusters):
                    data_df_copy = st.session_state['data'].copy() # Modify a copy
                    data_df_copy['cluster'] = clusters
                    st.session_state['data'] = data_df_copy # Assign back
                    st.success("Clustering complete and cluster labels added/updated in the main dataset.")
                else:
                    st.error(f"Mismatch in length: Data has {len(data_df)} rows, but {len(clusters)} clusters were generated. Cannot assign clusters to the main dataset.")
                    # Do not add 'cluster' column if lengths mismatch to avoid corruption
                    # st.session_state['job_clusters'] can still be kept for direct inspection if needed
            else:
                st.error("Clustering failed to produce results.")

        # Display results if 'cluster' column is in the data (implies successful assignment)
        if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns:
            current_k_display = st.session_state['data']['cluster'].nunique()
            st.subheader(f"Clustering Results (K={current_k_display})")
            st.write("Job Details and Cluster Assignments (Top 10):")
            st.dataframe(st.session_state['data'][['Job.ID', 'Title', 'text', 'cluster']].head(10), use_container_width=True)

            st.subheader("Sample Job Descriptions per Cluster")
            unique_clusters_display = sorted(st.session_state['data']['cluster'].unique())
            for cluster_num_display in unique_clusters_display:
                st.write(f"**Cluster {cluster_num_display}:**")
                cluster_data_display = st.session_state['data'][st.session_state['data']['cluster'] == cluster_num_display]
                if not cluster_data_display.empty:
                    sample_size = min(5, len(cluster_data_display))
                    cluster_sample_display = cluster_data_display.sample(sample_size, random_state=42) # Add random_state for consistency
                    st.dataframe(cluster_sample_display[['Job.ID', 'Title', 'text', 'cluster']], use_container_width=True)
                st.write("---")
        else:
            st.info("Clustering has not been performed or cluster labels were not assigned to the dataset. Click the button above.")
    # else: # Covered by initial checks
    #     st.warning("Embeddings are not available or empty. Cannot proceed with clustering.")


def upload_cv_page():
    st.header("Upload CV(s)")
    st.write("Upload your CV(s) in PDF or Word format (max 5 files). Processed text and embeddings will be generated.")
    
    uploaded_files = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True, key="cv_file_uploader")
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("You can upload a maximum of 5 CVs. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]

        if st.button("Process Uploaded CVs", key="process_cvs_button"):
            # Initialize/clear list for this batch of uploads
            processed_cvs_in_batch = [] 
            bert_model = load_bert_model() 

            if not bert_model:
                st.error("BERT model failed to load. Cannot process CVs.")
                return 

            with st.spinner("Processing uploaded CVs..."):
                cv_upload_progress_bar = st.progress(0)
                cv_upload_status_text = st.empty()
                
                for i, uploaded_file_obj in enumerate(uploaded_files):
                    cv_text_content = ""
                    processed_cv_text = ""
                    cv_embedding = None
                    try:
                        file_extension = uploaded_file_obj.name.split(".")[-1].lower()
                        if file_extension == "pdf":
                            cv_text_content = extract_text_from_pdf(uploaded_file_obj)
                        elif file_extension == "docx":
                            cv_text_content = extract_text_from_docx(uploaded_file_obj)
                        
                        if cv_text_content and cv_text_content.strip():
                            processed_cv_text = preprocess_text(cv_text_content)
                            if processed_cv_text and processed_cv_text.strip(): # Ensure processed text is also not empty
                                cv_embedding_array = generate_embeddings_with_progress(bert_model, [processed_cv_text])
                                cv_embedding = cv_embedding_array[0] if (cv_embedding_array is not None and cv_embedding_array.size > 0) else None
                            else:
                                st.warning(f"Processed text for {uploaded_file_obj.name} is empty. Skipping embedding.")
                        else:
                            st.warning(f"Could not extract text or extracted text is empty from {uploaded_file_obj.name}.")
                        
                        # Add to batch list even if embedding failed, to show it was attempted
                        processed_cvs_in_batch.append({
                            'filename': uploaded_file_obj.name,
                            'original_text': cv_text_content if cv_text_content else "", # Ensure not None
                            'processed_text': processed_cv_text if processed_cv_text else "", # Ensure not None
                            'embedding': cv_embedding 
                        })
                        if cv_embedding is not None:
                             st.success(f"Successfully processed and embedded: {uploaded_file_obj.name}")
                        elif cv_text_content and processed_cv_text: # Text extracted and processed, but embedding failed
                             st.warning(f"Processed {uploaded_file_obj.name}, but failed to generate embedding.")


                    except Exception as e:
                        st.error(f"Error processing {uploaded_file_obj.name}: {e}")
                        processed_cvs_in_batch.append({ # Add with error indication
                            'filename': uploaded_file_obj.name, 'original_text': "Error in processing.", 
                            'processed_text': "", 'embedding': None
                        })
                    
                    if len(uploaded_files) > 0:
                        cv_upload_progress_bar.progress((i + 1) / len(uploaded_files))
                        cv_upload_status_text.text(f"Processed {i + 1}/{len(uploaded_files)} CVs.")
                
                st.session_state['uploaded_cvs_data'] = processed_cvs_in_batch # Store all attempts from this batch
                cv_upload_progress_bar.empty()
                cv_upload_status_text.empty()
                st.success(f"CV processing batch complete. {len(processed_cvs_in_batch)} CVs attempted.")

    # Display currently stored CVs from session state
    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs in this Session:")
        for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_data.get('filename', 'N/A')}", expanded=False):
                st.text_area(f"Original Text", cv_data.get('original_text','N/A'), height=100, disabled=True, key=f"disp_orig_cv_text_{i}")
                st.text_area(f"Processed Text", cv_data.get('processed_text','N/A'), height=100, disabled=True, key=f"disp_proc_cv_text_{i}")
                if cv_data.get('embedding') is not None:
                    st.success("Embedding available.")
                else:
                    st.warning("Embedding not available for this CV.")
    else:
        st.info("No CVs have been processed and stored in this session yet.")


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("This page provides job recommendations based on your uploaded CV and the job postings.")

    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("Please upload and process CV(s) on the 'Upload CV' page first.")
        return

    data_df = st.session_state.get('data')
    if data_df is None or 'processed_text' not in data_df.columns:
        st.error("Job data not loaded or not preprocessed. Please ensure data is loaded on Home and preprocessed on Preprocessing page.")
        return
    
    job_embeddings_for_similarity = None
    embedding_source_msg = ""
    # Option to select which job embeddings to use for recommendation
    job_embedding_choice = st.radio(
        "Select Job Embeddings for Recommendation:",
        ("Standard BERT Job Embeddings", "TSDAE Job Embeddings"),
        key="rec_job_embedding_choice", horizontal=True
    )

    if job_embedding_choice == "TSDAE Job Embeddings":
        if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
            job_embeddings_for_similarity = st.session_state['tsdae_embeddings']
            embedding_source_msg = "Using TSDAE embeddings for recommendation."
        else:
            st.warning("TSDAE embeddings not available. Please generate them on the 'TSDAE' page or select Standard BERT embeddings.")
            return
    else: # Standard BERT Job Embeddings
        if st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
            job_embeddings_for_similarity = st.session_state['job_text_embeddings']
            embedding_source_msg = "Using standard BERT embeddings (from processed job text) for recommendation."
        else:
            st.warning("Standard BERT job embeddings not available. Please generate them on the 'BERT Model' page or select TSDAE embeddings if available.")
            return
    st.info(embedding_source_msg)

    if 'cluster' not in data_df.columns:
        st.warning("Job clusters are not available. Recommendations will be generated without cluster information. "
                   "Run clustering on the 'Clustering Job2Vec' page if desired.")

    if st.button("Generate Recommendations for All Stored CVs", key="generate_recs_button"):
        st.session_state['all_recommendations_for_annotation'] = {} 

        with st.spinner("Generating recommendations for stored CVs..."):
            valid_cvs_for_rec = [cv for cv in st.session_state.get('uploaded_cvs_data', []) if cv.get('embedding') is not None]
            if not valid_cvs_for_rec:
                st.warning("No CVs with embeddings found to generate recommendations.")
                return

            for i, cv_data in enumerate(valid_cvs_for_rec):
                cv_filename = cv_data.get('filename', f'CV_{i+1}')
                cv_embedding = cv_data['embedding'] # Already checked it's not None

                st.subheader(f"Recommendations for {cv_filename}")
                cv_embedding_2d = cv_embedding.reshape(1, -1) if cv_embedding.ndim == 1 else cv_embedding
                
                if job_embeddings_for_similarity.ndim == 1 or job_embeddings_for_similarity.shape[0] == 0:
                     st.error(f"Selected job embeddings ({embedding_source_msg.split(' ')[1]}) are invalid (1D or empty). Cannot compute similarity for {cv_filename}.")
                     continue 

                similarities = cosine_similarity(cv_embedding_2d, job_embeddings_for_similarity)[0]
                
                temp_rec_df = data_df.copy()
                if len(similarities) == len(temp_rec_df):
                    temp_rec_df['similarity_score'] = similarities
                else:
                    st.error(f"Length mismatch: {len(similarities)} similarities vs {len(temp_rec_df)} jobs for CV {cv_filename}. Skipping.")
                    continue
                    
                recommended_jobs = temp_rec_df.sort_values(by='similarity_score', ascending=False).head(20)

                if not recommended_jobs.empty:
                    display_columns = ['Job.ID', 'Title', 'similarity_score']
                    if 'cluster' in recommended_jobs.columns: # Check if 'cluster' column exists
                        display_columns.append('cluster')
                    display_columns.append('text') 
                    
                    st.dataframe(recommended_jobs[display_columns], use_container_width=True)
                    st.session_state['all_recommendations_for_annotation'][cv_filename] = recommended_jobs
                else:
                    st.info(f"No job recommendations found for {cv_filename}.")
                st.write("---") 
        st.success("Recommendation generation process complete!")


def annotation_page():
    st.header("Annotation")
    st.write("Annotate the relevance of the top 20 job recommendations for each CV.")

    if not st.session_state.get('all_recommendations_for_annotation'):
        st.warning("No recommendations available for annotation. Please generate recommendations on the 'Job Recommendation' page first.")
        return

    st.subheader("Annotation Form")
    
    if 'collected_annotations' not in st.session_state or not isinstance(st.session_state['collected_annotations'], pd.DataFrame):
        st.session_state['collected_annotations'] = pd.DataFrame()

    with st.form(key="annotation_form_all"):
        # This list will store dicts, each dict is a row for the final DataFrame
        current_form_input_data = [] 

        for cv_filename, recommendations_df in st.session_state['all_recommendations_for_annotation'].items():
            st.markdown(f"### Annotate Recommendations for CV: **{cv_filename}**")

            for _, row in recommendations_df.iterrows():
                job_id = str(row['Job.ID']) # Ensure Job.ID is string for keys and matching

                st.markdown(f"**Job ID:** {job_id} | **Title:** {row['Title']}")
                with st.expander("View Details", expanded=False):
                    st.write(f"**Description:** {row['text']}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
                    if 'cluster' in row and pd.notna(row['cluster']):
                        st.write(f"**Cluster:** {row['cluster']}")
                
                # Base info for this job, common to all its annotator entries in the form
                job_annotation_input = {
                    'cv_filename': cv_filename,
                    'job_id': job_id,
                    'job_title': row['Title'],
                    'job_text': row['text'],
                    'similarity_score': row['similarity_score'],
                    'cluster': row.get('cluster', pd.NA) 
                }

                # Inputs for each annotator for this specific job
                annotator_inputs_for_job = {}
                cols = st.columns(len(ANNOTATORS))
                for i, annotator_name in enumerate(ANNOTATORS):
                    with cols[i]:
                        st.markdown(f"**{annotator_name}**")
                        relevance_key = f"relevance_{cv_filename}_{job_id}_{annotator_name}"
                        feedback_key = f"feedback_{cv_filename}_{job_id}_{annotator_name}"

                        # Retrieve previous annotation for this specific job and annotator if it exists
                        default_relevance = 0
                        default_feedback = ""
                        if not st.session_state['collected_annotations'].empty:
                            mask = (st.session_state['collected_annotations']['cv_filename'] == cv_filename) & \
                                   (st.session_state['collected_annotations']['job_id'] == job_id)
                            # The ANNOTATORS list gives us "Annotator 1", "Annotator 2", etc.
                            # Column names are f'annotator_{i+1}_relevance'
                            annotator_relevance_col = f'annotator_{i+1}_relevance'
                            annotator_feedback_col = f'annotator_{i+1}_feedback'
                            
                            # Check if the column exists before trying to access it
                            if annotator_relevance_col in st.session_state['collected_annotations'].columns:
                                relevant_row = st.session_state['collected_annotations'][mask]
                                if not relevant_row.empty:
                                    val = relevant_row.iloc[0].get(annotator_relevance_col)
                                    if pd.notna(val): default_relevance = int(val)
                                    
                                    if annotator_feedback_col in relevant_row.columns:
                                         default_feedback = str(relevant_row.iloc[0].get(annotator_feedback_col, ""))


                        relevance = st.radio(
                            "Relevance:", options=[0, 1, 2, 3],
                            format_func=lambda x: f"{x} (" + ("Very Irrelevant" if x == 0 else "Slightly Relevant" if x == 1 else "Moderately Relevant" if x == 2 else "Very Relevant") + ")",
                            index=default_relevance, 
                            key=relevance_key, horizontal=True
                        )
                        qualitative_feedback = st.text_area(
                            "Feedback:", value=default_feedback, key=feedback_key, height=60
                        )
                        
                        annotator_inputs_for_job[f'annotator_{i+1}_name'] = annotator_name
                        annotator_inputs_for_job[f'annotator_{i+1}_relevance'] = relevance
                        annotator_inputs_for_job[f'annotator_{i+1}_feedback'] = qualitative_feedback
                
                # Combine base job info with all annotator inputs for that job
                job_annotation_input.update(annotator_inputs_for_job)
                current_form_input_data.append(job_annotation_input)
                st.markdown("---") 

        submitted = st.form_submit_button("Submit All Annotations")

    if submitted:
        if current_form_input_data:
            new_annotations_df = pd.DataFrame(current_form_input_data)
            
            # Strategy: Replace old annotations with new ones based on cv_filename and job_id
            # This ensures that submitting the form updates all annotator fields for the displayed jobs.
            if not st.session_state['collected_annotations'].empty:
                # Identify jobs that are in the new submission
                submitted_job_keys = new_annotations_df[['cv_filename', 'job_id']].drop_duplicates()
                
                # Filter out these jobs from the old annotations
                # Create a temporary multi-index for efficient filtering
                temp_old_df = st.session_state['collected_annotations'].copy()
                temp_old_df = temp_old_df.set_index(['cv_filename', 'job_id'])
                submitted_job_tuples = [tuple(x) for x in submitted_job_keys.to_numpy()]
                
                # Drop rows from old_df that are present in new_df based on the multi-index
                # This is a bit complex; simpler is to filter out rows that match cv_filename & job_id from new_df
                # and then concat.
                
                # Keep rows in old_df that are NOT for the cv/job pairs in the current submission
                # This preserves annotations for jobs/CVs not currently in 'all_recommendations_for_annotation'
                condition = ~pd.MultiIndex.from_frame(st.session_state['collected_annotations'][['cv_filename', 'job_id']]).isin(pd.MultiIndex.from_frame(submitted_job_keys))

                if condition.any(): # Check if condition is not all False
                    preserved_annotations = st.session_state['collected_annotations'][condition]
                    st.session_state['collected_annotations'] = pd.concat([preserved_annotations, new_annotations_df], ignore_index=True)
                else: # All old annotations were for jobs in the current submission, or old df was empty for these keys
                    st.session_state['collected_annotations'] = new_annotations_df

            else: # No prior annotations
                st.session_state['collected_annotations'] = new_annotations_df
            
            st.success("Annotations submitted/updated successfully!")
        else:
            st.warning("No annotation data was entered in the form.")

    # Display current collected annotations after potential submission
    if not st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        st.subheader("Collected Annotations Preview")
        st.dataframe(st.session_state['collected_annotations'], use_container_width=True)
        csv_buffer = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Annotations as CSV",
            data=csv_buffer,
            file_name="job_recommendation_annotations.csv",
            mime="text/csv",
            key="download_annotations_button"
        )
    else:
        st.info("No annotations collected yet.")


# --- NEW EVALUATION PAGE ---
def evaluation_page():
    st.header("Model Evaluation")
    st.write("Evaluate the recommendation model using collected annotations.")

    bert_model = load_bert_model()
    if bert_model is None:
        st.error("BERT model could not be loaded. Evaluation cannot proceed.")
        return

    # --- Prerequisite checks ---
    if 'data' not in st.session_state or 'processed_text' not in st.session_state['data'].columns:
        st.warning("Job data with 'processed_text' is not available. Please load and preprocess data first.")
        return
    
    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("No CVs uploaded or processed. Please upload CVs on the 'Upload CV' page.")
        return
    
    # Check if CVs have processed_text
    cvs_ready_for_eval = [cv for cv in st.session_state['uploaded_cvs_data'] if cv.get('processed_text') and cv['processed_text'].strip()]
    if not cvs_ready_for_eval:
        st.warning("No CVs with processed text available for evaluation. Ensure CVs are processed on the 'Upload CV' page.")
        return

    if st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        st.warning("No annotations collected yet. Please annotate recommendations on the 'Annotation' page.")
        return

    # --- Evaluation Parameters ---
    st.subheader("Evaluation Parameters")
    relevance_threshold = st.slider(
        "Relevance Threshold (Average score >= threshold is relevant)",
        min_value=0.0, max_value=3.0, value=1.5, step=0.1,
        key="eval_relevance_threshold"
    )

    if st.button("Run Evaluation", key="run_evaluation_button"):
        with st.spinner("Preparing data and running evaluation..."):
            # 1. Prepare Queries: CVs' processed text
            queries = {}
            for cv_data in cvs_ready_for_eval:
                # Use filename as query_id, ensure it's a string
                queries[str(cv_data['filename'])] = cv_data['processed_text']
            
            if not queries:
                st.error("No valid queries (CVs with processed text) found for evaluation.")
                return

            # 2. Prepare Corpus: Job descriptions' processed text
            corpus_df = st.session_state['data']
            # Ensure Job.ID is string and processed_text exists
            corpus = dict(zip(corpus_df['Job.ID'].astype(str), corpus_df['processed_text']))

            if not corpus:
                st.error("Job corpus is empty or 'processed_text' is missing. Cannot run evaluation.")
                return

            # 3. Prepare Relevant Docs (Ground Truth from Annotations)
            relevant_docs = {}
            annotations_df = st.session_state['collected_annotations']
            
            # Identify annotator relevance columns dynamically
            annotator_relevance_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS)) if f'annotator_{i+1}_relevance' in annotations_df.columns]

            if not annotator_relevance_cols:
                st.error("No annotator relevance columns found in the collected annotations. Cannot determine ground truth.")
                return

            # Group by CV and Job to calculate average relevance
            # Ensure job_id is string for consistency
            annotations_df['job_id'] = annotations_df['job_id'].astype(str)
            grouped_annotations = annotations_df.groupby(['cv_filename', 'job_id'])

            for (cv_file, job_id_str), group in grouped_annotations:
                scores = []
                for col in annotator_relevance_cols:
                    # Get scores, convert to numeric, filter NaNs
                    valid_scores = pd.to_numeric(group[col], errors='coerce').dropna().tolist()
                    scores.extend(valid_scores)
                
                if scores: # If there are any valid scores for this cv/job pair
                    average_score = np.mean(scores)
                    if average_score >= relevance_threshold:
                        cv_file_str = str(cv_file) # Ensure cv_file is string key
                        if cv_file_str not in relevant_docs:
                            relevant_docs[cv_file_str] = set()
                        relevant_docs[cv_file_str].add(job_id_str)
            
            if not relevant_docs:
                st.warning(f"No relevant documents found based on annotations and threshold {relevance_threshold}. Evaluation might yield all zeros.")
                # Still proceed, evaluator can handle empty relevant_docs for some queries.

            # Filter queries to only those present in relevant_docs or those for which we want to evaluate (all processed CVs)
            # Evaluator handles queries not in relevant_docs (they will have 0 relevant items)
            
            st.write(f"Number of queries (CVs) for evaluation: {len(queries)}")
            st.write(f"Number of documents in corpus (Jobs): {len(corpus)}")
            st.write(f"Number of CVs with at least one relevant job identified from annotations: {len(relevant_docs)}")
            
            # 4. Instantiate and Run Evaluator
            try:
                ir_evaluator = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name="job_recommendation_eval", # Custom name
                    show_progress_bar=True, # Show progress bar for encoding
                    main_score_function='cosine' # or 'dot'
                )
                results = ir_evaluator(bert_model, output_path=None) # output_path=None to not save to disk

                st.subheader("Evaluation Results")
                if results and isinstance(results, dict): # Check if results is a dict
                    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Score'])
                    results_df.index.name = "Metric"
                    st.dataframe(results_df)
                    
                    # Display primary metric if available in results (depends on evaluator version/config)
                    # primary_metric_key = ir_evaluator.primary_metric # This might not exist or be what we expect
                    # A common primary metric for IR is MAP
                    map_keys = [k for k in results.keys() if 'map' in k.lower()]
                    if map_keys:
                         st.metric(label=f"Primary Metric ({map_keys[0]})", value=f"{results[map_keys[0]]:.4f}")

                else: # Handle cases where results might not be a dict (e.g. older versions or error)
                    st.write("Raw results:", results)


            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")
                st.exception(e) # Show full traceback for debugging

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"] # Added Evaluation
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
elif page == "Evaluation": # New page
    evaluation_page()
