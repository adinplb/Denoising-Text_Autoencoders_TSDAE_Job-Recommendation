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
if 'data' not in st.session_state:
    st.session_state['data'] = None # Main job dataset
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None # Embeddings for processed job text
if 'job_text_embedding_job_ids' not in st.session_state: # NEW: Job.IDs corresponding to job_text_embeddings
    st.session_state['job_text_embedding_job_ids'] = None
if 'tsdae_embeddings' not in st.session_state:
    st.session_state['tsdae_embeddings'] = None # Embeddings for TSDAE processed job text
if 'tsdae_embedding_job_ids' not in st.session_state: # NEW: Job.IDs corresponding to tsdae_embeddings
    st.session_state['tsdae_embedding_job_ids'] = None
if 'job_clusters_raw' not in st.session_state: # Raw cluster labels from KMeans (matches embedding length)
    st.session_state['job_clusters_raw'] = None
# Note: 'job_clusters' in st.session_state.data['cluster'] will be the merged cluster assignments

if 'uploaded_cvs_data' not in st.session_state:
    st.session_state['uploaded_cvs_data'] = [] # Stores list of {'filename', 'original_text', 'processed_text', 'embedding'}
if 'all_recommendations_for_annotation' not in st.session_state:
    st.session_state['all_recommendations_for_annotation'] = {} 
if 'collected_annotations' not in st.session_state:
    st.session_state['collected_annotations'] = pd.DataFrame()


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading data...')
def load_data_from_url(url):
    """Loads job data from a given URL and selects relevant features."""
    try:
        df = pd.read_csv(url)
        st.success('Successfully loaded data!')
        if 'Job.ID' in df.columns:
            df['Job.ID'] = df['Job.ID'].astype(str) # Ensure Job.ID is string
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

def preprocess_text(text):
    """
    Performs text preprocessing steps: symbol removal, case folding, tokenization,
    stopwords removal, and stemming. Returns empty string if input is not string or result is empty.
    """
    if not isinstance(text, str) or not text.strip(): 
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words and w.isalnum()]
    if not filtered_words: # If all words are stopwords/punctuation
        return ""
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in filtered_words]
    return " ".join(stemmed_words)

def preprocess_text_with_intermediate(data_df):
    """
    Performs text preprocessing, stores intermediate steps, and adds 'processed_text' column.
    """
    processed_results_intermediate = [] # Renamed to avoid conflict
    if 'text' not in data_df.columns:
        st.warning("The 'text' column was not found in the dataset for preprocessing.")
        return data_df # Return original df if 'text' column is missing

    with st.spinner("Preprocessing 'text' column... This might take a moment."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)

        # Ensure 'text' column is string and fill NaNs for intermediate step display
        texts_for_intermediate_display = data_df['text'].fillna('').astype(str)

        for i, text_content in enumerate(texts_for_intermediate_display):
            intermediate = {'original': text_content}
            symbol_removed = text_content.translate(str.maketrans('', '', string.punctuation))
            symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed)
            intermediate['symbol_removed'] = symbol_removed
            
            case_folded = symbol_removed.lower()
            intermediate['case_folded'] = case_folded
            
            word_tokens_temp = word_tokenize(case_folded)
            intermediate['tokenized'] = " ".join(word_tokens_temp)
            
            stop_words_temp = set(stopwords.words('english'))
            # Further filter tokens for stopwords_removed and stemmed steps
            valid_tokens_for_stop_stem = [w for w in word_tokens_temp if w.isalnum()]
            filtered_temp = [w for w in valid_tokens_for_stop_stem if w not in stop_words_temp]
            intermediate['stopwords_removed'] = " ".join(filtered_temp)
            
            porter_temp = PorterStemmer()
            stemmed_temp = [porter_temp.stem(w) for w in filtered_temp]
            intermediate['stemmed'] = " ".join(stemmed_temp)
            
            processed_results_intermediate.append(intermediate)
            if total_rows > 0:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries (intermediate steps).")
        
        # Apply the robust preprocess_text function for the final 'processed_text' column
        data_df['processed_text'] = data_df['text'].fillna('').astype(str).apply(preprocess_text)
        data_df['preprocessing_steps'] = processed_results_intermediate

        st.success("Preprocessing of 'text' column complete!")
        progress_bar.empty()
        status_text.empty()
    return data_df

def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    if not isinstance(text_to_denoise, str) or not text_to_denoise.strip():
        return "" # Return empty if input is bad or empty after strip
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return "" 
    
    result_words = [] 
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0: 
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
        if method == 'c' and result_words: # Shuffle only if list is not empty
            random.shuffle(result_words)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result_words)


@st.cache_resource
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

@st.cache_data
def generate_embeddings_with_progress(_model, texts_list_to_embed): # Renamed
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([]) # Return empty array, not None
    
    # Input texts_list_to_embed is already filtered for non-empty strings by the caller
    if not texts_list_to_embed: 
        st.warning("Input text list for embedding is empty. Skipping embedding generation.")
        return np.array([])
        
    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = [] # Renamed
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size] # Renamed
                batch_embeddings_np_array = _model.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False) # Renamed
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
def cluster_embeddings_with_progress(embeddings_to_cluster_param, n_clusters_for_algo): # Renamed
    if embeddings_to_cluster_param is None or embeddings_to_cluster_param.size == 0:
        st.warning("No embeddings to cluster.")
        return None
    if n_clusters_for_algo > embeddings_to_cluster_param.shape[0]:
        st.warning(f"Number of clusters ({n_clusters_for_algo}) cannot exceed number of samples ({embeddings_to_cluster_param.shape[0]}). Adjusting K.")
        n_clusters_for_algo = embeddings_to_cluster_param.shape[0]
        if n_clusters_for_algo < 1: 
             st.error("Not enough samples to cluster (K < 1 after adjustment).")
             return None
    if n_clusters_for_algo < 2 and embeddings_to_cluster_param.shape[0] >=2 : # KMeans needs at least 2 clusters if possible
        st.warning(f"KMeans typically requires at least 2 clusters. Setting K to 2.")
        n_clusters_for_algo = 2


    try:
        with st.spinner(f"Clustering {embeddings_to_cluster_param.shape[0]} embeddings into {n_clusters_for_algo} clusters..."):
            # n_init='auto' is default in newer scikit-learn, explicit for clarity
            kmeans = KMeans(n_clusters=n_clusters_for_algo, random_state=42, n_init='auto')
            clusters_assigned = kmeans.fit_predict(embeddings_to_cluster_param) # Renamed
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
                    st.write(f"Found {len(search_results)} matching entries.")
                else:
                    st.info(f"No entries found for '{search_word}' in '{search_column}'.")
        st.subheader('Feature Information')
        feature_list = data_df.columns.tolist()
        st.write(f'Total Features: **{len(feature_list)}**')
        st.write('**Features:**', feature_list) # Display as list
        st.subheader('Explore Feature Details')
        selected_feature = st.selectbox('Select a Feature:', [''] + feature_list, key="home_feature_select")
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
                st.write(data_df[selected_feature].value_counts().nlargest(20))
    else:
        st.error("Data could not be loaded. Please check the data source or network connection.")

def preprocessing_page():
    st.header("Preprocessing Job Descriptions")
    st.write("This page performs text preprocessing on the 'text' column of the main job dataset.")
    if st.session_state.get('data') is None:
        st.info("Job data not loaded. Attempting to load now...")
        st.session_state['data'] = load_data_from_url(DATA_URL)
        if st.session_state.get('data') is None:
            st.error("Failed to load job data. Please try again or check the data source.")
            return
    
    data_df_to_preprocess = st.session_state['data'] # Use current data from session state

    if st.button("Run Preprocessing on Job Descriptions", key="run_job_preprocessing_button"):
        with st.spinner("Preprocessing job descriptions..."):
            # Create a copy to ensure modifications don't affect other uses of st.session_state.data prematurely
            data_copy = data_df_to_preprocess.copy()
            processed_data = preprocess_text_with_intermediate(data_copy)
            st.session_state['data'] = processed_data # Update session state with the processed data
        st.success("Job description preprocessing complete!")
    
    # Display results if preprocessing has been done (check for 'processed_text' in the session data)
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Job description preprocessing has been performed.")
        display_data = st.session_state['data']
        if 'preprocessing_steps' in display_data.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)")
            valid_steps = [step for step in display_data['preprocessing_steps'] if isinstance(step, dict)]
            if valid_steps:
                 intermediate_df = pd.DataFrame(valid_steps) # Renamed
                 st.dataframe(intermediate_df.head(), use_container_width=True)
            else:
                 st.warning("Preprocessing intermediate steps data is not in the expected format or is empty.")
        st.subheader("Final Preprocessed Job Text (Preview from last run)")
        st.dataframe(display_data[['Job.ID', 'text', 'processed_text']].head(), use_container_width=True)
        st.subheader('Search for a Word in Preprocessed Job Text')
        search_word_job = st.text_input("Enter a word to search in 'processed_text':", key="prep_job_search_word") # Renamed key
        if search_word_job:
            # Ensure 'processed_text' is treated as string for search
            search_results_job = display_data[display_data['processed_text'].astype(str).str.contains(search_word_job, case=False, na=False)]
            if not search_results_job.empty:
                st.subheader(f"Search results for '{search_word_job}' in 'processed_text':")
                st.dataframe(search_results_job[['Job.ID', 'Title', 'processed_text']], use_container_width=True)
                st.write(f"Found {len(search_results_job)} matching entries.")
            else:
                st.info(f"No entries found for '{search_word_job}' in 'processed_text'.")
    else:
        st.info("Job data loaded, but preprocessing has not been run yet. Click the button above.")

def tsdae_page():
    st.header("TSDAE (Sequential Noise Injection for Job Text)")
    st.write("Applies noise to preprocessed job text and generates TSDAE embeddings.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded and preprocessed first. Visit 'Preprocessing' page.")
        return
    
    bert_model = load_bert_model()
    if bert_model is None:
        st.error("BERT model could not be loaded. Cannot proceed.")
        return

    st.subheader("TSDAE Settings")
    deletion_ratio = st.slider("Deletion Ratio", 0.1, 0.9, 0.6, 0.1, key="tsdae_del_ratio")
    freq_threshold = st.slider("High Frequency Threshold", 10, 500, 100, 10, key="tsdae_freq_thresh")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="tsdae_run_button"):
        data_for_tsdae = st.session_state['data'].copy() # Work on a copy
        
        all_words_for_freq = []
        for text_content in data_for_tsdae['processed_text'].fillna('').astype(str).tolist():
            all_words_for_freq.extend(word_tokenize(text_content)) 
        word_freq_dict = {word.lower(): all_words_for_freq.count(word.lower()) for word in set(all_words_for_freq)}
        if not word_freq_dict:
            st.warning("Word frequency dictionary for TSDAE is empty. Methods 'b'/'c' might be affected.")

        with st.spinner("Applying Noise A (Random Deletion)..."):
            data_for_tsdae['noisy_text_a'] = data_for_tsdae['processed_text'].fillna('').astype(str).apply(
                lambda x: denoise_text(x, method='a', del_ratio=deletion_ratio))
        with st.spinner("Applying Noise B (High-Freq Removal)..."):
            data_for_tsdae['noisy_text_b'] = data_for_tsdae['noisy_text_a'].astype(str).apply(
                lambda x: denoise_text(x, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
        with st.spinner("Applying Noise C (High-Freq Removal + Shuffle)..."):
            data_for_tsdae['final_noisy_text'] = data_for_tsdae['noisy_text_b'].astype(str).apply(
                lambda x: denoise_text(x, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
        
        st.session_state['data'] = data_for_tsdae # Save data with noisy text columns
        st.success("Sequential noise application complete.")
        st.dataframe(data_for_tsdae[['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        # Filter for embedding generation
        final_noisy_texts_series = data_for_tsdae['final_noisy_text'].fillna('').astype(str)
        non_empty_mask_tsdae = final_noisy_texts_series.str.strip() != ''
        valid_texts_for_tsdae_emb = final_noisy_texts_series[non_empty_mask_tsdae].tolist()
        job_ids_for_tsdae_emb = data_for_tsdae.loc[non_empty_mask_tsdae, 'Job.ID'].tolist()

        if not valid_texts_for_tsdae_emb:
            st.warning("No valid (non-empty) 'final_noisy_text' found for TSDAE embedding generation.")
        else:
            st.session_state['tsdae_embeddings'] = generate_embeddings_with_progress(bert_model, valid_texts_for_tsdae_emb)
            st.session_state['tsdae_embedding_job_ids'] = job_ids_for_tsdae_emb
            if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
                st.success(f"TSDAE embeddings generated for {len(job_ids_for_tsdae_emb)} jobs!")
            else:
                st.warning("TSDAE embedding generation resulted in empty output.")
    
    if st.session_state.get('tsdae_embeddings') is not None:
        st.subheader("Current TSDAE Embeddings")
        st.write(f"Shape: {st.session_state['tsdae_embeddings'].shape} (for {len(st.session_state.get('tsdae_embedding_job_ids', []))} jobs)")
        st.write("Preview (first 3):", st.session_state['tsdae_embeddings'][:3])
    if 'final_noisy_text' in st.session_state.get('data', pd.DataFrame()).columns: # Check if noisy text exists
        st.subheader("Current Noisy Text Columns (Preview)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

def bert_model_page():
    st.header("Standard BERT Embeddings for Job Descriptions")
    st.write("Generates standard BERT embeddings from preprocessed job descriptions.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded and preprocessed. Visit 'Preprocessing' page.")
        return

    data_df_bert = st.session_state['data'] # Use current data
    bert_model = load_bert_model()
    if bert_model is None:
        st.error("BERT model could not be loaded.")
        return

    if st.button("Generate/Regenerate Standard Job Embeddings", key="gen_std_job_emb_button"):
        processed_texts_series = data_df_bert['processed_text'].fillna('').astype(str)
        non_empty_mask_bert = processed_texts_series.str.strip() != ''
        valid_texts_list_bert = processed_texts_series[non_empty_mask_bert].tolist()
        job_ids_for_bert_embeddings = data_df_bert.loc[non_empty_mask_bert, 'Job.ID'].tolist()

        if not valid_texts_list_bert:
            st.warning("No valid processed job texts found for embedding.")
        else:
            st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, valid_texts_list_bert)
            st.session_state['job_text_embedding_job_ids'] = job_ids_for_bert_embeddings
            if st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
                st.success(f"Standard job embeddings generated for {len(job_ids_for_bert_embeddings)} jobs!")
            else:
                st.warning("Standard job embedding generation resulted in empty output.")

    current_job_embeddings = st.session_state.get('job_text_embeddings') # Renamed
    current_job_ids = st.session_state.get('job_text_embedding_job_ids') # Renamed

    if current_job_embeddings is not None and current_job_embeddings.size > 0 and current_job_ids:
        st.subheader("Current Standard Job Embeddings")
        st.write(f"Shape: {current_job_embeddings.shape} (for {len(current_job_ids)} jobs)")
        st.write("Preview (first 3):", current_job_embeddings[:3])
        
        st.subheader("2D Visualization (PCA)")
        if len(current_job_embeddings) >= 2: 
            try:
                pca = PCA(n_components=2)
                reduced_embeddings_2d = pca.fit_transform(current_job_embeddings)
                plot_df_pca = pd.DataFrame(reduced_embeddings_2d, columns=['PC1', 'PC2'])
                plot_df_pca['Job.ID'] = current_job_ids
                
                # Merge with titles/descriptions from the main data_df for hover info
                titles_desc_df_pca = st.session_state['data'][st.session_state['data']['Job.ID'].isin(current_job_ids)][['Job.ID', 'Title', 'text']]
                plot_df_pca = pd.merge(plot_df_pca, titles_desc_df_pca, on='Job.ID', how='left')
                
                if not plot_df_pca.empty and 'Title' in plot_df_pca.columns:
                    fig = px.scatter(plot_df_pca, x='PC1', y='PC2', hover_name='Title', 
                                     hover_data={'text': True, 'Job.ID': True, 'PC1':False, 'PC2':False}, 
                                     title='2D PCA of Standard Job Embeddings')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not fully prepare data for PCA plot (e.g. missing titles).")
            except Exception as e:
                st.error(f"Error during PCA: {e}")
        else:
            st.warning("Need at least 2 data points for PCA.")
    else:
        st.info("Standard job embeddings not generated yet.")

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings (TSDAE or standard) using K-Means and merges results.")
    
    data_df_cluster = st.session_state.get('data') # Renamed
    if data_df_cluster is None:
        st.error("Job data not loaded. Cannot proceed.")
        return

    embeddings_for_clustering = None # Renamed
    job_ids_for_selected_embeddings = None # Renamed
    source_name_clustering = "" # Renamed

    # UI to select embedding type
    embedding_type_choice = st.radio(
        "Select embeddings for clustering:",
        ("TSDAE Embeddings", "Standard BERT Job Embeddings"),
        key="cluster_embedding_choice", horizontal=True
    )

    if embedding_type_choice == "TSDAE Embeddings":
        if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
            embeddings_for_clustering = st.session_state['tsdae_embeddings']
            job_ids_for_selected_embeddings = st.session_state.get('tsdae_embedding_job_ids')
            source_name_clustering = "TSDAE Embeddings"
            if not job_ids_for_selected_embeddings:
                st.error("TSDAE embeddings exist but corresponding Job IDs are missing. Cannot cluster.")
                return
        else:
            st.warning("TSDAE embeddings not available. Please generate them or choose Standard BERT.")
            return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
            embeddings_for_clustering = st.session_state['job_text_embeddings']
            job_ids_for_selected_embeddings = st.session_state.get('job_text_embedding_job_ids')
            source_name_clustering = "Standard BERT Job Embeddings"
            if not job_ids_for_selected_embeddings:
                st.error("Standard BERT embeddings exist but Job IDs are missing. Cannot cluster.")
                return
        else:
            st.warning("Standard BERT job embeddings not available. Please generate them or choose TSDAE.")
            return
    
    st.info(f"Using: {source_name_clustering} (Number of items to cluster: {len(job_ids_for_selected_embeddings)})")

    if embeddings_for_clustering is not None and job_ids_for_selected_embeddings:
        max_k = embeddings_for_clustering.shape[0]
        if max_k < 2:
            st.error("Not enough embedded items (need at least 2) to perform clustering.")
            return
        
        num_clusters_k = st.slider("Number of Clusters (K)", 2, min(50, max_k), min(N_CLUSTERS, max_k), key="k_slider_cluster") # Renamed

        if st.button(f"Run K-Means (K={num_clusters_k}) on {source_name_clustering}", key="run_kmeans_button"):
            cluster_labels = cluster_embeddings_with_progress(embeddings_for_clustering, num_clusters_k) # Renamed
            if cluster_labels is not None:
                if len(job_ids_for_selected_embeddings) == len(cluster_labels):
                    cluster_info_to_merge_df = pd.DataFrame({'Job.ID': job_ids_for_selected_embeddings, 'cluster': cluster_labels}) # Renamed
                    
                    data_copy_for_merge = st.session_state['data'].copy()
                    if 'cluster' in data_copy_for_merge.columns:
                        data_copy_for_merge = data_copy_for_merge.drop(columns=['cluster']) # Drop old cluster column
                    
                    updated_data_with_clusters_df = pd.merge(data_copy_for_merge, cluster_info_to_merge_df, on='Job.ID', how='left') # Renamed
                    st.session_state['data'] = updated_data_with_clusters_df
                    st.success(f"Clustering complete. 'cluster' column updated in the main dataset for {len(job_ids_for_selected_embeddings)} jobs.")
                else:
                    st.error("Mismatch between number of Job IDs and cluster labels. Cannot merge.")
            else:
                st.error("Clustering algorithm failed to return labels.")

    if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader(f"Current Clustering Results in Dataset (K={st.session_state['data']['cluster'].nunique(dropna=True)})") # dropna for nunique
        st.dataframe(st.session_state['data'][['Job.ID', 'Title', 'text', 'cluster']].head(10), height=300)
        # Display sample per cluster
        valid_clusters = st.session_state['data']['cluster'].dropna().unique()
        if valid_clusters.size > 0:
            st.subheader("Sample Job Descriptions per Cluster")
            for cl_num in sorted(valid_clusters): # Iterate over valid, sorted cluster numbers
                st.write(f"**Cluster {int(cl_num)}:**") # Ensure cl_num is int for display
                cluster_subset_df = st.session_state['data'][st.session_state['data']['cluster'] == cl_num] # Renamed
                if not cluster_subset_df.empty:
                    st.dataframe(cluster_subset_df[['Job.ID', 'Title', 'text']].sample(min(3, len(cluster_subset_df)), random_state=1), height=150)
                st.write("---")
    else:
        st.info("No 'cluster' column in the main dataset yet, or no clusters assigned.")


def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5). Original text, processed text, and embeddings will be stored.")
    
    uploaded_files_cv = st.file_uploader("Choose CV files:", type=["pdf", "docx"], accept_multiple_files=True, key="cv_uploader_widget") # Renamed
    
    if uploaded_files_cv:
        if len(uploaded_files_cv) > 5:
            st.warning("Max 5 CVs. Processing first 5.")
            uploaded_files_cv = uploaded_files_cv[:5]

        if st.button("Process Uploaded CVs", key="process_cvs_button_upload"):
            cv_batch_data = [] # Renamed
            bert_model_cv = load_bert_model() # Renamed
            if not bert_model_cv:
                st.error("BERT model load failed. Cannot process CVs.")
                return

            with st.spinner("Processing CVs..."):
                prog_bar_cv = st.progress(0) # Renamed
                status_txt_cv = st.empty() # Renamed
                for i, file_obj in enumerate(uploaded_files_cv): # Renamed
                    orig_txt, proc_txt, emb = "", "", None # Renamed variables
                    try:
                        ext = file_obj.name.split(".")[-1].lower()
                        if ext == "pdf": orig_txt = extract_text_from_pdf(file_obj)
                        elif ext == "docx": orig_txt = extract_text_from_docx(file_obj)
                        
                        if orig_txt and orig_txt.strip():
                            proc_txt = preprocess_text(orig_txt)
                            if proc_txt and proc_txt.strip():
                                emb_arr = generate_embeddings_with_progress(bert_model_cv, [proc_txt])
                                emb = emb_arr[0] if (emb_arr is not None and emb_arr.size > 0) else None
                            else: st.warning(f"Processed text for {file_obj.name} is empty.")
                        else: st.warning(f"Extracted text from {file_obj.name} is empty.")
                        
                        cv_batch_data.append({'filename': file_obj.name, 'original_text': orig_txt or "", 
                                              'processed_text': proc_txt or "", 'embedding': emb})
                        if emb is not None: st.success(f"Processed & embedded: {file_obj.name}")
                        elif proc_txt: st.warning(f"Processed {file_obj.name}, but embedding failed.")
                    except Exception as e:
                        st.error(f"Error with {file_obj.name}: {e}")
                        cv_batch_data.append({'filename': file_obj.name, 'original_text': "Processing Error", 
                                              'processed_text': "", 'embedding': None})
                    if uploaded_files_cv: prog_bar_cv.progress((i + 1) / len(uploaded_files_cv))
                    status_txt_cv.text(f"Done: {i+1}/{len(uploaded_files_cv)}")
                
                st.session_state['uploaded_cvs_data'] = cv_batch_data
                prog_bar_cv.empty()
                status_txt_cv.empty()
                st.success(f"CV batch processing finished. {len(cv_batch_data)} CVs attempted.")

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs in Session:")
        for i, cv_item in enumerate(st.session_state['uploaded_cvs_data']): # Renamed
            with st.expander(f"CV {i+1}: {cv_item.get('filename', 'Unknown')}", expanded=False):
                st.text_area(f"Original Text:", cv_item.get('original_text',''), height=100, disabled=True, key=f"disp_cv_orig_{i}")
                st.text_area(f"Processed Text:", cv_item.get('processed_text',''), height=100, disabled=True, key=f"disp_cv_proc_{i}")
                st.success("Embedding available.") if cv_item.get('embedding') is not None else st.warning("Embedding NOT available.")
    else:
        st.info("No CVs processed in this session yet.")

def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations for uploaded CVs using selected job embeddings.")

    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("Upload and process CVs first via 'Upload CV(s)' page.")
        return
    data_df_rec = st.session_state.get('data') # Renamed
    if data_df_rec is None or 'processed_text' not in data_df_rec.columns:
        st.error("Job data (with 'processed_text') not available. Load & preprocess first.")
        return
    
    job_emb_for_rec, emb_src_msg_rec = None, "" # Renamed
    choice_rec = st.radio("Job Embeddings for Recommendation:", ("Standard BERT", "TSDAE"), key="rec_emb_choice", horizontal=True) # Renamed

    if choice_rec == "TSDAE":
        if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
            job_emb_for_rec = st.session_state['tsdae_embeddings']
            # Ensure we have job_ids for these TSDAE embeddings
            if not st.session_state.get('tsdae_embedding_job_ids') or \
               len(st.session_state['tsdae_embedding_job_ids']) != job_emb_for_rec.shape[0]:
                st.error("TSDAE embeddings Job ID mismatch or missing. Cannot use for recommendation. Try regenerating TSDAE embeddings.")
                return
            emb_src_msg_rec = "Using TSDAE embeddings."
        else:
            st.warning("TSDAE embeddings unavailable. Generate them or choose Standard BERT.")
            return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
            job_emb_for_rec = st.session_state['job_text_embeddings']
            # Ensure we have job_ids for these standard embeddings
            if not st.session_state.get('job_text_embedding_job_ids') or \
               len(st.session_state['job_text_embedding_job_ids']) != job_emb_for_rec.shape[0]:
                st.error("Standard BERT embeddings Job ID mismatch or missing. Cannot use. Try regenerating standard embeddings.")
                return
            emb_src_msg_rec = "Using Standard BERT job embeddings."
        else:
            st.warning("Standard BERT job embeddings unavailable. Generate them or choose TSDAE.")
            return
    st.info(emb_src_msg_rec)

    # Get the Job.IDs corresponding to the selected job_emb_for_rec
    selected_job_emb_ids = st.session_state.get('tsdae_embedding_job_ids') if choice_rec == "TSDAE" else st.session_state.get('job_text_embedding_job_ids')
    
    # Filter the main data_df to only include jobs for which we have the selected embeddings
    # This ensures that similarity scores are calculated against the correct subset of jobs.
    # The order of job_emb_for_rec and selected_job_emb_ids must be consistent.
    # data_df_for_similarity must have rows corresponding to selected_job_emb_ids in the correct order.
    
    # Create a mapping from Job.ID to its original index in data_df_rec if needed for reordering,
    # or ensure data_df_rec is filtered and ordered by selected_job_emb_ids.
    # For simplicity, assume selected_job_emb_ids and job_emb_for_rec are already aligned.
    # We need to create a temporary DataFrame from the jobs that correspond to job_emb_for_rec
    
    # Create a DataFrame of jobs that have embeddings
    jobs_with_embeddings_df = data_df_rec[data_df_rec['Job.ID'].isin(selected_job_emb_ids)].copy()
    # Reorder this df to match the order of selected_job_emb_ids to align with job_emb_for_rec
    jobs_with_embeddings_df['Job.ID'] = pd.Categorical(jobs_with_embeddings_df['Job.ID'], categories=selected_job_emb_ids, ordered=True)
    jobs_with_embeddings_df = jobs_with_embeddings_df.sort_values('Job.ID').reset_index(drop=True)

    if len(jobs_with_embeddings_df) != len(job_emb_for_rec):
        st.error(f"Critical error: Number of jobs filtered for similarity ({len(jobs_with_embeddings_df)}) does not match number of selected job embeddings ({len(job_emb_for_rec)}). Cannot proceed with recommendations.")
        return


    if st.button("Generate Recommendations", key="gen_recs_btn"): # Renamed
        st.session_state['all_recommendations_for_annotation'] = {} 
        with st.spinner("Generating recommendations..."):
            valid_cvs = [cv for cv in st.session_state.get('uploaded_cvs_data', []) if cv.get('embedding') is not None] # Renamed
            if not valid_cvs:
                st.warning("No CVs with embeddings found.")
                return

            for cv_item_rec in valid_cvs: # Renamed
                cv_fname = cv_item_rec.get('filename', 'Unknown CV') # Renamed
                cv_emb = cv_item_rec['embedding'] # Renamed
                st.subheader(f"Recommendations for {cv_fname}")
                cv_emb_2d = cv_emb.reshape(1, -1) if cv_emb.ndim == 1 else cv_emb
                
                if job_emb_for_rec.ndim == 1 or job_emb_for_rec.shape[0] == 0:
                     st.error(f"Selected job embeddings are invalid. Cannot compute similarity for {cv_fname}.")
                     continue 
                
                # Similarities are calculated against job_emb_for_rec
                sim_scores = cosine_similarity(cv_emb_2d, job_emb_for_rec)[0] # Renamed
                
                # temp_rec_df is now jobs_with_embeddings_df, which is already filtered and ordered
                temp_rec_df_sim = jobs_with_embeddings_df.copy() # Renamed
                temp_rec_df_sim['similarity_score'] = sim_scores
                    
                recs_df = temp_rec_df_sim.sort_values(by='similarity_score', ascending=False).head(20) # Renamed

                if not recs_df.empty:
                    cols_to_display = ['Job.ID', 'Title', 'similarity_score'] # Renamed
                    if 'cluster' in recs_df.columns: cols_to_display.append('cluster')
                    cols_to_display.append('text') 
                    st.dataframe(recs_df[cols_to_display], use_container_width=True)
                    st.session_state['all_recommendations_for_annotation'][cv_fname] = recs_df
                else:
                    st.info(f"No recommendations found for {cv_fname}.")
                st.write("---") 
        st.success("Recommendation generation complete!")

def annotation_page():
    st.header("Annotation of Job Recommendations")
    st.write("Annotate relevance (0-3) and provide feedback for recommended jobs per CV.")
    if not st.session_state.get('all_recommendations_for_annotation'):
        st.warning("No recommendations generated yet. Go to 'Job Recommendation' page first.")
        return

    if 'collected_annotations' not in st.session_state or not isinstance(st.session_state['collected_annotations'], pd.DataFrame):
        st.session_state['collected_annotations'] = pd.DataFrame()

    with st.form(key="annotation_main_form"): # Renamed
        form_data_list = [] # Renamed
        for cv_f, rec_df in st.session_state['all_recommendations_for_annotation'].items(): # Renamed
            st.markdown(f"### CV: **{cv_f}**")
            for _, item_row in rec_df.iterrows(): # Renamed
                job_identifier = str(item_row['Job.ID']) # Renamed
                st.markdown(f"**Job ID:** {job_identifier} | **Title:** {item_row['Title']}")
                with st.expander("Details", expanded=False):
                    st.write(f"Desc: {item_row['text']}\nSim: {item_row['similarity_score']:.4f}")
                    if 'cluster' in item_row and pd.notna(item_row['cluster']): st.write(f"Cluster: {item_row['cluster']}")
                
                base_item_data = {'cv_filename': cv_f, 'job_id': job_identifier, 'job_title': item_row['Title'], 
                                  'job_text': item_row['text'], 'similarity_score': item_row['similarity_score'], 
                                  'cluster': item_row.get('cluster', pd.NA)} # Renamed
                
                annotator_data_for_item = {} # Renamed
                ann_cols = st.columns(len(ANNOTATORS)) # Renamed
                for idx, ann_name in enumerate(ANNOTATORS): # Renamed
                    with ann_cols[idx]:
                        st.markdown(f"**{ann_name}**")
                        rel_key = f"rel_{cv_f}_{job_identifier}_{ann_name}" # Renamed
                        fb_key = f"fb_{cv_f}_{job_identifier}_{ann_name}" # Renamed
                        def_rel, def_fb = 0, "" # Renamed
                        if not st.session_state['collected_annotations'].empty:
                            prev_mask = (st.session_state['collected_annotations']['cv_filename'] == cv_f) & \
                                       (st.session_state['collected_annotations']['job_id'] == job_identifier)
                            rel_col_name = f'annotator_{idx+1}_relevance' # Renamed
                            fb_col_name = f'annotator_{idx+1}_feedback' # Renamed
                            if rel_col_name in st.session_state['collected_annotations'].columns:
                                prev_entry = st.session_state['collected_annotations'][prev_mask] # Renamed
                                if not prev_entry.empty:
                                    val_rel = prev_entry.iloc[0].get(rel_col_name) # Renamed
                                    if pd.notna(val_rel): def_rel = int(val_rel)
                                    if fb_col_name in prev_entry.columns: def_fb = str(prev_entry.iloc[0].get(fb_col_name, ""))
                        
                        rel_val = st.radio("Relevance:", [0,1,2,3], index=def_rel, key=rel_key, horizontal=True, # Renamed
                                           format_func=lambda x: f"{x} ({'Poor' if x==0 else 'Fair' if x==1 else 'Good' if x==2 else 'Exc.'})") # Shorter labels
                        fb_val = st.text_area("Feedback:", value=def_fb, key=fb_key, height=60) # Renamed
                        annotator_data_for_item[f'annotator_{idx+1}_name'] = ann_name
                        annotator_data_for_item[f'annotator_{idx+1}_relevance'] = rel_val
                        annotator_data_for_item[f'annotator_{idx+1}_feedback'] = fb_val
                
                base_item_data.update(annotator_data_for_item)
                form_data_list.append(base_item_data)
                st.markdown("---")
        form_submitted = st.form_submit_button("Submit All Annotations") # Renamed

    if form_submitted:
        if form_data_list:
            new_ann_df = pd.DataFrame(form_data_list) # Renamed
            if not st.session_state['collected_annotations'].empty:
                submitted_keys_df = new_ann_df[['cv_filename', 'job_id']].drop_duplicates() # Renamed
                # Create boolean mask for rows in old annotations that are NOT in the new submission
                # This is complex. A simpler approach: if a (cv, job_id) is in new_ann_df, it replaces the old one.
                # We can achieve this by concatenating and then dropping duplicates, keeping the last.
                st.session_state['collected_annotations'] = pd.concat([st.session_state['collected_annotations'], new_ann_df]).drop_duplicates(subset=['cv_filename', 'job_id'], keep='last').reset_index(drop=True)
            else:
                st.session_state['collected_annotations'] = new_ann_df.reset_index(drop=True)
            st.success("Annotations submitted/updated.")
        else:
            st.warning("No annotation data entered.")

    if not st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        st.subheader("Collected Annotations Preview")
        st.dataframe(st.session_state['collected_annotations'], height=300)
        csv_out = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8') # Renamed
        st.download_button("Download Annotations (CSV)", csv_out, "job_annotations.csv", "text/csv", key="dl_ann_btn") # Renamed
    else:
        st.info("No annotations collected yet.")

def evaluation_page():
    st.header("Model Evaluation (Information Retrieval Metrics)")
    st.write("Evaluates recommendation performance based on collected annotations.")
    bert_model_eval = load_bert_model() # Renamed
    if bert_model_eval is None:
        st.error("BERT model load failed. Evaluation aborted.")
        return

    # Prerequisite checks
    data_df_eval = st.session_state.get('data') # Renamed
    uploaded_cvs_eval = st.session_state.get('uploaded_cvs_data', []) # Renamed
    annotations_eval_df = st.session_state.get('collected_annotations', pd.DataFrame()) # Renamed

    if data_df_eval is None or 'processed_text' not in data_df_eval.columns:
        st.warning("Job data with 'processed_text' unavailable. Load & preprocess first.")
        return
    if not uploaded_cvs_eval:
        st.warning("No CVs uploaded. Upload CVs first.")
        return
    cvs_for_eval_list = [cv for cv in uploaded_cvs_eval if cv.get('processed_text', '').strip()] # Renamed
    if not cvs_for_eval_list:
        st.warning("No CVs with processed text found for evaluation.")
        return
    if annotations_eval_df.empty:
        st.warning("No annotations collected. Annotate first.")
        return

    st.subheader("Evaluation Parameters")
    rel_thresh_eval = st.slider("Relevance Threshold (Avg score >= threshold is relevant)", 0.0, 3.0, 1.5, 0.1, key="eval_thresh_slider") # Renamed

    if st.button("Run Evaluation", key="eval_run_btn"): # Renamed
        with st.spinner("Preparing data & running IR evaluation..."):
            eval_queries = {str(cv['filename']): cv['processed_text'] for cv in cvs_for_eval_list} # Renamed
            eval_corpus = dict(zip(data_df_eval['Job.ID'].astype(str), data_df_eval['processed_text'])) # Renamed
            
            # Ensure Job.ID in annotations is string for matching with corpus keys
            annotations_eval_df['job_id'] = annotations_eval_df['job_id'].astype(str)
            
            eval_relevant_docs = {} # Renamed
            ann_rel_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS)) if f'annotator_{i+1}_relevance' in annotations_eval_df.columns] # Renamed
            if not ann_rel_cols:
                st.error("No annotator relevance columns in annotations. Cannot determine ground truth.")
                return

            grouped_anns = annotations_eval_df.groupby(['cv_filename', 'job_id']) # Renamed
            for (cv_name, j_id), group_data in grouped_anns: # Renamed
                cv_scores = [] # Renamed
                for rel_c in ann_rel_cols: # Renamed
                    cv_scores.extend(pd.to_numeric(group_data[rel_c], errors='coerce').dropna().tolist())
                if cv_scores:
                    avg_s = np.mean(cv_scores) # Renamed
                    if avg_s >= rel_thresh_eval:
                        cv_name_str = str(cv_name) # Renamed
                        if cv_name_str not in eval_relevant_docs: eval_relevant_docs[cv_name_str] = set()
                        eval_relevant_docs[cv_name_str].add(j_id)
            
            if not eval_relevant_docs: st.warning(f"No relevant docs found with threshold {rel_thresh_eval}. Metrics might be zero.")
            
            st.write(f"Queries (CVs): {len(eval_queries)}, Corpus (Jobs): {len(eval_corpus)}, CVs with relevant docs: {len(eval_relevant_docs)}")
            
            try:
                ir_eval = InformationRetrievalEvaluator(eval_queries, eval_corpus, eval_relevant_docs, name="job_rec_eval", show_progress_bar=True) # Renamed
                eval_results = ir_eval(bert_model_eval, output_path=None) # Renamed
                st.subheader("Evaluation Results")
                if eval_results and isinstance(eval_results, dict):
                    results_presentation_df = pd.DataFrame.from_dict(eval_results, orient='index', columns=['Score']) # Renamed
                    results_presentation_df.index.name = "Metric"
                    st.dataframe(results_presentation_df)
                    map_metric_keys = [k for k in eval_results.keys() if 'map' in k.lower()] # Renamed
                    if map_metric_keys: st.metric(label=f"Primary Metric ({map_metric_keys[0]})", value=f"{eval_results[map_metric_keys[0]]:.4f}")
                else: st.write("Raw results:", eval_results)
            except Exception as e_eval: # Renamed
                st.error(f"Error during IR evaluation: {e_eval}")
                st.exception(e_eval)

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio")

# Page routing
if page == "Home": home_page()
elif page == "Preprocessing": preprocessing_page()
elif page == "TSDAE (Noise Injection)": tsdae_page()
elif page == "BERT Model": bert_model_page()
elif page == "Clustering Job2Vec": clustering_page()
elif page == "Upload CV": upload_cv_page()
elif page == "Job Recommendation": job_recommendation_page()
elif page == "Annotation": annotation_page()
elif page == "Evaluation": evaluation_page()
