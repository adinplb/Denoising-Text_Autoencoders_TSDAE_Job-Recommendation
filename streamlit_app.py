import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import os # Not used directly, can be removed if not needed for other parts
from pdfminer.high_level import extract_text as pdf_extract_text
# from docx import Document as DocxDocument # Renamed to avoid conflict
from docx import Document as DocxDocumentPython # Renamed for clarity if you meant python-docx
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from tqdm import tqdm # Used for local progress bar simulation, not directly visible in Streamlit's st.progress
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
# from sklearn.preprocessing import normalize # Not used directly in provided snippet, check if needed
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
    # The following check for 'punkt_tab.pickle' might be too specific or internal.
    # A general check for 'punkt' availability is usually sufficient.
    # try:
    #     nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
    # except LookupError:
    #     st.info("Downloading NLTK punkt_tab resource...")
    #     nltk.download('punkt_tab') # This specific resource might not exist or be needed directly.
    st.success("NLTK resources checked/downloaded.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv'
RELEVANT_FEATURES = ['Job.ID', 'text', 'Title']
N_CLUSTERS = 20 # Default number of clusters for KMeans
# ANNOTATORS constant is no longer directly used in the form creation for individual pages,
# but the concept of 5 annotators is embedded in page creation and DataFrame structure.
# ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]
MAX_ANNOTATORS = 5


# --- Global Data Storage (using Streamlit Session State) ---
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
if 'collected_annotations' not in st.session_state: # Stores collected annotations from all annotators
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
        document = DocxDocumentPython(uploaded_file) # Using the renamed import
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
            for i, text_content in enumerate(data_df['text'].fillna('')): # Renamed variable to avoid conflict
                intermediate = {}
                if isinstance(text_content, str):
                    intermediate['original'] = text_content
                    symbol_removed = text_content.translate(str.maketrans('', '', string.punctuation))
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
def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100): # Renamed 'text' to 'text_to_denoise'
    """
    Applies noise to text based on specified method for TSDAE.
    Methods: 'a' (random deletion), 'b' (high-frequency word removal),
    'c' (high-frequency word removal + shuffle).
    """
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return text_to_denoise
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0: # Ensure at least one word remains
            keep_or_not[np.random.choice(n)] = True
        result_words = np.array(words)[keep_or_not]
    elif method == 'b':
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'. Please compute it from your corpus.")
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        if num_to_remove > len(high_freq_indices): # Handle cases where del_ratio is too high
            num_to_remove = len(high_freq_indices)
        to_remove_indices = set(random.sample(high_freq_indices, num_to_remove)) if high_freq_indices else set()
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result_words and words: # Ensure something remains if original wasn't empty
            result_words = [random.choice(words)]
    elif method == 'c':
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'. Please compute it from your corpus.")
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        if num_to_remove > len(high_freq_indices):
            num_to_remove = len(high_freq_indices)
        to_remove_indices = set(random.sample(high_freq_indices, num_to_remove)) if high_freq_indices else set()
        result_words = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result_words and words:
            result_words = [random.choice(words)]
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
        return None

@st.cache_data
def generate_embeddings_with_progress(_model, texts_list): # Renamed 'texts' to 'texts_list'
    """
    Generates embeddings for a list of texts using the provided model (cached).
    Includes a Streamlit spinner and a progress bar.
    _model argument is prefixed with underscore to prevent Streamlit hashing errors.
    """
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([])
    try:
        with st.spinner("Generating embeddings... This can take a few minutes."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_list = [] # Renamed 'embeddings'
            total_texts = len(texts_list)
            batch_size = 32 # Adjust based on your model and memory
            for i in range(0, total_texts, batch_size):
                batch_texts = texts_list[i:i + batch_size]
                batch_embeddings = _model.encode(batch_texts, convert_to_tensor=False) # convert_to_tensor=False for numpy
                embeddings_list.extend(batch_embeddings) # No .cpu().numpy() needed if convert_to_tensor=False
            
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
def cluster_embeddings_with_progress(embeddings_to_cluster, n_clusters_param): # Renamed params
    """Clusters embeddings using KMeans and displays a progress bar."""
    if embeddings_to_cluster is None or embeddings_to_cluster.size == 0:
        st.warning("No embeddings to cluster.")
        return None
    try:
        with st.spinner(f"Clustering embeddings into {n_clusters_param} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters_param, random_state=42, n_init='auto')
            clusters_result = kmeans.fit_predict(embeddings_to_cluster) # Renamed 'clusters'
            st.success(f"Clustering complete!")
            return clusters_result
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

# --- Page Functions ---

def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("This page provides an overview of the job dataset and allows you to explore its features.")

    if st.session_state['data'] is None:
        st.session_state['data'] = load_data_from_url(DATA_URL)

    data_df = st.session_state['data'] # Use data_df consistently

    if data_df is not None:
        st.subheader('Data Preview')
        st.dataframe(data_df.head(), use_container_width=True)

        st.subheader('Data Summary')
        st.write(f'Number of rows: {len(data_df)}')
        st.write(f'Number of columns: {len(data_df.columns)}')

        st.subheader('Search for a Word in a Feature')
        search_word = st.text_input("Enter a word to search:")
        search_column = st.selectbox("Select the feature to search in:", [''] + data_df.columns.tolist())

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                if not search_results.empty:
                    st.subheader(f"Search results for '{search_word}' in '{search_column}':")
                    st.dataframe(search_results, use_container_width=True)
                    st.write(f"Found {len(search_results)} matching entries in '{search_column}'.")
                else:
                    st.info(f"No entries found in '{search_column}' containing '{search_word}'.")
            else:
                st.error(f"Error: Column '{search_column}' not found in the data.")

        st.subheader('Feature Information')
        feature_list = data_df.columns.tolist()
        st.write(f'Total Features: **{len(feature_list)}**')
        st.write('**Features:**')
        st.code(str(feature_list))

        st.subheader('Explore Feature Details')
        selected_feature = st.selectbox('Select a Feature to see details:', [''] + feature_list)
        if selected_feature:
            st.write(f'**Feature:** `{selected_feature}`')
            st.write(f'**Data Type:** `{data_df[selected_feature].dtype}`')
            st.write(f'**Number of Unique Values:** `{data_df[selected_feature].nunique()}`')
            st.write('**Sample Unique Values:**')
            unique_values = data_df[selected_feature].unique()
            if len(unique_values) > 20:
                st.write(unique_values[:20])
                st.caption(f'(Showing first 20 of {len(unique_values)} unique values)')
            else:
                st.write(unique_values)

            if pd.api.types.is_numeric_dtype(data_df[selected_feature]):
                st.subheader(f'Descriptive Statistics for `{selected_feature}`')
                st.write(data_df[selected_feature].describe())
                fig = px.histogram(data_df, x=selected_feature, title=f'Distribution of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)
            elif pd.api.types.is_string_dtype(data_df[selected_feature]) or pd.api.types.is_object_dtype(data_df[selected_feature]):
                st.subheader(f'Value Counts for `{selected_feature}` (Top 20)')
                st.write(data_df[selected_feature].value_counts().head(20))
            else:
                st.info('No specific descriptive statistics or value counts for this data type.')

def preprocessing_page():
    st.header("Preprocessing")
    st.write("This page performs text preprocessing on the 'text' column, showing intermediate steps.")

    if st.session_state['data'] is None:
        st.session_state['data'] = load_data_from_url(DATA_URL)

    data_df = st.session_state['data']

    if data_df is not None:
        data_df = preprocess_text_with_intermediate(data_df)
        st.session_state['data'] = data_df # Update session state with preprocessed data

        if 'preprocessing_steps' in data_df.columns:
            st.subheader("Preprocessing Results (Intermediate Steps)")
            display_df = pd.DataFrame([step for step in data_df['preprocessing_steps']])
            st.dataframe(display_df.head(), use_container_width=True)

            st.subheader("Final Preprocessed Text (Preview)")
            st.dataframe(data_df[['text', 'processed_text']].head(), use_container_width=True)

            st.subheader('Search for a Word in Preprocessed Text')
            search_word_preprocessed = st.text_input("Enter a word to search in 'processed_text':")
            if search_word_preprocessed:
                search_results_preprocessed = data_df[data_df['processed_text'].str.contains(search_word_preprocessed, case=False, na=False)]
                if not search_results_preprocessed.empty:
                    st.subheader(f"Search results for '{search_word_preprocessed}' in 'processed_text':")
                    st.dataframe(search_results_preprocessed[['Job.ID', 'Title', 'processed_text']], use_container_width=True)
                    st.write(f"Found {len(search_results_preprocessed)} matching entries.")
                else:
                    st.info(f"No entries found in 'processed_text' containing '{search_word_preprocessed}'.")
        else:
            st.warning("Preprocessing steps not available. 'text' column might be missing or preprocessing failed.")
    else:
        st.info("Data not loaded. Please go to Home page first.")

def tsdae_page():
    st.header("TSDAE (Sequential Noise Injection)")
    st.write("This page applies noise to the preprocessed text using methods 'a', 'b', and 'c' sequentially.")

    if st.session_state['data'] is None or 'processed_text' not in st.session_state['data'].columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page.")
        return

    data_df = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is not None:
        st.subheader("TSDAE Settings")
        deletion_ratio = st.slider("Deletion Ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
        freq_threshold = st.slider("High Frequency Threshold", min_value=10, max_value=500, value=100, step=10)

        word_freq_dict = None
        all_words = []
        for text_content in data_df['processed_text'].fillna('').tolist():
            all_words.extend(word_tokenize(text_content))
        word_freq_dict = {word.lower(): all_words.count(word.lower()) for word in set(all_words)}

        if st.button("Apply Sequential Noise and Generate Embeddings"):
            noisy_text_stage_a = []
            # Using st.spinner for tqdm simulation as tqdm itself doesn't render in Streamlit directly
            with st.spinner("Applying Random Deletion (Method 'a')... This may take time."):
                # Simulating progress update for Streamlit
                status_text_a = st.empty()
                progress_bar_a = st.progress(0)
                total_items_a = len(data_df['processed_text'].fillna('').tolist())
                for idx, text_content in enumerate(data_df['processed_text'].fillna('').tolist()):
                    noisy_text_stage_a.append(denoise_text(text_content, method='a', del_ratio=deletion_ratio))
                    if idx % (total_items_a // 100 + 1) == 0: # Update progress roughly 100 times
                        progress_bar_a.progress(idx / total_items_a)
                        status_text_a.text(f"Method 'a': Processed {idx}/{total_items_a}")
                progress_bar_a.empty()
                status_text_a.empty()
                st.session_state['data']['noisy_text_a'] = noisy_text_stage_a
            st.success("Method 'a' noise application complete.")


            noisy_text_stage_b = []
            with st.spinner("Applying High-Frequency Word Removal (Method 'b')... This may take time."):
                status_text_b = st.empty()
                progress_bar_b = st.progress(0)
                total_items_b = len(st.session_state['data']['noisy_text_a'])
                for idx, text_content in enumerate(st.session_state['data']['noisy_text_a']):
                    noisy_text_stage_b.append(denoise_text(text_content, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
                    if idx % (total_items_b // 100 + 1) == 0:
                        progress_bar_b.progress(idx / total_items_b)
                        status_text_b.text(f"Method 'b': Processed {idx}/{total_items_b}")
                progress_bar_b.empty()
                status_text_b.empty()
                st.session_state['data']['noisy_text_b'] = noisy_text_stage_b
            st.success("Method 'b' noise application complete.")

            final_noisy_texts = []
            with st.spinner("Applying High-Frequency Word Removal + Shuffle (Method 'c')... This may take time."):
                status_text_c = st.empty()
                progress_bar_c = st.progress(0)
                total_items_c = len(st.session_state['data']['noisy_text_b'])
                for idx, text_content in enumerate(st.session_state['data']['noisy_text_b']):
                    final_noisy_texts.append(denoise_text(text_content, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
                    if idx % (total_items_c // 100 + 1) == 0:
                        progress_bar_c.progress(idx / total_items_c)
                        status_text_c.text(f"Method 'c': Processed {idx}/{total_items_c}")
                progress_bar_c.empty()
                status_text_c.empty()
                st.session_state['data']['final_noisy_text'] = final_noisy_texts
            st.success("Method 'c' noise application complete.")


            st.subheader("Sequentially Noisy Text (Preview)")
            st.dataframe(st.session_state['data'][['processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), use_container_width=True)

            st.subheader("Generating Embeddings for Final Noisy Text")
            final_noisy_embeddings = generate_embeddings_with_progress(bert_model, st.session_state['data']['final_noisy_text'].tolist())

            if final_noisy_embeddings.size > 0:
                st.session_state['tsdae_embeddings'] = final_noisy_embeddings
                st.subheader("Combined TSDAE Embeddings (Preview)")
                st.write("Shape of combined embeddings:", final_noisy_embeddings.shape)
                st.write("Preview of the first 3 combined embeddings:")
                st.write(final_noisy_embeddings[:3])
            else:
                st.warning("Failed to generate embeddings for the final noisy text.")
    else:
        st.warning("BERT model not loaded. Cannot proceed with TSDAE.")

def bert_model_page():
    st.header("BERT Model: Embedding Generation & Visualization")
    st.write("This page generates BERT embeddings from the preprocessed text and visualizes them.")

    if st.session_state['data'] is None or 'processed_text' not in st.session_state['data'].columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page.")
        return

    data_df = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is not None:
        texts_to_embed = data_df['processed_text'].fillna('').tolist()
        if texts_to_embed:
            # Generate embeddings only if they don't exist or user forces regeneration (add button for that if needed)
            if st.session_state.get('job_text_embeddings') is None:
                 st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, texts_to_embed)
            
            job_text_embeddings = st.session_state['job_text_embeddings']

            if job_text_embeddings is not None and job_text_embeddings.size > 0:
                st.subheader("Embeddings (Matrix Preview)")
                st.write("Shape of the embedding matrix:", job_text_embeddings.shape)
                st.write("Preview of the first 3 embeddings:")
                st.write(job_text_embeddings[:3])
                st.info(f"Each processed job description is now represented by a vector of {job_text_embeddings.shape[1]} dimensions.")

                st.subheader("2D Visualization of Embeddings (PCA)")
                pca = PCA(n_components=2)
                reduced_embeddings_2d = pca.fit_transform(job_text_embeddings)

                plot_df = pd.DataFrame(reduced_embeddings_2d, columns=['PC1', 'PC2'])
                # Ensure indices align if data_df was filtered or changed
                plot_df['title'] = data_df['Title'].tolist()[:len(plot_df)]
                plot_df['description'] = data_df['text'].tolist()[:len(plot_df)] # Use original text for hover

                fig = px.scatter(plot_df, x='PC1', y='PC2',
                                 hover_name='title', hover_data={'description': True, 'PC1':False, 'PC2':False}, # Clean hover data
                                 title='2D Visualization of Job Embeddings (PCA)',
                                 width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No embeddings generated. 'processed_text' column might be empty or generation failed.")
        else:
            st.warning("No text found in 'processed_text' column to generate embeddings.")
    else:
        st.warning("BERT model not loaded. Cannot proceed with BERT Model page.")

def clustering_page():
    st.header("Clustering Job2Vec")
    st.write("This page clusters the generated BERT embeddings using K-Means.")

    embeddings_to_cluster = None
    embedding_source_name = ""

    if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0 :
        embeddings_to_cluster = st.session_state['tsdae_embeddings']
        embedding_source_name = "TSDAE Embeddings"
    elif st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
        embeddings_to_cluster = st.session_state['job_text_embeddings']
        embedding_source_name = "BERT Embeddings (from processed text)"
    else:
        st.warning("No embeddings available for clustering. Please visit 'BERT Model' or 'TSDAE (Noise Injection)' page first.")
        return

    data_df = st.session_state['data'] 

    if embeddings_to_cluster.size > 0:
        st.subheader(f"Clustering Settings ({embedding_source_name})")
        # Ensure N_CLUSTERS is not more than the number of samples
        max_k_value = min(50, len(embeddings_to_cluster) -1 if len(embeddings_to_cluster) > 1 else 2)
        if max_k_value < 2: max_k_value = 2

        num_clusters_input = st.slider("Number of Clusters (K)", min_value=2, 
                                       max_value=max_k_value, 
                                       value=min(N_CLUSTERS, max_k_value))


        if st.button(f"Perform K-Means Clustering with K={num_clusters_input}"):
            clusters = cluster_embeddings_with_progress(embeddings_to_cluster, num_clusters_input)
            if clusters is not None:
                st.session_state['job_clusters'] = clusters
                # Ensure the 'cluster' column is added/updated correctly, aligning lengths
                if len(data_df) == len(clusters):
                    st.session_state['data']['cluster'] = clusters
                else:
                    st.error(f"Mismatch in length between data ({len(data_df)}) and clusters ({len(clusters)}). Cannot assign clusters.")
                    st.session_state['job_clusters'] = None # Reset if assignment failed

        if 'cluster' in st.session_state['data'].columns and st.session_state['job_clusters'] is not None:
            st.subheader(f"Clustering Results (K={num_clusters_input if 'num_clusters_input' in locals() else st.session_state['data']['cluster'].nunique()})")
            st.write("Original Text and Cluster Assignments:")
            # Display relevant columns, ensure 'text' is the original job description
            st.dataframe(st.session_state['data'][['Job.ID', 'Title', 'text', 'cluster']].head(10), use_container_width=True)

            st.subheader("Sample Job Descriptions per Cluster")
            for cluster_num in sorted(st.session_state['data']['cluster'].unique()):
                st.write(f"**Cluster {cluster_num}:**")
                cluster_data = st.session_state['data'][st.session_state['data']['cluster'] == cluster_num]
                if not cluster_data.empty:
                    cluster_sample = cluster_data.sample(min(5, len(cluster_data)))
                    st.dataframe(cluster_sample[['Job.ID', 'Title', 'text', 'cluster']], use_container_width=True)
                else:
                    st.info(f"No samples for Cluster {cluster_num}.")
                st.write("---")
        elif st.session_state.get('job_clusters') is None :
             st.info("No clustering performed yet or clustering failed. Click the button above.")
    else:
        st.warning("No embeddings available for clustering.")


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("This page provides job recommendations based on your uploaded CV and the clustered job postings.")

    st.subheader("Your Uploaded CVs")
    if st.session_state['uploaded_cvs_data']:
        for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_data['filename']}", expanded=False):
                st.text_area(f"CV {i+1} Content", cv_data['text'], height=150, disabled=True, key=f"cv_content_{i}")
                if cv_data.get('embedding') is not None:
                    st.write(f"CV {i+1} embedding available.")
                else:
                    st.warning(f"CV {i+1} embedding not generated.")
    else:
        st.info("Please upload your CV(s) on the 'Upload CV' page first.")
        return

    job_embeddings_for_similarity = None
    embedding_source_info = ""
    if st.session_state.get('tsdae_embeddings') is not None and st.session_state['tsdae_embeddings'].size > 0:
        job_embeddings_for_similarity = st.session_state['tsdae_embeddings']
        embedding_source_info = "Using TSDAE embeddings for recommendation."
    elif st.session_state.get('job_text_embeddings') is not None and st.session_state['job_text_embeddings'].size > 0:
        job_embeddings_for_similarity = st.session_state['job_text_embeddings']
        embedding_source_info = "Using standard BERT embeddings (from processed job text) for recommendation."
    else:
        st.warning("No job embeddings available for recommendation. Please generate them on 'BERT Model' or 'TSDAE' page.")
        return
    
    st.info(embedding_source_info)

    data_df = st.session_state.get('data')
    if data_df is None:
        st.error("Job data not loaded. Cannot proceed with recommendations.")
        return
    if 'cluster' not in data_df.columns and st.session_state.get('job_clusters') is None:
         st.warning("Job clusters are not available. Please run clustering on the 'Clustering Job2Vec' page. Recommendations will be generated without cluster info if you proceed.")


    if st.button("Generate Recommendations for All Uploaded CVs"):
        st.session_state['all_recommendations_for_annotation'] = {} 
        
        # Prepare the base structure for collected_annotations
        # This will include all job details and empty columns for annotators
        # This should be done ONCE when recommendations are generated.
        base_data_for_collection = []
        temp_all_recs_for_df = {} # Temporary holder for recommendations to build the base annotation df

        with st.spinner("Generating recommendations for all CVs..."):
            for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
                cv_filename = cv_data['filename']
                cv_embedding = cv_data.get('embedding')

                if cv_embedding is not None:
                    st.write(f"--- Processing CV: {cv_filename} ---")
                    cv_embedding_2d = cv_embedding.reshape(1, -1) if cv_embedding.ndim == 1 else cv_embedding
                    
                    similarities = cosine_similarity(cv_embedding_2d, job_embeddings_for_similarity)[0]
                    
                    temp_rec_df = data_df.copy()
                    if len(similarities) == len(temp_rec_df):
                        temp_rec_df['similarity_score'] = similarities
                    else:
                        st.error(f"Similarity scores length ({len(similarities)}) does not match job data length ({len(temp_rec_df)}) for CV {cv_filename}. Skipping.")
                        continue
                        
                    recommended_jobs = temp_rec_df.sort_values(by='similarity_score', ascending=False).head(20)

                    if not recommended_jobs.empty:
                        st.subheader(f"Top 20 Recommendations for {cv_filename}")
                        display_cols = ['Job.ID', 'Title', 'similarity_score', 'text']
                        if 'cluster' in recommended_jobs.columns:
                            display_cols.insert(3, 'cluster')
                        st.dataframe(recommended_jobs[display_cols], use_container_width=True)
                        
                        st.session_state['all_recommendations_for_annotation'][cv_filename] = recommended_jobs
                        temp_all_recs_for_df[cv_filename] = recommended_jobs # Store for base annotation DF
                    else:
                        st.info(f"No job recommendations found for {cv_filename}.")
                else:
                    st.warning(f"Skipping recommendations for {cv_filename}: CV embedding not generated.")
        
        # Initialize/Re-initialize collected_annotations DataFrame structure
        if temp_all_recs_for_df:
            all_job_recs_list = []
            for cv_fn, rec_df in temp_all_recs_for_df.items():
                for _, row in rec_df.iterrows():
                    job_entry = {
                        'cv_filename': cv_fn,
                        'job_id': row['Job.ID'],
                        'job_title': row['Title'],
                        'job_text': row['text'], 
                        'similarity_score': row['similarity_score'],
                        'cluster': row.get('cluster', pd.NA) # Use .get for cluster as it might be missing
                    }
                    all_job_recs_list.append(job_entry)
            
            if all_job_recs_list:
                base_annotations_df = pd.DataFrame(all_job_recs_list)
                # Add columns for all potential annotators
                for i in range(1, MAX_ANNOTATORS + 1):
                    base_annotations_df[f'annotator_{i}_name'] = pd.NA
                    base_annotations_df[f'annotator_{i}_relevance'] = pd.NA
                    base_annotations_df[f'annotator_{i}_feedback'] = pd.NA
                st.session_state['collected_annotations'] = base_annotations_df
                st.success("Annotation structure prepared based on new recommendations.")
            else:
                st.session_state['collected_annotations'] = pd.DataFrame() # Should not happen if recs were generated
        else:
            st.session_state['collected_annotations'] = pd.DataFrame() # No recs generated

        st.success("Recommendation generation complete for all CVs!")
        if not st.session_state['all_recommendations_for_annotation']:
             st.info("No recommendations were generated for any CV, or no CVs had embeddings.")


# MODIFIED: New individual_annotation_page function
def individual_annotation_page(annotator_number):
    st.header(f"Annotation Page - Annotator {annotator_number}")

    annotator_name = st.text_input(
        f"Please enter your name (Annotator {annotator_number}):",
        key=f"annotator_name_input_{annotator_number}"
    )

    if not annotator_name.strip():
        st.warning("Please enter your name to proceed with annotation.")
        # Display existing annotations if any, but disable form
        if 'collected_annotations' in st.session_state and not st.session_state['collected_annotations'].empty:
            st.subheader("Current State of All Collected Annotations (Read-Only)")
            st.dataframe(st.session_state['collected_annotations'], use_container_width=True)
        return

    if 'all_recommendations_for_annotation' not in st.session_state or \
       not st.session_state['all_recommendations_for_annotation']:
        st.warning("No recommendations available for annotation. "
                   "Please generate recommendations on the 'Job Recommendation' page first.")
        return

    # Ensure 'collected_annotations' DataFrame is initialized and has the correct structure
    # This should have been done in job_recommendation_page after generating recs.
    # This is a fallback/check.
    if 'collected_annotations' not in st.session_state or \
       not isinstance(st.session_state['collected_annotations'], pd.DataFrame) or \
       st.session_state['collected_annotations'].empty:
        
        st.warning("Collected annotations structure is missing or empty. Attempting to rebuild from recommendations.")
        # Try to rebuild from all_recommendations_for_annotation
        all_job_recs_list = []
        if st.session_state['all_recommendations_for_annotation']:
            for cv_fn, rec_df in st.session_state['all_recommendations_for_annotation'].items():
                for _, row in rec_df.iterrows():
                    job_entry = {
                        'cv_filename': cv_fn,
                        'job_id': row['Job.ID'],
                        'job_title': row['Title'],
                        'job_text': row['text'], 
                        'similarity_score': row['similarity_score'],
                        'cluster': row.get('cluster', pd.NA)
                    }
                    all_job_recs_list.append(job_entry)
            
            if all_job_recs_list:
                base_df = pd.DataFrame(all_job_recs_list)
                for i in range(1, MAX_ANNOTATORS + 1):
                    base_df[f'annotator_{i}_name'] = pd.NA
                    base_df[f'annotator_{i}_relevance'] = pd.NA
                    base_df[f'annotator_{i}_feedback'] = pd.NA
                st.session_state['collected_annotations'] = base_df
            else: # Should not happen if all_recommendations_for_annotation is not empty
                 st.session_state['collected_annotations'] = pd.DataFrame()
        else: # No recommendations to build from
            st.session_state['collected_annotations'] = pd.DataFrame()
        
        if st.session_state['collected_annotations'].empty:
            st.error("Could not initialize annotation structure because no recommendations are available.")
            return


    st.subheader(f"Annotate Recommendations (Annotator {annotator_number} - {annotator_name})")
    
    # Use a form for submission
    with st.form(key=f"annotation_form_annotator_{annotator_number}"):
        form_annotations_data = [] # To hold data from this form submission

        for cv_filename, recommendations_df in st.session_state['all_recommendations_for_annotation'].items():
            st.markdown(f"### CV: **{cv_filename}**")
            
            # Filter collected_annotations for this CV to get prior annotations for display (if any)
            # This is for display within the form item, actual update happens on submit
            current_job_annotations_df = st.session_state.get('collected_annotations', pd.DataFrame())
            if not current_job_annotations_df.empty:
                 current_job_annotations_df = current_job_annotations_df[current_job_annotations_df['cv_filename'] == cv_filename]

            for _, row in recommendations_df.iterrows(): # Iterate through recommendations
                job_id = row['Job.ID']
                
                st.markdown(f"**Job ID:** {job_id} | **Title:** {row['Title']}")
                with st.expander("View Job Details & Your Previous Annotation (if any)", expanded=False):
                    st.write(f"**Description (Original):** {row['text']}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
                    st.write(f"**Cluster:** {row.get('cluster', 'N/A')}")

                    # Display this annotator's previous annotation for this item, if it exists
                    if not current_job_annotations_df.empty:
                        prev_ann = current_job_annotations_df[current_job_annotations_df['job_id'] == job_id]
                        if not prev_ann.empty:
                            prev_relevance = prev_ann.iloc[0].get(f'annotator_{annotator_number}_relevance')
                            prev_feedback = prev_ann.iloc[0].get(f'annotator_{annotator_number}_feedback', "")
                            if pd.notna(prev_relevance):
                                st.info(f"Your previous relevance: {prev_relevance}\n\nYour previous feedback: {prev_feedback}")


                relevance_key = f"relevance_{cv_filename}_{job_id}_{annotator_number}"
                feedback_key = f"feedback_{cv_filename}_{job_id}_{annotator_number}"

                # Get default value from existing annotations if available
                default_relevance = 0 # Default to 0
                default_feedback = ""
                if not current_job_annotations_df.empty:
                    existing_row_ann = current_job_annotations_df[current_job_annotations_df['job_id'] == job_id]
                    if not existing_row_ann.empty:
                        val = existing_row_ann.iloc[0].get(f'annotator_{annotator_number}_relevance')
                        if pd.notna(val): default_relevance = int(val)
                        default_feedback = existing_row_ann.iloc[0].get(f'annotator_{annotator_number}_feedback', "")


                relevance = st.radio(
                    "Relevance:", options=[0, 1, 2, 3],
                    format_func=lambda x: f"{x} (" + ("Very Irrelevant" if x == 0 else "Slightly Relevant" if x == 1 else "Moderately Relevant" if x == 2 else "Very Relevant") + ")",
                    index=default_relevance, 
                    key=relevance_key, horizontal=True
                )
                qualitative_feedback = st.text_area(
                    "Feedback:", value=default_feedback, key=feedback_key, height=70
                )

                form_annotations_data.append({
                    'cv_filename': cv_filename,
                    'job_id': job_id,
                    'annotator_name_val': annotator_name, # Use the name from text input
                    'relevance_val': relevance,
                    'feedback_val': qualitative_feedback
                })
                st.markdown("---") 

        submitted = st.form_submit_button("Submit My Annotations")

    if submitted:
        df_collected_annotations = st.session_state.get('collected_annotations', pd.DataFrame()).copy()

        if df_collected_annotations.empty:
            st.error("Base annotation structure is missing. Cannot save annotations. Please re-generate recommendations.")
            return

        # Update the DataFrame
        updated_count = 0
        for ann_item in form_annotations_data:
            # Find the row to update
            match_condition = (df_collected_annotations['cv_filename'] == ann_item['cv_filename']) & \
                              (df_collected_annotations['job_id'] == ann_item['job_id'])
            
            row_indices = df_collected_annotations[match_condition].index

            if not row_indices.empty:
                idx_to_update = row_indices[0]
                df_collected_annotations.loc[idx_to_update, f'annotator_{annotator_number}_name'] = ann_item['annotator_name_val']
                df_collected_annotations.loc[idx_to_update, f'annotator_{annotator_number}_relevance'] = ann_item['relevance_val']
                df_collected_annotations.loc[idx_to_update, f'annotator_{annotator_number}_feedback'] = ann_item['feedback_val']
                updated_count +=1
            else:
                # This case should ideally not happen if df_collected_annotations is correctly initialized
                # from all_recommendations_for_annotation
                st.warning(f"Could not find job_id {ann_item['job_id']} for CV {ann_item['cv_filename']} in collected annotations. Skipping this item.")

        st.session_state['collected_annotations'] = df_collected_annotations
        st.success(f"Annotations from {annotator_name} (Annotator {annotator_number}) submitted successfully! {updated_count} items updated.")
        
        st.subheader("Current State of All Collected Annotations (Preview)")
        st.dataframe(st.session_state['collected_annotations'], use_container_width=True)

        if not st.session_state['collected_annotations'].empty:
            csv_buffer = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Collected Annotations as CSV",
                data=csv_buffer,
                file_name="all_job_recommendation_annotations.csv",
                mime="text/csv",
                key=f"download_all_annotations_annotator_{annotator_number}"
            )

def upload_cv_page():
    st.header("Upload CV(s)")
    st.write("Upload your CV(s) in PDF or Word format (max 5 files).")
    uploaded_files = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("You can upload a maximum of 5 CVs. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]

        # Clear previous CV data only if new files are uploaded and processed successfully later
        # st.session_state['uploaded_cvs_data'] = [] 
        
        bert_model = load_bert_model() 

        if not bert_model:
            st.error("BERT model failed to load. Cannot process CVs.")
            return

        processed_cvs_this_session = [] # Temporary list for newly processed CVs

        with st.spinner("Processing uploaded CVs..."):
            cv_upload_progress_bar = st.progress(0)
            cv_upload_status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                file_extension = uploaded_file.name.split(".")[-1].lower()
                cv_text_content = "" # Renamed variable
                try:
                    if file_extension == "pdf":
                        cv_text_content = extract_text_from_pdf(uploaded_file)
                    elif file_extension == "docx":
                        cv_text_content = extract_text_from_docx(uploaded_file)
                    
                    if cv_text_content:
                        processed_cv_text = preprocess_text(cv_text_content)
                        # generate_embeddings_with_progress expects a list
                        cv_embedding_array = generate_embeddings_with_progress(bert_model, [processed_cv_text])
                        cv_embedding = cv_embedding_array[0] if cv_embedding_array.size > 0 else None
                        
                        processed_cvs_this_session.append({
                            'filename': uploaded_file.name,
                            'text': cv_text_content,
                            'embedding': cv_embedding
                        })
                        st.success(f"Processed CV: {uploaded_file.name}")
                    else:
                        st.warning(f"Could not extract text from {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                cv_upload_progress_bar.progress((i + 1) / len(uploaded_files))
                cv_upload_status_text.text(f"Processed {i + 1}/{len(uploaded_files)} CVs.")
            
            cv_upload_progress_bar.empty()
            cv_upload_status_text.empty()
        
        if processed_cvs_this_session: # Only update if new CVs were processed
            st.session_state['uploaded_cvs_data'] = processed_cvs_this_session
            st.success("All selected CVs processed and stored!")


    # Display currently stored CVs
    if st.session_state.get('uploaded_cvs_data'):
        st.info("Currently uploaded and processed CVs:")
        for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
             with st.expander(f"CV {i+1}: {cv_data['filename']}", expanded=False):
                st.text_area(f"CV Content (Cached)", cv_data['text'], height=100, disabled=True, key=f"cached_cv_content_{i}")
                if cv_data.get('embedding') is not None:
                    st.write("Embedding available.")
                else:
                    st.warning("Embedding not generated for this CV.")
    else:
        st.info("No CVs uploaded yet in this session.")


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", "Clustering Job2Vec", "Upload CV", "Job Recommendation"]
annotation_page_names = [f"Annotation {i}" for i in range(1, MAX_ANNOTATORS + 1)] # Create 5 annotation pages
page_options.extend(annotation_page_names)

page = st.sidebar.radio("Go to", page_options)

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
elif page.startswith("Annotation"):
    try:
        annotator_num_str = page.split(" ")[-1]
        annotator_number = int(annotator_num_str)
        if 1 <= annotator_number <= MAX_ANNOTATORS:
            individual_annotation_page(annotator_number)
        else:
            st.error("Invalid annotator page number selected.")
    except (IndexError, ValueError):
        st.error("Invalid annotation page format selected.")
