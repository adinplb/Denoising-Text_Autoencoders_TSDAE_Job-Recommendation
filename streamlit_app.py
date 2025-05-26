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
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


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
def denoise_text(text, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    """
    Applies noise to text based on specified method for TSDAE.
    Methods: 'a' (random deletion), 'b' (high-frequency word removal),
    'c' (high-frequency word removal + shuffle).
    """
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0: # Ensure at least one word remains
            keep_or_not[np.random.choice(n)] = True
        result = np.array(words)[keep_or_not]
    elif method == 'b':
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'. Please compute it from your corpus.")
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        if num_to_remove > len(high_freq_indices): # Handle cases where del_ratio is too high
            num_to_remove = len(high_freq_indices)
        to_remove_indices = set(random.sample(high_freq_indices, num_to_remove)) if high_freq_indices else set()
        result = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result and words: # Ensure something remains if original wasn't empty
            result = [random.choice(words)]
    elif method == 'c':
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'. Please compute it from your corpus.")
        high_freq_indices = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        if num_to_remove > len(high_freq_indices):
            num_to_remove = len(high_freq_indices)
        to_remove_indices = set(random.sample(high_freq_indices, num_to_remove)) if high_freq_indices else set()
        result = [w for i, w in enumerate(words) if i not in to_remove_indices]
        if not result and words:
            result = [random.choice(words)]
        random.shuffle(result)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result)

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
def generate_embeddings_with_progress(_model, texts):
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
            embeddings = []
            total_texts = len(texts)
            batch_size = 32 # Adjust based on your model and memory
            for i in range(0, total_texts, batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = _model.encode(batch_texts, convert_to_tensor=True)
                embeddings.extend(batch_embeddings.cpu().numpy())
                progress_val = (i + len(batch_texts)) / total_texts
                embedding_progress_bar.progress(progress_val)
                embedding_status_text.text(f"Generated embeddings for {i + len(batch_texts)}/{total_texts} entries.")
            st.success("Embedding generation complete!")
            embedding_progress_bar.empty()
            embedding_status_text.empty()
            return np.array(embeddings)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

@st.cache_data
def cluster_embeddings_with_progress(embeddings, n_clusters):
    """Clusters embeddings using KMeans and displays a progress bar."""
    if embeddings is None or embeddings.size == 0:
        st.warning("No embeddings to cluster.")
        return None
    try:
        with st.spinner(f"Clustering embeddings into {n_clusters} clusters..."):
            # KMeans doesn't have a built-in progress callback, so we simulate it
            # by showing a spinner for the entire operation.
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(embeddings)
            st.success(f"Clustering complete!")
            return clusters
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

# --- Page Functions ---

def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("This page provides an overview of the job dataset and allows you to explore its features.")

    if st.session_state['data'] is None:
        st.session_state['data'] = load_data_from_url(DATA_URL)

    data = st.session_state['data']

    if data is not None:
        st.subheader('Data Preview')
        st.dataframe(data.head(), use_container_width=True)

        st.subheader('Data Summary')
        st.write(f'Number of rows: {len(data)}')
        st.write(f'Number of columns: {len(data.columns)}')

        st.subheader('Search for a Word in a Feature')
        search_word = st.text_input("Enter a word to search:")
        search_column = st.selectbox("Select the feature to search in:", [''] + data.columns.tolist())

        if search_word and search_column:
            if search_column in data.columns:
                search_results = data[data[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                if not search_results.empty:
                    st.subheader(f"Search results for '{search_word}' in '{search_column}':")
                    st.dataframe(search_results, use_container_width=True)
                    st.write(f"Found {len(search_results)} matching entries in '{search_column}'.")
                else:
                    st.info(f"No entries found in '{search_column}' containing '{search_word}'.")
            else:
                st.error(f"Error: Column '{search_column}' not found in the data.")

        st.subheader('Feature Information')
        feature_list = data.columns.tolist()
        st.write(f'Total Features: **{len(feature_list)}**')
        st.write('**Features:**')
        st.code(str(feature_list))

        st.subheader('Explore Feature Details')
        selected_feature = st.selectbox('Select a Feature to see details:', [''] + feature_list)
        if selected_feature:
            st.write(f'**Feature:** `{selected_feature}`')
            st.write(f'**Data Type:** `{data[selected_feature].dtype}`')
            st.write(f'**Number of Unique Values:** `{data[selected_feature].nunique()}`')
            st.write('**Sample Unique Values:**')
            unique_values = data[selected_feature].unique()
            if len(unique_values) > 20:
                st.write(unique_values[:20])
                st.caption(f'(Showing first 20 of {len(unique_values)} unique values)')
            else:
                st.write(unique_values)

            if pd.api.types.is_numeric_dtype(data[selected_feature]):
                st.subheader(f'Descriptive Statistics for `{selected_feature}`')
                st.write(data[selected_feature].describe())
                fig = px.histogram(data, x=selected_feature, title=f'Distribution of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)
            elif pd.api.types.is_string_dtype(data[selected_feature]) or pd.api.types.is_object_dtype(data[selected_feature]):
                st.subheader(f'Value Counts for `{selected_feature}` (Top 20)')
                st.write(data[selected_feature].value_counts().head(20))
            else:
                st.info('No specific descriptive statistics or value counts for this data type.')

def preprocessing_page():
    st.header("Preprocessing")
    st.write("This page performs text preprocessing on the 'text' column, showing intermediate steps.")

    if st.session_state['data'] is None:
        st.session_state['data'] = load_data_from_url(DATA_URL)

    data = st.session_state['data']

    if data is not None:
        data = preprocess_text_with_intermediate(data)
        st.session_state['data'] = data # Update session state with preprocessed data

        if 'preprocessing_steps' in data.columns:
            st.subheader("Preprocessing Results (Intermediate Steps)")
            # Create a DataFrame for display
            display_df = pd.DataFrame([step for step in data['preprocessing_steps']])
            st.dataframe(display_df.head(), use_container_width=True)

            st.subheader("Final Preprocessed Text (Preview)")
            st.dataframe(data[['text', 'processed_text']].head(), use_container_width=True)

            st.subheader('Search for a Word in Preprocessed Text')
            search_word_preprocessed = st.text_input("Enter a word to search in 'processed_text':")
            if search_word_preprocessed:
                search_results_preprocessed = data[data['processed_text'].str.contains(search_word_preprocessed, case=False, na=False)]
                if not search_results_preprocessed.empty:
                    st.subheader(f"Search results for '{search_word_preprocessed}' in 'processed_text':")
                    st.dataframe(search_results_preprocessed[['Job.ID', 'Title', 'processed_text']], use_container_width=True)
                    st.write(f"Found {len(search_results_preprocessed)} matching entries.")
                else:
                    st.info(f"No entries found in 'processed_text' containing '{search_word_preprocessed}'.")
        else:
            st.warning("Preprocessing steps not available.")
    else:
        st.info("Data not loaded. Please go to Home page first.")

def tsdae_page():
    st.header("TSDAE (Sequential Noise Injection)")
    st.write("This page applies noise to the preprocessed text using methods 'a', 'b', and 'c' sequentially.")

    if st.session_state['data'] is None or 'processed_text' not in st.session_state['data'].columns:
        st.warning("Please preprocess the data first by visiting the 'Preprocessing' page.")
        return

    data = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is not None:
        st.subheader("TSDAE Settings")
        deletion_ratio = st.slider("Deletion Ratio", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
        freq_threshold = st.slider("High Frequency Threshold", min_value=10, max_value=500, value=100, step=10)

        word_freq_dict = None
        # Create a word frequency dictionary from the processed text
        all_words = []
        for text in data['processed_text'].fillna('').tolist():
            all_words.extend(word_tokenize(text))
        word_freq_dict = {word.lower(): all_words.count(word.lower()) for word in set(all_words)}

        if st.button("Apply Sequential Noise and Generate Embeddings"):
            noisy_text_stage_a = []
            with st.spinner("Applying Random Deletion (Method 'a')..."):
                for text in tqdm(data['processed_text'].fillna('').tolist(), desc="Applying Noise A"):
                    noisy_text_stage_a.append(denoise_text(text, method='a', del_ratio=deletion_ratio))
                st.session_state['data']['noisy_text_a'] = noisy_text_stage_a

            noisy_text_stage_b = []
            with st.spinner("Applying High-Frequency Word Removal (Method 'b')..."):
                for text in tqdm(st.session_state['data']['noisy_text_a'], desc="Applying Noise B"):
                    noisy_text_stage_b.append(denoise_text(text, method='b', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
                st.session_state['data']['noisy_text_b'] = noisy_text_stage_b

            final_noisy_texts = []
            with st.spinner("Applying High-Frequency Word Removal + Shuffle (Method 'c')..."):
                for text in tqdm(st.session_state['data']['noisy_text_b'], desc="Applying Noise C"):
                    final_noisy_texts.append(denoise_text(text, method='c', del_ratio=deletion_ratio, word_freq_dict=word_freq_dict, freq_threshold=freq_threshold))
                st.session_state['data']['final_noisy_text'] = final_noisy_texts

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

    data = st.session_state['data']
    bert_model = load_bert_model()

    if bert_model is not None:
        texts_to_embed = data['processed_text'].fillna('').tolist()
        if texts_to_embed:
            st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, texts_to_embed)
            job_text_embeddings = st.session_state['job_text_embeddings']

            if job_text_embeddings.size > 0:
                st.subheader("Embeddings (Matrix Preview)")
                st.write("Shape of the embedding matrix:", job_text_embeddings.shape)
                st.write("Preview of the first 3 embeddings:")
                st.write(job_text_embeddings[:3])
                st.info(f"Each processed job description is now represented by a vector of {job_text_embeddings.shape[1]} dimensions.")

                st.subheader("2D Visualization of Embeddings (PCA)")
                pca = PCA(n_components=2)
                reduced_embeddings_2d = pca.fit_transform(job_text_embeddings)

                plot_df = pd.DataFrame(reduced_embeddings_2d, columns=['PC1', 'PC2'])
                plot_df['title'] = data['Title'].tolist()
                plot_df['description'] = data['text'].tolist() # Use original text for hover

                fig = px.scatter(plot_df, x='PC1', y='PC2',
                                 hover_name='title', hover_data={'description': True},
                                 title='2D Visualization of Job Embeddings (PCA)',
                                 width=800, height=600)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("No embeddings generated. 'processed_text' column might be empty.")
        else:
            st.warning("No text found in 'processed_text' column to generate embeddings.")
    else:
        st.warning("BERT model not loaded. Cannot proceed with BERT Model page.")

def clustering_page():
    st.header("Clustering Job2Vec")
    st.write("This page clusters the generated BERT embeddings using K-Means.")

    # Determine which embeddings to use for clustering
    embeddings_to_cluster = None
    embedding_source_name = ""

    if st.session_state['tsdae_embeddings'] is not None:
        embeddings_to_cluster = st.session_state['tsdae_embeddings']
        embedding_source_name = "TSDAE Embeddings"
    elif st.session_state['job_text_embeddings'] is not None:
        embeddings_to_cluster = st.session_state['job_text_embeddings']
        embedding_source_name = "BERT Embeddings"
    else:
        st.warning("No embeddings available for clustering. Please visit 'BERT Model' or 'TSDAE (Noise Injection)' page first.")
        return

    data = st.session_state['data'] # Get data with original text

    if embeddings_to_cluster.size > 0:
        st.subheader(f"Clustering Settings ({embedding_source_name})")
        num_clusters_input = st.slider("Number of Clusters (K)", min_value=2, max_value=min(50, len(embeddings_to_cluster)), value=N_CLUSTERS)

        if st.button(f"Perform K-Means Clustering with K={num_clusters_input}"):
            st.session_state['job_clusters'] = cluster_embeddings_with_progress(embeddings_to_cluster, num_clusters_input)
            if st.session_state['job_clusters'] is not None:
                st.session_state['data']['cluster'] = st.session_state['job_clusters'] # Add clusters to data

        if 'cluster' in st.session_state['data'].columns:
            st.subheader(f"Clustering Results (K={num_clusters_input})")
            st.write("Original Text and Cluster Assignments:")
            st.dataframe(st.session_state['data'][['text', 'cluster']].head(10), use_container_width=True)

            # Display a sample of each cluster
            st.subheader("Sample Job Descriptions per Cluster")
            for cluster_num in sorted(st.session_state['data']['cluster'].unique()):
                st.write(f"**Cluster {cluster_num}:**")
                cluster_data = st.session_state['data'][st.session_state['data']['cluster'] == cluster_num]
                if not cluster_data.empty:
                    cluster_sample = cluster_data.sample(min(5, len(cluster_data)))
                    st.dataframe(cluster_sample[['text', 'cluster']], use_container_width=True)
                else:
                    st.info(f"No samples for Cluster {cluster_num}.")
                st.write("---")
        else:
            st.info("No clustering performed yet. Click the button above.")
    else:
        st.warning("No embeddings available for clustering.")


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("This page provides job recommendations based on your uploaded CV and the clustered job postings.")

    # --- Display Uploaded CVs ---
    st.subheader("Your Uploaded CVs")
    if st.session_state['uploaded_cvs_data']:
        for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
            st.write(f"**CV {i+1}: {cv_data['filename']}**")
            st.text_area(f"CV {i+1} Content", cv_data['text'], height=150, disabled=True, key=f"cv_content_{i}")
            if cv_data['embedding'] is not None:
                st.write(f"CV {i+1} embedding generated.")
            else:
                st.warning(f"CV {i+1} embedding not generated.")
            st.write("---")
    else:
        st.info("Please upload your CV(s) on the 'Upload CV' page first.")
        return # Exit if no CVs are uploaded

    # --- Job Recommendation Logic ---
    if st.session_state['job_clusters'] is not None and st.session_state['data'] is not None:
        
        # Determine which job embeddings to use for similarity matching
        job_embeddings_for_similarity = None
        if st.session_state['tsdae_embeddings'] is not None:
            job_embeddings_for_similarity = st.session_state['tsdae_embeddings']
            st.info("Using TSDAE embeddings for recommendation.")
        elif st.session_state['job_text_embeddings'] is not None:
            job_embeddings_for_similarity = st.session_state['job_text_embeddings']
            st.info("Using standard BERT embeddings for recommendation.")
        else:
            st.warning("No job embeddings available for recommendation. Please generate them on 'BERT Model' or 'TSDAE' page.")
            return

        data = st.session_state['data']
        job_clusters = st.session_state['job_clusters']

        if st.button("Generate Recommendations for All Uploaded CVs"):
            st.session_state['all_recommendations_for_annotation'] = {} # Clear previous recs

            for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
                cv_filename = cv_data['filename']
                cv_embedding = cv_data['embedding']

                if cv_embedding is not None:
                    st.subheader(f"Recommendations for {cv_filename}")
                    # Ensure CV embedding is 2D for cosine_similarity
                    cv_embedding_2d = cv_embedding.reshape(1, -1) if cv_embedding.ndim == 1 else cv_embedding
                    
                    similarities = cosine_similarity(cv_embedding_2d, job_embeddings_for_similarity)[0]
                    
                    # Create a temporary DataFrame for this CV's recommendations
                    temp_rec_df = data.copy()
                    temp_rec_df['similarity_score'] = similarities
                    
                    # Sort and get top recommendations
                    recommended_jobs = temp_rec_df.sort_values(by='similarity_score', ascending=False).head(20)

                    if not recommended_jobs.empty:
                        st.dataframe(recommended_jobs[['Job.ID', 'Title', 'similarity_score', 'cluster', 'text']], use_container_width=True)
                        st.session_state['all_recommendations_for_annotation'][cv_filename] = recommended_jobs # Store for annotation
                    else:
                        st.info(f"No job recommendations found for {cv_filename}.")
                    st.write("---") # Separator for clarity
                else:
                    st.warning(f"Skipping recommendations for {cv_filename}: CV embedding not generated.")
    elif st.session_state['uploaded_cvs_data'] and st.session_state['job_clusters'] is None:
        st.info("Please cluster the job embeddings first on the 'Clustering Job2Vec' page to get recommendations.")
    else:
        st.info("Please upload your CV(s) and process job data/embeddings/clusters to get recommendations.")

def annotation_page():
    st.header("Annotation")
    st.write("Annotate the relevance of the top 20 job recommendations for each CV.")

    if not st.session_state['all_recommendations_for_annotation']:
        st.warning("No recommendations available for annotation. Please generate recommendations on the 'Job Recommendation' page first.")
        return

    st.subheader("Annotation Form")
    all_annotations_data = []
    
    for cv_filename, recommendations_df in st.session_state['all_recommendations_for_annotation'].items():
        st.markdown(f"### Annotate Recommendations for CV: **{cv_filename}**")
        
        for idx, row in recommendations_df.iterrows():
            st.write(f"**Job ID:** {row['Job.ID']}")
            st.write(f"**Title:** {row['Title']}")
            st.write(f"**Description:** {row['text']}")
            st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
            st.write(f"**Cluster:** {row['cluster']}")
            
            annotation_row_data = {
                'cv_filename': cv_filename,
                'job_id': row['Job.ID'],
                'job_title': row['Title'],
                'job_text': row['text'],
                'similarity_score': row['similarity_score'],
                'cluster': row['cluster']
            }

            cols = st.columns(len(ANNOTATORS))
            for i, annotator in enumerate(ANNOTATORS):
                with cols[i]:
                    # Numerical input (Relevant/Not Relevant)
                    relevance = st.radio(
                        f"{annotator} - Relevance Score:", # Changed label
                        options=[0, 1, 2, 3], # Changed options
                        index=0, # Default to 0
                        key=f"relevance_{cv_filename}_{row['Job.ID']}_{annotator}"
                    )
                    annotation_row_data[f'annotator_{i+1}_relevance'] = relevance # Store numerical value
                    
                    # Qualitative input (Text area)
                    qualitative_feedback = st.text_area(
                        f"{annotator} - Feedback",
                        key=f"feedback_{cv_filename}_{row['Job.ID']}_{annotator}",
                        height=68
                    )
                    annotation_row_data[f'annotator_{i+1}_feedback'] = qualitative_feedback
            
            all_annotations_data.append(annotation_row_data)
            st.markdown("---") # Separator for each job

    if st.button("Submit All Annotations"):
        st.session_state['collected_annotations'] = pd.DataFrame(all_annotations_data)
        st.success("Annotations submitted successfully!")
        st.subheader("Collected Annotations Preview")
        st.dataframe(st.session_state['collected_annotations'], use_container_width=True)
        
        # Optional: Save annotations to CSV
        csv_buffer = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Annotations as CSV",
            data=csv_buffer,
            file_name="job_recommendation_annotations.csv",
            mime="text/csv",
        )

def upload_cv_page():
    st.header("Upload CV(s)")
    st.write("Upload your CV(s) in PDF or Word format (max 5 files).")
    uploaded_files = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("You can upload a maximum of 5 CVs. Only the first 5 will be processed.")
            uploaded_files = uploaded_files[:5]

        st.session_state['uploaded_cvs_data'] = [] # Clear previous uploads
        bert_model = load_bert_model() # Load BERT model once for all CVs

        if not bert_model:
            st.error("BERT model failed to load. Cannot process CVs.")
            return

        with st.spinner("Processing uploaded CVs..."):
            cv_upload_progress_bar = st.progress(0)
            cv_upload_status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                file_extension = uploaded_file.name.split(".")[-1].lower()
                cv_text = ""
                try:
                    if file_extension == "pdf":
                        cv_text = extract_text_from_pdf(uploaded_file)
                    elif file_extension == "docx":
                        cv_text = extract_text_from_docx(uploaded_file)
                    
                    if cv_text:
                        processed_cv_text = preprocess_text(cv_text)
                        cv_embedding = generate_embeddings_with_progress(bert_model, [processed_cv_text])[0]
                        
                        st.session_state['uploaded_cvs_data'].append({
                            'filename': uploaded_file.name,
                            'text': cv_text,
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
            st.success("All selected CVs processed!")

    elif st.session_state['uploaded_cvs_data']:
        st.info("Currently uploaded CVs:")
        for i, cv_data in enumerate(st.session_state['uploaded_cvs_data']):
            st.write(f"- {cv_data['filename']}")
            st.text_area(f"CV {i+1} Content (Cached)", cv_data['text'], height=100, disabled=True, key=f"cached_cv_content_{i}")
            if cv_data['embedding'] is not None:
                st.info("Embedding generated.")
            else:
                st.warning("Embedding not generated.")


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", "Clustering Job2Vec", "Job Recommendation", "Annotation", "Upload CV"])

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
elif page == "Job Recommendation":
    job_recommendation_page()
elif page == "Annotation":
    annotation_page()
elif page == "Upload CV":
    upload_cv_page()
