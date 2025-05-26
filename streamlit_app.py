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
from tqdm import tqdm  # Import tqdm for progress bar (for local testing visibility)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA # For 2D visualization in BERT model page

# Download necessary NLTK resources (run once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
except LookupError:
    nltk.download('punkt_tab')

# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv'
RELEVANT_FEATURES = ['Job.ID', 'text', 'Title']
N_CLUSTERS = 20 # Default number of clusters

# --- Global Data Storage (using Streamlit Session State) ---
# This helps share data between pages without re-running heavy computations
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None
if 'job_clusters' not in st.session_state:
    st.session_state['job_clusters'] = None
if 'cv_text' not in st.session_state:
    st.session_state['cv_text'] = ""

# --- Helper Functions ---
@st.cache_data(show_spinner='Loading data...')
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        st.success('Successfully loaded data!')
        return df[RELEVANT_FEATURES].copy()
    except Exception as e:
        st.error(f'Error loading data from URL: {e}')
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
        document = Document(uploaded_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None

# --- Text Preprocessing Function with Intermediate Results ---
def preprocess_text_with_intermediate(data_df):
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
                    # Symbol Removal
                    symbol_removed = text.translate(str.maketrans('', '', string.punctuation))
                    symbol_removed = re.sub(r'[^\w\s]', '', symbol_removed)
                    intermediate['symbol_removed'] = symbol_removed
                    # Case Folding
                    case_folded = symbol_removed.lower()
                    intermediate['case_folded'] = case_folded
                    # Tokenize (for filtering/stopwords/stemming steps)
                    word_tokens = word_tokenize(case_folded)
                    intermediate['tokenized'] = " ".join(word_tokens) # Store tokenized for display
                    # Stopwords Removal
                    stop_words = set(stopwords.words('english'))
                    filtered = [w for w in word_tokens if w not in stop_words]
                    intermediate['stopwords_removed'] = " ".join(filtered)
                    # Stemming
                    porter = PorterStemmer()
                    stemmed = [porter.stem(w) for w in filtered]
                    intermediate['stemmed'] = " ".join(stemmed) # This is the final preprocessed text
                    processed_results.append(intermediate)
                else:
                    processed_results.append({
                        'original': '', 'symbol_removed': '', 'case_folded': '',
                        'tokenized': '', 'stopwords_removed': '', 'stemmed': ''
                    })
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
            data_df['preprocessing_steps'] = processed_results
            data_df['processed_text'] = [d['stemmed'] for d in processed_results] # Final preprocessed text
            st.success("Preprocessing of 'text' column complete!")
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("The 'text' column was not found in the dataset.")
    return data_df

# --- Embedding Generation Functions ---
@st.cache_resource
def load_bert_model(model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

@st.cache_data
def generate_embeddings_with_progress(_model, texts):
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([])
    try:
        with st.spinner("Generating embeddings... This can take a few minutes."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings = []
            total_texts = len(texts)
            batch_size = 32
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
    if embeddings is None or embeddings.size == 0:
        st.warning("No embeddings to cluster.")
        return None
    try:
        with st.spinner(f"Clustering embeddings into {n_clusters} clusters..."):
            cluster_progress_bar = st.progress(0)
            cluster_status_text = st.empty()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(embeddings)
            st.success(f"Clustering complete!")
            cluster_progress_bar.empty()
            cluster_status_text.empty()
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
        st.warning("BERT model not loaded. Cannot generate embeddings.")

def clustering_page():
    st.header("Clustering Job2Vec")
    st.write("This page clusters the generated BERT embeddings using K-Means.")

    if st.session_state['job_text_embeddings'] is None:
        st.warning("Please generate embeddings first by visiting the 'BERT Model' page.")
        return

    job_text_embeddings = st.session_state['job_text_embeddings']
    data = st.session_state['data'] # Get data with original text

    if job_text_embeddings.size > 0:
        st.subheader("Clustering Settings")
        num_clusters_input = st.slider("Number of Clusters (K)", min_value=2, max_value=min(50, len(job_text_embeddings)), value=N_CLUSTERS)

        if st.button(f"Perform K-Means Clustering with K={num_clusters_input}"):
            st.session_state['job_clusters'] = cluster_embeddings_with_progress(job_text_embeddings, num_clusters_input)
            st.session_state['data']['cluster'] = st.session_state['job_clusters'] # Add clusters to data

        if 'cluster' in st.session_state['data'].columns:
            st.subheader(f"Clustering Results (K={num_clusters_input})")
            st.write("Original Text and Cluster Assignments:")
            st.dataframe(st.session_state['data'][['text', 'cluster']].head(10), use_container_width=True)

            # Display a sample of each cluster
            st.subheader("Sample Job Descriptions per Cluster")
            for cluster_num in sorted(st.session_state['data']['cluster'].unique()):
                st.write(f"**Cluster {cluster_num}:**")
                cluster_sample = st.session_state['data'][st.session_state['data']['cluster'] == cluster_num].sample(min(5, len(st.session_state['data'][st.session_state['data']['cluster'] == cluster_num])))
                st.dataframe(cluster_sample[['text', 'cluster']], use_container_width=True)
                st.write("---")
        else:
            st.info("No clustering performed yet. Click the button above.")
    else:
        st.warning("No embeddings available for clustering.")

def upload_cv_page():
    st.header("Upload CV")
    st.write("Upload your CV in PDF or Word format.")
    uploaded_cv = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_cv is not None:
        file_extension = uploaded_cv.name.split(".")[-1].lower()
        try:
            if file_extension == "pdf":
                st.session_state['cv_text'] = extract_text_from_pdf(uploaded_cv)
            elif file_extension == "docx":
                st.session_state['cv_text'] = extract_text_from_docx(uploaded_cv)
            st.success("CV uploaded successfully!")
            if st.session_state['cv_text']:
                st.subheader("Uploaded CV Content (Preview)")
                st.text_area("CV Text", st.session_state['cv_text'], height=300)
            else:
                st.warning("Could not extract text from the uploaded CV.")
        except Exception as e:
            st.error(f"Error reading CV file: {e}")

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Preprocessing", "BERT Model", "Clustering Job2Vec", "Upload CV"])

if page == "Home":
    home_page()
elif page == "Preprocessing":
    preprocessing_page()
elif page == "BERT Model":
    bert_model_page()
elif page == "Clustering Job2Vec":
    clustering_page()
elif page == "Upload CV":
    upload_cv_page()









'''
import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# --- Data Loading ---
@st.cache_data
def load_job_data(url):
    try:
        df = pd.read_csv(url)
        if 'text' in df.columns and 'Title' in df.columns:
            return df[['Job.ID', 'text', 'Title']].rename(columns={'text': 'description', 'Title': 'title'})
        else:
            st.error("Error: 'text' and 'Title' columns not found in the job data.")
            return None
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

job_data_url = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
job_df = load_job_data(job_data_url)

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head())

# --- CV Upload Functionality ---
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

uploaded_cv = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
cv_text = ""
if uploaded_cv is not None:
    file_extension = uploaded_cv.name.split(".")[-1].lower()
    if file_extension == "pdf":
        cv_text = extract_text_from_pdf(uploaded_cv)
    elif file_extension == "docx":
        cv_text = extract_text_from_docx(uploaded_cv)

if cv_text:
    st.subheader("Uploaded CV Content (Preview)")
    st.text_area("CV Text", cv_text, height=300)

# --- Embedding using BERT ---
@st.cache_resource
def load_bert_model(model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data
def generate_embeddings(_model, texts):
    embeddings = _model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

bert_model = load_bert_model()
job_embeddings = None
normalized_job_embeddings = None
cv_embedding = None
normalized_cv_embedding = None

if job_df is not None and 'description' in job_df.columns:
    job_descriptions = job_df['description'].fillna('').tolist()
    job_embeddings = generate_embeddings(bert_model, job_descriptions)
    normalized_job_embeddings = normalize(job_embeddings)

if cv_text and bert_model is not None:
    cv_embedding = generate_embeddings(bert_model, [cv_text])[0]
    normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))

# --- Similarity Matching and Recommendations ---
if normalized_cv_embedding is not None and normalized_job_embeddings is not None and job_df is not None:
    st.subheader("Top 20 Job Recommendations")
    cosine_similarities = cosine_similarity(normalized_cv_embedding, normalized_job_embeddings)[0]
    similarity_df = pd.DataFrame({'title': job_df['title'], 'similarity_score': cosine_similarities})
    top_recommendations = similarity_df.sort_values(by='similarity_score', ascending=False).head(20)
    st.dataframe(top_recommendations)
elif uploaded_cv is not None:
    st.info("Please wait while job embeddings are being generated.")
elif job_df is not None:
    st.info("Upload your CV to get job recommendations.")
else:
    st.info("Job data not loaded.")

# --- Basic Evaluation (Illustrative - Simplified) ---
if normalized_cv_embedding is not None and normalized_job_embeddings is not None and job_df is not None:
    st.subheader("Basic Recommendation Statistics")
    avg_similarity = cosine_similarities.mean()
    max_similarity = cosine_similarities.max()
    min_similarity = cosine_similarities.min()

    st.write(f"Average Similarity: {avg_similarity:.4f}")
    st.write(f"Maximum Similarity: {max_similarity:.4f}")
    st.write(f"Minimum Similarity: {min_similarity:.4f}")
'''
'''
UMAP USED
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import plotly.express as px
import umap

# --- Constants ---
JOB_DATA_URL = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"

# --- Data Loading ---
@st.cache_data
def load_job_data(url):
    try:
        df = pd.read_csv(url)
        if 'text' in df.columns and 'Title' in df.columns:
            return df[['Job.ID', 'text', 'Title']].rename(columns={'text': 'description', 'Title': 'title'})
        else:
            st.error("Error: 'text' and 'Title' columns not found in the job data.")
            return None
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

job_df = load_job_data(JOB_DATA_URL)

# --- Sidebar for CV Upload ---
with st.sidebar:
    st.header("Upload Your CV")
    uploaded_cv = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_cv:
        st.success("CV uploaded successfully!")

# --- Main Dashboard ---
st.title("Job Posting Embedding Visualization (UMAP)")

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head())

    # --- Embedding Generation ---
    @st.cache_resource
    def load_bert_model(model_name="all-mpnet-base-v2"):
        model = SentenceTransformer(model_name)
        return model

    @st.cache_data
    def generate_job_embeddings(_model, df):
        if df is not None and 'description' in df.columns:
            job_descriptions = df['description'].fillna('').tolist()
            embeddings = _model.encode(job_descriptions, convert_to_tensor=True).cpu().numpy()
            normalized_embeddings = normalize(embeddings)
            return normalized_embeddings
        return None

    bert_model = load_bert_model()
    job_embeddings = generate_job_embeddings(bert_model, job_df)

    if job_embeddings is not None:
        st.subheader("Job Posting Embeddings (Normalized - Preview)")
        st.write(job_embeddings[:5])  # Display a snippet of the embeddings

        # --- Visualization of Embeddings (using UMAP for dimensionality reduction to 3D) ---
        st.subheader("Visualize Job Posting Embeddings (UMAP 3D)")
        @st.cache_data
        def reduce_dimensionality_umap(embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings

        reduced_embeddings_umap = reduce_dimensionality_umap(job_embeddings)

        fig_umap = px.scatter_3d(
            job_df,
            x=reduced_embeddings_umap[:, 0],
            y=reduced_embeddings_umap[:, 1],
            z=reduced_embeddings_umap[:, 2],
            hover_data=['title', 'description'],
            title='3D Visualization of Job Posting Embeddings (UMAP)'
        )
        st.plotly_chart(fig_umap)
    else:
        st.info("Job embeddings could not be generated.")

else:
    st.info("Job data not loaded. Please ensure the URL is correct.")

'''
