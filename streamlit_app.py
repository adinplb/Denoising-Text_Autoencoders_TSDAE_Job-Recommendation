import streamlit as st
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import os

# --- Constants ---
KAGGLE_DATASET_PATH = "kandij/job-recommendation-datasets"
KAGGLE_FILENAMES = [
    "Combined_Jobs_Final.csv",
    "Experience.csv",
    "Job_Views.csv",
    "Positions_Of_Interest.csv",
    "job_data.csv"
]
# Common encodings to try if utf-8 fails
COMMON_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']
COMMON_DELIMITERS = [',', ';', '\t']

# --- Function to Load Data from Kaggle ---
@st.cache_data(show_spinner="Downloading and loading data from Kaggle...")
def load_data_from_kaggle(dataset_path, filename):
    """
    Authenticates with Kaggle API, downloads a specified file from a dataset,
    and attempts to read it into a pandas DataFrame, trying common encodings and delimiters.
    """
    try:
        # Set Kaggle API credentials from Streamlit Secrets as environment variables
        os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle"]["username"]
        os.environ['KAGGLE_KEY'] = st.secrets["kaggle"]["key"]

        api = KaggleApi()
        api.authenticate() # Authenticate using environment variables

        # Create a directory to store the data if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Define the full path where the file will be saved
        filepath = os.path.join(data_dir, filename)

        # Download the specific file from the dataset if it's not already there
        if not os.path.exists(filepath):
            st.info(f"Downloading '{filename}' from Kaggle...")
            api.dataset_download_file(dataset_path, filename, path=data_dir, force=False)
        else:
            st.info(f"Using cached file for '{filename}'.")

        # Attempt to read the CSV with multiple encodings and delimiters
        df = None
        for encoding in COMMON_ENCODINGS:
            for delimiter in COMMON_DELIMITERS:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, sep=delimiter, error_bad_lines=False, warn_bad_lines=True)
                    st.success(f"Successfully loaded '{filename}' with encoding: {encoding} and delimiter: '{delimiter}'")
                    return df
                except pd.errors.ParserError as e_parser:
                    st.warning(f"ParserError with '{filename}', encoding: {encoding}, delimiter: '{delimiter}': {e_parser}")
                    continue # Try the next delimiter
                except UnicodeDecodeError:
                    st.warning(f"UnicodeDecodeError with '{filename}', encoding: {encoding}.")
                    break # Move to the next encoding
                except Exception as e_read:
                    st.error(f"Error reading file '{filename}' with encoding: {encoding} and delimiter: '{delimiter}': {e_read}")
                    return None # Other reading errors are fatal

            if df is not None: # If successfully loaded with any delimiter, break the encoding loop
                break

        # If loop finishes, no encoding/delimiter combination worked
        st.error(f"Failed to load '{filename}' with all attempted encodings and delimiters.")
        return None

    except KeyError:
        st.error("Kaggle API credentials not found in Streamlit Secrets. Please configure `kaggle.username` and `kaggle.key`.")
        return None
    except Exception as e_general:
        st.error(f"An unexpected error occurred during data loading: {e_general}")
        return None

# --- Main Dashboard ---
st.title("Kaggle Dataset Explorer")

st.subheader("Select a CSV File to Explore")
selected_filename = st.selectbox("Choose a CSV file:", KAGGLE_FILENAMES)

if selected_filename:
    data_to_display = load_data_from_kaggle(KAGGLE_DATASET_PATH, selected_filename)
    if data_to_display is not None:
        st.subheader(f"Data: {selected_filename}")
        st.dataframe(data_to_display, use_container_width=True)

        st.subheader("Information about Features")
        if not data_to_display.empty:
            feature_list = data_to_display.columns.tolist()
            st.write(f"Total Features: **{len(feature_list)}**")
            st.write("**Features:**")
            st.code(str(feature_list)) # Display as code for better readability of long lists

            st.subheader("Explore Feature Details")
            # Add an empty string to allow no selection initially
            selected_feature = st.selectbox("Select a Feature to see details:", [""] + feature_list)
            if selected_feature:
                st.write(f"**Feature:** `{selected_feature}`")
                st.write(f"**Data Type:** `{data_to_display[selected_feature].dtype}`")
                st.write(f"**Number of Unique Values:** `{data_to_display[selected_feature].nunique()}`")

                # Display first 20 unique values or all if less than 20
                unique_values = data_to_display[selected_feature].unique()
                st.write("**Sample Unique Values:**")
                if len(unique_values) > 20:
                    st.write(unique_values[:20])
                    st.caption(f"(Showing first 20 of {len(unique_values)} unique values)")
                else:
                    st.write(unique_values)

                if pd.api.types.is_numeric_dtype(data_to_display[selected_feature]):
                    st.subheader(f"Descriptive Statistics for `{selected_feature}`")
                    st.write(data_to_display[selected_feature].describe())
                elif pd.api.types.is_string_dtype(data_to_display[selected_feature]) or pd.api.types.is_object_dtype(data_to_display[selected_feature]):
                    st.subheader(f"Value Counts for `{selected_feature}` (Top 20)")
                    st.write(data_to_display[selected_feature].value_counts().head(20))
                else:
                    st.info("No specific descriptive statistics or value counts for this data type.")
        else:
            st.warning("The selected DataFrame is empty, so no feature information is available.")
    else:
        st.info(f"Could not load data from {selected_filename}. Please check the error messages above.")
else:
    st.info("Please select a CSV file from the dropdown to start exploring.")
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
