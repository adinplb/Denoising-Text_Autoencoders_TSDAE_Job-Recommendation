import streamlit as st
import pandas as pd
import numpy as np
import io
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer, util, evaluation
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --- NLTK Resource Downloads ---
# This block ensures necessary NLTK data is available when the app runs,
# especially important for Streamlit Cloud deployments.
import nltk
try:
    # Attempt to access a resource to check if stopwords are downloaded
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    # Attempt to access a resource to check if punkt tokenizer is downloaded
    word_tokenize("example text")
except LookupError:
    st.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# The punkt_tab resource might be implicitly needed by some NLTK operations
# or specific versions. Explicitly checking and downloading it for robustness.
try:
    # Check for a file that is part of the punkt_tab package
    nltk.data.find('tokenizers/punkt/PY3/punkt_tab.pickle')
except LookupError:
    st.info("Downloading NLTK punkt_tab resource...")
    nltk.download('punkt_tab')

# --- Data Loading ---
@st.cache_data
def load_job_data(url):
    """
    Loads job data from a CSV URL, selects relevant columns, and renames them.
    Includes error handling for file loading and column existence.
    """
    try:
        df = pd.read_csv(url)
        # Check if required columns exist
        if 'text' in df.columns and 'Title' in df.columns and 'Job.ID' in df.columns:
            return df[['Job.ID', 'text', 'Title']].rename(columns={'text': 'description', 'Title': 'title'})
        else:
            st.error("Error: 'Job.ID', 'text', and 'Title' columns not found in the job data. Please check the CSV structure.")
            return None
    except Exception as e:
        st.error(f"Error loading job data from {url}: {e}")
        return None

job_data_url = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
job_df = load_job_data(job_data_url)

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head(), use_container_width=True)

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Performs symbol removal, case folding, stopword tokenization, and stemming.
    """
    if isinstance(text, str):
        # 1. Symbol Removal: Remove punctuation and non-alphanumeric characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text) # Further remove non-word characters

        # 2. Case Folding: Convert text to lowercase
        text = text.lower()

        # 3. Stopwords Removal & Tokenization
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_words = [w for w in word_tokens if w not in stop_words]

        # 4. Stemming
        porter = PorterStemmer()
        stemmed_words = [porter.stem(w) for w in filtered_words]

        return " ".join(stemmed_words)
    return "" # Return empty string for non-string inputs

# Apply preprocessing to job descriptions if data is loaded correctly
if job_df is not None and 'description' in job_df.columns:
    job_df['processed_description'] = job_df['description'].apply(preprocess_text)
    st.subheader("Processed Job Descriptions (Preview)")
    st.dataframe(job_df[['title', 'description', 'processed_description']].head(), use_container_width=True)

# --- CV Upload Functionality ---
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

uploaded_cv = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
cv_text = ""
processed_cv_text = ""
cv_embedding = None
normalized_cv_embedding = None

if uploaded_cv is not None:
    file_extension = uploaded_cv.name.split(".")[-1].lower()
    if file_extension == "pdf":
        cv_text = extract_text_from_pdf(uploaded_cv)
    elif file_extension == "docx":
        cv_text = extract_text_from_docx(uploaded_cv)

    if cv_text:
        st.subheader("Uploaded CV Content (Preview)")
        # Added a label for accessibility
        st.text_area("Raw CV Text", cv_text, height=300, key="raw_cv_text")

        processed_cv_text = preprocess_text(cv_text)
        st.subheader("Processed CV Content (Preview)")
        # Added a label for accessibility
        st.text_area("Processed CV Text", processed_cv_text, height=200, key="processed_cv_text")

        # Load BERT model and generate embedding for the processed CV text
        # Using the cached model for consistency
        bert_model_for_cv = load_bert_model()
        cv_embedding = generate_embeddings(bert_model_for_cv, [processed_cv_text])[0]
        normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))
        st.success("CV processed and embedding generated!")
    else:
        st.warning("Could not extract text from the uploaded CV. Please try a different file.")

# --- Embedding using BERT ---
@st.cache_resource
def load_bert_model(model_name="all-mpnet-base-v2"):
    """Loads the SentenceTransformer model (cached)."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

@st.cache_data
def generate_embeddings(_model, texts):
    """
    Generates embeddings for a list of texts using the provided model (cached).
    _model argument is prefixed with underscore to prevent Streamlit hashing errors.
    """
    if _model is None:
        st.error("BERT model is not loaded. Cannot generate embeddings.")
        return np.array([])
    try:
        embeddings = _model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return np.array([])

bert_model_cached = load_bert_model() # Load the model once and cache it
job_embeddings = None
normalized_job_embeddings = None

if job_df is not None and 'processed_description' in job_df.columns and bert_model_cached is not None:
    job_processed_descriptions = job_df['processed_description'].fillna('').tolist()
    # Ensure there are descriptions to embed
    if job_processed_descriptions:
        job_embeddings = generate_embeddings(bert_model_cached, job_processed_descriptions)
        if job_embeddings.size > 0: # Check if embeddings were actually generated
            normalized_job_embeddings = normalize(job_embeddings)
        else:
            st.warning("No job embeddings generated. Check processed descriptions.")
    else:
        st.warning("No processed job descriptions found to generate embeddings.")

# --- CV Upload Functionality ---
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

uploaded_cv = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])
cv_text = ""
processed_cv_text = ""
cv_embedding = None
normalized_cv_embedding = None

if uploaded_cv is not None:
    file_extension = uploaded_cv.name.split(".")[-1].lower()
    if file_extension == "pdf":
        cv_text = extract_text_from_pdf(uploaded_cv)
    elif file_extension == "docx":
        cv_text = extract_text_from_docx(uploaded_cv)

    if cv_text:
        st.subheader("Uploaded CV Content (Preview)")
        st.text_area("Raw CV Text", cv_text, height=300, key="raw_cv_text")

        processed_cv_text = preprocess_text(cv_text)
        st.subheader("Processed CV Content (Preview)")
        st.text_area("Processed CV Text", processed_cv_text, height=200, key="processed_cv_text")

        # Load BERT model and generate embedding for the processed CV text
        # Using the cached model for consistency
        bert_model_for_cv = load_bert_model()
        if bert_model_for_cv: # Ensure model is loaded
            cv_embedding = generate_embeddings(bert_model_for_cv, [processed_cv_text])[0]
            normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))
            st.success("CV processed and embedding generated!")
        else:
            st.error("BERT model failed to load, cannot process CV.")
    else:
        st.warning("Could not extract text from the uploaded CV. Please try a different file.")

# --- Clustering the Embeddings ---
@st.cache_data
def cluster_embeddings(embeddings, n_clusters=5):
    """
    Performs K-Means clustering on the given embeddings.
    n_init='auto' is used for scikit-learn versions > 1.2.
    """
    if embeddings.size == 0 or len(embeddings) < n_clusters:
        st.warning("Not enough data points for clustering or embeddings are empty. Skipping clustering.")
        return np.array([]), np.array([])
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(embeddings)
        return clusters, kmeans.cluster_centers_
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return np.array([]), np.array([])


job_clusters = None
cluster_centers = None
if job_df is not None and normalized_job_embeddings is not None and normalized_job_embeddings.size > 0:
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=min(20, len(job_df)), value=min(5, len(job_df)))
    if len(job_df) >= num_clusters: # Ensure number of clusters is not more than data points
        job_clusters, cluster_centers = cluster_embeddings(normalized_job_embeddings, num_clusters)
        if job_clusters.size > 0:
            job_df['cluster'] = job_clusters
        else:
            st.warning("Clustering failed or returned empty clusters.")
    else:
        st.warning(f"Number of clusters ({num_clusters}) cannot be greater than the number of job postings ({len(job_df)}). Adjusting slider.")
        st.sidebar.slider("Number of Clusters", min_value=2, max_value=len(job_df), value=min(5, len(job_df)), key="adjusted_clusters")


# --- Visualize using 3D ---
if job_df is not None and normalized_job_embeddings is not None and normalized_job_embeddings.size > 0 and 'cluster' in job_df.columns:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    try:
        reduced_embeddings = pca.fit_transform(normalized_job_embeddings)

        fig = px.scatter_3d(
            job_df,
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            z=reduced_embeddings[:, 2],
            color='cluster',
            hover_data={'title': True, 'description': True, 'cluster': True,
                        reduced_embeddings[:, 0]: False, reduced_embeddings[:, 1]: False, reduced_embeddings[:, 2]: False},
            title='3D Visualization of Job Posting Clusters'
        )
        st.subheader("Job Posting Clusters (3D)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating 3D visualization: {e}")

# --- Similarity Matching ---
def get_top_recommendations(cv_norm_embedding, job_norm_embeddings, job_data_df, job_clusters_data, top_n=20):
    """
    Calculates cosine similarity and Euclidean distance between CV and job embeddings,
    and returns top N recommendations.
    """
    if cv_norm_embedding is None or job_norm_embeddings is None or job_data_df is None or job_clusters_data is None or job_norm_embeddings.size == 0:
        return pd.DataFrame()

    try:
        cosine_similarities = cosine_similarity(cv_norm_embedding, job_norm_embeddings)[0]
        distances = pairwise_distances(cv_norm_embedding, job_norm_embeddings, metric='euclidean')[0]

        results_df = pd.DataFrame({
            'title': job_data_df['title'].fillna('N/A'),
            'description': job_data_df['description'].fillna('N/A'),
            'cluster': job_clusters_data,
            'similarity_score': cosine_similarities,
            'distance': distances,
            'original_index': job_data_df.index # Keep original index for merging later
        })

        top_recommendations = results_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        return top_recommendations
    except Exception as e:
        st.error(f"Error during similarity matching: {e}")
        return pd.DataFrame()

top_recommendations_df = pd.DataFrame() # Initialize to empty DataFrame

if processed_cv_text and job_df is not None and normalized_job_embeddings is not None and job_clusters is not None and normalized_cv_embedding is not None:
    st.subheader("Top 20 Job Recommendations")
    top_recommendations_df = get_top_recommendations(normalized_cv_embedding, normalized_job_embeddings, job_df, job_clusters)

    if not top_recommendations_df.empty:
        st.dataframe(top_recommendations_df, use_container_width=True)

        # --- Manual Annotation ---
        st.subheader("Annotate Recommendations (Are these relevant to your CV?)")
        annotation_data = []
        annotators = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]

        for index, row in top_recommendations_df.iterrows():
            st.write(f"**Job Title:** {row['title']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
            st.write(f"**Distance:** {row['distance']:.4f}")
            st.write(f"**Cluster:** {row['cluster']}")

            annotation = {}
            # Use the original_index from top_recommendations_df to link back to job_df
            annotation['job_original_index'] = row['original_index']
            annotation['title'] = row['title']
            annotation['description'] = row['description']

            cols = st.columns(len(annotators))
            for i, annotator in enumerate(annotators):
                with cols[i]:
                    # Ensure unique and descriptive keys for radio buttons
                    relevant = st.radio(
                        f"{annotator} - Relevant?",
                        options=["Relevant", "Not Relevant"],
                        key=f"anno_job_{row['original_index']}_{annotator}"
                    )
                    annotation[annotator.lower().replace(" ", "_")] = 1 if relevant == "Relevant" else 0
            annotation_data.append(annotation)
            st.divider()

        if st.button("Submit Annotations", key="submit_annotations_button"):
            annotation_df = pd.DataFrame(annotation_data)
            st.subheader("Submitted Annotations")
            st.dataframe(annotation_df, use_container_width=True)
            st.session_state['annotations'] = annotation_df

            # --- Saving annotations to CSV ---
            annotations_file = "annotations.csv"
            try:
                if os.path.exists(annotations_file):
                    existing_df = pd.read_csv(annotations_file)
                    updated_df = pd.concat([existing_df, annotation_df], ignore_index=True)
                    updated_df.to_csv(annotations_file, index=False)
                    st.success(f"Annotations saved to {annotations_file} (appended).")
                else:
                    annotation_df.to_csv(annotations_file, index=False)
                    st.success(f"Annotations saved to {annotations_file}.")
            except Exception as e:
                st.error(f"Error saving annotations to CSV: {e}")

            # --- Option to download annotations ---
            csv_buffer = io.StringIO()
            if 'annotations' in st.session_state and not st.session_state['annotations'].empty:
                st.session_state['annotations'].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download All Submitted Annotations (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="all_submitted_annotations.csv",
                    mime="text/csv",
                    key="download_annotations_button"
                )
            else:
                st.info("No annotations to download yet.")
    else:
        st.info("No job recommendations found based on the uploaded CV. Please try uploading a different CV or check the job data.")

# --- Evaluate the Result ---
# Ensure top_recommendations_df is not empty and annotations exist before attempting evaluation
if 'annotations' in st.session_state and not st.session_state['annotations'].empty and not top_recommendations_df.empty and normalized_cv_embedding is not None:
    st.subheader("Evaluation of Recommendations")

    annotation_df_session = st.session_state['annotations']

    # Merge top_recommendations_df with annotation_df_session using the original index
    # This ensures we're evaluating the jobs that were actually recommended AND annotated
    merged_df = top_recommendations_df.merge(annotation_df_session, left_on='original_index', right_on='job_original_index', how='inner', suffixes=('_rec', '_anno'))

    # Drop duplicate columns from merge if any, keep relevant ones
    merged_df = merged_df.drop(columns=['job_original_index_anno'], errors='ignore')
    merged_df.rename(columns={'job_original_index_rec': 'job_original_index'}, inplace=True)


    # Assuming relevance if at least one annotator marked it as relevant
    annotator_cols = [col for col in merged_df.columns if col.startswith('annotator_')]
    if annotator_cols: # Check if any annotator columns exist after merge
        merged_df['ground_truth'] = merged_df[annotator_cols].max(axis=1)

        relevant_indices = merged_df[merged_df['ground_truth'] == 1]['original_index'].tolist()
        recommended_indices_for_eval = merged_df['original_index'].tolist() # Indices of recommended jobs that were annotated

        # Calculate Recall@k and Precision@k manually
        k_values = [5, 10, 20]
        results = {}

        for k in k_values:
            # Get the top k recommended jobs from the merged_df (which are also annotated)
            top_k_recommended_and_annotated = merged_df.head(k)['original_index'].tolist()
            relevant_in_top_k = len(set(top_k_recommended_and_annotated) & set(relevant_indices))
            precision_at_k = relevant_in_top_k / k if k > 0 else 0
            recall_at_k = relevant_in_top_k / len(relevant_indices) if relevant_indices else 0
            results[f'Precision@{k}'] = precision_at_k
            results[f'Recall@{k}'] = recall_at_k

        st.write("Evaluation Metrics (Based on Annotations):")
        st.dataframe(pd.DataFrame([results]), use_container_width=True)

        # More advanced evaluation using SentenceTransformers' InformationRetrievalEvaluator
        if not merged_df.empty and bert_model_cached is not None:
            queries = {"query": processed_cv_text}
            # Corpus should be the full set of job descriptions that were recommended and annotated
            corpus = {str(row['original_index']): row['processed_description_rec'] for index, row in merged_df.iterrows()}
            relevant_docs = defaultdict(set)
            for index, row in merged_df.iterrows():
                if row['ground_truth'] == 1:
                    relevant_docs["query"].add(str(row['original_index']))

            if relevant_docs["query"]: # Only run evaluator if there are relevant documents
                try:
                    evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs, top_k=[5, 10, 20])
                    evaluation_result = evaluator(bert_model_cached)
                    st.subheader("SBERT Information Retrieval Evaluator Results (Based on Annotations)")
                    st.write(evaluation_result)
                except Exception as e:
                    st.warning(f"Error during SBERT Evaluation: {e}")
                    st.warning("Ensure your annotation data and corpus are correctly aligned for evaluation.")
            else:
                st.info("No relevant documents found in annotations for SBERT evaluation.")
        else:
            st.info("Cannot perform SBERT evaluation: Merged data is empty or BERT model not loaded.")
    else:
        st.info("No annotator data found in the merged recommendations for evaluation.")
else:
    st.info("Upload a CV, get recommendations, and submit annotations to see evaluation results.")
