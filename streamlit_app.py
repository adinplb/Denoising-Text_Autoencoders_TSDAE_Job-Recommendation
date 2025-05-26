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

# Download necessary NLTK resources (run once)
import nltk
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize("example")
except LookupError:
    nltk.download('punkt')

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

# --- Preprocessing Function ---
def preprocess_text(text):
    if isinstance(text, str):
        # Symbol Removal
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^\w\s]', '', text)

        # Case Folding
        text = text.lower()

        # Stopwords Removal
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_words = [w for w in word_tokens if w not in stop_words]

        # Stemming
        porter = PorterStemmer()
        stemmed_words = [porter.stem(w) for w in filtered_words]

        return " ".join(stemmed_words)
    return ""

if job_df is not None and 'description' in job_df.columns:
    job_df['processed_description'] = job_df['description'].apply(preprocess_text)
    st.subheader("Processed Job Descriptions (Preview)")
    st.dataframe(job_df[['title', 'description', 'processed_description']].head())

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
    st.text_area("", cv_text, height=300)
    processed_cv_text = preprocess_text(cv_text)
    st.subheader("Processed CV Content (Preview)")
    st.text_area("", processed_cv_text, height=200)
    # Generate embedding for the processed CV text
    bert_model = SentenceTransformer("all-mpnet-base-v2")
    cv_embedding = bert_model.encode([processed_cv_text], convert_to_tensor=True).cpu().numpy()[0]
    normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))

# --- Embedding using BERT ---
@st.cache_resource
def load_bert_model(model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data
def generate_embeddings(_model, texts):
    embeddings = _model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

bert_model_cached = load_bert_model()
job_embeddings = None
normalized_job_embeddings = None

if job_df is not None and 'processed_description' in job_df.columns:
    job_processed_descriptions = job_df['processed_description'].fillna('').tolist()
    job_embeddings = generate_embeddings(bert_model_cached, job_processed_descriptions)
    normalized_job_embeddings = normalize(job_embeddings)

# --- Clustering the Embeddings ---
@st.cache_data
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') # Corrected KMeans initialization
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans.cluster_centers_

if job_df is not None and normalized_job_embeddings is not None:
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=20, value=5)
    job_clusters, cluster_centers = cluster_embeddings(normalized_job_embeddings, num_clusters)
    job_df['cluster'] = job_clusters

# --- Visualize using 3D ---
if job_df is not None and normalized_job_embeddings is not None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(normalized_job_embeddings)

    fig = px.scatter_3d(
        job_df,
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        color='cluster',
        hover_data=['title', 'description', 'cluster'],
        title='3D Visualization of Job Posting Clusters'
    )
    st.subheader("Job Posting Clusters (3D)")
    st.plotly_chart(fig)

# --- Similarity Matching ---
def get_top_recommendations(cv_embedding, job_embeddings, job_df, job_clusters, top_n=20):
    if cv_embedding is None or job_embeddings is None or job_df is None or job_clusters is None:
        return pd.DataFrame()

    cosine_similarities = cosine_similarity(normalized_cv_embedding, normalized_job_embeddings)[0]
    distances = pairwise_distances(normalized_cv_embedding, normalized_job_embeddings, metric='euclidean')[0]

    results_df = pd.DataFrame({
        'title': job_df['title'].fillna('N/A'),
        'description': job_df['description'].fillna('N/A'),
        'cluster': job_clusters,
        'similarity_score': cosine_similarities,
        'distance': distances
    })

    top_recommendations = results_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    return top_recommendations

if processed_cv_text and job_df is not None and normalized_job_embeddings is not None and job_clusters is not None and normalized_cv_embedding is not None:
    st.subheader("Top 20 Job Recommendations")
    top_recommendations_df = get_top_recommendations(normalized_cv_embedding, job_embeddings, job_df, job_clusters)
    if not top_recommendations_df.empty:
        st.dataframe(top_recommendations_df)

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
            annotation['job_index'] = row.name # Use DataFrame index for job_index
            annotation['title'] = row['title']
            annotation['description'] = row['description']

            cols = st.columns(len(annotators))
            for i, annotator in enumerate(annotators):
                with cols[i]:
                    relevant = st.radio(annotator, options=["Relevant", "Not Relevant"], key=f"anno_{row.name}_{annotator}")
                    annotation[annotator.lower().replace(" ", "_")] = 1 if relevant == "Relevant" else 0
            annotation_data.append(annotation)
            st.divider()

        if st.button("Submit Annotations"):
            annotation_df = pd.DataFrame(annotation_data)
            st.subheader("Submitted Annotations")
            st.dataframe(annotation_df)
            st.session_state['annotations'] = annotation_df

            # --- Saving annotations to CSV ---
            annotations_file = "annotations.csv"
            if os.path.exists(annotations_file):
                existing_df = pd.read_csv(annotations_file)
                updated_df = pd.concat([existing_df, annotation_df], ignore_index=True)
                updated_df.to_csv(annotations_file, index=False)
                st.success(f"Annotations saved to {annotations_file} (appended).")
            else:
                annotation_df.to_csv(annotations_file, index=False)
                st.success(f"Annotations saved to {annotations_file}.")

            # --- Option to download annotations ---
            csv_buffer = io.StringIO()
            annotation_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Current Annotations (CSV)",
                data=csv_buffer.getvalue(),
                file_name="current_annotations.csv",
                mime="text/csv",
            )
    else:
        st.info("No job recommendations found based on the uploaded CV.")

# --- Evaluate the Result ---
if 'annotations' in st.session_state and not top_recommendations_df.empty and normalized_cv_embedding is not None:
    st.subheader("Evaluation of Recommendations")

    annotation_df = st.session_state['annotations']
    merged_df = top_recommendations_df.merge(annotation_df, left_on=top_recommendations_df.index, right_on='job_index', how='inner') # Merge on index
    merged_df.set_index('key_0', inplace=True) # Reset index after merge

    # Assuming relevance if at least one annotator marked it as relevant
    if any(col.startswith('annotator_') for col in merged_df.columns):
        annotator_cols = [col for col in merged_df.columns if col.startswith('annotator_')]
        merged_df['ground_truth'] = merged_df[annotator_cols].max(axis=1)

        relevant_indices = merged_df[merged_df['ground_truth'] == 1].index.tolist()
        recommended_indices = merged_df.index.tolist()
        query_embedding = normalized_cv_embedding  # The processed CV embedding

        # Create a dummy evaluator (replace with a more sophisticated one if needed)
        # For simplicity, we'll calculate Recall@k and Precision@k manually here

        k_values = [5, 10, 20]
        results = {}

        for k in k_values:
            top_k_indices = merged_df.head(k).index.tolist()
            relevant_in_top_k = len(set(top_k_indices) & set(relevant_indices))
            precision_at_k = relevant_in_top_k / k if k > 0 else 0
            recall_at_k = relevant_in_top_k / len(relevant_indices) if relevant_indices else 0
            results[f'Precision@{k}'] = precision_at_k
            results[f'Recall@{k}'] = recall_at_k

        st.write("Evaluation Metrics (Based on Annotations):")
        st.write(results)

        # More advanced evaluation using SentenceTransformers' InformationRetrievalEvaluator
        # This requires creating a queries and corpus dictionary
        if not merged_df.empty:
            queries = {"query": processed_cv_text}
            corpus = {str(i): row['processed_description'] for i, row in merged_df.iterrows()}
            relevant_docs = defaultdict(set)
            for index in merged_df[merged_df['ground_truth'] == 1].index:
                relevant_docs["query"].add(str(index))

            try:
                evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs, top_k=[5, 10, 20])
                evaluation_result = evaluator(bert_model)
                st.subheader("SBERT Information Retrieval Evaluator Results (Based on Annotations)")
                st.write(evaluation_result)
            except Exception as e:
                st.warning(f"Error during SBERT Evaluation: {e}")
                st.warning("Ensure your annotation data is correctly aligned for evaluation.")
    else:
        st.warning("No annotation data found for evaluation.")
