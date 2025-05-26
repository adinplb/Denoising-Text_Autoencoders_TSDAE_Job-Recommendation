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

@st.cache_data
def load_job_data(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

job_data_url = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
job_df = load_job_data(job_data_url)

if job_df is not None:
    st.subheader("Job Data Preview")
    st.dataframe(job_df.head())

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
    st.text_area("", cv_text, height=300)



@st.cache_resource
def load_bert_model(model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model

@st.cache_data
def generate_embeddings(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

bert_model = load_bert_model()

if job_df is not None and 'description' in job_df.columns:
    job_descriptions = job_df['description'].fillna('').tolist()
    job_embeddings = generate_embeddings(job_descriptions, bert_model)
    normalized_job_embeddings = normalize(job_embeddings)

if cv_text:
    cv_embedding = generate_embeddings([cv_text], bert_model)[0]
    normalized_cv_embedding = normalize(cv_embedding.reshape(1, -1))



@st.cache_data
def cluster_embeddings(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters, kmeans.cluster_centers_

if job_df is not None and normalized_job_embeddings is not None:
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=20, value=5)
    job_clusters, cluster_centers = cluster_embeddings(normalized_job_embeddings, num_clusters)
    job_df['cluster'] = job_clusters




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
        hover_data=['title', 'company', 'location', 'cluster'],
        title='3D Visualization of Job Posting Clusters'
    )
    st.subheader("Job Posting Clusters (3D)")
    st.plotly_chart(fig)



def get_top_recommendations(cv_embedding, job_embeddings, job_df, job_clusters, top_n=20):
    if cv_embedding is None or job_embeddings is None:
        return pd.DataFrame()

    cosine_similarities = cosine_similarity(normalized_cv_embedding, normalized_job_embeddings)[0]
    distances = pairwise_distances(normalized_cv_embedding, normalized_job_embeddings, metric='euclidean')[0]

    results_df = pd.DataFrame({
        'title': job_df['title'],
        'company': job_df['company'],
        'location': job_df['location'],
        'cluster': job_clusters,
        'similarity_score': cosine_similarities,
        'distance': distances
    })

    top_recommendations = results_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    return top_recommendations

if cv_text and job_df is not None and normalized_job_embeddings is not None and job_clusters is not None:
    st.subheader("Top 20 Job Recommendations")
    top_recommendations_df = get_top_recommendations(normalized_cv_embedding, normalized_job_embeddings, job_df, job_clusters)
    st.dataframe(top_recommendations_df)




if cv_text and not top_recommendations_df.empty:
    st.subheader("Annotate Recommendations (Are these relevant to your CV?)")
    annotation_data = []
    annotators = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]

    for index, row in top_recommendations_df.iterrows():
        st.write(f"**Job Title:** {row['title']}")
        st.write(f"**Company:** {row['company']}")
        st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
        st.write(f"**Distance:** {row['distance']:.4f}")
        st.write(f"**Cluster:** {row['cluster']}")

        annotation = {}
        annotation['job_index'] = index
        annotation['title'] = row['title']

        cols = st.columns(len(annotators))
        for i, annotator in enumerate(annotators):
            with cols[i]:
                relevant = st.radio(annotator, options=["Relevant", "Not Relevant"], key=f"anno_{index}_{annotator}")
                annotation[annotator.lower().replace(" ", "_")] = 1 if relevant == "Relevant" else 0
        annotation_data.append(annotation)
        st.divider()

    if st.button("Submit Annotations"):
        annotation_df = pd.DataFrame(annotation_data)
        st.subheader("Submitted Annotations")
        st.dataframe(annotation_df)
        st.session_state['annotations'] = annotation_df



if 'annotations' in st.session_state and not top_recommendations_df.empty:
    st.subheader("Evaluation of Recommendations")

    annotation_df = st.session_state['annotations']
    merged_df = top_recommendations_df.merge(annotation_df, left_index=True, right_on='job_index', how='inner')

    # Assuming relevance if at least one annotator marked it as relevant
    merged_df['ground_truth'] = merged_df[['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4', 'annotator_5']].max(axis=1)

    relevant_indices = merged_df[merged_df['ground_truth'] == 1].index.tolist()
    recommended_indices = merged_df.index.tolist()
    query_embedding = normalized_cv_embedding  # The CV embedding

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

    st.write("Evaluation Metrics:")
    st.write(results)

    # More advanced evaluation using SentenceTransformers' InformationRetrievalEvaluator
    # This requires creating a queries and corpus dictionary
    if not merged_df.empty:
        queries = {"query": cv_text}
        corpus = {str(i): row['title'] + " " + row['company'] + " " + row['location'] for i, row in merged_df.iterrows()}
        relevant_docs = defaultdict(set)
        for index in merged_df[merged_df['ground_truth'] == 1].index:
            relevant_docs["query"].add(str(index))

        evaluator = evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs, top_k=[5, 10, 20])
        evaluation_result = evaluator(bert_model)
        st.subheader("SBERT Information Retrieval Evaluator Results")
        st.write(evaluation_result)

