import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random
import nltk
import torch

# --- Data URLs ---
JOB_DATA_URL = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/combined_jobs_2000.csv"
USER_DATA_URL = "https://raw.githubusercontent.com/adinplb/Denoising-Text_Autoencoders_TSDAE_Job-Recommendation/refs/heads/master/dataset/user_applicant_jobs.csv"

# --- NLTK Download (Important for Streamlit Cloud) ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- Helper Functions (from your Colab) ---
def denoise_text(text, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    words = word_tokenize(text)
    n = len(words)
    if n == 0:
        return text

    if method == 'a':
        # === (a) Random 60% Deletion ===
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True
        result = np.array(words)[keep_or_not]

    elif method == 'b':
        # === (b) Remove 60% of high-frequency words ===
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'")
        high_freq_words = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        to_remove = set(random.sample(high_freq_words, int(del_ratio * len(high_freq_words)))) if high_freq_words else set()
        result = [w for i, w in enumerate(words) if i not in to_remove]

    elif method == 'c':
        # === (c) Based on (b) + shuffle remaining words ===
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'")
        high_freq_words = [i for i, w in enumerate(words) if word_freq_dict.get(w.lower(), 0) > freq_threshold]
        to_remove = set(random.sample(high_freq_words, int(del_ratio * len(high_freq_words)))) if high_freq_words else set()
        result = [w for i, w in enumerate(words) if i not in to_remove]
        random.shuffle(result)  # simple shuffle, pair-aware shuffling can be added if needed

    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")

    return TreebankWordDetokenizer().detokenize(result)

def encode_batch(jobbert_model, texts):
    features = jobbert_model.tokenize(texts)
    features = batch_to_device(features, jobbert_model.device)
    features["text_keys"] = ["anchor"]
    with torch.no_grad():
        out_features = jobbert_model.forward(features)
    return out_features["sentence_embedding"].cpu().numpy()

def encode(jobbert_model, texts, batch_size: int = 8):
    # Sort texts by length and keep track of original indices
    sorted_indices = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in sorted_indices]

    embeddings = []

    # Encode in batches
    for i in range(0, len(sorted_texts), batch_size):
        batch = sorted_texts[i:i+batch_size]
        embeddings.append(encode_batch(jobbert_model, batch))

    # Concatenate embeddings and reorder to original indices
    sorted_embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_indices)
    return sorted_embeddings[original_order]

def calculate_relative_threshold(similarities, percentile=75):
    """
    Calculate the relative similarity threshold based on a given percentile.
    """
    return np.percentile(similarities, percentile)

def evaluate_with_relative_threshold(similarities, top_n_indices, df_clustered_jobbert, threshold):
    """
    Evaluate relevance based on cosine similarity and a dynamic threshold.
    """
    relevant_docs = set()
    for idx, similarity in zip(top_n_indices, similarities):
        if similarity >= threshold:
            relevant_docs.add(df_clustered_jobbert.iloc[idx]["Job.ID"])  # Mark as relevant
    return relevant_docs

def get_top_n_local_search(embeddings_user, df_clustered_jobbert, embedding_matrix, top_n_list=[3, 5, 10, 20]):
    # === Step 1: Find the closest cluster center to the user embedding ===
    cluster_centers = kmeans.cluster_centers_
    cluster_similarities = cosine_similarity(embeddings_user, cluster_centers)
    best_cluster_id = np.argmax(cluster_similarities)

    # === Step 2: Filter jobs in the best cluster ===
    cluster_subset = df_clustered_jobbert[df_clustered_jobbert['cluster'] == best_cluster_id]
    cluster_indices = cluster_subset.index.to_numpy()
    cluster_embeddings = embedding_matrix[cluster_indices]

    # === Step 3: Compute similarity between user and job postings in the cluster ===
    similarities = cosine_similarity(embeddings_user, cluster_embeddings).flatten()
    top_indices_within_cluster = np.argsort(similarities)[::-1]

    # === Step 4: Calculate relative similarity threshold ===
    threshold = calculate_relative_threshold(similarities)

    # === Step 5: Extract top-N matches with relevance evaluation based on threshold ===
    results = {}
    for n in top_n_list:
        top_n_idx = top_indices_within_cluster[:n]
        selected_indices = cluster_indices[top_n_idx]
        top_n_df = df_clustered_jobbert.loc[selected_indices].copy()
        top_n_df["similarity"] = similarities[top_n_idx]

        # Evaluate relevance based on relative similarity threshold
        relevant_docs = evaluate_with_relative_threshold(similarities[top_n_idx], top_n_idx, df_clustered_jobbert, threshold)
        top_n_df["relevance_label"] = top_n_df["similarity"].apply(lambda x: "relevant" if x >= threshold else "not relevant")

        results[f"top_{n}"] = top_n_df[['Job.ID', 'Title', 'text', 'cluster', 'similarity', 'relevance_label']]

    return results

# --- Data and Model Loading (Cached) ---
@st.cache_data(show_spinner="Loading job titles...")
def load_job_data():
    st.write("load_job_data: Starting")  # Debugging
    try:
        job_titles = pd.read_csv(JOB_DATA_URL)
        st.write("load_job_data: Finished loading job titles")  # Debugging
        return job_titles
    except Exception as e:
        st.error(f"Error loading job titles: {e}")
        st.write(f"load_job_data: Error in load_job_data: {e}")  # Debugging
        return None

@st.cache_data(show_spinner="Loading user data...")
def load_user_data():
    st.write("load_user_data: Starting")  # Debugging
    try:
        user_corpus = pd.read_csv(USER_DATA_URL)
        st.write("load_user_data: Finished loading user data")  # Debugging
        return user_corpus
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        st.write(f"load_user_data: Error in load_user_data: {e}")  # Debugging
        return None

@st.cache_resource(show_spinner="Loading the JobBERT model...")
def load_model():
    st.write("load_model: Starting")  # Debugging
    try:
        model = SentenceTransformer("TechWolf/JobBERT-v2")
        st.write("load_model: Finished loading model")  # Debugging
        return model
    except Exception as e:
        st.error(f"Error loading the JobBERT model: {e}")
        st.write(f"load_model: Error in load_model: {e}")  # Debugging
        return None

@st.cache_data(show_spinner="Processing job data...")
def process_job_data(job_titles, _model):  # Changed 'model' to '_model'
    st.write("process_job_data: Starting")  # Debugging
    if job_titles is None or _model is None:
        st.write("process_job_data: job_titles or _model is None")  # Debugging
        return None, None
    try:
        job_titles['noisy_text'] = job_titles['text'].fillna("").apply(lambda x: denoise_text(x))
        clean_texts = job_titles['text'].fillna("").tolist()
        noisy_texts = job_titles['noisy_text'].tolist()
        clean_embeddings = encode(_model, clean_texts) # Use _model here
        noisy_embeddings = encode(_model, noisy_texts) # And here
        tsdae_embeddings = (clean_embeddings + noisy_embeddings) / 2.0
        job_titles['jobbert_tsdae_embedding'] = tsdae_embeddings.tolist()
        st.write("process_job_data: Finished processing job data")  # Debugging
        return job_titles, tsdae_embeddings
    except Exception as e:
        st.error(f"Error processing job data: {e}")
        st.write(f"process_job_data: Error: {e}")  # Debugging
        return None, None

@st.cache_data(show_spinner="Clustering job embeddings...")
def cluster_embeddings(job_titles, num_clusters=20):
    st.write("cluster_embeddings: Starting")  # Debugging
    if job_titles is None:
        st.write("cluster_embeddings: job_titles is None")  # Debugging
        return None, None, None
    try:
        embedding_matrix = np.vstack(job_titles['jobbert_tsdae_embedding'].values)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) # Added n_init to suppress warning
        kmeans.fit(embedding_matrix)
        cluster_labels = kmeans.labels_
        df_clustered_jobbert = pd.DataFrame({
            'Job.ID': job_titles['Job.ID'].values if 'Job.ID' in job_titles.columns else range(len(job_titles)),
            'Title': job_titles['Title'].values if 'Title' in job_titles.columns else [None]*len(job_titles),
            'text': job_titles['text'].values,
            'cluster': cluster_labels,
            'original_index': job_titles.index
        })
        st.write("cluster_embeddings: Finished clustering")  # Debugging
        return df_clustered_jobbert, kmeans, embedding_matrix
    except Exception as e:
        st.error(f"Error clustering job embeddings: {e}")
        st.write(f"cluster_embeddings: Error: {e}")  # Debugging
        return None, None, None

@st.cache_data(show_spinner="Processing user data...")
def process_user_data(user_corpus, _model): # Changed 'model' to '_model'
    st.write("process_user_data: Starting")  # Debugging
    if user_corpus is None or _model is None:
        st.write("process_user_data: user_corpus or _model is None")  # Debugging
        return None, None
    try:
        texts_user = user_corpus["text"].fillna("").tolist()
        embeddings_user = encode(_model, texts_user) # Use _model here
        user_corpus['jobbert_embedding'] = embeddings_user.tolist()
        st.write("process_user_data: Finished processing user data")  # Debugging
        return user_corpus, embeddings_user
    except Exception as e:
        st.error(f"Error processing user data: {e}")
        st.write(f"process_user_data: Error: {e}")  # Debugging
        return None, None

# --- Main Streamlit App ---
def main():
    st.title("Job Recommendation Dashboard")
    st.write("main: App starting")  # Debugging

    # Load data and model
    job_titles = load_job_data()
    user_corpus = load_user_data()
    model = load_model()

    # Check for errors
    if job_titles is None or user_corpus is None or model is None:
        st.write("main: Error loading data or model. Stopping.")  # Debugging
        st.stop()

    # Process data and cluster embeddings
    job_titles, tsdae_embeddings = process_job_data(job_titles, model)
    df_clustered_jobbert, kmeans, embedding_matrix = cluster_embeddings(job_titles)
    user_corpus, embeddings_user = process_user_data(user_corpus, model)

    # Check if processing or clustering failed
    if df_clustered_jobbert is None or embeddings_user is None:
        st.write("main: Error processing data. Stopping.")  # Debugging
        st.stop()

    # Display raw data in expanders
    with st.expander("Show Job Titles Data"):
        st.dataframe(job_titles)

    with st.expander("Show User Corpus Data"):
        st.dataframe(user_corpus)

    # User input for job query
    query_text = st.text_input("Enter a job title to find recommendations:", "java developer")

    if query_text:
        # Find the matching row in user_corpus
        user_q_row = user_corpus[user_corpus['text'].str.lower() == query_text.lower()]

        # Safety check
        if user_q_row.empty:
            st.error(f"Text '{query_text}' not found in user_corpus.")
        else:
            # Extract existing embedding from user_corpus
            embeddings_user = np.array(user_q_row.iloc[0]['jobbert_embedding']).reshape(1, -1)

            # Get recommendations
            recommendations = get_top_n_local_search(embeddings_user, df_clustered_jobbert, embedding_matrix)

            st.subheader(f"Top Job Recommendations for '{query_text}'")
            for k, df in recommendations.items():
                st.write(f"**{k.upper()}**")
                st.dataframe(df[['Job.ID', 'Title', 'cluster', 'similarity', 'relevance_label']])

    st.write("main: App finished successfully (if you see this, it's a good sign!)")  # Debugging

if __name__ == "__main__":
    main()
