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
from sentence_transformers.evaluation import InformationRetrievalEvaluator 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLTK Resource Downloads ---
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
N_CLUSTERS = 20 
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
if 'data' not in st.session_state:
    st.session_state['data'] = None 
if 'job_text_embeddings' not in st.session_state:
    st.session_state['job_text_embeddings'] = None 
if 'job_text_embedding_job_ids' not in st.session_state: 
    st.session_state['job_text_embedding_job_ids'] = None
if 'tsdae_embeddings' not in st.session_state:
    st.session_state['tsdae_embeddings'] = None 
if 'tsdae_embedding_job_ids' not in st.session_state: 
    st.session_state['tsdae_embedding_job_ids'] = None
if 'job_clusters_raw' not in st.session_state: 
    st.session_state['job_clusters_raw'] = None
if 'uploaded_cvs_data' not in st.session_state:
    st.session_state['uploaded_cvs_data'] = [] 
if 'all_recommendations_for_annotation' not in st.session_state:
    st.session_state['all_recommendations_for_annotation'] = {} 
if 'collected_annotations' not in st.session_state:
    st.session_state['collected_annotations'] = pd.DataFrame()


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading data...')
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        st.success('Successfully loaded data!')
        if 'Job.ID' in df.columns:
            df['Job.ID'] = df['Job.ID'].astype(str) 
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
        document = DocxDocument(uploaded_file)
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return None

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip(): 
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [w for w in word_tokens if w not in stop_words and w.isalnum()]
    if not filtered_words: 
        return ""
    porter = PorterStemmer()
    stemmed_words = [porter.stem(w) for w in filtered_words]
    return " ".join(stemmed_words)

def preprocess_text_with_intermediate(data_df):
    processed_results_intermediate = [] 
    if 'text' not in data_df.columns:
        st.warning("The 'text' column was not found for preprocessing.")
        return data_df 

    with st.spinner("Preprocessing 'text' column..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)
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
        
        data_df['processed_text'] = data_df['text'].fillna('').astype(str).apply(preprocess_text)
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success("Preprocessing of 'text' column complete!")
        progress_bar.empty()
        status_text.empty()
    return data_df

def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    if not isinstance(text_to_denoise, str) or not text_to_denoise.strip():
        return "" 
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
        if method == 'c' and result_words: 
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
def generate_embeddings_with_progress(_model, texts_list_to_embed): 
    if _model is None:
        st.error("BERT model is not loaded for embedding generation.")
        return np.array([]) 
    if not texts_list_to_embed: 
        st.warning("Input text list for embedding is empty.")
        return np.array([])
    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = [] 
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size] 
                batch_embeddings_np_array = _model.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False) 
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
def cluster_embeddings_with_progress(embeddings_to_cluster_param, n_clusters_for_algo): 
    if embeddings_to_cluster_param is None or embeddings_to_cluster_param.size == 0:
        st.warning("No embeddings provided for clustering.")
        return None
    if n_clusters_for_algo > embeddings_to_cluster_param.shape[0]:
        st.warning(f"K ({n_clusters_for_algo}) > samples ({embeddings_to_cluster_param.shape[0]}). Adjusting K.")
        n_clusters_for_algo = embeddings_to_cluster_param.shape[0]
    if n_clusters_for_algo < 1 : 
         st.error("Not enough samples to cluster (K < 1).")
         return None
    if n_clusters_for_algo == 1 and embeddings_to_cluster_param.shape[0] > 1 : 
        st.warning(f"K=1 requested for >1 samples. Setting K=2 for meaningful clustering.")
        n_clusters_for_algo = 2
    elif embeddings_to_cluster_param.shape[0] == 1 and n_clusters_for_algo >1: 
        st.warning(f"Only 1 sample available. Setting K=1.")
        n_clusters_for_algo = 1


    try:
        with st.spinner(f"Clustering {embeddings_to_cluster_param.shape[0]} embeddings into {n_clusters_for_algo} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters_for_algo, random_state=42, n_init='auto')
            clusters_assigned = kmeans.fit_predict(embeddings_to_cluster_param) 
            st.success(f"Clustering complete!")
            return clusters_assigned
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None

# --- Page Functions ---
def home_page():
    st.header("Home: Exploratory Data Analysis")
    st.write("Overview of the job dataset and feature exploration.")
    if st.session_state.get('data') is None:
        st.session_state['data'] = load_data_from_url(DATA_URL)
    data_df = st.session_state.get('data')
    if data_df is not None:
        st.subheader('Data Preview')
        st.dataframe(data_df.head(), use_container_width=True)
        st.subheader('Data Summary')
        st.write(f'Rows: {len(data_df)}, Columns: {len(data_df.columns)}')
        st.subheader('Search Word in Feature')
        search_word = st.text_input("Search word:", key="home_search_word")
        column_options = [''] + [str(col) for col in data_df.columns.tolist()]
        search_column = st.selectbox("Search in column:", column_options, key="home_search_column")
        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                if not search_results.empty:
                    st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':")
                    st.dataframe(search_results, use_container_width=True)
                else:
                    st.info(f"No entries found for '{search_word}' in '{search_column}'.")
        st.subheader('Feature Information')
        st.write('**Features:**', data_df.columns.tolist())
        st.subheader('Explore Feature Details')
        selected_feature = st.selectbox('Select Feature:', [''] + data_df.columns.tolist(), key="home_feature_select")
        if selected_feature:
            st.write(f'**Feature:** `{selected_feature}`, **DType:** `{data_df[selected_feature].dtype}`, **Unique Values:** `{data_df[selected_feature].nunique()}`')
            st.write('**Sample Unique Values (first 20):**', data_df[selected_feature].unique()[:20])
            if pd.api.types.is_numeric_dtype(data_df[selected_feature]):
                st.write(data_df[selected_feature].describe())
                try: 
                    fig = px.histogram(data_df, x=selected_feature, title=f'Distribution of {selected_feature}')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e: st.warning(f"Histogram error for {selected_feature}: {e}")
            elif pd.api.types.is_string_dtype(data_df[selected_feature]) or pd.api.types.is_object_dtype(data_df[selected_feature]):
                st.write(data_df[selected_feature].value_counts().nlargest(20))
    else:
        st.error("Data could not be loaded. Check source/network.")
    return

def preprocessing_page():
    st.header("Preprocessing Job Descriptions")
    st.write("Performs text preprocessing on the 'text' column of the job dataset.")
    if st.session_state.get('data') is None:
        st.info("Job data not loaded. Attempting to load...")
        st.session_state['data'] = load_data_from_url(DATA_URL)
        if st.session_state.get('data') is None:
            st.error("Failed to load job data.")
            return
    
    if st.button("Run Preprocessing on Job Descriptions", key="run_job_prep_btn"):
        with st.spinner("Preprocessing jobs..."):
            data_copy = st.session_state['data'].copy()
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy)
        st.success("Job preprocessing complete!")
    
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Job preprocessing has been performed.")
        display_data = st.session_state['data']
        if 'preprocessing_steps' in display_data.columns:
            st.subheader("Intermediate Steps (last run)")
            valid_steps = [s for s in display_data['preprocessing_steps'] if isinstance(s, dict)]
            if valid_steps: st.dataframe(pd.DataFrame(valid_steps).head(), use_container_width=True)
            else: st.warning("Intermediate steps data missing/invalid.")
        st.subheader("Final Preprocessed Job Text (Preview)")
        st.dataframe(display_data[['Job.ID', 'text', 'processed_text']].head(), use_container_width=True)
        search_word_job = st.text_input("Search in 'processed_text':", key="prep_job_search")
        if search_word_job:
            results = display_data[display_data['processed_text'].astype(str).str.contains(search_word_job, na=False, case=False)]
            if not results.empty: st.dataframe(results[['Job.ID', 'Title', 'processed_text']],use_container_width=True)
            else: st.info(f"No matches for '{search_word_job}'.")
    else:
        st.info("Job data loaded, but preprocessing not run yet.")
    return

def tsdae_page():
    st.header("TSDAE (Noise Injection & Embedding for Job Text)")
    st.write("Applies sequential noise and generates TSDAE embeddings for preprocessed job text.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed first.")
        return
    bert_model = load_bert_model()
    if bert_model is None: return

    st.subheader("TSDAE Settings")
    del_ratio = st.slider("Deletion Ratio", 0.1, 0.9, 0.6, 0.1, key="tsdae_del_r")
    freq_thresh = st.slider("High Freq Threshold", 10, 500, 100, 10, key="tsdae_freq_t")

    if st.button("Apply Noise & Generate TSDAE Embeddings", key="tsdae_run_all_btn"):
        data_tsdae = st.session_state['data'].copy()
        words_for_freq = [w for txt in data_tsdae['processed_text'].fillna('').astype(str) for w in word_tokenize(txt)]
        word_freq = {w.lower(): words_for_freq.count(w.lower()) for w in set(words_for_freq)}
        if not word_freq: st.warning("Word freq dict for TSDAE empty.")

        with st.spinner("Noise A..."): data_tsdae['noisy_text_a'] = data_tsdae['processed_text'].fillna('').astype(str).apply(lambda x: denoise_text(x, 'a', del_ratio))
        with st.spinner("Noise B..."): data_tsdae['noisy_text_b'] = data_tsdae['noisy_text_a'].astype(str).apply(lambda x: denoise_text(x, 'b', del_ratio, word_freq, freq_thresh))
        with st.spinner("Noise C..."): data_tsdae['final_noisy_text'] = data_tsdae['noisy_text_b'].astype(str).apply(lambda x: denoise_text(x, 'c', del_ratio, word_freq, freq_thresh))
        st.session_state['data'] = data_tsdae
        st.success("Noise application complete.")
        st.dataframe(data_tsdae[['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)

        noisy_series = data_tsdae['final_noisy_text'].fillna('').astype(str)
        mask = noisy_series.str.strip() != ''
        valid_noisy_texts = noisy_series[mask].tolist()
        valid_noisy_job_ids = data_tsdae.loc[mask, 'Job.ID'].tolist()
        if not valid_noisy_texts: st.warning("No valid noisy texts for TSDAE embedding.")
        else:
            st.session_state['tsdae_embeddings'] = generate_embeddings_with_progress(bert_model, valid_noisy_texts)
            st.session_state['tsdae_embedding_job_ids'] = valid_noisy_job_ids
            if st.session_state.get('tsdae_embeddings', np.array([])).size > 0: st.success(f"TSDAE embeddings generated for {len(valid_noisy_job_ids)} jobs!")
            else: st.warning("TSDAE embedding output empty.")
    
    if st.session_state.get('tsdae_embeddings') is not None:
        st.subheader("Current TSDAE Embeddings")
        st.write(f"Shape: {st.session_state['tsdae_embeddings'].shape} (for {len(st.session_state.get('tsdae_embedding_job_ids',[]))} jobs)")
    if 'final_noisy_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader("Current Noisy Text Columns (Preview)")
        st.dataframe(st.session_state['data'][['Job.ID','processed_text', 'noisy_text_a', 'noisy_text_b', 'final_noisy_text']].head(), height=200)
    return

def bert_model_page():
    st.header("Standard BERT Embeddings (Job Descriptions)")
    st.write("Generates standard BERT embeddings from preprocessed job descriptions.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed.")
        return
    bert_model = load_bert_model()
    if bert_model is None: return

    if st.button("Generate/Regenerate Standard Job Embeddings", key="gen_std_emb_btn"):
        data_bert = st.session_state['data']
        proc_series = data_bert['processed_text'].fillna('').astype(str)
        mask = proc_series.str.strip() != ''
        valid_texts = proc_series[mask].tolist()
        valid_job_ids = data_bert.loc[mask, 'Job.ID'].tolist()
        if not valid_texts: st.warning("No valid processed job texts for embedding.")
        else:
            st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(bert_model, valid_texts)
            st.session_state['job_text_embedding_job_ids'] = valid_job_ids
            if st.session_state.get('job_text_embeddings', np.array([])).size > 0: st.success(f"Std job embeddings generated for {len(valid_job_ids)} jobs!")
            else: st.warning("Std job embedding output empty.")

    job_emb = st.session_state.get('job_text_embeddings')
    job_ids = st.session_state.get('job_text_embedding_job_ids')
    if job_emb is not None and job_emb.size > 0 and job_ids:
        st.subheader(f"Current Standard Job Embeddings ({len(job_ids)} jobs)")
        st.write(f"Shape: {job_emb.shape}")
        st.subheader("2D Visualization (PCA)")
        if len(job_emb) >= 2: 
            try:
                pca_2d = PCA(n_components=2).fit_transform(job_emb)
                plot_pca_df = pd.DataFrame(pca_2d, columns=['PC1','PC2'])
                plot_pca_df['Job.ID'] = job_ids
                hover_df = st.session_state['data'][st.session_state['data']['Job.ID'].isin(job_ids)][['Job.ID', 'Title', 'text']]
                plot_pca_df = pd.merge(plot_pca_df, hover_df, on='Job.ID', how='left')
                if not plot_pca_df.empty and 'Title' in plot_pca_df.columns:
                    fig_pca = px.scatter(plot_pca_df, 'PC1','PC2', hover_name='Title', hover_data={'text':True,'Job.ID':True,'PC1':False,'PC2':False}, title='2D PCA of Std Job Embeddings')
                    st.plotly_chart(fig_pca, use_container_width=True)
                else: st.warning("PCA plot data incomplete.")
            except Exception as e: st.error(f"PCA Error: {e}")
        else: st.warning("Need >= 2 data points for PCA.")
    else: st.info("Standard job embeddings not generated yet.")
    return

def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Clusters job embeddings using K-Means and merges results to main dataset.")
    if st.session_state.get('data') is None: st.error("Job data not loaded."); return

    emb_to_cluster, job_ids_clust, src_name_clust = None, None, ""
    choice = st.radio("Embeddings for clustering:", ("TSDAE", "Standard BERT"), key="clust_emb_choice", horizontal=True)

    if choice == "TSDAE":
        if st.session_state.get('tsdae_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['tsdae_embeddings']
            job_ids_clust = st.session_state.get('tsdae_embedding_job_ids')
            src_name_clust = "TSDAE Embeddings"
            if not job_ids_clust: st.error("TSDAE Job IDs missing."); return
        else: st.warning("TSDAE embeddings unavailable."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings', np.array([])).size > 0:
            emb_to_cluster = st.session_state['job_text_embeddings']
            job_ids_clust = st.session_state.get('job_text_embedding_job_ids')
            src_name_clust = "Standard BERT Job Embeddings"
            if not job_ids_clust: st.error("Std BERT Job IDs missing."); return
        else: st.warning("Std BERT embeddings unavailable."); return
    
    st.info(f"Using: {src_name_clust} ({len(job_ids_clust)} items)")
    if emb_to_cluster is not None and job_ids_clust:
        max_k_val = emb_to_cluster.shape[0]
        if max_k_val < 2: st.error("Need >= 2 items to cluster."); return
        k_val = st.slider("Number of Clusters (K)", 2, min(50, max_k_val), min(N_CLUSTERS, max_k_val), key="k_slider_c")
        if st.button(f"Run K-Means (K={k_val}) on {src_name_clust}", key="run_kmeans_c_btn"):
            labels = cluster_embeddings_with_progress(emb_to_cluster, k_val)
            if labels is not None:
                if len(job_ids_clust) == len(labels):
                    info_df = pd.DataFrame({'Job.ID': job_ids_clust, 'cluster': labels})
                    data_copy = st.session_state['data'].copy()
                    if 'cluster' in data_copy.columns: data_copy = data_copy.drop(columns=['cluster'])
                    st.session_state['data'] = pd.merge(data_copy, info_df, on='Job.ID', how='left')
                    st.success(f"'cluster' column updated for {len(job_ids_clust)} jobs.")
                else: st.error("Job ID / cluster label length mismatch.")
            else: st.error("Clustering failed.")

    if 'cluster' in st.session_state.get('data', pd.DataFrame()).columns:
        st.subheader(f"Current Clustering (K={st.session_state['data']['cluster'].nunique(dropna=True)})")
        st.dataframe(st.session_state['data'][['Job.ID', 'Title', 'text', 'cluster']].head(10), height=300)
        valid_cl = st.session_state['data']['cluster'].dropna().unique()
        if valid_cl.size > 0:
            st.subheader("Sample Job Descriptions per Cluster")
            for c_num in sorted(valid_cl):
                st.write(f"**Cluster {int(c_num)}:**")
                subset = st.session_state['data'][st.session_state['data']['cluster'] == c_num]
                if not subset.empty: st.dataframe(subset[['Job.ID', 'Title', 'text']].sample(min(3,len(subset)),random_state=1), height=150)
                st.write("---")
    else: st.info("No 'cluster' column in dataset or no clusters assigned.")
    return

def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5).")
    uploaded_cv_files = st.file_uploader("Choose CV files:", type=["pdf","docx"], accept_multiple_files=True, key="cv_upload_widget")
    if uploaded_cv_files:
        if len(uploaded_cv_files) > 5:
            st.warning("Max 5 CVs. Processing first 5.")
            uploaded_cv_files = uploaded_cv_files[:5]
        if st.button("Process Uploaded CVs", key="proc_cv_btn"):
            cv_data_batch = []
            bert_model_for_cv = load_bert_model()
            if not bert_model_for_cv: 
                st.error("BERT model load failed for CVs.")
                return # Exit if model not loaded
            with st.spinner("Processing CVs..."):
                cv_prog_bar = st.progress(0)
                cv_stat_txt = st.empty()
                for i, cv_file in enumerate(uploaded_cv_files):
                    o_txt, p_txt, cv_e = "", "", None
                    try:
                        file_ext = cv_file.name.split(".")[-1].lower()
                        if file_ext == "pdf": o_txt = extract_text_from_pdf(cv_file)
                        elif file_ext == "docx": o_txt = extract_text_from_docx(cv_file)
                        
                        if o_txt and o_txt.strip():
                            p_txt = preprocess_text(o_txt)
                            if p_txt and p_txt.strip():
                                e_arr = generate_embeddings_with_progress(bert_model_for_cv, [p_txt])
                                # Ensure e_arr is not None and has elements before accessing e_arr[0]
                                cv_e = e_arr[0] if (e_arr is not None and e_arr.size > 0) else None
                            else: 
                                st.warning(f"Processed text for {cv_file.name} is empty.")
                        else: 
                            st.warning(f"Extracted text for {cv_file.name} is empty.")
                        
                        cv_data_batch.append({'filename':cv_file.name, 'original_text':o_txt or "", 
                                              'processed_text':p_txt or "", 'embedding':cv_e})
                        
                        # CORRECTED LOGIC FOR DISPLAYING SUCCESS/FAILURE
                        if cv_e is not None and cv_e.size > 0: # Check if cv_e is a non-empty array
                            st.success(f"Processed & embedded: {cv_file.name}")
                        elif p_txt: # This means text was processed but embedding (cv_e) is None or empty
                            st.warning(f"Processed {cv_file.name}, but embedding failed or resulted in empty array.")
                        # else: (No original text or no processed text) - already warned above

                    except Exception as e:
                        st.error(f"Error with {cv_file.name}: {e}")
                        cv_data_batch.append({'filename':cv_file.name, 'original_text':"Error in processing", 
                                              'processed_text':"", 'embedding':None}) # Log error case
                    
                    if uploaded_cv_files: # Check if list is not empty before division
                        cv_prog_bar.progress((i+1)/len(uploaded_cv_files))
                    cv_stat_txt.text(f"Done: {i+1}/{len(uploaded_cv_files)}")

                st.session_state['uploaded_cvs_data'] = cv_data_batch
                cv_prog_bar.empty()
                cv_stat_txt.empty()
                st.success(f"CV batch processing done. {len(cv_data_batch)} attempted.")

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs:")
        for i, cv_d in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_d.get('filename', 'N/A')}"):
                st.text_area(f"Original:", cv_d.get('original_text',''), height=70, disabled=True, key=f"d_cv_o_{i}")
                st.text_area(f"Processed:", cv_d.get('processed_text',''), height=70, disabled=True, key=f"d_cv_p_{i}")
                if cv_d.get('embedding') is not None and cv_d.get('embedding').size > 0:
                    st.success("Embedding OK.") 
                else:
                    st.warning("Embedding missing or empty.")
    else: 
        st.info("No CVs processed yet.")
    return

def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Generates job recommendations for uploaded CVs.")
    if not st.session_state.get('uploaded_cvs_data'): st.warning("Upload & process CVs first."); return
    main_data = st.session_state.get('data')
    if main_data is None or 'processed_text' not in main_data.columns: st.error("Job data not ready."); return
    
    job_emb, emb_msg, job_emb_ids = None, "", None # Initialize job_emb_ids
    rec_choice = st.radio("Job Embeddings for Recs:", ("Standard BERT", "TSDAE"), key="rec_emb_c", horizontal=True)

    if rec_choice == "TSDAE":
        if st.session_state.get('tsdae_embeddings', np.array([])).size > 0:
            job_emb = st.session_state['tsdae_embeddings']
            job_emb_ids = st.session_state.get('tsdae_embedding_job_ids') # Get corresponding IDs
            if not job_emb_ids or len(job_emb_ids) != job_emb.shape[0]: st.error("TSDAE emb/ID mismatch."); return
            emb_msg = "Using TSDAE embeddings."
        else: st.warning("TSDAE embeddings unavailable."); return
    else: # Standard BERT
        if st.session_state.get('job_text_embeddings', np.array([])).size > 0:
            job_emb = st.session_state['job_text_embeddings']
            job_emb_ids = st.session_state.get('job_text_embedding_job_ids') # Get corresponding IDs
            if not job_emb_ids or len(job_emb_ids) != job_emb.shape[0]: st.error("Std BERT emb/ID mismatch."); return
            emb_msg = "Using Standard BERT job embeddings."
        else: st.warning("Std BERT job embeddings unavailable."); return
    st.info(emb_msg)

    # Filter main_data to only include jobs for which we have the selected embeddings and their IDs
    if not job_emb_ids: # Should be caught above, but defensive
        st.error("Job IDs for selected embeddings are missing."); return
        
    jobs_for_sim_df = main_data[main_data['Job.ID'].isin(job_emb_ids)].copy()
    # Ensure jobs_for_sim_df is ordered according to job_emb_ids to align with job_emb for cosine similarity
    jobs_for_sim_df['Job.ID'] = pd.Categorical(jobs_for_sim_df['Job.ID'], categories=job_emb_ids, ordered=True)
    jobs_for_sim_df = jobs_for_sim_df.sort_values('Job.ID').reset_index(drop=True)

    if len(jobs_for_sim_df) != len(job_emb): 
        st.error(f"Job emb/data alignment error for recs. Filtered jobs: {len(jobs_for_sim_df)}, Embeddings: {len(job_emb)}"); 
        return

    if st.button("Generate Recommendations", key="gen_recs_b"):
        st.session_state['all_recommendations_for_annotation'] = {} 
        with st.spinner("Generating recommendations..."):
            valid_cvs_rec = [cv for cv in st.session_state.get('uploaded_cvs_data', []) if cv.get('embedding') is not None and cv.get('embedding').size > 0]
            if not valid_cvs_rec: st.warning("No CVs with valid embeddings found."); return

            for cv_data_rec in valid_cvs_rec:
                cv_file_n = cv_data_rec.get('filename', 'Unknown CV')
                cv_embed = cv_data_rec['embedding']
                st.subheader(f"Recommendations for {cv_file_n}")
                cv_embed_2d = cv_embed.reshape(1, -1) if cv_embed.ndim == 1 else cv_embed
                
                if job_emb.ndim == 1 or job_emb.shape[0] == 0: 
                    st.error(f"Job embeddings invalid for {cv_file_n}.")
                    continue 
                
                similarities_rec = cosine_similarity(cv_embed_2d, job_emb)[0]
                
                # temp_df_rec should be jobs_for_sim_df which is already filtered and ordered
                temp_df_rec = jobs_for_sim_df.copy() 
                temp_df_rec['similarity_score'] = similarities_rec
                
                recommended_j_df = temp_df_rec.sort_values(by='similarity_score', ascending=False).head(20)
                if not recommended_j_df.empty:
                    display_c = ['Job.ID', 'Title', 'similarity_score']
                    if 'cluster' in recommended_j_df.columns: display_c.append('cluster')
                    display_c.append('text') 
                    st.dataframe(recommended_j_df[display_c], use_container_width=True)
                    st.session_state['all_recommendations_for_annotation'][cv_file_n] = recommended_j_df
                else: st.info(f"No recommendations for {cv_file_n}.")
                st.write("---") 
        st.success("Recommendation generation complete!")
    return

def annotation_page():
    st.header("Annotation of Job Recommendations")
    st.write("Annotate relevance (0-3) and provide feedback.")
    if not st.session_state.get('all_recommendations_for_annotation'): st.warning("Generate recommendations first."); return
    if 'collected_annotations' not in st.session_state: st.session_state['collected_annotations'] = pd.DataFrame()

    with st.form(key="ann_form"):
        form_input = []
        for cv_fn, recs in st.session_state['all_recommendations_for_annotation'].items():
            st.markdown(f"### CV: **{cv_fn}**")
            for _, row_item in recs.iterrows():
                jobid = str(row_item['Job.ID'])
                st.markdown(f"**Job ID:** {jobid} | **Title:** {row_item['Title']}")
                with st.expander("Details"): st.write(f"Desc: {row_item['text']}\nSim: {row_item['similarity_score']:.4f}")
                item_data = {'cv_filename':cv_fn,'job_id':jobid,'job_title':row_item['Title'],'job_text':row_item['text'],
                             'similarity_score':row_item['similarity_score'],'cluster':row_item.get('cluster',pd.NA)}
                ann_item_inputs = {}
                ann_cols_layout = st.columns(len(ANNOTATORS))
                for i, annotator in enumerate(ANNOTATORS):
                    with ann_cols_layout[i]:
                        st.markdown(f"**{annotator}**")
                        rel_k, fb_k = f"rel_{cv_fn}_{jobid}_{annotator}", f"fb_{cv_fn}_{jobid}_{annotator}"
                        def_r, def_f = 0, ""
                        if not st.session_state['collected_annotations'].empty:
                            mask_prev = (st.session_state['collected_annotations']['cv_filename'] == cv_fn) & \
                                       (st.session_state['collected_annotations']['job_id'] == jobid)
                            r_col, f_col = f'annotator_{i+1}_relevance', f'annotator_{i+1}_feedback'
                            if r_col in st.session_state['collected_annotations'].columns:
                                prev = st.session_state['collected_annotations'][mask_prev]
                                if not prev.empty:
                                    val_r = prev.iloc[0].get(r_col)
                                    if pd.notna(val_r): def_r = int(val_r)
                                    if f_col in prev.columns: def_f = str(prev.iloc[0].get(f_col, ""))
                        r_val = st.radio("Rel:", [0,1,2,3], index=def_r, key=rel_k, horizontal=True, format_func=lambda x:f"{x}")
                        f_val = st.text_area("FB:", value=def_f, key=fb_k, height=50)
                        ann_item_inputs[f'annotator_{i+1}_name'] = annotator
                        ann_item_inputs[f'annotator_{i+1}_relevance'] = r_val
                        ann_item_inputs[f'annotator_{i+1}_feedback'] = f_val
                item_data.update(ann_item_inputs)
                form_input.append(item_data)
                st.markdown("---")
        submitted_ann = st.form_submit_button("Submit All Annotations")
    if submitted_ann:
        if form_input:
            new_df_ann = pd.DataFrame(form_input)
            st.session_state['collected_annotations'] = pd.concat([st.session_state.get('collected_annotations', pd.DataFrame()), new_df_ann]).drop_duplicates(subset=['cv_filename', 'job_id'], keep='last').reset_index(drop=True)
            st.success("Annotations submitted/updated.")
        else: st.warning("No annotation data entered.")
    if not st.session_state.get('collected_annotations', pd.DataFrame()).empty:
        st.subheader("Collected Annotations Preview")
        st.dataframe(st.session_state['collected_annotations'], height=300)
        csv_data_out = st.session_state['collected_annotations'].to_csv(index=False).encode('utf-8')
        st.download_button("Download Annotations (CSV)", csv_data_out, "job_annotations.csv", "text/csv", key="dl_ann_csv_btn")
    else: st.info("No annotations collected yet.")
    return

def evaluation_page():
    st.header("Model Evaluation (IR Metrics)")
    st.write("Evaluates recommendation based on collected annotations.")
    model_eval = load_bert_model()
    if model_eval is None: st.error("BERT model load failed for evaluation."); return
    data_eval = st.session_state.get('data')
    cvs_eval = st.session_state.get('uploaded_cvs_data', [])
    anns_df = st.session_state.get('collected_annotations', pd.DataFrame())
    if data_eval is None or 'processed_text' not in data_eval.columns: st.warning("Job data not ready."); return
    if not cvs_eval: st.warning("No CVs uploaded."); return
    valid_cvs_list = [cv for cv in cvs_eval if cv.get('processed_text','').strip()]
    if not valid_cvs_list: st.warning("No CVs with processed text for evaluation."); return
    if anns_df.empty: st.warning("No annotations collected."); return

    st.subheader("Evaluation Parameters")
    threshold = st.slider("Relevance Threshold (Avg score >= threshold is relevant)", 0.0, 3.0, 1.5, 0.1, key="eval_thresh")
    if st.button("Run Evaluation", key="run_eval_btn"):
        with st.spinner("Preparing data & running IR evaluation..."):
            queries_dict = {str(cv['filename']): cv['processed_text'] for cv in valid_cvs_list}
            corpus_dict = dict(zip(data_eval['Job.ID'].astype(str), data_eval['processed_text']))
            anns_df['job_id'] = anns_df['job_id'].astype(str) 
            relevant_docs_dict = {}
            rel_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS)) if f'annotator_{i+1}_relevance' in anns_df.columns]
            if not rel_cols: st.error("No annotator relevance columns in annotations."); return
            
            for (cv_f_name, jb_id), grp in anns_df.groupby(['cv_filename', 'job_id']):
                scores_list = [pd.to_numeric(grp[col], errors='coerce').dropna().tolist() for col in rel_cols]
                flat_scores = [item for sublist in scores_list for item in sublist] 
                if flat_scores:
                    if np.mean(flat_scores) >= threshold:
                        cv_f_name_str = str(cv_f_name)
                        if cv_f_name_str not in relevant_docs_dict: relevant_docs_dict[cv_f_name_str] = set()
                        relevant_docs_dict[cv_f_name_str].add(jb_id)
            
            if not relevant_docs_dict: st.warning(f"No relevant docs found with threshold {threshold}.")
            st.write(f"Queries: {len(queries_dict)}, Corpus: {len(corpus_dict)}, CVs with relevant items: {len(relevant_docs_dict)}")
            try:
                evaluator = InformationRetrievalEvaluator(queries_dict, corpus_dict, relevant_docs_dict, name="job_rec_custom_eval", show_progress_bar=True)
                results_dict = evaluator(model_eval, output_path=None)
                st.subheader("Evaluation Results")
                if results_dict and isinstance(results_dict, dict):
                    results_df_display = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Score'])
                    results_df_display.index.name = "Metric"
                    st.dataframe(results_df_display)
                    map_keys_found = [k for k in results_dict.keys() if 'map' in k.lower()]
                    if map_keys_found: st.metric(label=f"Primary Metric ({map_keys_found[0]})", value=f"{results_dict[map_keys_found[0]]:.4f}")
                else: st.write("Raw results:", results_dict)
            except Exception as e: st.error(f"IR Evaluation Error: {e}"); st.exception(e)
    return

# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio")

# Page routing - Ensure consistent indentation (4 spaces used here)
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
elif page == "Evaluation":
    evaluation_page()
