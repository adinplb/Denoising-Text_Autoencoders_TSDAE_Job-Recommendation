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
# from tqdm import tqdm # Not needed for Streamlit's UI progress
from sentence_transformers import SentenceTransformer, InputExample, losses 
from sentence_transformers.datasets import DenoisingAutoEncoderDataset 
from sentence_transformers.losses.TripletLoss import TripletDistanceMetric # For TripletLoss
from torch.utils.data import DataLoader 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score 

# --- NLTK Resource Downloads ---
@st.cache_resource
def download_nltk_resources():
    resources_to_download = {
        "stopwords": "corpora/stopwords",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab" 
    }
    all_successful = True
    for resource_name, resource_path_fragment in resources_to_download.items():
        try:
            if resource_name == "stopwords":
                stopwords.words('english') 
            elif resource_name == "punkt": 
                word_tokenize("test sentence for punkt") 
            elif resource_name == "punkt_tab": 
                 try:
                     nltk.data.find(resource_path_fragment) 
                 except LookupError:
                     st.info(f"NLTK resource '{resource_name}' not found, attempting download...")
                     nltk.download(resource_name, quiet=True)
        except LookupError: 
            st.info(f"NLTK resource '{resource_name}' not found, attempting download...")
            try:
                nltk.download(resource_name, quiet=True)
            except Exception as e_download:
                st.warning(f"Could not download NLTK resource '{resource_name}': {e_download}")
                all_successful = False
        except Exception as e_check: 
            st.warning(f"Error checking NLTK resource '{resource_name}': {e_check}")
            all_successful = False
            
    if all_successful:
        st.success("NLTK resources checked/downloaded.")
    else:
        st.error("Some NLTK resources might be missing. Please check the console for details.")

download_nltk_resources()


# --- Constants ---
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'
ONET_DATA_URL = 'https://raw.githubusercontent.com/adinplb/tsdae-embeddings/refs/heads/master/dataset/Occupation%20Data.csv'


FEATURES_TO_COMBINE = [ 
    'Status', 'Title', 'Position', 'Company', 
    'City', 'State.Name', 'Industry', 'Job.Description', 
    'Employment.Type', 'Education.Required'
]
JOB_DETAIL_FEATURES = [ 
    'Company', 'Status', 'City', 'Job.Description', 'Employment.Type', 
    'Position', 'Industry', 'Education.Required', 'State.Name'
] 

N_CLUSTERS = 20 
ANNOTATORS = ["Annotator 1", "Annotator 2", "Annotator 3", "Annotator 4", "Annotator 5"]


# --- Global Data Storage (using Streamlit Session State) ---
default_values = {
    'data': None,
    'onet_data': None, 
    'job_text_embeddings': None,
    'job_text_embedding_job_ids': None,
    'uploaded_cvs_data': [],
    'all_recommendations_for_annotation': {},
    'collected_annotations': pd.DataFrame(),
    'annotator_details': {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS},
    'current_annotator_slot_for_input': ANNOTATORS[0] if ANNOTATORS else None,
    'annotators_saved_status': set(),
    'model_trained_flags': {
        "tsdae_trained_this_session": False, 
        "sbert_onet_finetuned_this_session": False, 
        "cv_job_finetuned_this_session": False
    },
    'bert_model_instance': None,
    'sbert_classified_jobs_df': None 
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---
@st.cache_data(show_spinner='Loading job data...')
def load_and_combine_data_from_url(url, features_to_combine_list, detail_features_to_ensure, data_name="Job Data"):
    try:
        df_full = pd.read_csv(url) 
        st.success(f'Successfully loaded {data_name} from URL!')

        if 'Job.ID' not in df_full.columns and data_name == "Job Data": 
            st.error("Column 'Job.ID' not found in the job dataset.")
            return None
        if 'Job.ID' in df_full.columns: 
            df_full['Job.ID'] = df_full['Job.ID'].astype(str)

        if data_name == "Job Data":
            existing_features_to_combine = [col for col in features_to_combine_list if col in df_full.columns]
            cols_to_load_set = set(['Job.ID', 'Title']) 
            cols_to_load_set.update(existing_features_to_combine)
            cols_to_load_set.update(detail_features_to_ensure) 
            actual_cols_to_load = [col for col in list(cols_to_load_set) if col in df_full.columns]
            df = df_full[actual_cols_to_load].copy() 

            for feature in existing_features_to_combine: 
                if feature in df.columns: 
                    df[feature] = df[feature].fillna('').astype(str)
            
            df['combined_jobs'] = df[existing_features_to_combine].agg(lambda x: ' '.join(x.dropna()), axis=1)
            df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
            st.success("Column 'combined_jobs' created successfully for job data.")
            return df
        else: 
            return df_full

    except Exception as e:
        st.error(f'Error loading or combining {data_name}: {e}')
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

def denoise_text(text_to_denoise, method='a', del_ratio=0.6, word_freq_dict=None, freq_threshold=100):
    if not isinstance(text_to_denoise, str) or not text_to_denoise.strip():
        return "" 
    words = word_tokenize(text_to_denoise)
    n = len(words)
    if n == 0:
        return "" 
    result_words_list = [] 
    if method == 'a':
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0 and n > 0 : 
            idx_to_keep = np.random.choice(n) 
            keep_or_not[idx_to_keep] = True
        result_words_list = np.array(words)[keep_or_not].tolist() 
    elif method in ('b', 'c'): 
        if word_freq_dict is None:
            raise ValueError("word_freq_dict is required for method 'b' or 'c'.")
        high_freq_indices = [i for i, w_inner in enumerate(words) if word_freq_dict.get(w_inner.lower(), 0) > freq_threshold]
        num_to_remove = int(del_ratio * len(high_freq_indices))
        to_remove_indices = set()
        if high_freq_indices and num_to_remove > 0 and num_to_remove <= len(high_freq_indices):
             to_remove_indices = set(random.sample(high_freq_indices, num_to_remove))
        result_words_list = [w_inner for i, w_inner in enumerate(words) if i not in to_remove_indices]
        if not result_words_list and words: 
            result_words_list = [random.choice(words)]
        if method == 'c' and result_words_list: 
            random.shuffle(result_words_list)
    else:
        raise ValueError("Unknown denoising method. Use 'a', 'b', or 'c'.")
    return TreebankWordDetokenizer().detokenize(result_words_list)


def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'):
    processed_results_intermediate = [] 
    if text_column_to_process not in data_df.columns:
        st.warning(f"Column '{text_column_to_process}' not found for preprocessing.")
        if 'processed_text' not in data_df.columns: data_df['processed_text'] = pd.Series(dtype='object')
        if 'preprocessing_steps' not in data_df.columns: data_df['preprocessing_steps'] = pd.Series(dtype='object')
        return data_df 

    with st.spinner(f"Preprocessing column '{text_column_to_process}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)
        
        processed_texts_col = [] 

        for i, text_content in enumerate(data_df[text_column_to_process].fillna('').astype(str)):
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
            final_processed_text = " ".join(stemmed_temp)
            intermediate['stemmed'] = final_processed_text
            
            processed_results_intermediate.append(intermediate)
            processed_texts_col.append(final_processed_text)

            if total_rows > 0:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
        
        data_df['processed_text'] = processed_texts_col 
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success(f"Preprocessing of column '{text_column_to_process}' complete! Column 'processed_text' has been created/updated.")
        progress_bar.empty()
        status_text.empty()
    return data_df

@st.cache_resource 
def load_bert_model_once(model_name="all-MiniLM-L6-v2"): 
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading BERT model '{model_name}': {e}")
        return None

if st.session_state.bert_model_instance is None:
    st.session_state.bert_model_instance = load_bert_model_once()


@st.cache_data
def generate_embeddings_with_progress(_model_ref_for_cache_key, texts_list_to_embed, model_state_indicator="base"):
    model_to_use = st.session_state.bert_model_instance 
    if model_to_use is None:
        st.error("BERT model is not available for embedding generation.")
        return np.array([]) 
    if not texts_list_to_embed: 
        st.warning("Input text list for embedding is empty.")
        return np.array([])
    try:
        with st.spinner(f"Generating embeddings for {len(texts_list_to_embed)} texts using current model state ({model_state_indicator})..."):
            embedding_progress_bar = st.progress(0)
            embedding_status_text = st.empty()
            embeddings_result_list = [] 
            total_texts_to_embed = len(texts_list_to_embed)
            batch_size = 32 
            for i in range(0, total_texts_to_embed, batch_size):
                batch_texts_segment = texts_list_to_embed[i:i + batch_size] 
                batch_embeddings_np_array = model_to_use.encode(batch_texts_segment, convert_to_tensor=False, show_progress_bar=False) 
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
    st.write("This page provides an overview of the job dataset and allows you to explore its features.") 

    if st.session_state.get('data') is None:
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, JOB_DETAIL_FEATURES)
    
    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Data Preview (including `combined_jobs`)') 
        cols_to_preview = ['Job.ID']
        if 'Title' in data_df.columns: cols_to_preview.append('Title')
        if 'combined_jobs' in data_df.columns: cols_to_preview.append('combined_jobs')
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Data Summary') 
        st.write(f'Number of rows: {len(data_df)}') 
        st.write(f'Number of columns: {len(data_df.columns)}') 
        
        if 'combined_jobs' in data_df.columns:
            st.subheader('Sample Content of `combined_jobs` Column') 
            for i in range(min(3, len(data_df))):
                title_display = data_df.iloc[i]['Title'] if 'Title' in data_df.columns else "N/A"
                with st.expander(f"Job.ID: {data_df.iloc[i]['Job.ID']} - {title_display}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Column 'combined_jobs' has not been created or is not in the data.") 

        st.subheader('Search Word in Feature') 
        search_word = st.text_input("Enter word to search:", key="home_search_word_new_v2") 
        all_available_cols_for_search = ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE + JOB_DETAIL_FEATURES
        searchable_cols = sorted(list(set(col for col in all_available_cols_for_search if col in data_df.columns)))
        search_column = st.selectbox("Select feature to search in:", [''] + searchable_cols, key="home_search_column_new_v2") 

        if search_word and search_column:
            if search_column in data_df.columns:
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                display_search_cols = ['Job.ID']
                if 'Title' in data_df.columns: display_search_cols.append('Title')
                if search_column not in display_search_cols: display_search_cols.append(search_column)

                if not search_results.empty:
                    st.write(f"Found {len(search_results)} entries for '{search_word}' in '{search_column}':") 
                    st.dataframe(search_results[display_search_cols].head(), use_container_width=True) 
                else: st.info(f"No entries found for '{search_word}' in '{search_column}'.") 
        
        st.subheader('Feature Information') 
        st.write('**Available Features (after processing):**', data_df.columns.tolist()) 
    else: st.error("Data could not be loaded.") 
    return

def preprocessing_page():
    st.header("Job Data Preprocessing") 
    st.write("This page performs preprocessing on the 'combined_jobs' column of the job dataset.") 
    if st.session_state.get('data') is None or 'combined_jobs' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data or 'combined_jobs' column not available. Please return to the 'Home' page to load data first.") 
        return
    data_df_to_preprocess = st.session_state['data']
    st.info("The 'combined_jobs' column will be processed to create the 'processed_text' column.") 
    if 'combined_jobs' in data_df_to_preprocess.columns:
        with st.expander("View 'combined_jobs' sample (before processing)"): 
            st.dataframe(data_df_to_preprocess[['Job.ID', 'combined_jobs']].head())
    if st.button("Run Preprocessing on 'combined_jobs' Column", key="run_job_col_prep_btn_v2"): 
        with st.spinner("Preprocessing 'combined_jobs'..."): 
            data_copy = data_df_to_preprocess.copy()
            st.session_state['data'] = preprocess_text_with_intermediate(data_copy, text_column_to_process='combined_jobs')
        st.success("Preprocessing of 'combined_jobs' column complete! 'processed_text' column has been created/updated.") 
    if 'processed_text' in st.session_state.get('data', pd.DataFrame()).columns:
        st.info("Preprocessing has been performed on 'combined_jobs'.") 
        display_data_processed = st.session_state['data'] 
        if 'preprocessing_steps' in display_data_processed.columns:
            st.subheader("Preprocessing Results (Intermediate Steps from last run)") 
            valid_intermediate_steps = [s for s in display_data_processed['preprocessing_steps'] if isinstance(s, dict)] 
            if valid_intermediate_steps: st.dataframe(pd.DataFrame(valid_intermediate_steps).head(), use_container_width=True)
            else: st.warning("Intermediate preprocessing steps data is not in the expected format or is empty.") 
        st.subheader("Final Preprocessed Text ('processed_text') (Preview)") 
        st.dataframe(display_data_processed[['Job.ID', 'combined_jobs', 'processed_text']].head(), use_container_width=True)
        search_word_in_processed = st.text_input("Search word in 'processed_text':", key="prep_job_proc_search_v2") 
        if search_word_in_processed:
            search_results_in_processed = display_data_processed[display_data_processed['processed_text'].astype(str).str.contains(search_word_in_processed, na=False, case=False)] 
            if not search_results_in_processed.empty:
                display_cols_search_proc = ['Job.ID', 'Title' if 'Title' in display_data_processed.columns else 'Job.ID', 'processed_text']
                st.dataframe(search_results_in_processed[display_cols_search_proc].head(), use_container_width=True)
            else: st.info(f"No results for '{search_word_in_processed}' in 'processed_text'.") 
    else: st.info("Column 'combined_jobs' is available, but preprocessing has not been run yet. Click the button above.") 
    return

def tsdae_page(): 
    st.header("TSDAE Pre-training on Job Descriptions") # Renamed
    st.write("""
    This page pre-trains (domain-adapts) the main BERT model using the Transformer Denoising Autoencoder (TSDAE) 
    objective on the 'processed_text' of job descriptions. 
    This modifies the main BERT model instance for subsequent use in this session.
    """)

    bert_model = st.session_state.bert_model_instance 
    if bert_model is None:
        st.error("Base BERT model not loaded. Cannot proceed."); return
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.data.columns:
        st.warning("Job data with 'processed_text' not available. Please load and preprocess data first."); return

    data_for_tsdae = st.session_state.data
    job_texts_for_tsdae = data_for_tsdae['processed_text'].fillna('').astype(str).tolist()
    job_texts_for_tsdae = [text for text in job_texts_for_tsdae if text.strip()] 

    if not job_texts_for_tsdae:
        st.warning("No valid 'processed_text' entries found in job data for TSDAE pre-training."); return

    st.info(f"Found {len(job_texts_for_tsdae)} job descriptions for TSDAE pre-training.")

    tsdae_batch_size = st.selectbox("Batch size for TSDAE pre-training:", options=[4, 8, 16, 32], index=2, key="tsdae_batch_size_pretrain")
    tsdae_epochs = st.number_input("Number of epochs for TSDAE pre-training:", min_value=1, max_value=5, value=1, key="tsdae_epochs_pretrain")
    
    if st.button("Start TSDAE Pre-training on Main Model", key="start_tsdae_pretraining_main"):
        with st.spinner(f"Pre-training model with TSDAE for {tsdae_epochs} epoch(s)... This may take some time."):
            try:
                # DenoisingAutoEncoderDataset handles noise creation internally based on the model's tokenizer
                train_dataset_tsdae = DenoisingAutoEncoderDataset(job_texts_for_tsdae)
                train_dataloader_tsdae = DataLoader(train_dataset_tsdae, batch_size=tsdae_batch_size, shuffle=True)
                
                train_loss_tsdae = losses.DenoisingAutoEncoderLoss(
                    bert_model, 
                    decoder_name_or_path=bert_model.tokenizer.name_or_path, 
                    tie_encoder_decoder=True 
                )
                warmup_steps_tsdae = int(len(train_dataloader_tsdae) * tsdae_epochs * 0.1)

                bert_model.fit(
                    train_objectives=[(train_dataloader_tsdae, train_loss_tsdae)],
                    epochs=tsdae_epochs, warmup_steps=warmup_steps_tsdae,
                    output_path=None, show_progress_bar=True,
                )
                st.session_state.model_trained_flags["tsdae_trained_this_session"] = True
                # Invalidate job_text_embeddings as the model has changed
                if 'job_text_embeddings' in st.session_state: del st.session_state['job_text_embeddings']
                if 'job_text_embedding_job_ids' in st.session_state: del st.session_state['job_text_embedding_job_ids']
                st.success("TSDAE pre-training complete! The main BERT model instance has been updated.")
                st.info("You may now proceed to 'SBERT O*NET Fine-tuning' or other fine-tuning/embedding generation steps.")
            except Exception as e:
                st.error(f"Error during TSDAE pre-training: {e}")
                st.exception(e)

    if st.session_state.model_trained_flags.get("tsdae_trained_this_session", False):
        st.success("The main BERT model has been pre-trained with TSDAE in this session.")
    return

def sbert_onet_finetuning_page():
    st.header("SBERT Fine-tuning (Job Text to O*NET Alignment)")
    st.write("""
    This page fine-tunes the current BERT model (potentially after TSDAE pre-training) 
    to better align job descriptions from your dataset with standard O*NET occupation descriptions.
    It uses the 'category' (O*NET Title) from your uploaded `sbert_classified_jobs.csv` to create training pairs.
    """)

    bert_model_to_ft = st.session_state.bert_model_instance
    if bert_model_to_ft is None:
        st.error("BERT model not loaded. Cannot proceed."); return

    main_job_data = st.session_state.get('data')
    if main_job_data is None or 'processed_text' not in main_job_data.columns:
        st.warning("Main job data with 'processed_text' not available. Please load and preprocess data first."); return
    
    # Load O*NET data
    if st.session_state.get('onet_data') is None:
        with st.spinner("Loading O*NET standard occupation data..."):
            st.session_state.onet_data = load_and_combine_data_from_url(ONET_DATA_URL, [], ['O*NET-SOC Code', 'Title', 'Description'], data_name="O*NET Data")
    
    onet_df = st.session_state.get('onet_data')
    if onet_df is None:
        st.error("Failed to load O*NET data. Cannot proceed."); return
    
    expected_onet_cols = ['O*NET-SOC Code', 'Title', 'Description']
    if not all(col in onet_df.columns for col in expected_onet_cols):
        st.error(f"O*NET data is missing required columns: {', '.join(expected_onet_cols)}"); return

    # Preprocess O*NET data if not already done
    if 'processed_onet_text' not in onet_df.columns:
        with st.spinner("Preprocessing O*NET data for fine-tuning..."):
            onet_df['onet_combined_text'] = onet_df['Title'].fillna('').astype(str) + " " + onet_df['Description'].fillna('').astype(str)
            onet_df['processed_onet_text'] = onet_df['onet_combined_text'].apply(preprocess_text)
            st.session_state.onet_data = onet_df 
    
    valid_onet_entries = onet_df[onet_df['processed_onet_text'].str.strip() != ''].copy()
    if valid_onet_entries.empty:
        st.error("No valid processed text entries found in O*NET data after preprocessing."); return
    
    st.info(f"Using {len(valid_onet_entries)} O*NET entries for matching.")

    st.subheader("Upload Your Pre-classified Job Data (CSV)")
    st.markdown("This CSV should contain `Job.ID` and `category` columns, where `category` is the O\*NET Title your jobs were matched to.")
    uploaded_sbert_classified_file = st.file_uploader("Upload `sbert_classified_jobs.csv`", type="csv", key="sbert_classified_jobs_uploader")

    if uploaded_sbert_classified_file is not None:
        try:
            sbert_classified_df_uploaded = pd.read_csv(uploaded_sbert_classified_file)
            sbert_classified_df_uploaded['Job.ID'] = sbert_classified_df_uploaded['Job.ID'].astype(str) 
            if not all(col in sbert_classified_df_uploaded.columns for col in ['Job.ID', 'category']):
                st.error("Uploaded CSV must contain 'Job.ID' and 'category' columns.")
                st.session_state.sbert_classified_jobs_df = None
            else:
                st.session_state.sbert_classified_jobs_df = sbert_classified_df_uploaded
                st.success(f"Uploaded '{uploaded_sbert_classified_file.name}' successfully.")
                st.dataframe(sbert_classified_df_uploaded.head())
        except Exception as e:
            st.error(f"Error reading or processing uploaded SBERT classified CSV: {e}")
            st.session_state.sbert_classified_jobs_df = None
    
    if st.session_state.sbert_classified_jobs_df is not None:
        sbert_classified_df = st.session_state.sbert_classified_jobs_df
        
        sbert_onet_epochs_ft = st.number_input("Epochs for SBERT O*NET fine-tuning:", min_value=1, max_value=10, value=1, key="sbert_onet_epochs_ft_v4")
        sbert_onet_batch_size_ft = st.selectbox("Batch size for SBERT O*NET fine-tuning:", options=[4, 8, 16, 32], index=2, key="sbert_onet_batch_size_ft_v4")

        if st.button("Start SBERT O*NET Fine-tuning", key="start_sbert_onet_ft_btn_v4"):
            with st.spinner("Preparing training data and fine-tuning with SBERT O*NET... This may take a while."):
                try:
                    train_examples_sbert_onet = []
                    onet_title_to_processed_text = pd.Series(valid_onet_entries.processed_onet_text.values, index=valid_onet_entries.Title).to_dict()
                    
                    # Ensure main_job_data has 'processed_text'
                    if 'processed_text' not in main_job_data.columns:
                        st.error("'processed_text' column missing from main job data. Please run preprocessing.")
                        return

                    merged_for_ft = pd.merge(sbert_classified_df[['Job.ID', 'category']], 
                                             main_job_data[['Job.ID', 'processed_text']], 
                                             on='Job.ID', 
                                             how='inner')
                    
                    if merged_for_ft.empty:
                        st.error("No matching Job.IDs found between uploaded classified data and main job data."); return

                    for _, row in merged_for_ft.iterrows():
                        job_proc_text = row['processed_text']
                        onet_category_title = row['category'] 
                        matched_onet_proc_text = onet_title_to_processed_text.get(onet_category_title)
                        
                        if job_proc_text and matched_onet_proc_text:
                            train_examples_sbert_onet.append(InputExample(texts=[job_proc_text, matched_onet_proc_text]))
                        else:
                            st.warning(f"Skipping Job.ID {row['Job.ID']}: Missing text for job or O*NET category '{onet_category_title}'.")
                    
                    if not train_examples_sbert_onet:
                        st.error("No training examples created. Check CSV 'category' values and O*NET Titles."); return

                    st.write(f"Created {len(train_examples_sbert_onet)} (job_text, matched_onet_text) training pairs.")
                    
                    train_dataloader_sbert_onet = DataLoader(train_examples_sbert_onet, shuffle=True, batch_size=sbert_onet_batch_size_ft)
                    train_loss_sbert_onet = losses.MultipleNegativesRankingLoss(model=bert_model_to_ft)
                    warmup_steps_sbert_onet = int(len(train_dataloader_sbert_onet) * sbert_onet_epochs_ft * 0.1)

                    st.info(f"Starting SBERT O*NET fine-tuning for {sbert_onet_epochs_ft} epoch(s)...")
                    bert_model_to_ft.fit(
                        train_objectives=[(train_dataloader_sbert_onet, train_loss_sbert_onet)],
                        epochs=sbert_onet_epochs_ft, warmup_steps=warmup_steps_sbert_onet,
                        output_path=None, show_progress_bar=True
                    )
                    st.session_state.model_trained_flags["sbert_onet_finetuned_this_session"] = True
                    if 'job_text_embeddings' in st.session_state: del st.session_state['job_text_embeddings'] 
                    if 'job_text_embedding_job_ids' in st.session_state: del st.session_state['job_text_embedding_job_ids']
                    st.success("SBERT O*NET fine-tuning complete! The main BERT model has been updated.")
                    st.info("You may now want to re-generate 'Standard BERT Embeddings' on the 'BERT Model & Embeddings' page.")

                except Exception as e_sbert_ft:
                    st.error(f"An error occurred during SBERT O*NET fine-tuning: {e_sbert_ft}")
                    st.exception(e_sbert_ft)
        
        if st.session_state.model_trained_flags.get("sbert_onet_finetuned_this_session", False):
            st.success("The main BERT model has been fine-tuned with SBERT O*NET data in this session.")
    elif uploaded_annotations_file is None: # Corrected variable name
        st.info("Please upload your 'sbert_classified_jobs.csv' file to enable SBERT O*NET fine-tuning.")
    return

def fine_tuning_page(): 
    st.header("Model Fine-tuning (CV-Job Matching)")
    st.write("This page allows for further fine-tuning the current BERT model using uploaded CVs and relevant job descriptions.")
    
    bert_model_to_ft = st.session_state.bert_model_instance 
    if bert_model_to_ft is None:
        st.error("BERT model not loaded. Cannot proceed with fine-tuning."); return

    if not st.session_state.get('uploaded_cvs_data'):
        st.warning("Please upload and process CVs first on the 'Upload CV' page."); return
    
    data_for_ft = st.session_state.get('data')
    if data_for_ft is None or 'processed_text' not in data_for_ft.columns:
        st.warning("Job data with 'processed_text' is not available. Load/preprocess data first."); return
    
    job_embeddings_for_pos_selection = st.session_state.get('job_text_embeddings')
    job_ids_for_pos_selection = st.session_state.get('job_text_embedding_job_ids')

    if job_embeddings_for_pos_selection is None or job_ids_for_pos_selection is None:
        st.warning("Job text embeddings are not available. Please generate them on the 'BERT Model & Embeddings' page first. These are used to find initial positive examples for fine-tuning."); return

    st.info("Fine-tuning will use processed CV texts as queries and identify top similar processed job texts as positive examples using the current model's similarity scores for jobs.")
    if st.session_state.model_trained_flags.get("tsdae_trained_this_session", False):
        st.info("Note: The current BERT model may have been previously fine-tuned with TSDAE.")
    if st.session_state.model_trained_flags.get("sbert_onet_finetuned_this_session", False):
        st.info("Note: The current BERT model may have been previously fine-tuned with SBERT O*NET data.")


    num_epochs_ft_cv = st.number_input("Number of fine-tuning epochs (CV-Job):", min_value=1, max_value=10, value=1, key="ft_cv_epochs_main_v2")
    batch_size_ft_cv = st.selectbox("Batch size for fine-tuning (CV-Job):", options=[4, 8, 16, 32], index=2, key="ft_cv_batch_size_main_v2")
    top_n_positive_ft_cv = st.number_input("Top N similar jobs per CV as positive examples:", min_value=1, max_value=10, value=2, key="ft_cv_top_n_main_v2")

    loss_function_choice = st.selectbox(
        "Choose Loss Function for CV-Job Fine-tuning:",
        ("MultipleNegativesRankingLoss", "TripletLoss"),
        key="cv_job_ft_loss_choice_v2"
    )
    triplet_margin = 0.5
    if loss_function_choice == "TripletLoss":
        triplet_margin = st.slider("Triplet Margin:", 0.1, 1.0, 0.5, 0.1, key="ft_triplet_margin_v2")


    if st.button("Start CV-Job Fine-tuning", key="start_cv_job_ft_btn_main_v2"):
        with st.spinner("Preparing fine-tuning data and training... This may take a while."):
            train_examples_cv = []
            
            job_texts_map_for_pos_selection = pd.Series(data_for_ft.processed_text.values, index=data_for_ft['Job.ID']).to_dict()
            all_job_ids_list = list(job_texts_map_for_pos_selection.keys())


            for cv_data_item in st.session_state.uploaded_cvs_data:
                cv_proc_text = cv_data_item.get('processed_text')
                cv_emb_for_pos = cv_data_item.get('embedding') 

                if not cv_proc_text or cv_emb_for_pos is None:
                    st.warning(f"CV {cv_data_item.get('filename')} lacks processed text or embedding. Skipping.")
                    continue

                if job_embeddings_for_pos_selection is not None and len(job_embeddings_for_pos_selection) > 0:
                    similarities_for_pos = cosine_similarity(cv_emb_for_pos.reshape(1, -1), job_embeddings_for_pos_selection)[0]
                    top_n_indices_for_pos = np.argsort(similarities_for_pos)[::-1][:top_n_positive_ft_cv]
                    
                    positive_job_ids_for_cv = [job_ids_for_pos_selection[idx] for idx in top_n_indices_for_pos]
                    
                    for pos_job_id in positive_job_ids_for_cv:
                        positive_job_proc_text = job_texts_map_for_pos_selection.get(pos_job_id)
                        if not positive_job_proc_text: continue

                        if loss_function_choice == "MultipleNegativesRankingLoss":
                            train_examples_cv.append(InputExample(texts=[cv_proc_text, positive_job_proc_text]))
                        
                        elif loss_function_choice == "TripletLoss":
                            possible_negatives = [jid for jid in all_job_ids_list if jid not in positive_job_ids_for_cv]
                            if possible_negatives:
                                negative_job_id = random.choice(possible_negatives)
                                negative_job_proc_text = job_texts_map_for_pos_selection.get(negative_job_id)
                                if negative_job_proc_text:
                                    train_examples_cv.append(InputExample(texts=[cv_proc_text, positive_job_proc_text, negative_job_proc_text]))
                            else:
                                st.warning(f"Could not find a negative sample for CV {cv_data_item.get('filename')} and positive job {pos_job_id}. Skipping this triplet.")
            
            if not train_examples_cv:
                st.error("No training examples created for CV-Job fine-tuning. Check CVs, job data, embeddings, and sampling strategy."); return

            st.write(f"Created {len(train_examples_cv)} training examples using {loss_function_choice}.")
            
            train_dataloader_ft_cv = DataLoader(train_examples_cv, shuffle=True, batch_size=batch_size_ft_cv)
            
            if loss_function_choice == "MultipleNegativesRankingLoss":
                train_loss_ft_cv = losses.MultipleNegativesRankingLoss(model=bert_model_to_ft)
            elif loss_function_choice == "TripletLoss":
                train_loss_ft_cv = losses.TripletLoss(model=bert_model_to_ft, 
                                                      distance_metric=TripletDistanceMetric.COSINE, 
                                                      triplet_margin=triplet_margin)
            else: 
                st.error("Invalid loss function selected.")
                return

            warmup_steps_ft_cv = int(len(train_dataloader_ft_cv) * num_epochs_ft_cv * 0.1)
            
            st.info(f"Starting CV-Job fine-tuning for {num_epochs_ft_cv} epoch(s) with {loss_function_choice}...")
            try:
                bert_model_to_ft.fit(train_objectives=[(train_dataloader_ft_cv, train_loss_ft_cv)],
                                     epochs=num_epochs_ft_cv, warmup_steps=warmup_steps_ft_cv,
                                     output_path=None, show_progress_bar=True)
                st.session_state.model_trained_flags["cv_job_finetuned_this_session"] = True
                st.success("CV-Job fine-tuning complete! The main BERT model has been further updated.")
                st.info("You may now want to re-generate 'Standard BERT Embeddings' on the 'BERT Model & Embeddings' page.")
                if 'job_text_embeddings' in st.session_state: del st.session_state['job_text_embeddings']
                if 'job_text_embedding_job_ids' in st.session_state: del st.session_state['job_text_embedding_job_ids']
            except Exception as e_ft_cv:
                st.error(f"An error occurred during CV-Job fine-tuning: {e_ft_cv}")
                st.exception(e_ft_cv)
    
    if st.session_state.model_trained_flags.get("cv_job_finetuned_this_session", False):
        st.success("Model has been fine-tuned with CV-Job data in this session.")
    elif st.session_state.model_trained_flags.get("sbert_onet_finetuned_this_session", False):
        st.info("Model was previously fine-tuned with SBERT O*NET data in this session. You can fine-tune it further with CV-Job data.")
    elif st.session_state.model_trained_flags.get("tsdae_trained_this_session", False):
        st.info("Model was previously fine-tuned with TSDAE in this session. You can fine-tune it further with CV-Job data.")
    return

def bert_model_page():
    st.header("BERT Model & Embeddings (Job Descriptions)") 
    st.write("Generates/Re-generates BERT embeddings for 'processed_text' of jobs using the current state of the main BERT model (which may have been fine-tuned).") 
    
    bert_model = st.session_state.bert_model_instance 
    if bert_model is None: 
        st.error("BERT model not available. Please check loading sequence."); return
    
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Job data must be loaded & preprocessed. Visit 'Preprocessing' page.") 
        return
    
    data_bert = st.session_state['data']
    
    if st.session_state.model_trained_flags.get("cv_job_finetuned_this_session", False):
        st.success("Current BERT model includes CV-Job fine-tuning.")
    elif st.session_state.model_trained_flags.get("sbert_onet_finetuned_this_session", False):
        st.success("Current BERT model includes SBERT O*NET fine-tuning.")
    elif st.session_state.model_trained_flags.get("tsdae_trained_this_session", False):
        st.success("Current BERT model includes TSDAE fine-tuning.")
    else:
        st.info("Currently using the base pre-trained BERT model (no in-session fine-tuning applied yet).")


    if st.button("Generate/Regenerate Job Embeddings with Current Model", key="gen_std_emb_btn_main_v3"): 
        if 'processed_text' not in data_bert.columns or data_bert['processed_text'].isnull().all():
            st.error("Column 'processed_text' is empty or missing. Please run preprocessing first.")
            return

        proc_series = data_bert['processed_text'].fillna('').astype(str)
        mask = proc_series.str.strip() != ''
        valid_texts = proc_series[mask].tolist()
        valid_job_ids = data_bert.loc[mask, 'Job.ID'].tolist()
        if not valid_texts: st.warning("No valid processed job texts for embedding.")
        else:
            model_state_indicator_str = (
                f"tsdae:{st.session_state.model_trained_flags['tsdae_trained_this_session']}_"
                f"sbert_onet:{st.session_state.model_trained_flags['sbert_onet_finetuned_this_session']}_"
                f"cvjob:{st.session_state.model_trained_flags['cv_job_finetuned_this_session']}"
            )
            
            st.session_state['job_text_embeddings'] = generate_embeddings_with_progress(
                bert_model, 
                valid_texts,
                model_state_indicator=model_state_indicator_str 
            )
            st.session_state['job_text_embedding_job_ids'] = valid_job_ids
            if st.session_state.get('job_text_embeddings', np.array([])).size > 0: 
                st.success(f"Job embeddings generated/re-generated for {len(valid_job_ids)} jobs using the current model state!")
            else: st.warning("Job embedding output empty.")

    job_emb = st.session_state.get('job_text_embeddings')
    job_ids = st.session_state.get('job_text_embedding_job_ids')
    if job_emb is not None and job_emb.size > 0 and job_ids:
        st.subheader(f"Current Job Embeddings ({len(job_ids)} jobs)")
        st.write(f"Shape: {job_emb.shape}")
        if len(job_emb) >= 2: 
            try:
                pca_2d = PCA(n_components=2).fit_transform(job_emb)
                plot_pca_df = pd.DataFrame(pca_2d, columns=['PC1','PC2'])
                plot_pca_df['Job.ID'] = job_ids
                
                hover_data_cols = ['Job.ID']
                main_data_for_hover = st.session_state['data']
                if 'Title' in main_data_for_hover.columns: hover_data_cols.append('Title')
                description_col_for_hover = 'combined_jobs' if 'combined_jobs' in main_data_for_hover.columns else 'text' 
                if description_col_for_hover in main_data_for_hover.columns : hover_data_cols.append(description_col_for_hover)
                
                hover_df = main_data_for_hover[main_data_for_hover['Job.ID'].isin(job_ids)][hover_data_cols]
                plot_pca_df = pd.merge(plot_pca_df, hover_df, on='Job.ID', how='left')
                
                hover_data_config = {'Job.ID':True, 'PC1':False, 'PC2':False}
                if description_col_for_hover in plot_pca_df.columns:
                    hover_data_config[description_col_for_hover] = True

                if not plot_pca_df.empty and 'Title' in plot_pca_df.columns:
                    fig_pca = px.scatter(plot_pca_df, 'PC1','PC2', 
                                         hover_name='Title', 
                                         hover_data=hover_data_config,
                                         title='2D PCA of Current Job Embeddings')
                    st.plotly_chart(fig_pca, use_container_width=True)
            except Exception as e: st.error(f"PCA Error: {e}")
        else: st.warning("Need >= 2 data points for PCA.")
    else: st.info("Job embeddings not generated yet with the current model state.")
    return

def clustering_page(): # Placeholder
    st.header("Clustering Job Embeddings")
    st.info("Implementation from previous version.")
    return

def upload_cv_page(): # Placeholder
    st.header("Upload & Process CV(s)")
    st.info("Implementation from previous version.")
    return

def job_recommendation_page(): # Placeholder
    st.header("Job Recommendation")
    st.info("Implementation from previous version.")
    return
    
def annotation_page(): # Placeholder
    st.header("Annotation of Job Recommendations")
    st.info("Implementation from previous version, with CSV upload.")
    return

def _calculate_average_precision(ranked_relevance_binary, k_val): # Placeholder
    if not ranked_relevance_binary: return 0.0
    ranked_relevance_binary = ranked_relevance_binary[:k_val] 
    relevant_hits, sum_precisions = 0, 0.0
    for i, is_relevant in enumerate(ranked_relevance_binary):
        if is_relevant:
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1)
    return sum_precisions / relevant_hits if relevant_hits > 0 else 0.0
    
def evaluation_page(): # Placeholder
    st.header("Model Evaluation")
    st.info("Implementation from previous version, with per-CV table and averages.")
    return


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", 
                "TSDAE Fine-tuning", 
                "SBERT O*NET Fine-tuning", 
                "BERT Model & Embeddings", 
                "CV-Job Fine-tuning", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio_v5")

if page == "Home":
    home_page()
elif page == "Preprocessing":
    preprocessing_page()
elif page == "TSDAE Fine-tuning": 
    tsdae_page()
elif page == "SBERT O*NET Fine-tuning": 
    sbert_onet_finetuning_page()
elif page == "BERT Model & Embeddings": 
    bert_model_page()
elif page == "CV-Job Fine-tuning": 
    fine_tuning_page()
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
