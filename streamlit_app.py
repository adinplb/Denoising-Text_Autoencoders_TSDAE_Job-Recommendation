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
from tqdm import tqdm 
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score 

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
# UPDATED DATA_URL
DATA_URL = 'https://raw.githubusercontent.com/adinplb/largedataset-JRec/refs/heads/main/Filtered_Jobs_4000.csv'

# Features to be combined to create the 'combined_jobs' column
FEATURES_TO_COMBINE = [
    'Status', 'Title', 'Position', 'Company', 
    'City', 'State.Name', 'Industry', 'Job.Description', 
    'Employment.Type', 'Education.Required'
]
# RELEVANT_FEATURES will now include Job.ID and the features to combine, plus the new combined_jobs column later
# For initial loading, we'll select Job.ID and FEATURES_TO_COMBINE.
# The 'combined_jobs' will be created dynamically.
# 'text' from the original RELEVANT_FEATURES might no longer be needed if 'Job.Description' is part of FEATURES_TO_COMBINE
# and the goal is to use 'combined_jobs' as the primary text source.

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
if 'annotator_details' not in st.session_state:
    st.session_state['annotator_details'] = {slot: {'actual_name': '', 'profile_background': ''} for slot in ANNOTATORS}
if 'current_annotator_slot_for_input' not in st.session_state: 
    st.session_state['current_annotator_slot_for_input'] = ANNOTATORS[0] if ANNOTATORS else None
if 'annotators_saved_status' not in st.session_state: 
    st.session_state['annotators_saved_status'] = set()


# --- Helper Functions ---
@st.cache_data(show_spinner='Memuat data pekerjaan...')
def load_and_combine_data_from_url(url, features_to_load, features_to_combine):
    """Loads job data, selects specified features, and creates a combined text column."""
    try:
        # Membaca semua kolom karena beberapa mungkin tidak ada di 'features_to_load' tapi dibutuhkan untuk penggabungan
        df_full = pd.read_csv(url) 
        st.success('Berhasil memuat data dari URL!')

        # Pastikan Job.ID adalah string
        if 'Job.ID' in df_full.columns:
            df_full['Job.ID'] = df_full['Job.ID'].astype(str)
        else:
            st.error("Kolom 'Job.ID' tidak ditemukan dalam dataset.")
            return None

        # Pilih fitur yang relevan untuk disimpan, pastikan semua ada
        cols_to_keep = ['Job.ID'] + [col for col in features_to_combine if col in df_full.columns]
        df = df_full[cols_to_keep].copy()

        # Buat kolom 'combined_jobs'
        # Isi NaN dengan string kosong sebelum menggabungkan
        for feature in features_to_combine:
            if feature in df.columns:
                df[feature] = df[feature].fillna('').astype(str)
            else:
                # Jika fitur untuk digabungkan tidak ada, buat kolom kosong agar penggabungan tidak error
                df[feature] = '' 
                st.warning(f"Fitur '{feature}' tidak ditemukan di dataset, akan diabaikan dalam penggabungan.")
        
        # Gabungkan fitur-fitur menjadi satu kolom teks, dipisahkan spasi
        df['combined_jobs'] = df[features_to_combine].agg(' '.join, axis=1)
        
        # Hapus spasi berlebih yang mungkin muncul dari penggabungan string kosong
        df['combined_jobs'] = df['combined_jobs'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        st.success("Kolom 'combined_jobs' berhasil dibuat.")
        return df
    except Exception as e:
        st.error(f'Error memuat atau menggabungkan data: {e}')
        return None

# Fungsi lain tetap sama (extract_text_from_pdf, dll.)
# ... (Fungsi-fungsi helper lainnya seperti extract_text_from_pdf, extract_text_from_docx, preprocess_text, dll. tetap sama)
# PASTIKAN FUNGSI-FUNGSI INI ADA DI KODE LENGKAP ANDA
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

def preprocess_text_with_intermediate(data_df, text_column_to_process='combined_jobs'): # Default ke combined_jobs
    processed_results_intermediate = [] 
    if text_column_to_process not in data_df.columns:
        st.warning(f"Kolom '{text_column_to_process}' tidak ditemukan untuk preprocessing.")
        return data_df 

    with st.spinner(f"Preprocessing kolom '{text_column_to_process}'..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(data_df)
        
        # Buat kolom 'processed_text' berdasarkan text_column_to_process
        # dan simpan langkah-langkah intermediate
        data_df['processed_text'] = "" # Inisialisasi kolom baru

        for i, text_content in enumerate(data_df[text_column_to_process].fillna('').astype(str)):
            intermediate = {'original': text_content}
            # ... (langkah-langkah preprocessing seperti sebelumnya, diterapkan pada text_content) ...
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
            data_df.loc[data_df.index[i], 'processed_text'] = final_processed_text # Simpan hasil akhir

            if total_rows > 0:
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"Processed {i + 1}/{total_rows} entries.")
        
        data_df['preprocessing_steps'] = processed_results_intermediate
        st.success(f"Preprocessing kolom '{text_column_to_process}' selesai!")
        progress_bar.empty()
        status_text.empty()
    return data_df

# --- Page Functions ---
def home_page():
    st.header("Beranda: Analisis Data Eksploratif")
    st.write("Halaman ini menyediakan ringkasan dataset pekerjaan dan memungkinkan Anda untuk menjelajahi fiturnya.")

    if st.session_state.get('data') is None:
        # Memanggil fungsi yang memuat dan menggabungkan data
        st.session_state['data'] = load_and_combine_data_from_url(DATA_URL, FEATURES_TO_COMBINE, FEATURES_TO_COMBINE)
    
    data_df = st.session_state.get('data')

    if data_df is not None:
        st.subheader('Pratinjau Data (termasuk `combined_jobs`)')
        # Tampilkan kolom penting termasuk 'combined_jobs' dan 'Job.ID' di awal
        cols_to_preview = ['Job.ID', 'Title', 'combined_jobs'] + [col for col in FEATURES_TO_COMBINE if col in data_df.columns and col not in ['Title']]
        st.dataframe(data_df[cols_to_preview].head(), use_container_width=True)

        st.subheader('Ringkasan Data')
        st.write(f'Jumlah baris: {len(data_df)}')
        st.write(f'Jumlah kolom: {len(data_df.columns)}')
        
        st.subheader('Contoh Isi Kolom `combined_jobs`')
        if 'combined_jobs' in data_df.columns:
            for i in range(min(3, len(data_df))):
                with st.expander(f"Job.ID: {data_df.iloc[i]['Job.ID']} - {data_df.iloc[i]['Title']}"):
                    st.text(data_df.iloc[i]['combined_jobs'])
        else:
            st.warning("Kolom 'combined_jobs' belum dibuat.")


        st.subheader('Cari Kata dalam Fitur')
        search_word = st.text_input("Masukkan kata untuk dicari:", key="home_search_word_new")
        # Filter kolom agar hanya menampilkan yang ada di DataFrame saat ini
        available_cols = [col for col in ['Job.ID', 'Title', 'combined_jobs'] + FEATURES_TO_COMBINE if col in data_df.columns]
        search_column = st.selectbox("Pilih fitur untuk dicari:", [''] + available_cols, key="home_search_column_new")

        if search_word and search_column:
            if search_column in data_df.columns:
                # Pastikan pencarian dilakukan pada tipe data string
                search_results = data_df[data_df[search_column].astype(str).str.contains(search_word, case=False, na=False)]
                if not search_results.empty:
                    st.write(f"Ditemukan {len(search_results)} entri untuk '{search_word}' di '{search_column}':")
                    st.dataframe(search_results[['Job.ID', 'Title', search_column]].head(), use_container_width=True) # Tampilkan kolom yang dicari
                else:
                    st.info(f"Tidak ada entri ditemukan untuk '{search_word}' di '{search_column}'.")
        
        st.subheader('Informasi Fitur')
        st.write('**Fitur yang tersedia (setelah pemrosesan):**', data_df.columns.tolist())
        # ... (sisa dari home_page bisa disesuaikan jika perlu) ...
    else:
        st.error("Data tidak dapat dimuat. Mohon periksa sumber data atau koneksi Anda.")
    return

# ... (Definisi fungsi halaman lainnya: preprocessing_page, tsdae_page, bert_model_page, dst.)
# Anda perlu memastikan bahwa semua halaman ini sekarang menggunakan kolom 'combined_jobs'
# sebagai input utama untuk pemrosesan teks, dan kemudian 'processed_text' setelah preprocessing.
# Contohnya, di preprocessing_page, text_column_to_process harus diatur ke 'combined_jobs'.
# Di bert_model_page dan tsdae_page, mereka harus mengambil teks dari 'processed_text' (hasil dari preprocessing 'combined_jobs').

# --- (Fungsi-fungsi halaman lainnya seperti preprocessing_page, tsdae_page, dll. tetap ada) ---
# Pastikan untuk menyesuaikan halaman-halaman tersebut agar menggunakan 'combined_jobs'
# sebagai input awal untuk preprocessing, dan 'processed_text' (hasil dari preprocessing 'combined_jobs')
# untuk pembuatan embedding dan langkah selanjutnya.

# Placeholder for other page functions to ensure script runs
# You'll need to adapt these to use 'combined_jobs' and then 'processed_text'

def tsdae_page():
    st.header("TSDAE (Sequential Noise Injection)")
    st.write("Halaman ini akan menggunakan kolom 'processed_text' (dari 'combined_jobs') untuk TSDAE.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Data belum diproses. Silakan ke halaman 'Preprocessing' terlebih dahulu.")
        return
    # ... (Implementasi TSDAE dengan input dari data['processed_text']) ...
    st.info("Implementasi TSDAE akan menggunakan kolom 'processed_text' yang berasal dari 'combined_jobs'.")
    if 'data' in st.session_state and st.session_state.data is not None and 'processed_text' in st.session_state.data:
        st.write("Contoh 'processed_text':")
        st.dataframe(st.session_state.data[['Job.ID', 'processed_text']].head())


def bert_model_page():
    st.header("BERT Model: Embedding Generation & Visualization")
    st.write("Halaman ini akan menggunakan kolom 'processed_text' (dari 'combined_jobs') untuk membuat embedding.")
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data', pd.DataFrame()).columns:
        st.warning("Data belum diproses. Silakan ke halaman 'Preprocessing' terlebih dahulu.")
        return
    # ... (Implementasi BERT model page dengan input dari data['processed_text']) ...
    st.info("Implementasi halaman Model BERT akan menggunakan kolom 'processed_text' yang berasal dari 'combined_jobs'.")
    if 'data' in st.session_state and st.session_state.data is not None and 'processed_text' in st.session_state.data:
        st.write("Contoh 'processed_text':")
        st.dataframe(st.session_state.data[['Job.ID', 'processed_text']].head())


def clustering_page():
    st.header("Clustering Job Embeddings")
    st.write("Halaman ini akan mengelompokkan embedding yang dihasilkan dari 'processed_text'.")
    # ... (Implementasi Clustering) ...
    st.info("Implementasi Clustering akan menggunakan embedding dari 'processed_text'.")


def upload_cv_page():
    st.header("Upload & Process CV(s)")
    st.write("Upload CVs (PDF/DOCX, max 5).")
    uploaded_cv_files = st.file_uploader("Choose CV files:", type=["pdf","docx"], accept_multiple_files=True, key="cv_upload_widget_main")
    if uploaded_cv_files:
        if len(uploaded_cv_files) > 5:
            st.warning("Max 5 CVs. Processing first 5.")
            uploaded_cv_files = uploaded_cv_files[:5]
        if st.button("Process Uploaded CVs", key="proc_cv_btn_main"):
            cv_data_batch = []
            bert_model_for_cv = load_bert_model()
            if not bert_model_for_cv: 
                st.error("BERT model load failed for CVs."); return 
            with st.spinner("Processing CVs..."):
                # ... (logika pemrosesan CV seperti sebelumnya) ...
                for i, cv_file in enumerate(uploaded_cv_files):
                    o_txt, p_txt, cv_e = "", "", None
                    try:
                        file_ext = cv_file.name.split(".")[-1].lower()
                        if file_ext == "pdf": o_txt = extract_text_from_pdf(cv_file)
                        elif file_ext == "docx": o_txt = extract_text_from_docx(cv_file)
                        if o_txt and o_txt.strip():
                            p_txt = preprocess_text(o_txt) # Preprocess CV text
                            if p_txt and p_txt.strip():
                                e_arr = generate_embeddings_with_progress(bert_model_for_cv, [p_txt])
                                cv_e = e_arr[0] if (e_arr is not None and e_arr.size > 0) else None
                        cv_data_batch.append({'filename':cv_file.name, 'original_text':o_txt or "", 
                                              'processed_text':p_txt or "", 'embedding':cv_e})
                        if cv_e is not None: st.success(f"Processed & embedded: {cv_file.name}")
                        else: st.warning(f"Failed to process/embed: {cv_file.name}")
                    except Exception as e:
                        st.error(f"Error with {cv_file.name}: {e}")
                st.session_state['uploaded_cvs_data'] = cv_data_batch
                st.success(f"CV batch processing done.")

    if st.session_state.get('uploaded_cvs_data'):
        st.subheader("Stored CVs:")
        for i, cv_d in enumerate(st.session_state['uploaded_cvs_data']):
            with st.expander(f"CV {i+1}: {cv_d.get('filename', 'N/A')}"):
                st.text_area(f"Original:", cv_d.get('original_text',''), height=70, disabled=True, key=f"disp_cv_o_{i}")
                st.text_area(f"Processed:", cv_d.get('processed_text',''), height=70, disabled=True, key=f"disp_cv_p_{i}")
                st.success("Embedding OK.") if cv_d.get('embedding') is not None and cv_d.get('embedding').size > 0 else st.warning("Embedding missing.")
    return


def job_recommendation_page():
    st.header("Job Recommendation")
    st.write("Halaman ini akan merekomendasikan pekerjaan berdasarkan CV yang diunggah dan data pekerjaan yang telah diproses (menggunakan 'processed_text' dari 'combined_jobs').")
    # ... (Implementasi Job Recommendation) ...
    st.info("Implementasi Job Recommendation akan menggunakan CV embeddings dan job embeddings (dari 'processed_text').")
    if not st.session_state.get('uploaded_cvs_data'): 
        st.warning("Upload & process CVs first."); return
    if st.session_state.get('data') is None or 'processed_text' not in st.session_state.get('data').columns:
        st.error("Job data (with 'processed_text') not available. Load & preprocess first."); return
    # ... (lanjutkan dengan logika rekomendasi) ...


def annotation_page():
    st.header("Annotation of Job Recommendations")
    # ... (Implementasi Anotasi seperti sebelumnya) ...
    st.info("Halaman anotasi akan menampilkan rekomendasi untuk dianotasi.")
    # (Pastikan semua referensi ke teks pekerjaan dan CV menggunakan field yang benar)


def _calculate_average_precision(ranked_relevance_binary, k_val):
    if not ranked_relevance_binary: return 0.0
    ranked_relevance_binary = ranked_relevance_binary[:k_val] 
    relevant_hits, sum_precisions = 0, 0.0
    for i, is_relevant in enumerate(ranked_relevance_binary):
        if is_relevant:
            relevant_hits += 1
            sum_precisions += relevant_hits / (i + 1)
    return sum_precisions / relevant_hits if relevant_hits > 0 else 0.0

def evaluation_page():
    st.header("Model Evaluation")
    st.write("Evaluates top 20 recommendations based on human annotations.")
    all_recommendations = st.session_state.get('all_recommendations_for_annotation', {})
    anns_df = st.session_state.get('collected_annotations', pd.DataFrame())
    if not all_recommendations: st.warning("No recommendations to evaluate."); return
    if anns_df.empty: st.warning("No annotations collected."); return

    st.subheader("Evaluation Parameters")
    st.info("The 'Binary Relevance Threshold' converts average graded annotator scores (0-3) into binary 'relevant' (1) or 'not relevant' (0) for P@20, MAP@20, MRR@20, HR@20, and Binary NDCG@20.")
    relevance_threshold_binary = st.slider("Binary Relevance Threshold", 0.0, 3.0, 1.5, 0.1, key="eval_thresh_binary_hg_v2")
    
    if st.button("Run Evaluation on Top 20 Recommendations", key="run_manual_eval_btn_v2"):
        with st.spinner("Calculating human-grounded evaluation metrics..."):
            all_p_at_20, all_map_at_20, all_mrr_at_20, all_hr_at_20 = [], [], [], []
            all_binary_ndcg_at_20, all_graded_ndcg_at_20 = [], []

            relevance_cols = [f'annotator_{i+1}_relevance' for i in range(len(ANNOTATORS)) if f'annotator_{i+1}_relevance' in anns_df.columns]
            if not relevance_cols: st.error("No annotator relevance columns in annotations."); return

            num_cvs_evaluated = 0
            for cv_filename, recommended_jobs_df in all_recommendations.items():
                if recommended_jobs_df.empty: continue
                
                recommended_jobs_df['Job.ID'] = recommended_jobs_df['Job.ID'].astype(str)
                cv_anns_subset = anns_df[anns_df['cv_filename'] == cv_filename].copy()
                if cv_anns_subset.empty: continue 
                
                num_cvs_evaluated +=1
                cv_anns_subset['job_id'] = cv_anns_subset['job_id'].astype(str)
                
                top_20_recs_df = recommended_jobs_df.head(20) # Ensure only top 20
                ranked_job_ids_list = top_20_recs_df['Job.ID'].tolist()
                model_similarity_scores = top_20_recs_df['similarity_score'].tolist()

                binary_relevance_scores = []
                graded_relevance_scores = []
                
                for job_id in ranked_job_ids_list:
                    job_specific_annotations = cv_anns_subset[cv_anns_subset['job_id'] == job_id]
                    avg_annotator_score = 0.0 
                    if not job_specific_annotations.empty:
                        annotator_scores_for_job = []
                        for rel_col_name in relevance_cols:
                            if rel_col_name in job_specific_annotations.columns:
                                annotator_scores_for_job.extend(pd.to_numeric(job_specific_annotations[rel_col_name], errors='coerce').dropna().tolist())
                        if annotator_scores_for_job:
                            avg_annotator_score = np.mean(annotator_scores_for_job)
                    
                    graded_relevance_scores.append(avg_annotator_score)
                    binary_relevance_scores.append(1 if avg_annotator_score >= relevance_threshold_binary else 0)
                
                k_cutoff = 20 
                
                if binary_relevance_scores: 
                    all_p_at_20.append(sum(binary_relevance_scores) / len(binary_relevance_scores))
                    if any(binary_relevance_scores): 
                        all_hr_at_20.append(1)
                    else:
                        all_hr_at_20.append(0)

                all_map_at_20.append(_calculate_average_precision(binary_relevance_scores, k_cutoff))
                
                current_rr = 0.0
                for r, is_rel in enumerate(binary_relevance_scores): 
                    if is_rel: current_rr = 1.0 / (r + 1); break
                all_mrr_at_20.append(current_rr)

                # Ensure lengths match for ndcg_score
                actual_k = len(binary_relevance_scores) # Number of items actually in the list (max 20)
                if actual_k == len(model_similarity_scores) and actual_k > 0:
                    all_binary_ndcg_at_20.append(ndcg_score([binary_relevance_scores], [model_similarity_scores[:actual_k]], k=actual_k))
                
                if actual_k == len(graded_relevance_scores) and actual_k == len(model_similarity_scores) and actual_k > 0:
                     all_graded_ndcg_at_20.append(ndcg_score([graded_relevance_scores], [model_similarity_scores[:actual_k]], k=actual_k))


            eval_results = {
                'Precision@20': np.mean(all_p_at_20) if all_p_at_20 else 0.0, # Default to 0.0 if no data
                'MAP@20': np.mean(all_map_at_20) if all_map_at_20 else 0.0,
                'MRR@20': np.mean(all_mrr_at_20) if all_mrr_at_20 else 0.0,
                'HR@20': np.mean(all_hr_at_20) if all_hr_at_20 else 0.0,
                'NDCG@20 (Binary)': np.mean(all_binary_ndcg_at_20) if all_binary_ndcg_at_20 else 0.0,
                'NDCG@20 (Graded)': np.mean(all_graded_ndcg_at_20) if all_graded_ndcg_at_20 else 0.0
            }

            st.subheader("Human-Grounded Evaluation Metrics Summary")
            if num_cvs_evaluated > 0:
                st.write(f"Calculated based on {num_cvs_evaluated} CVs with recommendations and annotations.")
            else:
                st.warning("No CVs with annotations were found to calculate metrics."); return 

            metric_config = {
                'Precision@20': {'fmt': "{:.2%}", 'help': "Avg P@20. Proportion of top 20 relevant items (binary).", 'color': "off"},
                'MAP@20': {'fmt': "{:.2%}", 'help': "Mean Average Precision@20 (binary relevance).", 'color': "off"},
                'MRR@20': {'fmt': "{:.4f}", 'help': "Mean Reciprocal Rank@20 (binary relevance).", 'color': "normal"},
                'HR@20': {'fmt': "{:.2%}", 'help': "Hit Ratio@20. Proportion of CVs with at least one relevant item in top 20.", 'color': "normal"},
                'NDCG@20 (Binary)': {'fmt': "{:.4f}", 'help': "Avg NDCG@20 using binary relevance.", 'color': "inverse"},
                'NDCG@20 (Graded)': {'fmt': "{:.4f}", 'help': "Avg NDCG@20 using average annotator scores as graded relevance.", 'color': "inverse"}
            }
            
            keys_to_display = ['Precision@20', 'MAP@20', 'MRR@20', 'HR@20', 'NDCG@20 (Binary)', 'NDCG@20 (Graded)']
            
            # Display in 2 rows, 3 cols each
            cols_for_metrics = st.columns(3) # Create 3 columns per row
            row_idx = 0
            col_idx = 0

            for key in keys_to_display:
                if key in eval_results:
                    value = eval_results[key]
                    cfg = metric_config[key]
                    
                    current_col = cols_for_metrics[col_idx % 3]
                    if col_idx > 0 and col_idx % 3 == 0 : # Start a new row of columns after every 3 metrics
                        cols_for_metrics = st.columns(3)
                        current_col = cols_for_metrics[0]


                    val_str = "N/A"
                    if isinstance(value, (int, float, np.number)) and not (isinstance(value, float) and np.isnan(value)):
                        val_str = cfg['fmt'].format(value * 100 if '%' in cfg['fmt'] else value)
                    elif isinstance(value, str):
                        val_str = value
                    
                    current_col.metric(label=key, value=val_str, delta_color=cfg['color'], help=cfg['help'])
                    col_idx +=1
    return


# --- Main App Logic (Page Navigation) ---
st.sidebar.title("Navigation")
page_options = ["Home", "Preprocessing", "TSDAE (Noise Injection)", "BERT Model", 
                "Clustering Job2Vec", "Upload CV", "Job Recommendation", "Annotation", "Evaluation"]
page = st.sidebar.radio("Go to", page_options, key="main_nav_radio")

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
    annotation_page() # Assuming annotation_page definition exists
elif page == "Evaluation":
    evaluation_page()
