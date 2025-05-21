import streamlit as st
import pandas as pd

st.title('👨‍⚕️ Optimizing Job Recommendation Using TSDAE at Job2Vec')

st.info('TSDAE menghasilkan representasi embedding yang lebih robust terhadap data pekerjaan yang tidak berlabel')


with st.expander ("Postingan Lowongan Pekerjaan Dataset (Kaggle)"):
  st.write ("**Fitur sudah diseleksi dan dimerging**")
  df_jobs = pd.read_csv("https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/tech_jobs.csv")
  df_jobs

with st.expander ("Profil Pengguna Dataset (Kaggle)"):
  st.write ("**Fitur sudah diseleksi dan dimerging**")
  df_user = pd.read_csv("https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/user_applicant_jobs.csv")
  df_user

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    df = extract_data(uploaded_file)

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None # build more code to return a dataframe 



