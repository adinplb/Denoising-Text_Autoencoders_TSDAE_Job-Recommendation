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



