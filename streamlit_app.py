import streamlit as st
import pandas as pd

st.title('👨‍⚕️ Optimizing Job Recommendation Using TSDAE at Job2Vec')

st.info('TSDAE Mmenghasilkan representasi embedding yang lebih robust terhadap data pekerjaan yang tidak berlabel')

df_jobs = pd.read_csv("https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/tech_jobs.csv")
df_jobs

df_user = pd.read_csv("https://raw.githubusercontent.com/adinplb/dp-machinelearning-ai/refs/heads/master/dataset/user_applicant_jobs.csv")
df_user
