# 👨‍⚕️ OPTIMIZING JOB RECOMMENDATION SYSTEM USING TRANSFORMER BASED SEQUENTIAL DENOISING AUTO-ENCODER ON JOB2VEC

Artificial Intelligence-based job recommendation systems are faced with a major task in handling job data which is generally unlabeled, inconsistent, and noisy. Job titles and categories posted by companies often differ, leading to overlapping content and difficulties in automatic classification. In addition, the lack of universal classification standards and the presence of semantic noise can degrade the quality of job representations (Job2Vec). This research aims to optimize the job recommendation system by applying Transformer-based Sequential Denoising Auto-Encoder (TSDAE) to the Job2Vec representation, which is a type of unsupervised sentence embedding learning that is able to produce more robust embedding by reconstructing input which has been given noise, making it suitable for domain specific like job advertisement. The dataset to be used is secondary data from Kaggle which consists of LinkedIn Job Posting Scraped corpus and user profiles corpus. Some features will be selected, merged and preprocessed before being converted into Job2Vec embedding using TSDAE. The TSDAE results will be clustered using K-Means and similarity calculation using Nearest Neighbors (Local Search) with a K value of 20. The system performance will be evaluated with 2 step evaluation approach: Application-Grounded Evaluation using metric NDCG@20, MRR@20. PRECISION@20, MAP@20 and Human-Grounded Evaluation using real-world resumes and Human Resources (HR) Expert feedback. This model will be compared with traditional embedding baselines such as TF-IDF, FastText, CountVectorizer, Word2Vec, GloVe and BERT. It will be compared with embedding with and without TSDAE. Our experimental results demonstrate that TSDAE outperforms both traditional and state-of-art embedding approaches while showing robust effectiveness in human expert evaluations. This research is expected to improve the quality of work recommendation systems that are more adaptive, scalable, and relevant to users  unique preferences, especially in the context of noise and unlabeled data.

Keywords: Job Recommendation System, Job2Vec, TSDAE, K-Means Clustering, K-Nearest Neighbors, Top-N Evaluation


## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://Denoising-Text_Autoencoders_TSDAE_Job-Recommendation.streamlit.app/) 

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Section Heading

This is filler text, please replace this with text for this section.

## Further Reading

This is filler text, please replace this with a explanatory text about further relevant resources for this repo
- Resource 1
- Resource 2
- Resource 3
