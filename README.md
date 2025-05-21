# 👨‍⚕️ Optimizing Job Recommendation Using TSDAE at Job2Vec

Sistem rekomendasi kerja berbasis kecerdasan buatan menghadapi tantangan dalam mengelola data lowongan pekerjaan yang umumnya tidak berlabel, tidak konsisten, dan mengandung banyak noise. Judul serta kategori pekerjaan yang diunggah oleh perusahaan sering kali bervariasi, menyebabkan overlapping content dan menyulitkan proses klasifikasi otomatis. Selain itu, ketiadaan standar klasifikasi yang baku serta adanya noise semantik dapat menurunkan kualitas representasi embedding vektor informasi lowongan kerja (Job2Vec). Penelitian ini bertujuan untuk mengoptimalkan sistem rekomendasi kerja dengan menerapkan Transformer-based Sequential Denoising Auto-Encoder (TSDAE) pada representasi Job2Vec. TSDAE merupakan metode self-supervised learning yang mampu menghasilkan embedding yang lebih tahan terhadap gangguan (robust) dengan merekonstruksi input yang telah diberikan noise, sehingga sangat sesuai untuk data pekerjaan yang tidak berlabel. Dataset yang digunakan merupakan data sekunder dari Kaggle berupa hasil scraping informasi lowongan kerja di LinkedIn dan profil pengguna. Data ini melalui proses seleksi fitur, penggabungan, dan pra-pemrosesan sebelum diubah menjadi embedding Job2Vec menggunakan TSDAE. Hasil embedding kemudian dikluster menggunakan algoritma K-Means dan dievaluasi kemiripannya menggunakan Nearest Neighbors dengan nilai K=20. Evaluasi sistem dilakukan melalui skenario Human-Grounded Evaluation yang melibatkan lima annotator pada lima CV nyata. Penilaian annotator digunakan sebagai ground truth dan dievaluasi menggunakan metrik NDCG@20, MAP@20, Precision@20, dan MRR@20. Pengujian membandingkan model dengan dan tanpa TSDAE. Proses anotasi didukung antarmuka Streamlit, sedangkan pelacakan performa sistem dilakukan melalui platform Wandb. Penelitian ini diharapkan menghasilkan sistem rekomendasi kerja yang adaptif, skalabel, dan relevan terhadap preferensi pengguna.

Kata Kunci: Sistem Rekomendasi Kerja, Job2Vec, TSDAE, K-Means Clustering, K-Nearest Neighbors, Evaluasi Top-N

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dp-machinelearning-ai.streamlit.app/) 

## GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/app-starter-kit?quickstart=1)

## Section Heading

This is filler text, please replace this with text for this section.

## Further Reading

This is filler text, please replace this with a explanatory text about further relevant resources for this repo
- Resource 1
- Resource 2
- Resource 3
