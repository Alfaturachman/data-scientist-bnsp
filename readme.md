# Portofolio Uji Sertifikasi Kompetensi Data Scientist (BNSP)

## Repositori Proyek Klasifikasi Medis & Analisis Jurnal (Replikasi Eksperimental)

Dokumentasi ini merupakan berkas master repositori untuk memenuhi syarat kelayakan/bukti pendukung (_evidence_) dalam **Uji Sertifikasi Kompetensi Skema Data Scientist (BNSP)**. Seluruh proyek di dalam repositori ini dirancang secara terstruktur dan disiplin untuk menyelesaikan permasalahan klasifikasi medis riil dengan mengacu pada jurnal-jurnal ilmiah Scopus terkini (2025/2026), serta menerapkan standar baku keamanan data (_data leakage-safe_).

---

## Informasi Peserta

- **Nama**: Alfaturachman Maulana Pahlevi
- **NIM**: A11.2025.16609
- **Kelas**: D.2.J
- **Program**: Workshop & Sertifikasi Data Scientist

---

## Struktur Repositori & Workspace

```text
d:/project sertifikasi/
│
├── readme.md                   <- Dokumentasi master repositori & pemetaan BNSP (berkas ini)
├── index.html                  <- Dashboard Infografis Eksekutif (Apresiasi Desain & Ringkasan Eksperimen)
│
├── diabetes/                   <- Proyek A: Klasifikasi Penyakit Diabetes Mellitus
│   ├── diabetes.csv            <- Pima Indians Diabetes Dataset (768 baris × 9 kolom)
│   ├── Diabetes_Prediction.ipynb <- Notebook Jupyter (Imputasi Bersyarat, Rekayasa Fitur, Ensemble, Optuna)
│   └── readme.md               <- Dokumentasi rinci proyek diabetes & replikasi jurnal
│
├── chronic kidney disease/     <- Proyek B: Deteksi Penyakit Ginjal Kronis (CKD)
│   ├── kidney_disease.csv      <- Dataset CKD dari UCI ML Repository (400 baris × 25 kolom)
│   ├── Kidney_Disease.ipynb    <- Notebook Jupyter (K-Means Stratified Split, Polynomial Features)
│   └── readme.md               <- Dokumentasi rinci proyek ginjal & evaluasi homogenitas
│
├── breast cancer/              <- Proyek C: Klasifikasi Tumor Payudara Wisconsin (WDBC)
│   ├── breast_cancer.csv       <- Dataset Wisconsin Diagnostic Breast Cancer (569 baris × 32 kolom)
│   ├── Breast_Cancer.ipynb     <- Notebook Jupyter (Interpretable Logistic Regression, SVM, KNN, RF)
│   └── readme.md               <- Dokumentasi rinci interpretasi koefisien klinis
│
└── heart_disease/              <- Proyek D: Deteksi Penyakit Jantung (UCI 4 Regional Cohort)
    ├── heart.csv               <- Dataset Penyakit Jantung Cleveland, Hungarian, Swiss, VA (920 baris)
    ├── Heart_Disease.ipynb     <- Notebook Jupyter (Outlier Isolation Forest, Multiregional Imputation)
    └── readme.md               <- Dokumentasi rinci proyek jantung & pemetaan kompetensi
```

---

## Ringkasan Eksekutif & Kinerja Pemodelan

Tabel berikut menunjukkan perbandingan pendekatan metodologis, rekayasa fitur, model terbaik, dan performa evaluasi pada data uji (_held-out test set_) untuk masing-masing proyek di dalam repositori:

| Proyek Klasifikasi         | Dataset & Sampel           | Algoritma Utama & Model Terbaik             | Metrik Utama (Test Set)                                           | Inovasi / Fitur Kunci                                                                        | Jurnal Acuan                              |
| :------------------------- | :------------------------- | :------------------------------------------ | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :---------------------------------------- |
| **Diabetes Prediction**    | Pima Indians (768 baris)   | Soft-Voting Ensemble (XGBoost + LightGBM)   | **Accuracy**: 71.43%<br>**ROC-AUC**: 77.33%                       | Class-Conditional Median Imputer, 16 Composite Features, Optuna Hyperparameter Tuning        | Ali et al. (Minia Univ, Egypt - 2025)     |
| **Chronic Kidney Disease** | UCI CKD (400 baris)        | Random Forest, SVM, XGBoost, & Logistic Reg | **Accuracy**: 100%<br>**F1-Score**: 100%                          | Feature-Based K-Means Stratified Split, Polynomial Features (Degree 2)                       | Dong Phuong et al. (Bioinformatics, 2025) |
| **Breast Cancer**          | WDBC Wisconsin (569 baris) | L2-Regularized Logistic Regression          | **Accuracy**: 98.25%<br>**F1-Score**: 97.56%                      | Z-Score Scaler terpisah, Transparansi Koefisien Ko-Variat untuk Interpretasi Medis           | Cheng & Yu (medRxiv Preprint, 2025)       |
| **Heart Disease**          | UCI 4 Kohort (920 baris)   | XGBoost Classifier (Eksperimen 1)           | **Accuracy**: 81.52%<br>**Recall**: 83.01%<br>**AUC-ROC**: 0.8765 | Multi-cohort Imputation (convert Chol=0 to NaN), Outlier Removal Analysis (Isolation Forest) | Tabassum et al. (Scopus Q2, 2025)         |

---

## Penjelasan Detail Eksperimen & Metodologi

### 1. Prediksi Diabetes Mellitus (Pima Indians)

- **Masalah Utama**: Adanya nilai `0` tidak logis pada indikator vital medis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) yang jika diabaikan atau diimputasi secara global akan mendistorsi prediksi.
- **Metode Solusi**: Menangani nilai nol sebagai data hilang (_missing value_) menggunakan `ClassConditionalMedianImputer` (imputasi median terpisah untuk kelas sehat dan diabetes) yang di-_fit_ secara aman pada data latih. Rekayasa **16 Fitur Komposit** (contoh: indeks metabolisme insulin, interaksi glukosa-usia) dilakukan untuk menangkap korelasi non-linier fitur.
- **Pencegahan Data Leakage (Kebocoran Data)**:
  Eksperimen ini secara krusial menguji dampak data leakage melalui 3 skenario:
    1.  **Skenario 1 (Split-First)**: Pembagian train-test secara aman terlebih dahulu sebelum imputasi. ROC-AUC: **74.81%**.
    2.  **Skenario 2 (Preprocess-First - LEAKAGE)**: Imputasi dan SMOTE dilakukan secara global sebelum data dibagi. Hal ini memberikan hasil tinggi palsu (**ROC-AUC 96.30% / Akurasi 90.50%**), yang membuktikan target leakage berbahaya jika diterapkan pada keputusan medis riil.
    3.  **Skenario 3 (Optimized)**: Skenario Split-First yang dioptimalkan secara legal menggunakan hyperparameter tuning berbasis **Optuna Bayesian Optimization** (5-Fold CV pada training set). ROC-AUC naik secara valid menjadi **77.33%** dengan Akurasi **71.43%**.

### 2. Deteksi Penyakit Ginjal Kronis (CKD)

- **Masalah Utama**: Heterogenitas pembagian data acak (train-test split) yang dapat memicu bias evaluasi kinerja pada dataset berdimensi tinggi dengan banyak fitur kategorikal.
- **Metode Solusi**: Penerapan **Feature-Based Stratified Splitting Combined With K-Means Clustering** (K-Stratified Split). Metode ini mengelompokkan data terlebih dahulu ke dalam cluster homogen berbasis K-Means, lalu melakukan stratifikasi sampel dari masing-masing cluster.
- **Hasil**: Pembagian data terbukti sangat homogen (tervalidasi lewat tes Kolmogorov-Smirnov). Dipasangkan dengan **Polynomial Features (Degree 2)** pada variabel numerik kunci, model Random Forest, SVM, Logistic Regression, dan XGBoost berhasil mencapai akurasi **100%** pada test set.

### 3. Klasifikasi Tumor Payudara Wisconsin (WDBC)

- **Masalah Utama**: Keengganan praktisi medis menggunakan model black-box (seperti Deep Learning atau Ensemble kompleks) karena tidak adanya penjelasan rasio odds klinis.
- **Metode Solusi**: Replikasi **Logistic Regression (LR) Framework** dengan regularisasi L2 (Ridge) untuk mencegah multikolinearitas antar 30 metrik seluler tumor.
- **Hasil & Interpretasi**: Model mencapai akurasi test sebesar **98.25%**, bersaing ketat dengan SVM RBF (**98.25%**) dan Random Forest (**97.37%**). Koefisien model LR memberikan interpretasi medis instan: fitur `worst area` dan `radius error` memiliki koefisien positif terbesar, yang secara langsung berkorelasi dengan peningkatan probabilitas tumor ganas (_malignant_).

### 4. Prediksi Penyakit Jantung (UCI 4 Kohort)

- **Masalah Utama**: Inkonsistensi data antar 4 cohort regional (Cleveland, Hungarian, Switzerland, VA Long Beach), khususnya tingginya missing values pada kolom `ca` (pembuluh darah) dan `thal` (thalasemia), serta anomali pembacaan nilai kolesterol `0`.
- **Metode Solusi**: Mengubah kolesterol bernilai `0` pada cohort Swiss/VA menjadi `NaN` lalu mengimputasinya dengan median kohort. Proyek ini membandingkan kinerja model tanpa outlier versus dengan outlier removal menggunakan **Isolation Forest** (contamination=3%).
- **Temuan Kunci**: Penghapusan outlier justru menurunkan performa model (Akurasi XGBoost turun dari **81.52% ke 77.54%**; Recall turun dari **83.01% ke 73.21%**). Secara klinis, disimpulkan bahwa data ekstrem (outlier numerik seperti oldpeak atau tekanan darah sangat tinggi) merupakan sinyal patologis parah penyakit jantung yang sangat berharga dan tidak boleh dihapus.

---

## Executive Dashboard & Infographics (`index.html`)

Repositori ini dilengkapi dengan berkas **`index.html`** yang merupakan infografis web visual premium interaktif yang berfungsi sebagai lembar ringkasan eksekutif (_Executive Summary Sheet_).

- **Desain**: Minimalis, elegan, berpusat pada kegunaan, mengadopsi bahasa desain **Apple Human Interface Guidelines (HIG)** dengan palet warna HSL premium (iOS Blue, iOS Slate, Green, Dark Grey).
- **Fitur Visual**:
    - _DNA helix & Medical ECG vector header illustrations_ (SVG asli).
    - _Unified Metrics Panel_ (6 metrik dataset dan model kunci dengan ikon SVG inline).
    - _Experiment Comparison Grid_ dilengkapi dengan representasi visual mini-charts dinamis.
    - _Interactive Workflow Steps_ (Tahapan 1-11 dipetakan dalam 3-kolom grid yang rapi dan responsif).
    - _Insight Utama Banner_ yang menyoroti temuan kritis eksperimental secara informatif.

Untuk membukanya, cukup klik dua kali berkas `index.html` pada peramban (_web browser_) Anda.

---

## Pemetaan Unit Kompetensi Sertifikasi (BNSP)

Seluruh pengerjaan notebook pada repositori ini dirancang untuk mematuhi **Standar Kompetensi Kerja Nasional Indonesia (SKKNI)** Bidang Data Science. Berikut adalah pemetaan kode unit kompetensi ke dalam bukti pengerjaan:

| Kode Unit Kompetensi | Judul Unit Kompetensi                 | Bukti Implementasi & Lokasi pada Notebook / Repositori                                                                                                                                                        |
| :------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **J.62DMI00.001.1**  | Menentukan Objektif Bisnis            | Rincian latar belakang medis, objektif diagnosis, risiko proyek, dan penentuan metrik evaluasi medis (misal: penekanan Recall untuk meminimalkan _false negative_). Lokasi: Bab I di semua Notebook.          |
| **J.62DMI00.002.1**  | Menentukan Tujuan Teknis Data Science | Pendefinisian tujuan teknis sebagai klasifikasi biner dan target metrik performa (Akurasi, F1-score, ROC-AUC). Lokasi: Bab I & II di semua Notebook.                                                          |
| **J.62DMI00.005.1**  | Menelaah Data                         | Deskripsi statistik (`.describe()`, tipe data, visualisasi countplot kelas target, KDE plot sebaran usia, dan matriks korelasi heatmap). Lokasi: Bab EDA & Telaah Data.                                       |
| **J.62DMI00.006.1**  | Memvalidasi Data                      | Analisis kualitas data, persentase data hilang (_missing values_), anomali data (nilai nol tidak logis, kolesterol nol). Lokasi: Bab Validasi Data.                                                           |
| **J.62DMI00.007.1**  | Menentukan Objek Data                 | Pemilahan fitur klinis relevan (14 fitur dari 76 fitur pada penyakit jantung; 9 fitur utama pada diabetes). Lokasi: Bab Data Selection.                                                                       |
| **J.62DMI00.008.1**  | Membersihkan Data                     | Implementasi pengisian data hilang menggunakan median per kohort, median bersyarat kelas (`ClassConditionalMedianImputer`), dan koreksi anomali. Lokasi: Bab Data Cleaning.                                   |
| **J.62DMI00.009.1**  | Mengkonstruksi Data                   | Pembuatan 16 Fitur Komposit Klinis baru (Diabetes), perluasan fitur kuadratik non-linier (CKD), dan One-Hot Encoding variabel kategorikal. Lokasi: Bab Feature Engineering.                                   |
| **J.62DMI00.012.1**  | Membangun Skenario Model              | Pengujian multi-skenario (Split-First vs Preprocess-First, Tanpa Outlier vs Isolation Forest Outlier Removal), Stratified 5-Fold Cross Validation. Lokasi: Bab Model Scenarios.                               |
| **J.62DMI00.013.1**  | Membangun Model                       | Pelatihan 8+ algoritma klasifikasi, optimasi parameter dengan Optuna Bayesian Tuning, dan perancangan Soft-Voting Ensemble Classifier. Lokasi: Bab Modeling & Training.                                       |
| **J.62DMI00.014.1**  | Mengevaluasi Hasil Pemodelan          | Penyusunan tabel perbandingan metrik, penggambaran kurva ROC-AUC, grafik perbandingan performa, dan Confusion Matrix (TP/TN/FP/FN). Lokasi: Bab Evaluation.                                                   |
| **J.62DMI00.015.1**  | Melakukan Proses Review Pemodelan     | Ulasan analitik kegagalan outlier removal (Penyakit Jantung), visualisasi pentingnya fitur via koefisien (Breast Cancer), dan analisis pembuktian target leakage (Diabetes). Lokasi: Bab Review & Discussion. |

---

## Lisensi

Repositori ini dilisensikan di bawah **MIT License** - lihat berkas [LICENSE](LICENSE) untuk detail lebih lanjut. Seluruh dataset bersumber dari data publik UCI Machine Learning Repository dan Kaggle.
