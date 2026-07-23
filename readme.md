# Uji Sertifikasi Kompetensi Data Scientist (BNSP)

## Repositori Proyek Klasifikasi Medis & Analisis Jurnal (Replikasi Eksperimental)

Dokumentasi ini merupakan berkas master repositori untuk memenuhi syarat kelayakan/bukti pendukung (_evidence_) dalam **Uji Sertifikasi Kompetensi Skema Data Scientist (BNSP)**. Seluruh proyek di dalam repositori ini dirancang secara terstruktur dan disiplin untuk menyelesaikan permasalahan klasifikasi medis riil dengan mengacu pada jurnal-jurnal ilmiah Scopus terkini (2025/2026), serta menerapkan standar baku keamanan data (_data leakage-safe_).

---

## Informasi Peserta

- **Nama**: Alfaturachman Maulana Pahlevi
- **NIM**: A11.2025.16609
- **Program**: Workshop & Uji Sertifikasi Data Scientist
- **Skema Sertifikasi**: Ilmuwan Data (Data Scientist)
- **Lembaga Sertifikasi**: Badan Nasional Sertifikasi Profesi (BNSP)
- **Tahun**: 2026

---

## Struktur Repositori & Workspace

```text
d:/project sertifikasi/
│
├── readme.md                   <- Dokumentasi master repositori & pemetaan BNSP (berkas ini)
│
├── stroke/                     <- Proyek Utama: Skrining Risiko Stroke Posbindu PTM Kota Semarang
│   ├── stroke.csv              <- Dataset Rekam Medis Posbindu (10.000 sampel × 10 fitur)
│   ├── Stroke-Prediction.ipynb <- Notebook Jupyter (CRISP-DM & 11 Unit SKKNI Leak-Free Pipeline)
│   └── readme.md               <- Dokumentasi rinci proyek stroke & audit kritis jurnal Scopus Q1
│
├── diabetes/                   <- Proyek A: Klasifikasi Penyakit Diabetes Mellitus
│   ├── diabetes.csv            <- Pima Indians Diabetes Dataset (768 baris × 9 kolom)
│   ├── Diabetes_Prediction.ipynb <- Notebook Jupyter (Imputasi Bersyarat, Rekayasa Fitur, Ensemble, Optuna)
│   └── readme.md               <- Dokumentasi rinci proyek diabetes & replikasi jurnal
│
└── heart_disease/              <- Proyek B: Deteksi Penyakit Jantung (UCI 4 Regional Cohort)
    ├── heart.csv               <- Dataset Penyakit Jantung Cleveland, Hungarian, Swiss, VA (920 baris)
    ├── Heart_Disease.ipynb     <- Notebook Jupyter (Outlier Isolation Forest, Multiregional Imputation)
    └── readme.md               <- Dokumentasi rinci proyek jantung & pemetaan kompetensi
```

---

## Ringkasan Eksekutif & Kinerja Pemodelan

Tabel berikut menunjukkan perbandingan pendekatan metodologis, rekayasa fitur, model terbaik, dan performa evaluasi pada data uji (_held-out test set_) untuk masing-masing proyek di dalam repositori:

| Proyek Klasifikasi | Dataset & Sampel | Algoritma Utama & Model Terbaik | Metrik Utama (Test Set) | Inovasi / Fitur Kunci | Jurnal Acuan |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stroke Risk Screening** | Rekam Medis Posbindu (10.000 baris × 10 fitur) | **Logistic Regression Pipeline** *(Best Overall)* | **Recall**: 76,51%<br>**F1-Score**: 0,6538<br>**PR-AUC**: 0,7245 | **0% Leakage Guarantee** (Split 80:20 awal), ImbPipeline SMOTE + 5-Fold Stratified CV + GridSearchCV, SHAP TreeExplainer, Audit Confusion Matrix Paper Acuan (+11,55% lebih unggul). | Tang et al. (*Frontiers in Neurology*, Scopus Q1 2026) |
| **Diabetes Prediction** | Pima Indians (768 baris) | LightGBM (Optimized) | **Accuracy**: 75.32%<br>**ROC-AUC**: 81.70% | KNN Imputer (k=5), 16 Composite Features, Optuna Hyperparameter Tuning | Ali et al. (Minia Univ, Egypt - 2025) |
| **Heart Disease** | UCI 4 Kohort (920 baris) | XGBoost Classifier (Eksperimen 1) | **Accuracy**: 81.52%<br>**Recall**: 83.01%<br>**AUC-ROC**: 0.8765 | Multi-cohort Imputation (convert Chol=0 to NaN), Outlier Removal Analysis (Isolation Forest) | Tabassum et al. (Scopus Q2, 2025) |

---

## Penjelasan Detail Eksperimen & Metodologi

### 1. Skrining Risiko Stroke Posbindu PTM Kota Semarang (Replikasi & Audit Frontiers in Neurology 2026)

- **Masalah Utama**: Deteksi dini risiko stroke medis pada 10.000 sampel rekam medis Posbindu PTM Kota Semarang dengan *class imbalance* (29,78% kejadian stroke). Banyak studi acuan terjangkit *data leakage* (kebocoran data) akibat imputasi/oversampling global sebelum pembagian data train-test.
- **Metode Solusi**:
  1. **Alur 11 Unit SKKNI Data Scientist**: Menerapkan alur CRISP-DM & 11 Unit SKKNI secara penuh, diawali *Stratified 80:20 Train-Test Split* di awal (Unit 08: `J.62DMI00.012.1`) untuk menjamin **0% Data Leakage**.
  2. **Pipeline Bebas Leakage**: Mengintegrasikan `ColumnTransformer` (Imputasi Median terisolasi + `StandardScaler` + `OneHotEncoder`) dan `ImbPipeline` (SMOTE + 5-Fold Stratified Cross Validation + `GridSearchCV`).
  3. **Model Terbaik**: **Logistic Regression Pipeline** meraih metrik klinis tertinggi dengan **Recall 76,51%**, **F1-Score 0,6538**, dan **PR-AUC 0,7245**.
  4. **Audit Kritis Paper Acuan**: Audit Confusion Matrix terhadap paper Tang et al. (*Frontiers in Neurology*, 2026) membuktikan F1-score sejati paper acuan sebenarnya hanyalah **0,5861** (bukan 0,74 sebagaimana diklaim akibat kebocoran data). Model eksperimen ini terbukti **+11,55% lebih unggul secara valid**.
  5. **Interpretabilitas & Rekomendasi Medis**: Analisis SHAP TreeExplainer menempatkan *Hipertensi* (mean |SHAP| 0,9087) dan *Penyakit Jantung* (0,4574) sebagai prediktor utama. Rekomendasi threshold klinis Posbindu ditetapkan pada **0.35** (Skrining Massal, Recall >85%), **0.50** (Standar Puskesmas, F1 Max 0.6538), dan **0.65** (Rujukan Spesialis, Precision >75%).

### 2. Prediksi Diabetes Mellitus (Pima Indians)

- **Masalah Utama**: Adanya nilai `0` tidak logis pada indikator vital medis (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) yang jika diabaikan atau diimputasi secara global akan mendistorsi prediksi.
- **Metode Solusi**: Menangani nilai nol sebagai data hilang (_missing value_) menggunakan **KNN Imputer (k=5)**. Rekayasa **16 Fitur Komposit** dilakukan untuk menangkap korelasi non-linier fitur.
- **Pencegahan Data Leakage**: Pengujian 3 skenario membuktikan Split-First yang dioptimalkan secara legal menggunakan Optuna Bayesian Optimization menghasilkan ROC-AUC **81.70%** dengan Akurasi **75.32%**.

### 3. Prediksi Penyakit Jantung (UCI 4 Kohort)

- **Masalah Utama**: Inkonsistensi data antar 4 cohort regional (Cleveland, Hungarian, Switzerland, VA Long Beach).
- **Metode Solusi**: Mengubah kolesterol bernilai `0` pada cohort Swiss/VA menjadi `NaN` lalu mengimputasinya dengan median kohort. Outlier removal via **Isolation Forest** membuktikan data ekstrem medis merupakan sinyal patologis parah penyakit jantung yang berharga dan tidak boleh dihapus.

---

## Pemetaan Unit Kompetensi Sertifikasi (BNSP)

Seluruh pengerjaan notebook pada repositori ini dirancang untuk mematuhi **Standar Kompetensi Kerja Nasional Indonesia (SKKNI)** Bidang Data Science. Berikut adalah pemetaan 11 kode unit kompetensi ke dalam bukti pengerjaan:

| Kode Unit Kompetensi | Judul Unit Kompetensi | Bukti Implementasi & Lokasi pada Notebook / Repositori |
| :--- | :--- | :--- |
| **J.62DMI00.001.1** | Menentukan Objektif Bisnis | Formulasi objektif skrining cepat Posbindu PTM & rujukan presisi Puskesmas Kota Semarang. Penekanan Recall medis. Lokasi: `Stroke-Prediction.ipynb` & Bab I di semua Notebook. |
| **J.62DMI00.002.1** | Menentukan Tujuan Teknis Data Science | Pendefinisian metrik evaluasi F1-Score, Recall, dan PR-AUC (menghindari jebakan akurasi naif pada imbalanced data). Lokasi: `Stroke-Prediction.ipynb` & Bab I. |
| **J.62DMI00.005.1** | Menelaah Data | EDA 10.000 sampel rekam medis Posbindu, analisis sebaran *Class Imbalance* 29,78% kejadian stroke. Lokasi: Bab EDA. |
| **J.62DMI00.006.1** | Memvalidasi Data | Pengujian konsistensi klinis, persentase missing BMI 2,01%, verifikasi 0 duplikasi data. Lokasi: Bab Validasi Data. |
| **J.62DMI00.007.1** | Menentukan Objek Data | Pemisahan variabel prediktor $X$ dan target $y$, keputusan etis melakukan *drop* variabel SES (*Socioeconomic Status*). Lokasi: Bab Objek Data. |
| **J.62DMI00.008.1** | Membersihkan Data | Imputasi median BMI terisolasi di dalam `ColumnTransformer` (terpisah aman per-fold). Lokasi: Bab Cleaning. |
| **J.62DMI00.009.1** | Mengkonstruksi Data | Transformasi fitur numerik (`StandardScaler`) & kategorikal (`OneHotEncoder`) terbungkus aman dalam pipeline. Lokasi: Bab Feature Engineering. |
| **J.62DMI00.012.1** | Membangun Skenario Model | **Stratified 80:20 Train-Test Split** dieksekusi di tahap awal sebelum preprocessing untuk menjamin **0% Data Leakage**. Lokasi: Bab Split Data. |
| **J.62DMI00.013.1** | Membangun Model | Pelatihan 5 algoritma (Logistic Reg, Random Forest, XGBoost, LightGBM, MLP) via `ImbPipeline` + SMOTE + 5-Fold Stratified CV + `GridSearchCV`. Lokasi: Bab Modeling. |
| **J.62DMI00.014.1** | Mengevaluasi Hasil Pemodelan | Evaluasi 5 model pada *test set* 2.000 sampel menggunakan 7 metrik (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, Brier Score). Lokasi: Bab Evaluasi. |
| **J.62DMI00.015.1** | Melakukan Proses Review Pemodelan | Analisis *SHAP TreeExplainer* (Hipertensi & Jantung sebagai prediktor utama), audit Confusion Matrix paper acuan Tang et al. (2026), dan penyusunan rekomendasi threshold klinis Posbindu. Lokasi: Bab Review. |

---

## Lisensi

Repositori ini dilisensikan di bawah **MIT License** - lihat berkas [LICENSE](LICENSE) untuk detail lebih lanjut. Seluruh dataset bersumber dari data publik rekam medis klinis, UCI Machine Learning Repository, dan Kaggle.
