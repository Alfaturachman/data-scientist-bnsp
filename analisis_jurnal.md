# Analisis Komprehensif Jurnal Pendukung Penelitian

## Topik: Prediksi Penyakit Jantung menggunakan Machine Learning

Dokumen ini merupakan analisis lintas-jurnal dari 4 paper ilmiah yang digunakan sebagai landasan penelitian dalam proyek **Heart Disease Prediction** untuk Uji Sertifikasi Data Scientist (BNSP).

---

## 📚 Daftar Jurnal yang Dianalisis

| No. |       Peran        | Judul Paper (Singkat)                                   | Jurnal                             |  Indeks   | Tahun |
| :-: | :----------------: | :------------------------------------------------------ | :--------------------------------- | :-------: | :---: |
|  1  | **Landasan Utama** | Unified Approach for Accurate Heart Disease Prediction  | Springer (Information Retrieval)   | Scopus Q2 | 2026  |
|  2  |  **Pendukung A**   | A ML-Based Framework for Heart Disease Diagnosis        | CMC (Tech Science Press)           | Scopus Q2 | 2025  |
|  3  |  **Pendukung B**   | Sustainable and Interpretable Heart Disease Prediction  | Scientific Reports (Nature)        | Scopus Q1 | 2026  |
|  4  |  **Pendukung C**   | A Systematic Review of ML Techniques for CVD Prediction | Cureus Journal of Computer Science | Scopus Q1 | 2026  |

---

## 🔬 Analisis Per-Jurnal

---

### 1. Paper Landasan Utama

**"Unified Approach for Accurate Heart Disease Prediction Using Machine Learning Techniques"**

> **Penulis:** R.V. Raghavendra Rao, Ch. Ram Mohan Reddy, K. Hemanth, D. Hruthik Chavan
> **Institusi:** B.M.S. College of Engineering, Bengaluru, India
> **Published:** Springer – _Information Retrieval_ (Scopus Q2, 2026) | Online: 04 Maret 2026

#### Tujuan Penelitian

Mengembangkan pipeline machine learning terpadu yang mampu memprediksi penyakit jantung dengan tingkat akurasi tertinggi, mengintegrasikan tiga komponen utama: seleksi fitur berbasis XGBoost, penanganan ketidakseimbangan kelas (SMOTE), dan optimasi hyperparameter otomatis (Optuna).

#### Dataset

- **Sumber:** Kaggle (`johnsmith88/heart-disease-dataset`) — versi pre-merged dari 4 kohort UCI
- **Ukuran:** 1.025 instance, 14 fitur, tanpa missing values
- **Target:** Biner (0 = tidak sakit, 1 = sakit jantung)
- **Distribusi Kelas:** ~48.7% negatif, 51.3% positif (hampir seimbang)

#### Metodologi (Pipeline Utama)

```
Raw Data → Preprocessing → SMOTE (dalam fold CV) → Feature Selection (XGBoost + SelectFromModel)
→ Optuna Hyperparameter Tuning (TPE Sampler) → Nested K-Fold CV → Evaluasi & SHAP
```

| Komponen                        | Detail                                                                       |
| :------------------------------ | :--------------------------------------------------------------------------- |
| **Algoritma Utama**             | XGBoost Classifier                                                           |
| **Feature Selection**           | Embedded method (XGBoost importance scores, threshold=0.033)                 |
| **Fitur Terpilih**              | age, sex, cp, restecg, thalach, exang, oldpeak, slope, ca, thal (10 dari 13) |
| **Class Balancing**             | SMOTE (diterapkan hanya dalam training fold, mencegah data leakage)          |
| **Hyperparameter Optimization** | Optuna + TPE Sampler (100 trials, target: ROC-AUC)                           |
| **Validasi**                    | Nested Stratified K-Fold Cross-Validation                                    |
| **Explainability**              | SHAP values + Partial Dependence Plots (PDP)                                 |

#### Hasil Utama (Test Set)

|    Metrik     |    Nilai    |
| :-----------: | :---------: |
| **Accuracy**  | **99.02%**  |
| **Precision** |   99.813%   |
|  **Recall**   | **100.00%** |
| **F1-Score**  |   99.05%    |
|  **ROC-AUC**  | **0.9998**  |

_Confusion Matrix: TP=105, TN=98, FP=2, FN=0 (Zero false negatives)_

#### Ablation Study (Incremental Pipeline Contribution)

| Konfigurasi                       | ROC-AUC  |  Recall  |
| :-------------------------------- | :------: | :------: |
| Baseline XGBoost (tanpa tambahan) |   0.90   |   0.82   |
| + SMOTE                           |   0.92   |   0.91   |
| + SMOTE + Feature Selection       |   0.95   |   0.93   |
| **Full Pipeline (proposed)**      | **0.95** | **0.97** |

#### SHAP Feature Importance

Fitur paling berpengaruh: **sex > ca > cp > thal > oldpeak**

#### Kelebihan & Kekurangan

| Kelebihan                                  | Kekurangan                                               |
| :----------------------------------------- | :------------------------------------------------------- |
| Pipeline modular dan dapat direproduksi    | Hanya divalidasi pada satu dataset (Kaggle)              |
| Nested CV mencegah bias estimasi           | Temporal patient data di luar scope                      |
| Ablation study terstruktur                 | Potensi optimism bias (dataset sudah dibersihkan Kaggle) |
| SHAP meningkatkan interpretabilitas klinis | Ukuran dataset relatif kecil (1.025 instance)            |

---

### 2. Paper Pendukung A

**"A Machine Learning-Based Framework for Heart Disease Diagnosis Using a Comprehensive Patient Cohort"**

> **Penulis:** Saadia Tabassum, Fazal Muhammad, Muhammad Ayaz Khan, et al.
> **Institusi:** University of Engineering & Technology, Mardan, Pakistan; Princess Nourah bint Abdulrahman University, Saudi Arabia
> **Published:** _Computers, Materials & Continua (CMC)_ (Scopus Q2, 2025) | Published: 09 Juni 2025

#### Tujuan Penelitian

Membandingkan performa 5 model ML (XGBoost, AdaBoost, Naïve Bayes, Logistic Regression, Bagging) dengan dan tanpa penghapusan outlier menggunakan **Isolation Forest**, untuk menentukan model terbaik dalam diagnosis penyakit jantung.

#### Dataset

- **Sumber:** Kaggle (`fedesoriano/heart-failure-prediction`) — gabungan 4 institusi
- **Ukuran:** 918 instance, 12 atribut
- **Missing Values:** 173 (pada kolom RestingBP: 1, Cholesterol: 172) → Diimputasi dengan mean
- **Split:** 70% training, 30% testing

#### Metodologi

```
Data Acquisition → Missing Value Imputation (Mean, per-kelas) → Outlier Detection (Isolation Forest)
→ Outlier Removal → Feature Scaling (Min-Max) → One-Hot Encoding → Model Training → Evaluasi
```

| Komponen                     | Detail                                                                    |
| :--------------------------- | :------------------------------------------------------------------------ |
| **Model Diuji**              | XGBoost, AdaBoost, Logistic Regression, Naïve Bayes, Bagging              |
| **Penanganan Missing Value** | Mean imputation terpisah untuk kelas sehat & sakit                        |
| **Deteksi Outlier**          | Isolation Forest (ditemukan pada: RestingBP=45, Cholesterol=45, MaxHR=46) |
| **Feature Scaling**          | Min-Max Normalization untuk: Age, RestingBP, Cholesterol, MaxHR, Oldpeak  |
| **Encoding**                 | One-Hot Encoding untuk variabel kategorikal (11 → 21 fitur)               |
| **Eksplorasi**               | 2 eksperimen: tanpa outlier removal vs. dengan outlier removal            |

#### Hasil Perbandingan (Experiment 2: Dengan Outlier Removal)

| Model               |  Accuracy  | Precision  |   Recall   |  F1-Score  | AUC-ROC  |
| :------------------ | :--------: | :--------: | :--------: | :--------: | :------: |
| Naïve Bayes         |   88.05%   |   95.06%   |   83.70%   |   89.02%   |   0.89   |
| AdaBoost            |   91.82%   |   91.58%   |   94.57%   |   93.05%   |   0.91   |
| Bagging             |   92.45%   |   92.55%   |   94.57%   |   93.55%   |   0.92   |
| Logistic Regression |   93.08%   |   94.51%   |   93.48%   |   93.99%   |   0.93   |
| **XGBoost**         | **94.34%** | **93.68%** | **96.74%** | **95.19%** | **0.94** |

#### Temuan Utama

- Penghapusan outlier via Isolation Forest meningkatkan akurasi XGBoost sebesar **+6.17%** (dari 88.17% → 94.34%).
- XGBoost secara konsisten unggul dalam semua metrik, terutama **Recall (96.74%)**.
- Naïve Bayes memiliki presisi tertinggi (95.06%) tetapi recall terendah (83.70%) → tidak ideal untuk diagnosis medis.

#### Kelebihan & Kekurangan

| Kelebihan                                                | Kekurangan                                                   |
| :------------------------------------------------------- | :----------------------------------------------------------- |
| Eksperimen terkontrol (with vs. without outlier removal) | Tidak menggunakan seleksi fitur                              |
| Evaluasi komprehensif dengan 5 metrik                    | Tidak ada hyperparameter tuning                              |
| Menggunakan Isolation Forest yang lebih canggih dari IQR | Dataset publik, tidak merepresentasikan variasi klinis nyata |
| Mengevaluasi Bagging Classifier (jarang diulas)          | Model Bagging kurang umum digunakan dalam konteks klinis     |

---

### 3. Paper Pendukung B

**"Sustainable and Interpretable Heart Disease Prediction: A Clinical Decision Support Approach"**

> **Penulis:** Tanzila Kehkashan, Maha Abdelhaq, Ahmad Sami Al-Shamayleh, et al.
> **Published:** _Scientific Reports (Nature Publishing Group)_ (Scopus Q1, 2026) | Published: 04 Februari 2026

#### Tujuan Penelitian

Mengembangkan model **1D Convolutional Neural Network (CNN)** yang mampu memprediksi penyakit jantung dengan akurasi tinggi sekaligus mempertahankan interpretabilitas melalui **LIME dan SHAP**, serta efisiensi komputasional untuk implementasi di fasilitas kesehatan dengan sumber daya terbatas.

#### Dataset

- **Sumber:** Kaggle (Cleveland Heart Disease — `johnsmith88/heart-disease-dataset`)
- **Ukuran:** 303 instance, 14 atribut (13 fitur + 1 target)
- **Distribusi:** 165 positif (54.5%), 138 negatif (45.5%)
- **Split:** 80% training (242 + augmentasi menjadi 484), 20% test (61 instance)

#### Arsitektur Model 1D CNN

```
Input (14 fitur) → Reshape (3D: samples×timesteps×features)
→ Conv1D (64 filters, kernel=3, ReLU) → Batch Normalization → MaxPooling (2)
→ Conv1D (128 filters, kernel=3, ReLU) → Batch Normalization → MaxPooling (2)
→ Flatten → Dense (128, ReLU) → Dropout (0.5) + L2 Regularization
→ Output Dense (2 units, Softmax)
```

| Komponen              | Detail                                   |
| :-------------------- | :--------------------------------------- |
| **Optimizer**         | Adam (learning_rate = 0.0005)            |
| **Loss Function**     | Categorical Cross-Entropy                |
| **Epochs**            | 50 (dengan Early Stopping, patience=15)  |
| **Batch Size**        | 32                                       |
| **Regularisasi**      | Dropout(0.5) + L2 + ReduceLROnPlateau    |
| **Data Augmentation** | Gaussian noise (σ=0.1) pada training set |
| **Explainability**    | LIME (tabular) + SHAP (KernelExplainer)  |

#### Ablation Study (Model Iterations)

| Iterasi             | Konfigurasi            |  Accuracy  |
| :------------------ | :--------------------- | :--------: |
| Model A (Initial)   | 2 Conv1D + 1 Dense     |   88.35%   |
| Model B (Depth++)   | 3 Conv1D + 2 Dense     |   89.10%   |
| Model C (+Dropout)  | +Dropout Layer         |   90.55%   |
| Model D (LR↓)       | LR = 0.001 → 0.0005    |   91.40%   |
| **Model E (Final)** | **+L2 Regularization** | **92.10%** |

#### Hasil Utama (Test Set, Model Final)

|        Metrik         |    Nilai    |
| :-------------------: | :---------: |
|     **Accuracy**      | **98.05%**  |
|     **Precision**     | **100.00%** |
|      **Recall**       |   96.12%    |
|     **F1-Score**      |   98.02%    |
|        **MCC**        |    0.963    |
| **Kappa Coefficient** |    0.961    |
|      **AUC-ROC**      |    0.99     |

_Confusion Matrix: TP=99, TN=102, FP=0, FN=4_

#### LIME & SHAP Analysis

Fitur terpenting menurut kedua metode interpretabilitas:

- **`sex`** (jenis kelamin) → faktor risiko terkuat
- **`ca`** (jumlah pembuluh darah utama yang diwarnai fluoroskopi)
- **`thal`** (thalassemia)
- **`oldpeak`** (depresi ST)
- **`cp`** (tipe nyeri dada) — memiliki efek paradoksal (berbeda antara LIME dan SHAP)

#### Kelebihan & Kekurangan

| Kelebihan                                         | Kekurangan                                             |
| :------------------------------------------------ | :----------------------------------------------------- |
| Interpretabilitas ganda (LIME + SHAP)             | Dataset kecil (303 instance)                           |
| Efisien secara komputasional (2-3 menit training) | Dari satu sumber (Cleveland, 1988)                     |
| Cocok untuk setting healthcare rendah sumber daya | Validasi prospektif klinis belum dilakukan             |
| Presisi 100% (zero false positive)                | Data augmentation berbasis Gaussian noise (artificial) |
| MCC & Kappa = metrik robust untuk imbalanced data | Bias demografis mungkin ada (data 1988)                |

---

### 4. Paper Pendukung C

**"A Systematic Review of ML Techniques for Cardiovascular Disease Prediction With Performance Analysis and Future Perspectives"**

> **Published:** _Cureus Journal of Computer Science_ (Scopus Q1, 2026)

#### Tujuan

Review sistematis berdasarkan **PRISMA guidelines** untuk merangkum, membandingkan, dan mengevaluasi berbagai teknik ML yang diterapkan pada prediksi CVD, serta mengidentifikasi gap penelitian dan arahan masa depan.

#### Temuan Utama dari Review

**Algoritma Terpopuler:**

- **Random Forest** & **XGBoost** → dominan dalam literature, akurasi 88–99%
- **Logistic Regression** → paling sering digunakan sebagai baseline
- **CNN & Deep Learning** → kinerja kompetitif dengan otomatisasi ekstraksi fitur

**Dataset Terpopuler:**

- Cleveland Heart Disease (UCI) — paling banyak dikutip
- Framingham Heart Study — dataset longitudinal terpanjang
- Kaggle (gabungan UCI) — semakin populer karena tidak ada missing values

**Tantangan Utama yang Diidentifikasi:**

1. **Imbalanced datasets** → SMOTE dan oversampling diperlukan
2. **Interpretabilitas model** → SHAP & LIME menjadi standar XAI
3. **Generalisasi lintas populasi** → perlunya multi-center validation
4. **Heterogenitas dataset** → sulit perbandingan langsung antar studi
5. **Kurangnya validasi klinis prospektif** → model belum teruji di workflow nyata

**Rekomendasi Masa Depan:**

- Federated Learning untuk privasi data klinis
- Integrasi data multimodal (EHR + ECG + genomik + wearable)
- Transformer architecture untuk CVD prediction
- Standarisasi benchmark dataset dan protokol evaluasi

---

## 📊 Analisis Perbandingan Lintas-Jurnal

### Perbandingan Metodologi

| Aspek                     |        Unified (Utama)         |  ML Framework (A)   |  CNN Interpretable (B)  | Systematic Review (C) |
| :------------------------ | :----------------------------: | :-----------------: | :---------------------: | :-------------------: |
| **Tipe Studi**            |           Eksperimen           |     Eksperimen      |       Eksperimen        |        Review         |
| **Dataset**               |         Kaggle (1,025)         |    Kaggle (918)     |     Cleveland (303)     |       Multiple        |
| **Algoritma Terbaik**     |            XGBoost             |       XGBoost       |         1D CNN          |      XGBoost/CNN      |
| **Feature Selection**     |     ✅ Embedded (XGBoost)      |      ❌ Tidak       | ❌ Tidak (otomatis CNN) |        Varies         |
| **Hyperparameter Tuning** |         ✅ Optuna/TPE          |      ❌ Tidak       |   ✅ Manual ablation    |        Varies         |
| **Imbalanced Handling**   |            ✅ SMOTE            | ❌ Tidak (balanced) | ✅ Gaussian augmentasi  |     SMOTE dominan     |
| **Outlier Handling**      | ❌ Tidak (Kaggle sudah bersih) | ✅ Isolation Forest |   ❌ Tidak disebutkan   |        Varies         |
| **Explainability (XAI)**  |         ✅ SHAP + PDP          |      ❌ Tidak       |     ✅ LIME + SHAP      |   Direkomendasikan    |
| **Cross-Validation**      |        ✅ Nested K-Fold        |   ❌ Single split   |     ❌ Single split     |        Varies         |

### Perbandingan Hasil (Metrik Terbaik)

| Paper                     | Model                      |  Accuracy  |  Recall  | F1-Score |    AUC     |
| :------------------------ | :------------------------- | :--------: | :------: | :------: | :--------: |
| **Unified (Utama)**       | XGBoost + Optuna + SMOTE   | **99.02%** | **100%** |  99.05%  | **0.9998** |
| **ML Framework (A)**      | XGBoost + Isolation Forest |   94.34%   |  96.74%  |  95.19%  |    0.94    |
| **CNN Interpretable (B)** | 1D CNN + LIME/SHAP         |   98.05%   |  96.12%  |  98.02%  |    0.99    |
| **Systematic Review (C)** | Various (best: XGBoost)    |  ~88–99%   | ~90–100% | ~89–99%  | 0.92–0.999 |

> **Kesimpulan Komparatif:** Paper Utama (_Unified_) mencapai akurasi tertinggi (99.02%) berkat kombinasi tiga teknik: embedded feature selection, SMOTE, dan Optuna tuning. Paper B (_CNN_) menonjol dalam hal **interpretabilitas klinis** melalui LIME+SHAP meski dataset lebih kecil.

---

## 🔗 Relevansi dengan Proyek Heart Disease EDA

### Pemetaan Kontribusi Jurnal ke Proyek

| Komponen Proyek                                              |    Referensi Jurnal    | Kontribusi                                                           |
| :----------------------------------------------------------- | :--------------------: | :------------------------------------------------------------------- |
| **Penggunaan dataset UCI (Cleveland, Hungarian, Swiss, VA)** |    Paper C (Review)    | Memvalidasi dataset sebagai yang paling banyak digunakan dalam riset |
| **Penggabungan 4 kohort menjadi satu dataset**               | Paper A (ML Framework) | Menunjukkan keunggulan kombinasi cohort: diversity, bias reduction   |
| **Penanganan missing values kolesterol = 0**                 |        Paper A         | Mean imputation per-kelas adalah praktek terbaik                     |
| **Seleksi 14 fitur dari 76 atribut asli**                    |        Paper C         | Cleveland 14-feature set adalah benchmark standar                    |
| **Target biner (0/1) dari `num` (0-4)**                      |      Semua paper       | Konversi ke biner adalah standar industri                            |
| **Pemilihan XGBoost sebagai model utama**                    |  Paper Utama + A + C   | XGBoost dominan dalam 3 dari 4 paper                                 |
| **Penggunaan SMOTE**                                         |    Paper Utama + C     | Standar untuk menangani class imbalance                              |
| **Optimasi Hyperparameter (Optuna)**                         |      Paper Utama       | TPE sampler terbukti superior vs Grid/Random Search                  |
| **SHAP untuk interpretabilitas**                             |    Paper Utama + B     | SHAP adalah standar XAI untuk model klinis                           |
| **Metrik evaluasi: Recall + F1 + AUC**                       |      Paper B + C       | Recall paling kritis dalam diagnosis medis                           |

---

## 💡 Sintesis & Rekomendasi untuk Pengembangan Model

Berdasarkan analisis lintas-jurnal, berikut rekomendasi langkah implementasi model untuk proyek ini:

### Pipeline yang Direkomendasikan

```
Dataset UCI (4 kohort) → Preprocessing (missing value handling, encoding, scaling)
→ Feature Selection (XGBoost importance) → SMOTE (fold-wise dalam CV)
→ Optuna + TPE Hyperparameter Tuning → XGBoost Classifier
→ Nested K-Fold CV → Evaluasi (Acc, Recall, F1, AUC)
→ SHAP Explainability → Clinical Insights
```

### Algoritma yang Direkomendasikan

1. **XGBoost + Optuna** ← _berdasarkan Paper Utama_ (target accuracy: >95%)
2. **Random Forest** ← sebagai baseline pembanding (_Paper A, C_)
3. **Logistic Regression** ← sebagai baseline interpretatif (_Paper A, C_)

### Metrik Evaluasi yang Diprioritaskan

1. **Recall** (Sensitivity) — paling penting dalam konteks medis (minimasi false negative)
2. **F1-Score** — keseimbangan precision-recall
3. **ROC-AUC** — diskriminasi keseluruhan
4. **Accuracy** — sebagai metrik pelengkap

---

_Dokumen ini disusun sebagai bagian dari bukti pendukung sertifikasi BNSP Data Scientist._
_Referensi lengkap tersedia di folder `journal/` dalam workspace proyek ini._
