# Analisis Jurnal — _Diabetes Prediction Framework on Pima Indians Dataset_

> **Role**: Senior Machine Learning Researcher & Scopus Journal Reviewer  
> **Dataset**: [Pima Indians Diabetes Database (Kaggle / UCI ML Repository)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
> **Paper**: Ali, A. A.; Galal, G. R.; Hassan, H. S. _Diabetes Prediction on Pima Indians Dataset Using Machine Learning Techniques_. **Minia University, Egypt (2025)**.  
> **Tanggal Analisis**: 12 Juli 2026

---

## 1. Ringkasan Executive

Paper ini membahas tentang rancangan pipeline machine learning yang aman dari kebocoran data (leakage-safe) untuk mendeteksi diabetes tipe 2 menggunakan dataset **Pima Indians Diabetes Database (PIDD)**. Kontribusi utama paper ini adalah menangani masalah umum data klinis yaitu adanya nilai nol tidak masuk akal (implausible zeros) pada fitur vital seperti Glucose, Blood Pressure, Skin Thickness, Insulin, dan BMI. Fitur-fitur tersebut diperlakukan sebagai missing values dan diimputasi menggunakan class-conditional medians. Selain itu, paper ini menguji efektivitas 16 fitur komposit buatan klinis untuk memperkuat sinyal prediksi, serta membandingkan performa 8 algoritma klasifikasi dengan model soft-voting ensemble (LightGBM + XGBoost) yang mencapai akurasi **89.61%** dan ROC-AUC **94.52%**.

---

## 2. Informasi Bibliografis

| Atribut          | Detail                                                                             |
| ---------------- | ---------------------------------------------------------------------------------- |
| **Judul**        | Diabetes Prediction on Pima Indians Dataset Using Machine Learning Techniques      |
| **Penulis**      | Abdelmgeid A. Ali¹, Galal R. Galal², Hassan S. Hassan³                             |
| **Afiliasi**     | Faculty of Computers and Information, Minia University, Minia 61519, Egypt         |
| **Tahun**        | 2025                                                                               |
| **Jurnal**       | International Journal of Electrical Systems and Information Technology (Scopus Q3) |
| **Status Akses** | Open Access                                                                        |

---

## 3. Metodologi & Arsitektur Model

Pipeline eksperimental yang diusulkan oleh penulis terdiri dari langkah-langkah terstruktur berikut:

1. **Pembagian Data (Stratified Split)**: Pembagian data 80% pelatihan (614 sampel) dan 20% pengujian (154 sampel) menggunakan stratified sampling untuk menjaga distribusi kelas target (Outcome: 0 = sehat, 1 = diabetes).
2. **Missing-Data Imputation**: Nilai nol tidak logis pada kolom `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` diganti dengan `NaN` dan diimputasi menggunakan median kelas. Untuk mencegah kebocoran data (_data leakage_), proses ini di-fit hanya menggunakan training set.
3. **Penyusunan 16 Fitur Komposit**: Engineering fitur baru yang mewakili kondisi klinis seperti indikator BMI sehat, normal insulin, serta rasio pregnancies terhadap usia.
4. **Standardisasi Fitur**: Dilakukan hanya untuk algoritma linier/jarak (Logistic Regression, SVM, k-NN) dengan fit StandardScaler pada training set saja.
5. **Ensemble Soft-Voting**: Memadukan probabilitas output dari dua model terbaik (LightGBM dan XGBoost) dengan bobot setara.

---

## 4. Evaluasi Kinerja & Temuan Kunci (Dari Jurnal)

### A. Kinerja Klasifikasi (Held-Out Test Set, n=154)

- **Akurasi**: 89.61%
- **ROC-AUC**: 94.52%
- **F1-Score**: 85.19%
- **Recall (Sensitivitas)**: 85.19%
- **Precision**: 85.19%

### B. Hasil Perbandingan Model (Single Models)

| Model                   | Test Accuracy | Test F1-Score | ROC-AUC    |
| ----------------------- | ------------- | ------------- | ---------- |
| **Ensemble (LGBM+XGB)** | **89.61%**    | **85.19%**    | **94.52%** |
| LightGBM                | 88.96%        | 84.11%        | 94.72%     |
| XGBoost                 | 88.31%        | 83.64%        | 94.63%     |
| Gradient Boosting       | 87.66%        | 82.24%        | 95.57%     |
| Random Forest           | 85.06%        | 78.90%        | 93.69%     |
| SVM                     | 83.77%        | 76.64%        | 90.05%     |
| k-NN                    | 83.12%        | 75.00%        | 86.62%     |
| Decision Tree           | 81.82%        | 75.44%        | 81.31%     |
| Logistic Regression     | 79.22%        | 69.81%        | 87.35%     |

---

## 5. Critical Review & Perspektif Reviewer Scopus

### Kekuatan Utama (Key Strengths)

1. **Aman dari Kebocoran Data**: Pipeline dirancang dengan sangat disiplin, di mana penggantian nilai nol, imputasi median, dan penskalaan data dilakukan terpisah antara training set dan test set.
2. **Rekayasa Fitur Klinis**: Mengkonstruksi 16 fitur komposit memberikan representasi yang lebih baik terhadap interaksi klinis tersembunyi dibandingkan hanya menggunakan 8 fitur mentah.
3. **Analisis Interoperabilitas (SHAP)**: Penggunaan SHAP memberikan wawasan mengenai kontribusi fitur secara global dan lokal yang sangat krusial bagi interpretasi klinis dokter.

### Keterbatasan & Ruang Evaluasi (Limitations & Weaknesses)

1. **Evaluasi pada Satu Split Data**: Uji performa akhir dilaporkan pada satu test split saja, tanpa cross-validation berulang yang dapat menimbulkan bias variansi split.
2. **Transportabilitas Terbatas**: Dataset Pima Indians berasal dari demografi tunggal (wanita suku Pima di Arizona). Generalisasi model pada ras dan kelompok usia lain masih memerlukan validasi eksternal.

---

## 6. Rekomendasi Roadmap Eksperimen Selanjutnya

1. **Gunakan Validasi Silang Berulang (Repeated K-Fold)** untuk memperoleh estimasi error yang lebih stabil.
2. **Terapkan Model Kalibrasi Probabilitas** (seperti Platt Scaling atau Isotonic Regression) untuk menjamin probabilitas prediksi mencerminkan probabilitas medis riil.
3. **Optimasi Berbasis Bayesian Optimization** untuk menyempurnakan hyperparameter ensemble XGBoost dan LightGBM.

---

## 7. Hasil Replikasi Eksperimental (Notebook: Diabetes_Prediction.ipynb)

Untuk menguji performa model secara terstandardisasi dan membandingkan hasil dengan teknik penataan data lainnya, kami mereplikasi eksperimen menggunakan **Model**: LightGBM, XGBoost, Gradient Boosting, dan Random Forest dengan KNN Imputation, 16 Composite Features, dan Leakage-Free Optuna Hyperparameter Tuning melalui 3 konfigurasi pipeline:

### A. Metodologi Eksperimen

1. **Eksperimen 1 — Split-First Pipeline**
    - **Alur**: `Split Data → Preprocessing → Training → Evaluation`
    - **Keterangan**: Dataset dibagi terlebih dahulu menjadi data latih (80%) dan data uji (20%). Preprocessing (menggunakan **KNN Imputer (k=5)** yang fit pada training saja + StandardScaler + 16 Fitur Komposit) dilakukan terpisah secara legal. Ini adalah **pipeline yang benar** bebas dari data leakage dan bebas dari target-conditional mismatch.
2. **Eksperimen 2 — Preprocess-First Pipeline**
    - **Alur**: `Preprocessing → Split Data → Training → Evaluation`
    - **Keterangan**: Preprocessing dengan class-conditional imputer yang dipandu label kelas (`y`) dan resampling SMOTE dilakukan pada seluruh dataset secara global sebelum pemisahan data train-test dilakukan. Ini memicu terjadinya **Data Leakage (Kebocoran Data) parah** dan mereplikasi metodologi bias dari jurnal acuan.
3. **Eksperimen 3 — Optimized Pipeline**
    - **Alur**: `Split Data → Preprocessing → Hyperparameter Tuning → Training → Evaluation`
    - **Keterangan**: Menggunakan alur Split-First yang aman dengan KNN Imputer (k=5), kemudian ditambahkan proses optimasi hyperparameter untuk XGBoost, LightGBM, dan Random Forest secara terpisah menggunakan **Optuna** (50 trials) pada data latih dengan Cross-Validation 5-fold untuk mencari kombinasi parameter terbaik secara legal.

### B. Perbandingan Hasil Evaluasi

| Model             | Exp1 Split-First Acc | Exp1 Split-First AUC | Exp2 Leakage Acc | Exp2 Leakage AUC | Exp3 Optimized Acc | Exp3 Optimized AUC |
| :---------------- | -------------------: | -------------------: | ---------------: | ---------------: | -----------------: | -----------------: |
| **LightGBM**      |               73.38% |               0.8019 |       **90.50%** |           0.9644 |         **75.32%** |         **0.8170** |
| XGBoost           |               72.08% |               0.8076 |           89.50% |           0.9595 |             72.08% |             0.8076 |
| Gradient Boosting |               72.08% |               0.8106 |           90.50% |       **0.9686** |             72.08% |             0.8106 |
| Random Forest     |               69.48% |               0.7931 |           91.00% |           0.9684 |             69.48% |             0.7931 |

### C. Temuan Kunci & Analisis Kritis

1. **Analisis Kebocoran Data (Eksperimen 2)**:
   Terlihat lonjakan performa yang sangat drastis pada seluruh model di Eksperimen 2 (misalnya LightGBM Akurasi **90.50%** dan ROC-AUC **0.9644**, Random Forest mencapai akurasi **91.00%**). Hal ini disebabkan oleh masuknya informasi dari data uji melalui imputasi bersyarat kelas target dan SMOTE secara global. Ini mendemonstrasikan secara nyata mengapa hasil tinggi pada beberapa paper sering kali bersifat bias dan semu.
2. **Peningkatan Performa Tanpa Mismatch (Eksperimen 3)**:
   Dengan beralih dari class-conditional imputer ke **KNN Imputer (k=5)** dan mengoptimalkan parameter model dengan **Optuna (50 trials)** secara sah (hanya pada fold training), performa LightGBM mencapai akurasi **75.32%** dan ROC-AUC **0.8170**. Hal ini menunjukkan performa sesungguhnya (riil) dari model saat ditangani tanpa data leakage.
