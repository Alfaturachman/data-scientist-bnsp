# Proyek Skrining Risiko Stroke Medis (Posbindu PTM Kota Semarang)

## Replikasi & Audit Kritis Jurnal Scopus Q1 (*Frontiers in Neurology*, 2026)
**Bukti Kelayakan Uji Sertifikasi Kompetensi BNSP - Skema Ilmuwan Data (Data Scientist)**

---

## 📌 Informasi Peserta & Sertifikasi

- **Nama**: Alfaturachman Maulana Pahlevi
- **NIM**: A11.2025.16609
- **Skema Sertifikasi**: Ilmuwan Data (Data Scientist)
- **Lembaga Sertifikasi**: Badan Nasional Sertifikasi Profesi (BNSP)
- **Tahun**: 2026

---

## 📋 Ringkasan Eksekutif Proyek

Proyek ini bertujuan untuk membangun model machine learning terinterpretasikan (_explainable AI_) untuk **skrining risiko stroke medis** pada layanan kesehatan primer (Posbindu PTM & Puskesmas Kota Semarang). 

Eksperimen dieksekusi secara ketat menggunakan **10.000 sampel rekam medis klinis** dengan menerapkan alur baku **CRISP-DM** dan **11 Unit SKKNI Data Scientist**. Eksperimen ini memprioritaskan keamanan data (_0% Data Leakage Guarantee_) serta melakukan audit kritis terhadap metodologi paper acuan internasional.

### 🎯 Highlight Pencapaian Utama:
1. **Model Terbaik**: **Logistic Regression Pipeline** meraih **Recall Skrining 76,51%**, **F1-Score 0,6538**, dan **PR-AUC 0,7245** pada *held-out test set* 2.000 sampel.
2. **Keunggulan Atas Paper Acuan**: Audit Confusion Matrix terhadap paper acuan Tang et al. (*Frontiers in Neurology*, Scopus Q1 2026) membuktikan F1-Score sejati paper acuan sebenarnya hanyalah **0,5861** (bukan 0,74 sebagaimana diklaim akibat kebocoran data). Model eksperimen ini terbukti **+11,55% lebih unggul secara valid**.
3. **0% Data Leakage Guarantee**: Pemisahan *Stratified 80:20 Holdout Split* dilakukan di tahap paling awal sebelum preprocessing/SMOTE dengan mengintegrasikan `ColumnTransformer` dan `ImbPipeline`.
4. **Interpretabilitas Klinis (SHAP)**: Analisis SHAP TreeExplainer mengidentifikasi *Hipertensi* (mean \|SHAP\| 0,9087) dan *Penyakit Jantung* (0,4574) sebagai faktor risiko stroke paling dominan.
5. **Rekomendasi Threshold Operasional**: Penetapan threshold medis fleksibel: **0.35** (Skrining Massal Posbindu, Recall >85%), **0.50** (Standar Puskesmas, F1 Max 0.6538), dan **0.65** (Rujukan Spesialis, Presisi >75%).

---

## 📊 Perbandingan Hasil Performa 5 Algoritma Machine Learning

Evaluasi dilakukan secara jujur dan transparan pada **2.000 sampel data uji independen** (_Held-Out Test Set_):

| Algoritma Model | Akurasi | Presisi | Recall (Sensitivitas) | F1-Score | PR-AUC | Brier Score | Status / Peran Klinis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🏆 **Logistic Regression** | 0,7585 | 0,5707 | **0,7651** | **0,6538** | **0,7245** | 0,1570 | **Juara Umum (Recall & F1 Medis Best)** |
| 🥇 **LightGBM Classifier** | **0,7710** | **0,6012** | 0,6879 | 0,6416 | 0,7184 | 0,1549 | **Akurasi & Presisi Tertinggi** |
| 🥈 **XGBoost Classifier** | 0,7685 | 0,5927 | 0,7131 | 0,6474 | 0,7127 | **0,1530** | Brier Score / Kalibrasi Terbaik |
| 🥉 **Random Forest** | 0,7695 | 0,5968 | 0,6980 | 0,6435 | 0,7084 | 0,1533 | Model Ensemble Klasik |
| 🎗️ **MLP (Deep Learning)** | 0,7475 | 0,5601 | 0,7114 | 0,6268 | 0,6813 | 0,1668 | Multi-Layer Perceptron |

---

## 🧐 Audit Kritis Paper Acuan (Frontiers in Neurology, 2026)

- **Judul Paper Acuan**: *"Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation"* (Tang et al., *Frontiers in Neurology*, 2026).
- **Temuan Leakage**: Paper acuan mengklaim pencapaian F1-Score 0,74. Namun, re-kalkulasi Confusion Matrix pada data uji terisolasi membuktikan bahwa F1-Score sejati paper acuan adalah **0,5861**. Nilai tinggi yang diklaim paper acuan dipicu oleh *Data Leakage* (penerapan imputasi & SMOTE secara global sebelum pembagian train-test).
- **Pembuktian Eksperimen Ini**: Dengan menerapkan `ImbPipeline` yang mengisolasi SMOTE dan Scaling hanya di dalam *training fold* (0% Leakage), model **Logistic Regression Pipeline** berhasil mencapai **F1-Score 0,6538**, membuktikan keunggulan murni **+11,55%** secara sah.

---

## 🩺 Interpretabilitas SHAP & Rekomendasi Threshold Medis

### 1. Top 5 Fitur Prediktor Stroke (SHAP Mean |SHAP| Value)
1. **Hipertensi (`hypertension`)**: `0,9087` (Faktor risiko paling signifikan mendongkrak probabilitas stroke).
2. **Penyakit Jantung (`heart_disease`)**: `0,4574` (Meningkatkan risiko secara signifikan).
3. **Diabetes (`diabetes`)**: `0,3244` (Prediktor komplikasi metabolik utama).
4. **Usia (`age`)**: `0,2896` (Risiko meningkat secara eksponensial seiring bertambahnya usia).
5. **Indeks Massa Tubuh (`bmi`)**: `0,1084` (Indikator obesitas klinis).

### 2. Panduan Threshold Operasional Posbindu & Puskesmas
- **Threshold 0.35 (Skrining Massal Posbindu)**: Mengoptimalkan **Recall >85%** untuk menjaring sebanyak mungkin warga berisiko stroke tanpa ada kasus terlewat (*minimalkan False Negative*).
- **Threshold 0.50 (Standar Puskesmas)**: Keseimbangan optimal **F1 Max (0,6538)** dan Recall **76,51%** untuk penanganan standar dokter umum.
- **Threshold 0.65 (Rujukan Spesialis)**: Mengoptimalkan **Presisi >75%** untuk konfirmasi rujukan cepat ke dokter spesialis saraf/rumah sakit.

---

## 🗺️ Pemetaan 11 Unit Kompetensi SKKNI Data Scientist (BNSP)

Eksperimen ini memenuhi 11 unit kompetensi SKKNI Data Science secara penuh:

| Kode Unit SKKNI | Judul Unit Kompetensi | Bukti Implemetasi dalam Notebook & Pipeline |
| :--- | :--- | :--- |
| **J.62DMI00.001.1** | Menentukan Objektif Bisnis | Formulasi objektif skrining cepat Posbindu PTM Kota Semarang & rujukan presisi Puskesmas. Penekanan Recall medis. |
| **J.62DMI00.002.1** | Menentukan Tujuan Teknis | Pendefinisian metrik evaluasi F1-Score, Recall, dan PR-AUC (menghindari jebakan akurasi naif pada imbalanced data). |
| **J.62DMI00.005.1** | Menelaah Data (EDA) | EDA 10.000 sampel rekam medis Posbindu, analisis sebaran *Class Imbalance* 29,78% kejadian stroke. |
| **J.62DMI00.006.1** | Memvalidasi Data | Pengujian konsistensi klinis, persentase missing BMI 2,01%, verifikasi 0 duplikasi data. |
| **J.62DMI00.007.1** | Menentukan Objek Data | Pemisahan variabel prediktor $X$ dan target $y$, keputusan etis melakukan *drop* variabel SES (*Socioeconomic Status*). |
| **J.62DMI00.008.1** | Membersihkan Data | Imputasi median BMI terisolasi di dalam `ColumnTransformer` (terpisah aman per-fold). |
| **J.62DMI00.009.1** | Mengkonstruksi Data | Transformasi fitur numerik (`StandardScaler`) & kategorikal (`OneHotEncoder`) terbungkus aman dalam pipeline. |
| **J.62DMI00.012.1** | Membangun Skenario Model | **Stratified 80:20 Train-Test Split** dieksekusi di tahap awal sebelum preprocessing untuk menjamin **0% Data Leakage**. |
| **J.62DMI00.013.1** | Membangun Model | Pelatihan 5 algoritma (Logistic Reg, Random Forest, XGBoost, LightGBM, MLP) via `ImbPipeline` + SMOTE + 5-Fold Stratified CV + `GridSearchCV`. |
| **J.62DMI00.014.1** | Mengevaluasi Hasil Pemodelan | Evaluasi 5 model pada *test set* 2.000 sampel menggunakan 7 metrik (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, Brier Score). |
| **J.62DMI00.015.1** | Melakukan Review Pemodelan | Analisis *SHAP TreeExplainer*, audit Confusion Matrix paper acuan Tang et al. (2026), dan penyusunan rekomendasi threshold klinis Posbindu. |

---

## 📚 Dataset & Jurnal Acuan

1. **Dataset**: Rekam Medis Klinis Skrining Stroke (10.000 sampel × 10 fitur), bersumber dari Kaggle (*Stroke Diagnosis and Health Metrics Data* oleh Shriyash Jagtap).
2. **Jurnal Acuan**: Tang et al. (2026), *"Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation"*, *Frontiers in Neurology* (Scopus Q1).

---

## 📁 Berkas Proyek

- 📓 **`Stroke-Prediction.ipynb`**: Notebook Jupyter Lengkap (Alur CRISP-DM, 11 Unit SKKNI, Modeling & Evaluation).
- 📊 **`stroke.csv`**: Dataset Rekam Medis Klinis 10.000 Sampel.
- 📄 **`readme.md`**: Ringkasan Dokumentasi Proyek (Berkas ini).