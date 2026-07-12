# Analisis Jurnal — _Logistic Regression Framework untuk Klasifikasi Tumor Payudara_

> **Role**: Senior Machine Learning Researcher & Scopus Journal Reviewer  
> **Dataset**: [Breast Cancer Wisconsin (Diagnostic) (UCI ML Repository)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
> **Paper**: Cheng, W.; Yu, Z. _A Fast and Interpretable Logistic Regression Framework for Breast Tumor Classification Using the Wisconsin Diagnostic Dataset_. **medRxiv Preprint 2025**. https://doi.org/10.64898/2025.12.23.25342946  
> **Tanggal Analisis**: 12 Juli 2026

---

## 1. Ringkasan Eksekutif

Paper ini membahas tentang rancangan pipeline machine learning yang sederhana, cepat, dan terinterpretasi (interpretable) untuk klasifikasi tumor payudara (jinak vs ganas) menggunakan dataset **Breast Cancer Wisconsin (Diagnostic) / WDBC**. Kontribusi utama dari paper ini adalah menunjukkan bahwa model linier klasik seperti **Logistic Regression (LR)**, jika dipasangkan dengan standarisasi fitur yang tepat, mampu bersaing ketat dengan model yang lebih kompleks (seperti SVM, Random Forest, dan k-NN) dalam hal akurasi diagnostik, sembari menawarkan transparansi koefisien yang krusial untuk aplikasi medis.

---

## 2. Informasi Bibliografis

| Atribut          | Detail                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Judul**        | A Fast and Interpretable Logistic Regression Framework for Breast Tumor Classification Using the Wisconsin Diagnostic Dataset |
| **Penulis**      | Weihao Cheng¹²\*, Zekai Yu¹                                                                                                   |
| **Afiliasi 1**   | School of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, China                                        |
| **Afiliasi 2**   | School of Communication Engineering, Hangzhou Dianzi University, Hangzhou, China                                              |
| **Preprint**     | medRxiv (Desember 2025)                                                                                                       |
| **DOI**          | 10.64898/2025.12.23.25342946                                                                                                  |
| **Status Akses** | Open Access (CC BY 4.0)                                                                                                       |

---

## 3. Metodologi & Arsitektur Model

Pipeline eksperimental yang diusulkan oleh penulis terdiri dari langkah-langkah terstruktur berikut:

1. **Pembagian Data (Stratified Split)**: Pembagian data 80% pelatihan (455 sampel) dan 20% pengujian (114 sampel) menggunakan metode stratified sampling untuk menjaga distribusi kelas tumor (ganas vs jinak) tetap proporsional.
2. **Standardisasi Fitur (Z-score Normalization)**: Standardisasi rata-rata nol dan varians unit diaplikasikan pada 30 fitur kontinu untuk memastikan konvergensi model LR dan interpretabilitas koefisien. Untuk mencegah kebocoran informasi (information leakage), scaler disesuaikan (fitted) hanya pada data pelatihan.
3. **Pelatihan Model**: Penerapan model Regresi Logistik dengan regularisasi L2 (Ridge) untuk mengendalikan overfitting dan meningkatkan stabilitas koefisien di ruang fitur yang saling berkorelasi.
4. **Pembandingan Model (Benchmarking)**: Penulis membandingkan kinerja Logistic Regression dengan tiga model baseline:
    - SVM dengan kernel Radial Basis Function (RBF)
    - Random Forest (RF)
    - k-Nearest Neighbors (k-NN)

---

## 4. Evaluasi Kinerja & Temuan Kunci

### A. Kinerja Klasifikasi (Held-Out Test Set, n=114)

Hasil eksperimen menunjukkan performa yang sangat kuat:

- **Akurasi**: 98,25%
- **ROC-AUC**: 0.9954
- **Sensitivitas (Malignant Recall)**: 97,62% (41 dari 42 kasus ganas berhasil dideteksi)
- **Spesifisitas**: 98,61% (71 dari 72 kasus jinak berhasil diidentifikasi)
- **Kesalahan Klasifikasi**: Hanya 2 sampel yang salah diklasifikasikan (1 tumor ganas diprediksi jinak, 1 tumor jinak diprediksi ganas).

### B. Hasil Perbandingan Model (Benchmarking)

| Model                   | Akurasi    | ROC-AUC    |
| ----------------------- | ---------- | ---------- |
| **Logistic Regression** | **0.9825** | **0.9954** |
| SVM (RBF)               | 0.9825     | 0.9950     |
| Random Forest           | 0.9561     | 0.9931     |
| k-NN                    | 0.9561     | 0.9788     |

_Temuan Utama_: Kompleksitas model yang lebih tinggi (seperti SVM RBF atau Random Forest) tidak memberikan peningkatan kinerja yang signifikan pada dataset WDBC. Hal ini mengindikasikan bahwa fitur-fitur morfologi nuklir dalam dataset ini sudah hampir dapat dipisahkan secara linier (linearly separable) setelah standarisasi.

### C. Interpretasi Fitur (Feature Importance)

Melalui analisis koefisien regresi terstandardisasi, paper ini mengidentifikasi 10 fitur paling berpengaruh dalam klasifikasi. Sebagian besar fitur ini terkait dengan pengukuran nilai ekstrem ("worst" measurements) dari ukuran dan ketidakteraturan batas inti sel:

1. **worst texture** (Koefisien: -1.255)
2. **radius error** (Koefisien: -1.083)
3. **worst concavity** (Koefisien: -0.954)
4. **worst area** (Koefisien: -0.948)
5. **worst radius** (Koefisien: -0.948)
6. **worst symmetry** (Koefisien: -0.939)
7. **area error** (Koefisien: -0.929)
8. **worst concave points** (Koefisien: -0.823)
9. **worst perimeter** (Koefisien: -0.763)
10. **worst smoothness** (Koefisien: -0.747)

_Kesesuaian Medis_: Hasil pemeringkatan fitur ini sangat sejalan dengan pengetahuan patologi klinis, di mana tumor ganas ditandai dengan variabilitas bentuk nuklir yang ekstrem, pembesaran ukuran nuklir, dan ketidakteraturan membran sel.

---

## 5. Critical Review & Perspektif Reviewer Scopus

### Kekuatan Utama (Key Strengths)

1. **Prinsip Parsimoni (Occam's Razor)**: Penulis berhasil membuktikan bahwa model yang lebih sederhana (Logistic Regression) sering kali lebih disukai daripada model kompleks karena menawarkan performa yang setara namun jauh lebih mudah diinterpretasikan secara klinis.
2. **Metodologi yang Disiplin**: Pembagian data menggunakan stratified split dan pencegahan kebocoran data (fitting scaler hanya pada training set) dilakukan dengan sangat ketat untuk menghindari bias.
3. **Aplikasi Medis yang Nyata**: Penilaian interpretabilitas koefisien secara langsung memberikan alasan biologis yang masuk akal di balik keputusan model.

### Keterbatasan & Ruang Evaluasi (Limitations & Weaknesses)

1. **Evaluasi pada Satu Split Data saja**: Penggunaan satu pembagian train-test split tanpa K-Fold cross-validation rentan terhadap bias keberuntungan split data.
2. **Skalabilitas & Generalisasi**: Dataset Wisconsin (WDBC) berukuran kecil (569 sampel) dan berasal dari satu pusat medis tunggal. Generalisasi model pada data eksternal dari berbagai rumah sakit dengan demografi berbeda belum teruji.
3. **Ketiadaan Kalibrasi Probabilitas**: Model tidak dilengkapi dengan kalibrasi probabilitas (seperti Platt scaling), padahal estimasi probabilitas yang akurat sangat krusial dalam pengambilan keputusan klinis medis.

---

## 6. Rekomendasi Roadmap Eksperimen Selanjutnya

Bagi Anda yang ingin mengembangkan riset ini lebih lanjut, berikut adalah langkah yang disarankan:

1. **Terapkan K-Fold Cross-Validation**: Gunakan minimal 5-Fold atau 10-Fold Cross-Validation berulang untuk mendapatkan estimasi performa yang lebih stabil dan tepercaya.
2. **Lakukan Kalibrasi Probabilitas**: Uji dan visualisasikan diagram keandalan (reliability diagram) serta gunakan teknik kalibrasi probabilitas untuk memastikan output probabilitas model mencerminkan risiko klinis yang sebenarnya.
3. **Eksplorasi Metode Seleksi Fitur**: Uji efektivitas pengurangan dimensi (seperti PCA) atau seleksi fitur (seperti RFE atau Lasso L1 penalty) untuk membandingkan apakah penyusutan jumlah fitur dapat mempertahankan akurasi sekaligus meningkatkan interpretabilitas model.

---

## 7. Hasil Replikasi Eksperimental (Notebook: Breast_Cancer.ipynb)

Untuk menguji performa model secara terstandardisasi dan membandingkan hasil dengan teknik penataan data lainnya, kami mereplikasi eksperimen menggunakan model **Logistic Regression (L2 Regularized)** melalui 3 konfigurasi pipeline:

### A. Metodologi Eksperimen

1. **Eksperimen 1 — Split-First Pipeline**
   - **Alur**: `Split Data → Preprocessing → Training → Evaluation`
   - **Keterangan**: Dataset dibagi terlebih dahulu menjadi data latih (80%) dan data uji (20%). Preprocessing (StandardScaler) di-*fit* hanya menggunakan data latih dan diterapkan pada data uji untuk mencegah kebocoran informasi. Ini adalah **pipeline yang benar** bebas dari data leakage.
2. **Eksperimen 2 — Preprocess-First Pipeline**
   - **Alur**: `Preprocessing → Split Data → Training → Evaluation`
   - **Keterangan**: StandardScaler dilakukan pada seluruh dataset sebelum data dibagi menjadi data latih dan data uji. Ini memicu terjadinya **Data Leakage (Kebocoran Data)**.
3. **Eksperimen 3 — Optimized Pipeline**
   - **Alur**: `Split Data → Preprocessing → Hyperparameter Tuning → Training → Evaluation`
   - **Keterangan**: Menggunakan alur Split-First yang aman seperti Eksperimen 1, kemudian ditambahkan proses optimasi parameter menggunakan **Grid Search** pada data latih dengan Cross-Validation 5-fold untuk mencari parameter regularisasi $C$ terbaik.

### B. Perbandingan Hasil Evaluasi

| Metrik Evaluasi | Eksperimen 1 (Split-First) | Eksperimen 2 (Preprocess-First) | Eksperimen 3 (Optimized) |
| :--- | :---: | :---: | :---: |
| **Accuracy (Akurasi)** | 98.25% | 98.25% | 97.37% |
| **ROC-AUC** | 0.9954 | 0.9954 | 0.9947 |
| **Sensitivity (Malignant Recall)** | 97.62% | 97.62% | 95.24% |
| **Specificity (Benign Recall)** | 98.61% | 98.61% | 98.61% |
| **Salah Klasifikasi (Error)** | 2 sampel | 2 sampel | 3 sampel |

### C. Temuan Kunci & Analisis Kritis

1. **Analisis Kebocoran Data (Eksperimen 2)**:
   Pada dataset yang bersih dan teratur ini, kebocoran data tidak mengubah hasil akurasi uji secara numerik (keduanya tetap 98.25%). Namun, secara metodologi, Eksperimen 2 adalah **salah** karena mean/std deviasi dari data uji ikut mempengaruhi standardisasi data latih. Hal ini terbukti merusak validitas generalisasi pada dataset yang lebih kompleks/kotor.
2. **Optimasi & Regularisasi (Eksperimen 3)**:
   Grid Search dengan validasi silang 5-fold mendeteksi parameter optimal pada $C=0.1$ dengan skor validasi silang sebesar **98.02%**. Model dengan $C=0.1$ lebih terregularisasi (lebih sederhana) untuk meminimalisasi overfitting lintas lipatan. Pada test set tunggal ini, akurasi uji tercatat sebesar **97.37%** (2 tumor ganas terlewat), merepresentasikan trade-off bias-variance yang objektif.
3. **Penyelarasan Patologis**:
   Koefisien terstandardisasi menunjukkan bahwa **texture_worst**, **radius_se**, **concave points_worst**, **area_worst**, dan **radius_worst** merupakan fitur penentu utama. Nilai negatif yang tinggi pada koefisien ini (karena target model Benign=1, Malignant=0) menunjukkan bahwa peningkatan ukuran sel dan ketidakteraturan morfologi nuklir berkorelasi kuat dengan keganasan tumor (Malignant).

