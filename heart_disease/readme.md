# Portofolio Uji Sertifikasi Data Scientist (BNSP)

## Proyek: Analisis & Prediksi Penyakit Jantung (Heart Disease Prediction)

Dokumentasi ini disusun untuk memenuhi bukti pendukung (evidence) dalam **Uji Sertifikasi Kompetensi Skema Data Scientist**. Proyek ini menggunakan dataset komprehensif **Heart Disease** dari **UCI Machine Learning Repository** (Cleveland, Hungarian, Switzerland, dan VA Long Beach).

---

## 📂 Struktur Proyek & Berkas

Berikut adalah struktur folder dan berkas pendukung dalam workspace ini:

```text
d:/project sertifikasi/
│
├── readme.md                   <- Dokumentasi utama & pemetaan kompetensi (berkas ini)
├── analisis_jurnal.md          <- Analisis lintas-jurnal 4 paper pendukung
├── Heart_Disease_EDA.ipynb     <- Notebook Jupyter: EDA, Preprocessing, Modeling & Evaluasi
├── journal/                    <- Folder berkas jurnal pendukung (format .md)
│   ├── Unified approach...md   <- Paper Landasan Utama (Scopus Q2, 2026)
│   ├── A MachineLearning-Based Framework...md  <- Paper Pendukung A (Scopus Q2, 2025)
│   ├── Sustainable and interpretable...md      <- Paper Pendukung B (Scopus Q1, 2026)
│   └── A Systematic Review...md               <- Paper Pendukung C (Scopus Q1, 2026)
└── heart_disease/              <- Folder dataset mentah dari UCI
    ├── processed.cleveland.data
    ├── processed.hungarian.data
    ├── processed.switzerland.data
    ├── processed.va.data
    └── heart-disease.names
```

---

## 📊 Pemetaan Unit Kompetensi (Bukti Pendukung)

Tabel berikut memetakan setiap **Unit Kompetensi (UK)** dan **Kriteria Unjuk Kerja (KUK)** ke bukti pendukung yang ada di dalam berkas proyek Anda:

| Kode Unit           | Judul Unit Kompetensi                     | Kriteria Unjuk Kerja (KUK)                                                                                                                      | Deskripsi & Lokasi Bukti Pendukung                                                                                                                                                                                                                 | Status |
| :------------------ | :---------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: |
| **J.62DMI00.001.1** | **Menentukan Objektif Bisnis**            | K1: Identifikasi latar belakang bisnis<br>K2: Identifikasi objektif bisnis<br>K3: Membuat metrik kesuksesan<br>K4: Menetapkan objektif & metrik | **`Heart_Disease_EDA.ipynb` (Cell 1 & 2)**<br>Dokumentasi di bagian awal notebook merinci latar belakang medis, objektif diagnosis, dan metrik akurasi target klasifikasi.                                                                         | **K**  |
|                     |                                           | K1-K2: Daftar terminologi & batasan<br>K1-K3: Risiko & rencana mitigasi<br>K1-K3: Biaya & keuntungan                                            | **`Heart_Disease_EDA.ipynb` (Cell 2 & Lampiran)**<br>Daftar 14 fitur (terminologi), asumsi data klinis (batasan), serta mitigasi data kosong (_missing values_).                                                                                   | **K**  |
| **J.62DMI00.002.1** | **Menentukan Tujuan Teknis Data Science** | K1-K2: Menentukan task & tujuan teknis<br>K1-K2: Kriteria kesuksesan teknis                                                                     | **`Heart_Disease_EDA.ipynb` (Cell 2)**<br>Tujuan teknis didefinisikan sebagai klasifikasi biner menggunakan metrik evaluasi seperti Akurasi, F1-Score, dan Recall.                                                                                 | **K**  |
| **J.62DMI00.005.1** | **Menelaah Data**                         | K1: Identifikasi tipe data<br>K2: Uraian nilai atribut<br>K3: Relasi antar data                                                                 | **`Heart_Disease_EDA.ipynb` (Cell 3 & 5)**<br>Memuat tipe data gabungan, deskripsi atribut medis (14 fitur), dan pengecekan korelasi fitur.                                                                                                        | **K**  |
|                     |                                           | K1-K3: Deskripsi statistik & visualisasi                                                                                                        | **`Heart_Disease_EDA.ipynb` (Cell 7, 8, 9, 10, 11, 12, 13)**<br>Penyajian tabel statistik dasar (`.describe()`), grafik countplot, KDE plot sebaran usia/klinis, dan matriks korelasi heatmap.                                                     | **K**  |
|                     |                                           | K1-K2: Laporan telaah & hipotesis                                                                                                               | **`Heart_Disease_EDA.ipynb` (Cell 14 & Kesimpulan)**<br>Hipotesis mengenai faktor usia, jenis kelamin (sex), dan tipe nyeri dada (cp = 4) dituliskan secara deskriptif.                                                                            | **K**  |
| **J.62DMI00.006.1** | **Memvalidasi Data**                      | K1-K2: Kualitas & kecukupan data<br>K1-K2: Rekomendasi kelengkapan data                                                                         | **`Heart_Disease_EDA.ipynb` (Cell 6 & 8)**<br>Analisis persentase missing values untuk setiap kolom dan rekomendasi imputasi berbasis statistik.                                                                                                   | **K**  |
| **J.62DMI00.007.1** | **Menentukan Objek Data**                 | K1-K2: Kriteria & teknik pemilahan data<br>K1-K2: Penentuan baris & kolom                                                                       | **`Heart_Disease_EDA.ipynb` (Cell 5 & 10)**<br>Pemilihan 14 fitur dari 76 atribut asli berdasarkan kesepakatan penelitian medis terdahulu, serta penggabungan 4 cohort regional.                                                                   | **K**  |
| **J.62DMI00.008.1** | **Membersihkan Data**                     | K1-K2: Strategi & koreksi data kotor<br>K1-K3: Laporan, evaluasi & dokumentasi                                                                  | **`Heart_Disease_EDA.ipynb` (Cell 8, 9 & 10)**<br>**Koreksi Anomali Kolesterol:** Mengidentifikasi kolesterol bernilai 0 pada dataset Swiss/VA sebagai _missing value_ dan mengubahnya menjadi `NaN`, kemudian mengimputasinya menggunakan median. | **K**  |
| **J.62DMI00.009.1** | **Mengkonstruksi Data**                   | K1-K2: Representasi fitur & rekayasa fitur<br>K1-K2: Dokumentasi transformasi data                                                              | **`Heart_Disease_EDA.ipynb` (Cell 10 & 12)**<br>Konversi variabel target `num` (0-4) menjadi target klasifikasi biner (0 = Sehat, 1 = Sakit Jantung).                                                                                              | **K**  |
| **J.62DMI00.012.1** | **Membangun Skenario Model**              | K1-K2: Asumsi data & teknik pemodelan<br>K1-K3: Skenario & metrik pengujian                                                                     | **`Heart_Disease_EDA.ipynb` (Sec. 4 & 4.4)**<br>**Skenario Komparatif:** Klasifikasi biner dengan 5 model kandidat: **XGBoost**, **AdaBoost**, **Bagging**, **Logistic Regression**, dan **Naïve Bayes**. Data dibagi **70% training / 30% testing (stratified)**. <br>**2 Skenario Pengujian:** (1) Eksperimen 1: Tanpa Outlier Removal, (2) Eksperimen 2: Dengan Outlier Removal via Isolation Forest. Metrik sukses: Accuracy ≥ 80%, **Recall ≥ 82%** (prioritas klinis), F1-Score ≥ 82%, AUC ≥ 0.85. _(Referensi: Tabassum et al. 2025)_ | **K**  |
| **J.62DMI00.013.1** | **Membangun Model**                       | K1-K2: Parameter & batas toleransi model<br>K1-K4: Eksplorasi tools & optimasi model                                                            | **`Heart_Disease_EDA.ipynb` (Sec. 4.1, 4.2, 4.3, 4.4)**<br>Pipeline lengkap: **(1) One-Hot Encoding** pada fitur kategorikal multikelas (`cp`, `restecg`, `slope`, `ca`, `thal`) → **(2) Train/Test Split (70:30 stratified)** → **(3) Deteksi Outlier** via **Isolation Forest** (contamination=3%) hanya pada fitur numerik kontinu → **(4) Min-Max Scaling** ke rentang `[0, 1]` → **(5) Training 5 Model**. Library: `scikit-learn`, `xgboost`. _(Referensi: Tabassum et al. 2025)_ | **K**  |
| **J.62DMI00.014.1** | **Mengevaluasi Hasil Pemodelan**          | K1-K2: Pengujian data riil<br>K1-K2: Penilaian output vs metrik                                                                                 | **`Heart_Disease_EDA.ipynb` (Sec. 4.5)**<br>Evaluasi pada test set (30% data): **Tabel Perbandingan Metrik** (Accuracy, Precision, Recall, F1, AUC-ROC), **Grouped Bar Chart** perbandingan kedua eksperimen, **Confusion Matrix** (TP/TN/FP/FN), dan **ROC-AUC Curve** untuk XGBoost (model terbaik). **Prioritas Recall** mengikuti standar diagnosis medis guna meminimalkan false negative. _(Referensi: Tabassum et al. 2025)_ | **K**  |
| **J.62DMI00.015.1** | **Melakukan Proses Review Pemodelan**     | K1-K2: Kesesuaian proses & tindak lanjut<br>K1-K2: Kualitas proses berdasarkan rencana                                                          | **`Heart_Disease_EDA.ipynb` (Sec. 4.6)**<br>Review kualitas model melalui: **(1) Perbandingan Eksperimen** — analisis mengapa outlier removal menurunkan performa (Akurasi XGBoost turun -3.98%, Recall turun -9.80%), **(2) Justifikasi Klinis** — outlier numerik (seperti oldpeak/trestbps tinggi) merupakan sinyal patologis parah yang penting bagi model, **(3) Analisis One-Hot Encoding** — meningkatkan akurasi dari 79.35% ke 81.52%, **(4) Gap Analisis Jurnal** — perbedaan hasil vs Paper A (82% vs 94%) karena bias imputasi pada 65% data hilang `ca`/`thal` di cohort Swiss/VA. | **K**  |

_Catatan: Kolom **Status** ditandai dengan **K (Kompeten)** untuk penilaian oleh Asesor._

---

## 🔑 Highlight Temuan Analisis (EDA)

Beberapa poin utama hasil analisis data pada dataset penyakit jantung:

1. **Prevalensi Penyakit Jantung:** Dari 920 pasien gabungan, **55.3%** positif mengalami penyakit jantung.
2. **Korelasi Usia:** Usia rata-rata pasien terdiagnosa penyakit jantung adalah **58 tahun**, lebih tinggi dibanding pasien sehat (median **52 tahun**).
3. **Faktor Jenis Kelamin:** Proporsi diagnosis penyakit jantung pada laki-laki jauh lebih tinggi (**~64.3%**) dibandingkan perempuan (**~26.0%**).
4. **Anomali Nyeri Dada (Asymptomatic):** Sebanyak **75.9%** pasien yang tidak merasakan gejala nyeri dada fisik (Asymptomatic - cp 4) justru didiagnosis positif memiliki penyakit jantung. Ini membuktikan bahwa nyeri dada bukanlah satu-satunya acuan deteksi awal.
5. **Koreksi Data Kolesterol:**## 🤖 Rancangan Pipeline Machine Learning (Paper Pendukung A)

Pipeline ML diimplementasikan dengan mereplikasi dan memvalidasi secara kritis metodologi dari **Paper Pendukung A (Tabassum et al., 2025)** untuk menguji pengaruh penghapusan outlier klinis.

```
[1] INPUT: Dataset UCI 4 Kohort (920 instance, 13 fitur + 1 target)
         ↓
[2] PREPROCESSING
    ├─ Missing Value Imputation  → Median & Mode Imputation per-kohort
    │                              (kolesterol=0 dikonversi ke NaN)
    └─ Categorical Encoding      → One-Hot Encoding untuk cp, restecg, slope, ca, thal
         ↓
[3] TRAIN-TEST SPLIT
    └─ 70% Training : 30% Testing (Stratified)
         ↓
[4] DETEKSI OUTLIER (Hanya pada Training Set)
    └─ Isolation Forest (contamination=3% pada fitur numerik kontinu:
                         age, trestbps, chol, thalach, oldpeak)
         ↓
[5] FEATURE SCALING
    └─ Min-Max Normalization (rentang [0, 1])
         ↓
[6] MODEL TRAINING & COMPARISON (5 Algoritma × 2 Eksperimen)
    ├─ Eksperimen 1: Tanpa Outlier Removal (Training Set penuh)
    ├─ Eksperimen 2: Dengan Outlier Removal (Training Set bersih)
    └─ Model Kandidat: XGBoost, AdaBoost, Bagging, Logistic Regression, Naïve Bayes
         ↓
[7] EVALUASI & REVIEW (Test Set)
    ├─ Metrik: Accuracy, Precision, Recall (prioritas utama), F1-Score, AUC-ROC
    ├─ Visualisasi: Grouped Bar Chart, Confusion Matrix, ROC-AUC Curve
    └─ Review Analitis: Analisis dampak klinis penghapusan outlier & bias imputasi
```

### Justifikasi Pilihan Teknis dari Jurnal

| Komponen Pipeline | Teknik yang Dipilih | Justifikasi Jurnal (Paper A) |
| :--- | :--- | :--- |
| **Algoritma Utama** | XGBoost | Performa tertinggi di Paper A (94.34% Accuracy, 96.74% Recall) |
| **Deteksi Outlier** | Isolation Forest (contamination=3%) | Mengisolasi data abnormal secara multidimensi |
| **Feature Scaling** | Min-Max Normalization | Digunakan sebelum pelatihan model agar bobot fitur setara |
| **Categorical Encoding**| One-Hot Encoding | Mengubah data nominal menjadi representasi numerik tanpa asumsi urutan |
| **Model Pembanding** | AdaBoost, Bagging, LR, Naïve Bayes | Membandingkan boosting, bagging, linear, dan probabilistik |
| **Eksperimen Ganda** | Without vs With Outlier Removal | Menguji secara empiris kontribusi riil Isolation Forest |

### Target Performa & Hasil Eksperimen 1 (Model Terbaik)

| Metrik | Target Minimum | Hasil Proyek (XGBoost) | Benchmark Paper A (XGBoost) |
| :---: | :---: | :---: | :---: |
| Accuracy | ≥ 80% | **81.52%** | 94.34% |
| **Recall** | **≥ 82%** | **83.01%** | 96.74% |
| F1-Score | ≥ 82% | **83.28%** | 95.19% |
| AUC-ROC | ≥ 0.85 | **0.8765** | 0.9400 |

*(Catatan: Model AdaBoost mencapai Akurasi tertinggi di proyek kita sebesar **82.25%** dengan Recall **82.35%**).*
