# PRAKTIK DEMONSTRASI

## Metadata Administrasi

| Parameter        | Keterangan                    |
| :--------------- | :---------------------------- |
| **Judul Skema**  | ILMUWAN DATA (DATA SCIENTIST) |
| **Tempat Kerja** | Ruangan D.5.1                 |
| **Nama Asesor**  | Adhitya Nugraha               |
| **Nama Asesi**   | Alfaturachman Maulana Pahlevi |
| **Tanggal**      | 23 Juni 2026                  |
| **Durasi**       | 240 menit                     |

---

## A. Petunjuk Kerja

1. Baca dan pelajari setiap instruksi kerja di bawah ini dengan cermat sebelum melaksanakan praktek.
2. Klarifikasi kepada asesor kompetensi apabila ada hal-hal yang belum jelas.
3. Laksanakan pekerjaan sesuai dengan urutan proses yang sudah ditetapkan.
4. Seluruh proses kerja mengacu kepada SOP/WI yang dipersyaratkan (Jika Ada).

---

## B. Skenario Tugas Praktik Demonstrasi

### Pemetaan Unit Kompetensi

| Kelompok Pekerjaan                                                           | No. | Kode Unit       | Judul Unit                            |
| :--------------------------------------------------------------------------- | :-: | :-------------- | :------------------------------------ |
| **Kelompok Pekerjaan 1:**<br>Analisis Kebutuhan & Perencanaan Proyek         | 1.  | J.62DMI00.001.1 | Menentukan Objektif Bisnis            |
|                                                                              | 2.  | J.62DMI00.002.1 | Menentukan Tujuan Teknis Data Science |
| **Kelompok Pekerjaan 2:**<br>Persiapan & Pengelolaan Data (Data Preparation) | 3.  | J.62DMI00.005.1 | Menelaah Data                         |
|                                                                              | 4.  | J.62DMI00.006.1 | Memvalidasi Data                      |
|                                                                              | 5.  | J.62DMI00.007.1 | Menentukan Objek Data                 |
|                                                                              | 6.  | J.62DMI00.008.1 | Membersihkan Data                     |
|                                                                              | 7.  | J.62DMI00.009.1 | Mengkonstruksi Data                   |
| **Kelompok Pekerjaan 3:**<br>Pemodelan & Evaluasi (Modeling & Evaluation)    | 8.  | J.62DMI00.012.1 | Membangun Skenario Model              |
|                                                                              | 9.  | J.62DMI00.013.1 | Membangun Model                       |
|                                                                              | 10. | J.62DMI00.014.1 | Mengevaluasi Hasil Pemodelan          |
|                                                                              | 11. | J.62DMI00.015.1 | Melakukan Proses Review Pemodelan     |

---

### Situasi (Situation)

Sebuah Puskesmas Kecamatan di Kota Semarang bekerja sama dengan Dinas Kesehatan Kota untuk membangun sistem bantu skrining risiko Stroke pada program Posbindu PTM (Pos Pembinaan Terpadu Penyakit Tidak Menular). Petugas kesehatan menghadapi keterbatasan waktu untuk menilai risiko stroke setiap warga yang datang berkunjung secara manual, sehingga dibutuhkan alat bantu berbasis data untuk memprioritaskan warga yang perlu dirujuk ke pemeriksaan lanjutan.

### Tugas (Task)

Anda diminta mendemonstrasikan seluruh 11 unit kompetensi secara berurutan dengan menggunakan acuan referensi ilmiah dan dataset berikut:

- **Referensi Ilmiah:** Tang, X., Tang, M., Liu, W., & Cui, S. (2026). _"Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation"_. Frontiers in Neurology, 16:1716984.
    - DOI: [10.3389/fneur.2025.1716984](https://doi.org/10.3389/fneur.2025.1716984)
    - URL: [Frontiers in Neurology](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1716984/full)
- **Dataset:** _"Stroke Diagnosis and Health Metrics Data"_
    - Link Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/stroke-diagnosis-and-health-metrics-data)
    - Dimensi: 10.000 baris x 10 kolom
    - Sebaran Kelas: 2.978 kasus stroke (29,78%) vs 7.022 kasus non-stroke (70,22%)
    - Fitur Numerik: `age`, `avg_glucose`, `bmi`
    - Fitur Kategorikal: `gender`, `hypertension`, `heart_disease`, `diabetes`, `smoking_status`, `ses` (socioeconomic status)
    - Target: `stroke` (1 = stroke, 0 = tidak stroke)

---

### Detail Instruksi Unit Kompetensi

#### 1) Menentukan Objektif Bisnis (`J.62DMI00.001.1`)

Meningkatkan cakupan deteksi dini warga berisiko tinggi stroke sebesar 30% dalam 12 bulan melalui skrining berbasis data pada kunjungan Posbindu PTM, guna menurunkan keterlambatan rujukan kasus stroke di wilayah kerja Puskesmas.

#### 2) Menentukan Tujuan Teknis Data Science (`J.62DMI00.002.1`)

Klasifikasi biner untuk memprediksi risiko stroke (kolom target: `stroke`) menggunakan 9 fitur dan 1 target. Metrik utama: Recall dan PR-AUC (bukan hanya ROC-AUC, mengingat risiko 'ROC illusion' pada data tidak seimbang 29,78%:70,22% sebagaimana ditemukan pada paper rujukan).

#### 3) Menelaah Data (`J.62DMI00.005.1`)

Lakukan telaah menyeluruh terhadap 10.000 baris data:

- Identifikasi tipe tiap kolom (numerik/kategorikal/biner).
- Hitung statistik deskriptif (mean, median, std, skewness) untuk `age`, `avg_glucose`, dan `bmi`.
- Buat visualisasi distribusi (histogram) untuk ketiga fitur numerik tersebut.
- Buat heatmap korelasi Pearson antar seluruh variabel.
- Tampilkan proporsi kelas target (stroke vs tidak stroke).
- Susun minimal 3 hipotesis awal berdasarkan pola yang ditemukan (contoh: hubungan usia dengan hipertensi dan penyakit jantung).

#### 4) Memvalidasi Data (`J.62DMI00.006.1`)

Periksa kelengkapan data:

- Identifikasi kolom mana yang memiliki missing value (`bmi`).
- Hitung persentase dan pola missingness-nya (apakah random atau sistematis terhadap kelompok umur/lainnya).
- Nilai kecukupan volume data (10.000 baris) terhadap tujuan teknis yang ditetapkan pada Unit 2.
- Susun rekomendasi tertulis: apakah data layak digunakan, dan syarat apa yang harus dipenuhi sebelum lanjut ke tahap berikutnya.

#### 5) Menentukan Objek Data (`J.62DMI00.007.1`)

- Pisahkan dataset menjadi $X$ (9 fitur: `age`, `gender`, `hypertension`, `heart_disease`, `diabetes`, `avg_glucose`, `bmi`, `smoking_status`, `ses`) dan $y$ (kolom target: `stroke`).
- Diskusikan dan putuskan secara tertulis: apakah kolom socioeconomic status (SES) relevan dipertahankan dalam konteks skrining Posbindu PTM di Indonesia, atau berpotensi menimbulkan bias jika datanya berasal dari konteks negara lain.

#### 6) Membersihkan Data (`J.62DMI00.008.1`)

Tangani missing value pada kolom `bmi` menggunakan `SimpleImputer` (strategi median) yang **DITEMPATKAN DI DALAM PIPELINE** bersama tahap scaling pada Unit 7. **JANGAN** menghitung median dari seluruh dataset sebelum data dibagi latih/uji. Periksa baris duplikat dan konsistensi label kategori teks (mis. penulisan `smoking_status` yang tidak konsisten kapitalisasinya).

#### 7) Mengkonstruksi Data (`J.62DMI00.009.1`)

Lakukan One-Hot Encoding pada fitur kategorikal nominal (`gender`, `smoking_status`, `ses`) dan standardisasi Z-score pada fitur numerik (`age`, `avg_glucose`, `bmi`) menggunakan `StandardScaler` di dalam pipeline yang sama dengan imputer pada Unit 6.

- _Catatan:_ `hypertension`, `heart_disease`, dan `diabetes` sudah berbentuk biner (0/1) sehingga **TIDAK** memerlukan encoding tambahan -- ketiganya langsung dipakai apa adanya sebagai fitur numerik biner dalam pipeline.

#### 8) Membangun Skenario Model (`J.62DMI00.012.1`)

Tentukan skenario pengujian:

- Stratified train-test split 80:20.
- Stratified 5-Fold Cross Validation pada data latih.
- Pilih minimal 5 algoritma kandidat dari kategori berbeda (contoh: Logistic Regression sebagai baseline, Random Forest dan XGBoost/LightGBM sebagai ensemble, MLP sebagai deep learning).
- Tentukan daftar metrik evaluasi: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, dan Brier Score.

#### 9) Membangun Model (`J.62DMI00.013.1`)

Latih model dengan SMOTE untuk menangani class imbalance (29,78% vs 70,22%). SMOTE **WAJIB** ditempatkan di dalam pipeline (`imblearn.pipeline.Pipeline`) agar dieksekusi ulang per fold cross-validation, bukan sekali di awal sebelum pembagian data. Lakukan hyperparameter tuning dengan `GridSearchCV` untuk tiap model.

#### 10) Mengevaluasi Hasil Pemodelan (`J.62DMI00.014.1`)

Uji seluruh model pada data uji yang dikunci.

- Tampilkan confusion matrix.
- Tampilkan kurva ROC dan kurva Precision-Recall **SECARA BERDAMPINGAN** untuk setiap model.
- Amati apakah ada model dengan ROC-AUC tinggi namun PR-AUC rendah (indikasi 'ROC illusion' pada data tidak seimbang).
- Hitung Brier Score untuk **SELURUH** model (bukan hanya satu) sebagai ukuran kalibrasi probabilitas, lalu soroti nilai Brier Score model terbaik pada ringkasan akhir -- konsisten dengan tabel perbandingan performa pada RESULT.

#### 11) Melakukan Proses Review Pemodelan (`J.62DMI00.015.1`)

- Periksa potensi data leakage: pastikan imputasi BMI dan SMOTE benar-benar hanya menggunakan data latih tiap fold.
- Bandingkan skor data latih vs data uji untuk mendeteksi overfitting.
- Lakukan interpretasi SHAP: tampilkan summary plot untuk model terbaik, identifikasi 5 fitur paling berpengaruh, dan hitung SHAP interaction values untuk menemukan minimal 2 pasangan fitur dengan interaksi terkuat (contoh pada paper rujukan: hipertensi-penyakit jantung).
- Susun rekomendasi threshold klasifikasi yang sesuai konteks (skrining massal vs sumber daya terbatas) dan rekomendasi kelayakan model.
- _Catatan teknis:_ SHAP interaction values secara native hanya didukung penuh oleh `TreeExplainer` (model berbasis pohon: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost). Jika model terbaik hasil evaluasi Anda **BUKAN** model berbasis pohon (mis. Logistic Regression atau MLP), hitung SHAP interaction values dari model tree-based terbaik yang tersedia sebagai gantinya, dan jelaskan secara tertulis alasan substitusi tersebut pada laporan review pemodelan.
- _PENTING:_ Secara teknis, pembagian data (Unit 8 - train-test split) **HARUS** dieksekusi **SEBELUM** Unit 6 (Membersihkan Data) dan Unit 7 (Mengkonstruksi Data), karena imputer dan encoder/scaler pada kedua unit tersebut ditempatkan di dalam pipeline yang baru boleh di-fit setelah data dibagi latih/uji. Urutan eksekusi kode yang benar adalah: **Unit 1 → 2 → 3 → 4 → 5 → 8 (split) → 6 → 7 (pipeline fit data latih) → 9 → 10 → 11**. Unit 6 dan 7 tetap didiskusikan dan dilaporkan sesuai urutan penomoran (agar laporan tertulis tetap runtut 1-11), namun kode programnya wajib mengikuti urutan eksekusi teknis di atas untuk mencegah data leakage.

_End of Scenario_

---

## C. Action (Aksi) - Perlengkapan dan Peralatan

1. Aplikasi pengolah kata & Spreadsheet.
2. Aplikasi Notepad++ / VSCode (text editor) & SQL (jika diperlukan).
3. Python 3 (`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`/`lightgbm`, `matplotlib`/`seaborn`, `shap`) pada Jupyter Notebook atau Google Colab.
4. Berkas dataset "Stroke Diagnosis and Health Metrics Data" yang dapat diunduh dari:
    - [Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/stroke-diagnosis-and-health-metrics-data)

---

## D. Result (Hasil)

1. Notebook/script Python seluruh 11 tahapan, dapat dijalankan ulang (reproducible), dengan imputer & SMOTE terbukti berada di dalam pipeline.
2. Ringkasan tertulis objektif bisnis & tujuan teknis data science.
3. Laporan telaah & validasi data, termasuk analisis pola missingness pada BMI.
4. Tabel perbandingan performa minimal 5 model (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score), confusion matrix, dan kurva ROC & PR berdampingan.
5. Ringkasan interpretasi SHAP (summary plot + interaction values), rekomendasi threshold klasifikasi, serta kelayakan model.

---

## E. Waktu Pelaksanaan

- **Waktu:** 10.00 - 14.00 WIB (240 menit / 4 jam)
- **Catatan prioritas waktu:** Jika 240 menit tetap tidak mencukupi, prioritaskan penyelesaian **Unit 1-7 dan 10 (wajib tuntas)**. Pada **Unit 8-9**, cukup lakukan GridSearchCV pada 3 dari 5 algoritma kandidat (prioritaskan 1 model baseline linear + 2 model ensemble/boosting). Pada **Unit 11**, minimal tuntaskan SHAP summary plot (interaction values dapat disusulkan jika waktu tidak cukup, dengan dicatat sebagai keterbatasan pada laporan review).
