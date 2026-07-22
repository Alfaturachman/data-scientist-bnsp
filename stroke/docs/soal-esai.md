# Soal & Jawaban Singkat Sertifikasi - Skrining Risiko Stroke

**Studi Kasus:**
Sebuah Puskesmas Kecamatan di Kota Semarang ingin membangun sistem bantu skrining risiko stroke berbasis data pada program Posbindu PTM. Dataset yang digunakan adalah "Stroke Diagnosis and Health Metrics Data", 10.000 baris x 10 kolom, link dataset: https://www.kaggle.com/datasets/shriyashjagtap/stroke-diagnosis-and-health-metrics-data. Acuan referensi ilmiah: Tang, X., Tang, M., Liu, W. & Cui, S. (2026), "Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation", Frontiers in Neurology 16:1716984, DOI: 10.3389/fneur.2025.1716984. URL: https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1716984/full.

---

### 1) Soal & Jawaban [Kode Unit: J.62DMI00.001.1]

- **Soal:** Sebuah Puskesmas Kecamatan di Kota Semarang ingin membangun sistem bantu skrining risiko stroke pada program Posbindu PTM. Jelaskan mengapa objektif bisnis harus dirumuskan lebih dahulu sebelum menentukan tujuan teknis data science, dan rumuskan satu objektif bisnis yang terukur (SMART) untuk kasus tersebut!
- **Jawaban:**
    - **Alasan:** Objektif bisnis berfokus pada dampak klinis operasional (mencegah keterlambatan rujukan), sedangkan tujuan teknis hanya pada metrik model (ROC-AUC/akurasi). Bisnis harus didahulukan agar model memiliki nilai guna nyata di lapangan.
    - **Objektif SMART:** Menurunkan angka keterlambatan rujukan kasus stroke baru di wilayah Puskesmas Semarang sebesar 15% dalam waktu 12 bulan melalui skrining berkala Posbindu PTM.

### 2) Soal & Jawaban [Kode Unit: J.62DMI00.002.1]

- **Soal:** Dataset yang dipakai memiliki sebaran kelas 2.978 kasus stroke (29,78%) berbanding 7.022 kasus non-stroke (70.22%). Jelaskan mengapa recall dan PR-AUC (bukan hanya accuracy atau ROC-AUC) yang harus dijadikan metrik utama pada kasus ini, dan apa risikonya jika hanya mengandalkan accuracy!
- **Jawaban:**
    - **Recall & PR-AUC:** Recall memastikan sedikit mungkin pasien berisiko stroke yang terlewat (False Negative). PR-AUC lebih objektif mengevaluasi performa kelas minoritas pada data imbalanced tanpa terpengaruh oleh jumlah besar True Negative (non-stroke).
    - **Risiko Akurasi:** Pada data imbalanced (70:30), model naif yang menebak semua pasien "sehat" akan mendapat akurasi tinggi (70%), namun memiliki Recall 0% (gagal mendeteksi stroke sama sekali).

### 3) Soal & Jawaban [Kode Unit: J.62DMI00.005.1]

- **Soal:** Dataset berjumlah 10.000 baris x 10 kolom, mencakup fitur demografi, klinis, perilaku, dan sosial-ekonomi. Sebutkan minimal 4 aktivitas telaah data (EDA) yang wajib dilakukan beserta satu temuan konkret yang mungkin muncul dari masing-masing aktivitas pada dataset ini!
- **Jawaban:**
    1. **Distribusi Kelas:** Menemukan ketidakseimbangan kelas target (Stroke 29,78% vs Non-stroke 70,22%).
    2. **Korelasi Numerik:** Menemukan korelasi positif sedang antara usia (`age`) dengan hipertensi/penyakit jantung ($r \approx 0.5$).
    3. **Analisis Sebaran Usia vs Target:** Median usia penderita stroke bergeser signifikan ke usia lanjut (>65 tahun).
    4. **Cek Kualitas Data:** Mendeteksi adanya nilai kosong (missing values) dan pencilan ekstrem pada variabel BMI (>50).

### 4) Soal & Jawaban [Kode Unit: J.62DMI00.006.1]

- **Soal:** Sebelum digunakan, kolom bmi pada dataset ini PERLU DIVERIFIKASI TERLEBIH DAHULU apakah benar-benar memiliki missing value (NaN) pada versi data yang Anda unduh - karena versi dataset di sumber publik dapat berubah dari waktu ke waktu. Jelaskan langkah-langkah memvalidasi kualitas dan kecukupan data ini (termasuk cara memverifikasi keberadaan missing value secara empiris, bukan berasumsi), serta susun rekomendasi tertulis kelayakan data untuk digunakan, baik jika missing value ditemukan maupun jika tidak!
- **Jawaban:**
    - **Validasi Empiris:** Cek jumlah NaN dengan `df['bmi'].isnull().sum()`, serta periksa statistik deskriptif (`df['bmi'].describe()`) untuk mendeteksi nilai tidak logis (seperti 0 atau negatif).
    - **Rekomendasi Kelayakan:** **Layak** jika missing value sedikit (<10%) dengan menerapkan imputasi median/KNN di dalam pipeline. **Kurang Layak** jika >30% karena memicu bias data tinggi. Jika tidak ada NaN, data **Sangat Layak** digunakan langsung.

### 5) Soal & Jawaban [Kode Unit: J.62DMI00.007.1]

- **Soal:** Dataset memiliki fitur socioeconomic status (SES) di samping fitur klinis seperti hypertension dan heart_disease. Jelaskan bagaimana Anda menentukan objek data (fitur X dan label y) pada kasus ini, dan diskusikan pertimbangan etis/kontekstual dalam memutuskan apakah fitur SES perlu dipertahankan untuk konteks Indonesia!
- **Jawaban:**
    - **Objek Data:** Fitur ($X$) = indikator klinis, perilaku, dan demografi. Label ($y$) = `stroke` (biner: 0 atau 1).
    - **Pertimbangan Etis SES:** Secara sosial-ekonomi, SES berpengaruh terhadap kepatuhan pengobatan dan gizi. Namun, secara etis, menggunakan SES dalam model klasifikasi klinis faskes primer (Puskesmas) memicu bias diskriminatif pelayanan publik yang melanggar asas kesetaraan. Rekomendasi: **Keluarkan SES** dari model prediksi klinis Posbindu.

### 6) Soal & Jawaban [Kode Unit: J.62DMI00.008.1]

- **Soal:** Jelaskan mengapa imputasi missing value pada kolom bmi harus dihitung hanya dari data latih (bukan dari seluruh dataset sebelum pembagian data), dan jelaskan konsekuensi teknisnya (data leakage) jika hal ini diabaikan!
- **Jawaban:**
    - **Alasan:** Parameter statistik imputasi (seperti median BMI) hanya boleh dipelajari dari data latih agar sebaran data uji tidak bocor sebelum pengujian.
    - **Konsekuensi:** Terjadi _data leakage_ (kebocoran data) yang membuat hasil evaluasi validasi terlihat terlalu optimis secara palsu, padahal model aslinya akan gagal melakukan generalisasi pada data pasien baru.

### 7) Soal & Jawaban [Kode Unit: J.62DMI00.009.1]

- **Soal:** Jelaskan langkah konstruksi data pada dataset ini yang meliputi: (a) One-Hot Encoding untuk fitur kategorikal nominal ((gender, smoking_status, ses); dan (b) standardisasi Z-score untuk fitur numerik (age, avg_glucose, bmi). Mengapa kedua proses ini harus digabung dalam satu pipeline bersama imputer?
- **Jawaban:**
    - **Langkah:** (a) _One-Hot Encoding:_ Mengubah kategori nominal menjadi kolom biner 0/1 untuk diproses algoritma. (b) _Z-score:_ Menyamakan skala fitur numerik agar rata-rata = 0 dan std dev = 1.
    - **Integrasi Pipeline:** Menjamin urutan preprocessing berjalan konsisten, mencegah kebocoran informasi (_data leakage_) dengan membatasi proses fitting parameter transformasi hanya pada data latih.

### 8) Soal & Jawaban [Kode Unit: J.62DMI00.012.1]

- **Soal:** Paper rujukan (Tang et al., 2026) membandingkan 9 algoritma sekaligus: Logistic Regression, KNN, Decision Tree, Naive Bayes, Random Forest, XGBoost, LightGBM, CatBoost, dan MLP, ditambah strategi ensemble (Voting, Stacking) dan AutoML. Susunlah skenario pemodelan untuk kasus ini yang memuat minimal 5 algoritma kandidat, skema pembagian data, dan strategi validasi. Jelaskan pertimbangan Anda memilih algoritma-algoritma tersebut!
- **Jawaban:**
    - **Skenario:** Pembagian data 80% Train Set (menggunakan 5-Fold Stratified Cross-Validation) dan 20% Test Set independen.
    - **Kandidat Model:** (1) Logistic Regression (interpretasi klinis mudah), (2) Random Forest (stabil, kebal outliers), (3) XGBoost (akurasi optimal), (4) LightGBM (training cepat & memori rendah), (5) MLP (menangkap interaksi non-linier kompleks).

### 9) Soal & Jawaban [Kode Unit: J.62DMI00.013.1]

- **Soal:** Paper rujukan menerapkan SMOTE secara otomatis melalui pipeline dalam proses cross-validation, dan hanya diterapkan pada subset data latih tiap fold, tidak memengaruhi fold validasi atau data uji. Jelaskan mengapa pendekatan ini penting untuk mencegah data leakage, dan apa yang akan terjadi pada hasil evaluasi jika SMOTE diterapkan sebelum data dibagi menjadi fold-fold cross-validation!
- **Jawaban:**
    - **Pentingnya:** Mencegah kebocoran data dengan memastikan pembentukan sampel sintetis baru hanya didasarkan pada data latih aktif, sedangkan data validasi/uji tetap asli mewakili distribusi klinis riil.
    - **Dampak Jika Sebelum Split:** Terjadi _overfitting_ dan bias evaluasi yang terlalu optimis karena data sintetis di data latih dibuat dari data yang juga masuk ke fold validasi.

### 10) Soal & Jawaban [Kode Unit: J.62DMI00.014.1]

- **Soal:** Paper rujukan menemukan bahwa model Naive Bayes memiliki ROC-AUC tinggi (0,84) namun PR-AUC rendah (0,70), fenomena yang mereka sebut sebagai 'ROC illusion'. Jelaskan mengapa fenomena ini bisa terjadi pada data dengan class imbalance, dan mengapa PR-AUC dianggap metrik yang lebih jujur dibanding ROC-AUC pada kasus seperti ini!
- **Jawaban:**
    - **Penyebab ROC Illusion:** Pada kelas negatif (non-stroke) yang dominan, jumlah TN yang besar meredam nilai FPR ($\frac{FP}{FP+TN}$), sehingga grafik ROC terlihat bagus meski model menghasilkan banyak False Positive.
    - **PR-AUC Lebih Jujur:** Menggunakan Precision ($\frac{TP}{TP+FP}$) yang secara langsung terpengaruh oleh kenaikan FP tanpa dipengaruhi oleh jumlah TN yang masif.

### 11) Soal & Jawaban [Kode Unit: J.62DMI00.015.1]

- **Soal:** Paper rujukan melaporkan model terbaik (LightGBM, F1=0,74 menurut narasi hasil) diinterpretasi menggunakan SHAP, dengan temuan bahwa interaksi hipertensi-penyakit jantung memiliki kekuatan interaksi tertinggi (0,086). Jelaskan bagaimana Anda melakukan proses review pemodelan yang mencakup: pengecekan data leakage/overfitting, interpretasi SHAP (termasuk SHAP interaction values, bukan hanya feature importance tunggal), dan susunlah rekomendasi threshold klasifikasi yang berbeda untuk skenario skrining massal versus skenario sumber daya terbatas! (Opsional tingkat lanjut: jika Anda menghitung ulang F1 dari confusion matrix yang dilaporkan paper, apakah hasilnya konsisten dengan angka F1=0,74 yang diklaim pada narasi teks? Diskusikan implikasinya terhadap keandalan pelaporan hasil penelitian.)
- **Jawaban:**
    - **Review & SHAP:** Cek keselarasan training vs testing score. Interaksi hipertensi-jantung (0,086) menandakan peningkatan risiko eksponensial saat kedua penyakit diderita bersamaan.
    - **Rekomendasi Threshold:** Skrining massal diturunkan ke **0,30 - 0,40** (mengutamakan Recall). Sumber daya terbatas dinaikkan ke **0,55 - 0,65** (mengutamakan Precision).
    - **Audit Inkonsistensi Paper:** **Tidak konsisten**. Hitung manual dari matrix LightGBM (TP=325, FP=197, TN=1216, FN=262) menghasilkan F1 **0,5861** (klaim paper 0,74). Batas atas F1 teoritis XGBoost dengan Recall 0,53 hanya 0,697 (klaim paper 0,72). Implikasi: adanya potensi kesalahan penulisan/analisis oleh penulis dan pentingnya verifikasi independen.
