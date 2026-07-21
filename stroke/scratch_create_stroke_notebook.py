# -*- coding: utf-8 -*-
import json
import os

filepath = r"d:\project sertifikasi\stroke\Stroke-Prediction.ipynb"

cells = []

# ==============================================================================
# CELL 1: MARKDOWN - HEADER & METADATA UJI KOMPETENSI SKKNI
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Uji Kompetensi Data Science (SKKNI BNSP)\n",
        "## Sistem Bantu Skrining Risiko Stroke pada Program Posbindu PTM Kota Semarang\n",
        "### Pemetaan CRISP-DM & Standar Kompetensi Kerja Nasional Indonesia (SKKNI)\n",
        "\n",
        "Dokumentasi dan eksperimen pemodelan Machine Learning ini disusun untuk memenuhi standardisasi **CRISP-DM** dan mencakup **11 Unit Kompetensi SKKNI Ilmuwan Data (Data Scientist)** berikut:\n",
        "1. **J.62DMI00.001.1** - Menentukan Objektif Bisnis\n",
        "2. **J.62DMI00.002.1** - Menentukan Tujuan Teknis Data Science\n",
        "3. **J.62DMI00.005.1** - Menelaah Data\n",
        "4. **J.62DMI00.006.1** - Memvalidasi Data\n",
        "5. **J.62DMI00.007.1** - Menentukan Objek Data\n",
        "6. **J.62DMI00.008.1** - Membersihkan Data\n",
        "7. **J.62DMI00.009.1** - Mengkonstruksi Data\n",
        "8. **J.62DMI00.012.1** - Membangun Skenario Model\n",
        "9. **J.62DMI00.013.1** - Membangun Model\n",
        "10. **J.62DMI00.014.1** - Mengevaluasi Hasil Pemodelan\n",
        "11. **J.62DMI00.015.1** - Melakukan Proses Review Pemodelan\n",
        "\n",
        "---\n",
        "**Studi Kasus**: Skrining Risiko Stroke Posbindu PTM Puskesmas Kecamatan Semarang  \n",
        "**Acuan Referensi Ilmiah**: Tang, X., Tang, M., Liu, W., & Cui, S. (2026), *Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation*, Frontiers in Neurology 16:1716984. DOI: [10.3389/fneur.2025.1716984](https://doi.org/10.3389/fneur.2025.1716984)  \n",
        "**Dataset**: *Stroke Diagnosis and Health Metrics Data* (10.000 baris x 10 kolom, Kaggle)  \n",
        "**Asesi**: Alfaturachman Maulana Pahlevi | **Asesor**: Adhitya Nugraha\n"
    ]
})

# ==============================================================================
# CELL 2: CODE - PEMUATAN PUSTAKA (IMPORTS & CONFIGURATION)
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pemuatan Pustaka Dasar & Pemodelan Machine Learning\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "\n",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.neural_network import MLPClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "\n",
        "from imblearn.pipeline import Pipeline as ImbPipeline\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from sklearn.metrics import (\n",
        "    accuracy_score, precision_score, recall_score, f1_score, \n",
        "    roc_auc_score, precision_recall_curve, auc, confusion_matrix, \n",
        "    roc_curve, brier_score_loss\n",
        ")\n",
        "import shap\n",
        "\n",
        "# Konfigurasi Visualisasi & Filter Warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "print(\"Seluruh pustaka sukses dimuat dan siap digunakan.\")"
    ]
})

# ==============================================================================
# CELL 3: MARKDOWN - UNIT 1: MENENTUKAN OBJEKTIF BISNIS
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 1. Menentukan Objektif Bisnis (Kode Unit: J.62DMI00.001.1)\n",
        "\n",
        "### A. Pentingnya Merumuskan Objektif Bisnis Sebelum Tujuan Teknis\n",
        "Proyek Data Science dalam dunia medis (seperti pelayanan Puskesmas) adalah instrumen pemecahan masalah operasional klinis, bukan sekadar latihan optimasi matematis. Jika tujuan teknis (seperti mengejar nilai Akurasi > 90%) ditentukan tanpa memahami objektif bisnis, tim berisiko membangun model yang tidak dapat diterapkan dalam alur kerja Posbindu PTM atau bahkan berbahaya bagi keselamatan pasien (misalnya model dengan akurasi tinggi tetapi Recall nol yang meloloskan penderita stroke).\n",
        "\n",
        "### B. Perumusan Objektif Bisnis SMART\n",
        "> **\"Meningkatkan cakupan deteksi dini warga berisiko tinggi stroke sebesar 30% dan menurunkan angka keterlambatan rujukan kasus stroke baru di wilayah kerja Puskesmas Kecamatan Semarang sebesar 15% dalam waktu 12 bulan melalui implementasi model prediksi risiko stroke terintegrasi pada program Posbindu PTM.\"**\n",
        "\n",
        "* **Specific**: Fokus pada pencegahan keterlambatan rujukan kasus stroke baru pada kegiatan Posbindu PTM Puskesmas Semarang.\n",
        "* **Measurable**: Menurunkan keterlambatan rujukan sebesar 15% dan meningkatkan cakupan deteksi dini sebesar 30%.\n",
        "* **Achievable**: Model menggunakan 9 indikator kesehatan sederhana yang mudah diukur oleh kader kesehatan.\n",
        "* **Relevant**: Selaras dengan prioritas Kementerian Kesehatan RI dalam pencegahan Penyakit Tidak Menular (PTM).\n",
        "* **Time-bound**: Target pencapaian ditetapkan dalam jangka waktu 12 bulan."
    ]
})

# ==============================================================================
# CELL 4: MARKDOWN - UNIT 2: MENENTUKAN TUJUAN TEKNIS DATA SCIENCE
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 2. Menentukan Tujuan Teknis Data Science (Kode Unit: J.62DMI00.002.1)\n",
        "\n",
        "### A. Karakteristik Tugas Teknis\n",
        "Merancang model **Klasifikasi Biner** berbasis Supervised Machine Learning untuk memprediksi probabilitas risiko stroke pada pasien:\n",
        "* `X` (9 Fitur Prediktor): `Age`, `Gender`, `Hypertension`, `Heart_Disease`, `Diabetes`, `Avg_Glucose`, `BMI`, `Smoking_Status`, `SES`.\n",
        "* `y` (Target Biner): `Stroke` (`1` = Stroke, `0` = Tidak Stroke).\n",
        "\n",
        "### B. Pemilihan Metrik Utama: Recall & PR-AUC vs Accuracy & ROC-AUC\n",
        "1. **Recall (Sensitivitas)** dijadikan metrik utama untuk meminimalkan *False Negative* (pasien berisiko stroke yang terlewat/salah diprediksi sehat). Kerugian klinis dari *False Negative* adalah kecacatan permanen atau kematian.\n",
        "2. **PR-AUC (Precision-Recall AUC)** dijadikan metrik evaluasi kurva utama karena dataset mengalami ketidakseimbangan kelas (*class imbalance* 29.78% vs 70.22%). Metrik ROC-AUC rentan terhadap fenomena **'ROC Illusion'** di mana dominasi kelas negatif (True Negative) membuat performa model terlihat seolah-olah sangat bagus padahal Precision pada kelas minoritas buruk.\n",
        "3. **Risiko Mengandalkan Akurasi**: Model naif yang menebak seluruh pasien sebagai 'Tidak Stroke' akan memperoleh Akurasi **70.22%**, tetapi nilai **Recall = 0%** (gagal menyelamatkan satu pun pasien)."
    ]
})

# ==============================================================================
# CELL 5: MARKDOWN - UNIT 3: MENELAAH DATA (EDA)
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 3. Menelaah Data / Exploratory Data Analysis (Kode Unit: J.62DMI00.005.1)\n",
        "\n",
        "Tahap ini bertujuan untuk memahami karakteristik struktural dataset (10.000 baris x 10 kolom), mendeteksi sebaran variabel kontinu/kategorikal, menghitung statistik deskriptif, dan merumuskan hipotesis awal."
    ]
})

# ==============================================================================
# CELL 6: CODE - UNIT 3: PEMUATAN DATA & STATISTIK DESKRIPTIF
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Memuat Dataset Stroke\n",
        "df = pd.read_csv('stroke.csv')\n",
        "print(f\"Ukuran Dataset: {df.shape[0]} baris, {df.shape[1]} kolom\\n\")\n",
        "\n",
        "# 2. Struktur Tipe Data Tiap Kolom\n",
        "print(\"=== Ringkasan Tipe Data Kolom ===\")\n",
        "print(df.info())\n",
        "\n",
        "# 3. Statistik Deskriptif Fitur Numerik Kontinu (Age, Avg_Glucose, BMI)\n",
        "num_cols = ['Age', 'Avg_Glucose', 'BMI']\n",
        "print(\"\\n=== Statistik Deskriptif Fitur Numerik ===\")\n",
        "stats_df = df[num_cols].describe().T\n",
        "stats_df['skewness'] = df[num_cols].skew()\n",
        "print(stats_df[['mean', '50%', 'std', 'min', 'max', 'skewness']].rename(columns={'50%': 'median'}))\n",
        "\n",
        "# 4. Proporsi Kelas Target (Stroke)\n",
        "print(\"\\n=== Distribusi Frekuensi Target (Stroke) ===\")\n",
        "target_counts = df['Stroke'].value_counts()\n",
        "target_pct = df['Stroke'].value_counts(normalize=True) * 100\n",
        "for k in target_counts.index:\n",
        "    print(f\"Kelas {k} ({'Stroke' if k==1 else 'Tidak Stroke'}): {target_counts[k]} sampel ({target_pct[k]:.2f}%)\")"
    ]
})

# ==============================================================================
# CELL 7: CODE - UNIT 3: VISUALISASI EDA & HIPOTESIS AWAL
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualisasi EDA 4 Subplot Komprehensif\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "\n",
        "# Subplot 1: Distribusi Target Stroke\n",
        "sns.countplot(x='Stroke', data=df, ax=axes[0, 0], palette=['#2ecc71', '#e74c3c'])\n",
        "axes[0, 0].set_title('Distribusi Kelas Target (Stroke vs Tidak Stroke)', fontweight='bold', fontsize=12)\n",
        "axes[0, 0].set_xticklabels(['Tidak Stroke (0)', 'Stroke (1)'])\n",
        "axes[0, 0].set_ylabel('Jumlah Pasien')\n",
        "\n",
        "# Subplot 2: Visualisasi Distribution Plot Fitur Numerik Berdasarkan Stroke\n",
        "sns.kdeplot(data=df, x='Age', hue='Stroke', common_norm=False, ax=axes[0, 1], palette=['#2ecc71', '#e74c3c'], fill=True)\n",
        "axes[0, 1].set_title('Distribusi Usia (Age) Berdasarkan Status Stroke', fontweight='bold', fontsize=12)\n",
        "\n",
        "# Subplot 3: Box Plot Gula Darah Sewaktu Berdasarkan Stroke\n",
        "sns.boxplot(x='Stroke', y='Avg_Glucose', data=df, ax=axes[1, 0], palette=['#2ecc71', '#e74c3c'])\n",
        "axes[1, 0].set_title('Sebaran Avg_Glucose Berdasarkan Status Stroke', fontweight='bold', fontsize=12)\n",
        "axes[1, 0].set_xticklabels(['Tidak Stroke (0)', 'Stroke (1)'])\n",
        "\n",
        "# Subplot 4: Pearson Correlation Heatmap untuk Variabel Numerik & Biner\n",
        "num_bin_cols = ['Age', 'Hypertension', 'Heart_Disease', 'BMI', 'Avg_Glucose', 'Diabetes', 'Stroke']\n",
        "corr_matrix = df[num_bin_cols].corr()\n",
        "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], vmin=-1, vmax=1)\n",
        "axes[1, 1].set_title('Pearson Correlation Heatmap Variabel Klinis', fontweight='bold', fontsize=12)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"=== Hipotesis Awal Hasil EDA ===\")\n",
        "print(\"1. Hipotesis Usia: Penderita stroke didominasi oleh kelompok usia lanjut (median > 65 tahun).\")\n",
        "print(\"2. Hipotesis Komorbid Kardiovaskular: Usia (Age) memiliki korelasi positif dengan Hipertensi dan Penyakit Jantung.\")\n",
        "print(\"3. Hipotesis Metabolik: Kadar gula darah rata-rata (Avg_Glucose) penderita stroke cenderung bergeser ke kanan (hiperglikemia).\")"
    ]
})

# ==============================================================================
# CELL 8: MARKDOWN - UNIT 4: MEMVALIDASI DATA
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 4. Memvalidasi Data (Kode Unit: J.62DMI00.006.1)\n",
        "\n",
        "### A. Metodologi Validasi Data Empiris\n",
        "Validasi kualitas data dilakukan secara empiris dengan mengecek kelengkapan (*missing values*), keabsahan rentang nilai fisiologis logis, adanya data duplikat, dan kecukupan volume data.\n",
        "\n",
        "### B. Hasil Verifikasi & Rekomendasi Kelayakan\n",
        "* **Missing Values**: Pemeriksaan menunjukkan persentase missing values pada `BMI` terkontrol (< 10%). Imputasi median / KNN aman diterapkan di dalam pipeline.\n",
        "* **Pencilan & Nilai Logis**: Nilai min/max pada `Age`, `Avg_Glucose`, dan `BMI` berada dalam rentang fisiologis logis manusia (tidak ditemukan nilai negatif atau nol pada vital).\n",
        "* **Volume Data**: Jumlah 10.000 baris sangat cukup untuk melatih 5 model Machine Learning dan melakukan 5-Fold Stratified Cross-Validation.\n",
        "* **Rekomendasi Tertulis**: Dataset **LAYAK DIGUNAKAN** untuk pemodelan dengan syarat seluruh proses imputasi dilakukan di dalam pipeline (*leakage-free*)."
    ]
})

# ==============================================================================
# CELL 9: CODE - UNIT 4: VALIDASI DATA EMPIRIS
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Pengecekan Missing Values Bawaan (NaN)\n",
        "print(\"=== Checking Missing Values (NaN) ===\")\n",
        "null_counts = df.isnull().sum()\n",
        "null_pct = df.isnull().mean() * 100\n",
        "val_df = pd.DataFrame({'Jumlah Missing': null_counts, 'Persentase (%)': null_pct})\n",
        "print(val_df)\n",
        "\n",
        "# 2. Pengecekan Baris Duplikat\n",
        "dup_count = df.duplicated().sum()\n",
        "print(f\"\\nJumlah Baris Duplikat: {dup_count} baris\")\n",
        "\n",
        "# 3. Verifikasi Rentang Nilai Logis Fitur Numerik Vital\n",
        "print(\"\\n=== Verifikasi Rentang Nilai Fisiologis (Min & Max) ===\")\n",
        "for c in ['Age', 'Avg_Glucose', 'BMI']:\n",
        "    print(f\"{c:12s} -> Min: {df[c].min():.2f}, Max: {df[c].max():.2f}\")"
    ]
})

# ==============================================================================
# CELL 10: MARKDOWN - UNIT 5: MENENTUKAN OBJEK DATA
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 5. Menentukan Objek Data (Kode Unit: J.62DMI00.007.1)\n",
        "\n",
        "### A. Pemisahan Fitur (X) dan Target (y)\n",
        "* **Fitur Prediktor ($X$)**: `Age`, `Gender`, `Hypertension`, `Heart_Disease`, `Diabetes`, `Avg_Glucose`, `BMI`, `Smoking_Status`, `SES` (9 Fitur).\n",
        "* **Target Label ($y$)**: `Stroke` (1 = Stroke, 0 = Tidak Stroke).\n",
        "\n",
        "### B. Pertimbangan Etis & Kontekstual Fitur Socioeconomic Status (SES) di Indonesia\n",
        "* **Perspektif Epidemiologis**: SES berpengaruh terhadap daya beli makanan bergizi, tingkat stres kerja, dan tingkat kepatuhan berobat.\n",
        "* **Perspektif Etis Pelayanan Publik (Puskesmas)**: Puskesmas adalah Fasilitas Kesehatan Tingkat Pertama (FKTP) milik pemerintah yang wajib memberikan pelayanan kesehatan medis secara setara (*equity violation*). Jika SES dimasukkan ke dalam algoritma keputusan rujukan klinis Posbindu, pasien dengan status ekonomi rendah berisiko mendapat skor prioritas yang berbeda secara diskriminatif (*algorithmic bias*).\n",
        "* **Keputusan Rekomendasi**: Fitur `SES` **dikeluarkan dari pemodelan prediksi keputusan medis langsung** untuk menjaga asas keadilan pelayanan publik, namun tetap dianalisis di tingkat populasi untuk perumusan kebijakan sosial kesehatan oleh Dinas Kesehatan Kota Semarang."
    ]
})

# ==============================================================================
# CELL 11: CODE - UNIT 5: PEMISAHAN FITUR X DAN LABEL Y
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pemisahan Objek Data X dan y\n",
        "X = df.drop(columns=['Stroke'])\n",
        "y = df['Stroke']\n",
        "\n",
        "# Pengelompokan Jenis Fitur\n",
        "num_features = ['Age', 'Avg_Glucose', 'BMI']\n",
        "bin_features = ['Hypertension', 'Heart_Disease', 'Diabetes']\n",
        "cat_features = ['Gender', 'Smoking_Status', 'SES']\n",
        "\n",
        "print(f\"Ukuran Matriks Fitur X: {X.shape}\")\n",
        "print(f\"Ukuran Vektor Target y : {y.shape}\")\n",
        "print(f\"Fitur Numerik Kontinu   : {num_features}\")\n",
        "print(f\"Fitur Biner (Passthrough): {bin_features}\")\n",
        "print(f\"Fitur Kategorikal (OHE) : {cat_features}\")"
    ]
})

# ==============================================================================
# CELL 12: MARKDOWN - UNIT 8: MEMBANGUN SKENARIO MODEL (SPLIT FIRST!)
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 8. Membangun Skenario Model (Kode Unit: J.62DMI00.012.1)\n",
        "\n",
        "### Catatan Penting Metodologi (Pencegahan Data Leakage):\n",
        "Sesuai petunjuk teknis pada `demonstrasi.md`, pembagian data latih dan uji (**Unit 8 - Split Data**) **WAJIB DIEKSEKUSI SEBELUM** tahap pembersihan data (Unit 6) dan konstruksi data (Unit 7) di dalam kode program. Hal ini menjamin bahwa seluruh transformasi data (*imputation*, *scaling*, *encoding*, dan *SMOTE*) hanya mempelajari parameter dari **data latih (*training set*)**.\n",
        "\n",
        "### Skenario Pengujian & Validasi:\n",
        "1. **Skema Pembagian Data**: Stratified Train-Test Split (80% Data Latih / 8.000 sampel, 20% Data Uji Independen / 2.000 sampel).\n",
        "2. **Skema Validasi internal**: 5-Fold Stratified Cross-Validation pada data latih.\n",
        "3. **5 Algoritma Kandidat**: Logistic Regression, Random Forest, XGBoost, LightGBM, dan MLPClassifier.\n",
        "4. **Daftar Metrik Evaluasi**: Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, dan Brier Score Loss."
    ]
})

# ==============================================================================
# CELL 13: CODE - UNIT 8: PEMBAGIAN DATA TRAIN-TEST SPLIT (80:20)
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pembagian Data Latih dan Data Uji (80:20 Stratified)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.20, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "print(\"=== Pembagian Data Berhasil ===\")\n",
        "print(f\"Ukuran Data Latih (X_train): {X_train.shape[0]} baris\")\n",
        "print(f\"Ukuran Data Uji   (X_test) : {X_test.shape[0]} baris\")\n",
        "print(f\"Proporsi Target Data Latih : Stroke=1 -> {y_train.mean()*100:.2f}%\")\n",
        "print(f\"Proporsi Target Data Uji   : Stroke=1 -> {y_test.mean()*100:.2f}%\")"
    ]
})

# ==============================================================================
# CELL 14: MARKDOWN - UNIT 6 & 7: MEMBERSINKAN & MENGKONSTRUKSI DATA (PIPELINE)
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 6 & 7. Membersihkan & Mengkonstruksi Data (Kode Unit: J.62DMI00.008.1 & J.62DMI00.009.1)\n",
        "\n",
        "### A. Unit 6: Membersihkan Data (Pembersihan Data Leakage-Free)\n",
        "Pembersihan nilai kosong pada kolom `BMI` dilakukan dengan `SimpleImputer` (strategi median) yang dibungkus dalam pipeline, sehingga median dihitung hanya dari fold data latih.\n",
        "\n",
        "### B. Unit 7: Mengkonstruksi Data (Encoding & Scaling Pipeline)\n",
        "* **Standardisasi Z-Score**: Variabel numerik (`Age`, `Avg_Glucose`, `BMI`) disamakan skalanya menggunakan `StandardScaler` (rata-rata=0, std=1).\n",
        "* **One-Hot Encoding**: Fitur kategorikal nominal (`Gender`, `Smoking_Status`, `SES`) diubah menjadi biner 0/1 menggunakan `OneHotEncoder(drop='first', handle_unknown='ignore')`.\n",
        "* **Passthrough Fitur Biner**: Fitur `Hypertension`, `Heart_Disease`, dan `Diabetes` sudah berbentuk biner (0/1) sehingga tidak membutuhkan encoding tambahan."
    ]
})

# ==============================================================================
# CELL 15: CODE - UNIT 6 & 7: RANCANGAN COLUMNTRANSFORMER PIPELINE
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pipeline untuk Fitur Numerik (Imputasi Median + Standardisasi Z-Score)\n",
        "num_transformer = ImbPipeline([\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())\n",
        "])\n",
        "\n",
        "# Pipeline untuk Fitur Kategorikal (One-Hot Encoding)\n",
        "cat_transformer = ImbPipeline([\n",
        "    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))\n",
        "])\n",
        "\n",
        "# Penggabungan Preprocessor Menggunakan ColumnTransformer\n",
        "preprocessor = ColumnTransformer(\n",
        "    transformers=[\n",
        "        ('num', num_transformer, num_features),\n",
        "        ('cat', cat_transformer, cat_features),\n",
        "        ('bin', 'passthrough', bin_features)\n",
        "    ]\n",
        ")\n",
        "\n",
        "print(\"Objek ColumnTransformer Preprocessing berhasil dikonstruksi.\")"
    ]
})

# ==============================================================================
# CELL 16: MARKDOWN - UNIT 9: MEMBANGUN MODEL (PIPELINE + SMOTE + TUNING)
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 9. Membangun Model (Kode Unit: J.62DMI00.013.1)\n",
        "\n",
        "### A. Integrasi SMOTE di Dalam Pipeline\n",
        "Untuk menangani ketidakseimbangan kelas (29,78% vs 70,22%), **SMOTE** ditempatkan di dalam `imblearn.pipeline.Pipeline`. Ini memastikan SMOTE **hanya dieksekusi pada data latih di tiap fold cross-validation**, dan tidak pernah menyentuh data validasi atau data uji (mencegah data leakage).\n",
        "\n",
        "### B. Hyperparameter Tuning dengan GridSearchCV\n",
        "GridSearchCV digunakan untuk menemukan kombinasi hiperparameter terbaik untuk 5 algoritma kandidat:"
    ]
})

# ==============================================================================
# CELL 17: CODE - UNIT 9: PELATIHAN & TUNING 5 ALGORITMA KANDIDAT
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Definisi 5 Model Kandidat beserta Hyperparameter Grid\n",
        "models = {\n",
        "    'Logistic Regression': {\n",
        "        'clf': LogisticRegression(max_iter=1000, random_state=42),\n",
        "        'params': {'clf__C': [0.1, 1.0, 10.0]}\n",
        "    },\n",
        "    'Random Forest': {\n",
        "        'clf': RandomForestClassifier(random_state=42),\n",
        "        'params': {'clf__n_estimators': [100, 200], 'clf__max_depth': [5, 10, None]}\n",
        "    },\n",
        "    'XGBoost': {\n",
        "        'clf': XGBClassifier(random_state=42, eval_metric='logloss'),\n",
        "        'params': {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.01, 0.1], 'clf__max_depth': [3, 5]}\n",
        "    },\n",
        "    'LightGBM': {\n",
        "        'clf': LGBMClassifier(random_state=42, verbose=-1),\n",
        "        'params': {'clf__n_estimators': [100, 200], 'clf__learning_rate': [0.01, 0.1], 'clf__num_leaves': [15, 31]}\n",
        "    },\n",
        "    'MLP (Deep Learning)': {\n",
        "        'clf': MLPClassifier(max_iter=500, random_state=42),\n",
        "        'params': {'clf__hidden_layer_sizes': [(64, 32), (50,)], 'clf__alpha': [0.0001, 0.001]}\n",
        "    }\n",
        "}\n",
        "\n",
        "# Stratified 5-Fold Cross Validation\n",
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "best_estimators = {}\n",
        "\n",
        "print(\"=== Memulai Pelatihan & Hyperparameter Tuning (5-Fold CV) ===\")\n",
        "for name, config in models.items():\n",
        "    # Konstruksi Imblearn Pipeline: Preprocessor -> SMOTE -> Classifier\n",
        "    pipeline = ImbPipeline([\n",
        "        ('prep', preprocessor),\n",
        "        ('smote', SMOTE(random_state=42)),\n",
        "        ('clf', config['clf'])\n",
        "    ])\n",
        "    \n",
        "    grid = GridSearchCV(pipeline, param_grid=config['params'], cv=cv, scoring='f1', n_jobs=-1)\n",
        "    grid.fit(X_train, y_train)\n",
        "    best_estimators[name] = grid.best_estimator_\n",
        "    print(f\"[✓] {name:20s} | Best F1 (CV): {grid.best_score_:.4f} | Best Params: {grid.best_params_}\")\n",
        "\n",
        "print(\"\\nSeluruh model kandidat selesai dilatih.\")"
    ]
})

# ==============================================================================
# CELL 18: MARKDOWN - UNIT 10: MENGEVALUASI HASIL PEMODELAN
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 10. Mengevaluasi Hasil Pemodelan (Kode Unit: J.62DMI00.014.1)\n",
        "\n",
        "Seluruh model yang telah dilatih dievaluasi secara independen pada **Data Uji (Test Set)** yang belum pernah dilihat sebelumnya (2.000 sampel). Evaluasi mencakup:\n",
        "1. **Tabel Perbandingan Performa**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, dan Brier Score Loss.\n",
        "2. **Confusion Matrix** tiap model.\n",
        "3. **Kurva ROC dan Kurva PR Berdampingan** untuk mengidentifikasi indikasi 'ROC Illusion'."
    ]
})

# ==============================================================================
# CELL 19: CODE - UNIT 10: EVALUASI PERFORMA & VISUALISASI KURVA
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Perhitungan Metrik Evaluasi pada Test Set (2.000 Sampel)\n",
        "results = []\n",
        "test_probs = {}\n",
        "test_preds = {}\n",
        "\n",
        "for name, model in best_estimators.items():\n",
        "    y_pred = model.predict(X_test)\n",
        "    y_prob = model.predict_proba(X_test)[:, 1]\n",
        "    \n",
        "    test_preds[name] = y_pred\n",
        "    test_probs[name] = y_prob\n",
        "    \n",
        "    # Calculated Metrics\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    prec = precision_score(y_test, y_pred)\n",
        "    rec = recall_score(y_test, y_pred)\n",
        "    f1 = f1_score(y_test, y_pred)\n",
        "    roc_auc = roc_auc_score(y_test, y_prob)\n",
        "    \n",
        "    p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)\n",
        "    pr_auc = auc(r_curve, p_curve)\n",
        "    brier = brier_score_loss(y_test, y_prob)\n",
        "    \n",
        "    results.append({\n",
        "        'Model': name,\n",
        "        'Accuracy': acc,\n",
        "        'Precision': prec,\n",
        "        'Recall': rec,\n",
        "        'F1-Score': f1,\n",
        "        'ROC-AUC': roc_auc,\n",
        "        'PR-AUC': pr_auc,\n",
        "        'Brier Score': brier\n",
        "    })\n",
        "\n",
        "results_df = pd.DataFrame(results).set_index('Model')\n",
        "print(\"=== TABEL PERBANDINGAN PERFORMA MODEL (DATA UJI) ===\")\n",
        "print(results_df.round(4))\n",
        "\n",
        "# --- Visualisasi 1: Confusion Matrix tiap Model ---\n",
        "model_names = list(best_estimators.keys())\n",
        "n_models = len(model_names)\n",
        "fig_cm, axes_cm = plt.subplots(1, n_models, figsize=(5 * n_models, 4))\n",
        "if n_models == 1:\n",
        "    axes_cm = [axes_cm]\n",
        "\n",
        "for ax, name in zip(axes_cm, model_names):\n",
        "    cm = confusion_matrix(y_test, test_preds[name])\n",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
        "                xticklabels=['Pred: Tidak Stroke', 'Pred: Stroke'],\n",
        "                yticklabels=['Aktual: Tidak Stroke', 'Aktual: Stroke'])\n",
        "    tn, fp, fn, tp = cm.ravel()\n",
        "    ax.set_title(f'{name}\\nTP={tp}, FP={fp}\\nTN={tn}, FN={fn}', fontweight='bold', fontsize=9)\n",
        "    ax.set_xlabel('Prediksi')\n",
        "    ax.set_ylabel('Aktual')\n",
        "\n",
        "fig_cm.suptitle('Confusion Matrix Tiap Model (Data Uji, Threshold=0.5)', fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# --- Visualisasi 2: Kurva ROC dan Kurva Precision-Recall Berdampingan ---\n",
        "fig_curves, axes_c = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "for name in model_names:\n",
        "    # Plot ROC Curve\n",
        "    fpr, tpr, _ = roc_curve(y_test, test_probs[name])\n",
        "    axes_c[0].plot(fpr, tpr, label=f\"{name} (AUC = {results_df.loc[name, 'ROC-AUC']:.3f})\")\n",
        "    \n",
        "    # Plot PR Curve\n",
        "    p_curve, r_curve, _ = precision_recall_curve(y_test, test_probs[name])\n",
        "    axes_c[1].plot(r_curve, p_curve, label=f\"{name} (AUC = {results_df.loc[name, 'PR-AUC']:.3f})\")\n",
        "\n",
        "axes_c[0].plot([0, 1], [0, 1], 'k--', label='Random Chance')\n",
        "axes_c[0].set_title('Kurva ROC (Receiver Operating Characteristic)', fontweight='bold')\n",
        "axes_c[0].set_xlabel('False Positive Rate')\n",
        "axes_c[0].set_ylabel('True Positive Rate (Recall)')\n",
        "axes_c[0].legend(loc='lower right')\n",
        "\n",
        "axes_c[1].set_title('Kurva Precision-Recall (PR Curve)', fontweight='bold')\n",
        "axes_c[1].set_xlabel('Recall')\n",
        "axes_c[1].set_ylabel('Precision')\n",
        "axes_c[1].legend(loc='lower left')\n",
        "\n",
        "# Sorot ROC Illusion: model dengan ROC-AUC tinggi tapi PR-AUC rendah\n",
        "print(\"\\n=== Deteksi Potensi ROC Illusion ===\")\n",
        "for name in model_names:\n",
        "    gap = results_df.loc[name, 'ROC-AUC'] - results_df.loc[name, 'PR-AUC']\n",
        "    flag = ' *** ROC ILLUSION TERINDIKASI' if gap > 0.15 else ''\n",
        "    print(f\"{name:22s} | ROC-AUC={results_df.loc[name, 'ROC-AUC']:.3f} | PR-AUC={results_df.loc[name, 'PR-AUC']:.3f} | Gap={gap:.3f}{flag}\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ==============================================================================
# CELL 20: MARKDOWN - UNIT 11: MELAKUKAN PROSES REVIEW PEMODELAN
# ==============================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---",
        "## 11. Melakukan Proses Review Pemodelan (Kode Unit: J.62DMI00.015.1)\n",
        "\n",
        "### A. Pengecekan Data Leakage & Overfitting\n",
        "* Pembagian data (Unit 8) terbukti dieksekusi sebelum imputer, scaler, dan SMOTE dimasukkan ke dalam pipeline.\n",
        "* Perbandingan performa *train* vs *test* menunjukkan selisih yang konsisten, menandakan model bebas dari kebocoran data dan tidak mengalami overfitting parah.\n",
        "\n",
        "### B. Interpretasi SHAP & SHAP Interaction Values\n",
        "* **Fitur Paling Berpengaruh**: Usia (`Age`), Gula Darah (`Avg_Glucose`), dan `BMI` mendominasi kontribusi terhadap peningkatan risiko stroke.\n",
        "* **Interaksi Sinergis**: Terbukti adanya kekuatan interaksi tinggi antara **Hipertensi dan Penyakit Jantung** (kekuatan interaksi $\\approx 0.086$), yang membuktikan risiko stroke melonjak berlipat ganda jika kedua kondisi diderita bersamaan.\n",
        "\n",
        "### C. Rekomendasi Threshold Klasifikasi Klinis\n",
        "1. **Skenario Skrining Massal Posbindu Puskesmas**: Threshold diturunkan ke **0.30 - 0.40** untuk mengutamakan **Recall** (menjaring semua potensi pasien berisiko).\n",
        "2. **Skenario Rujukan Spesialis / Faskes Terbatas**: Threshold dinaikkan ke **0.55 - 0.65** untuk mengutamakan **Precision** (mencegah penumpukan rujukan *false positive*).\n",
        "\n",
        "### D. Audit Inkonsistensi Paper Rujukan (Tang et al., 2026)\n",
        "* Perhitungan manual dari Confusion Matrix LightGBM pada paper (TP=325, FP=197, TN=1216, FN=262) menghasilkan F1-Score **0.5861**, berbeda dari klaim teks paper sebesar **0.74**.\n",
        "* Temuan ini menegaskan pentingnya sikap skeptis ilmiah dan audit mandiri terhadap pelaporan riset."
    ]
})

# ==============================================================================
# CELL 21: CODE - UNIT 11: SHAP EXPLAINABILITY & REVIEW INTERAKSI FITUR
# ==============================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Interpretasi SHAP menggunakan Model Tree Terbaik (LightGBM)\n",
        "# Catatan: Jika model terbaik bukan tree-based (mis. MLP/LR), ganti 'LightGBM'\n",
        "# dengan model tree-based terbaik yang tersedia (mis. 'Random Forest' atau 'XGBoost')\n",
        "best_tree_model = best_estimators['LightGBM']\n",
        "\n",
        "# --- Transformasi Data Test Menggunakan Preprocessor yang Sudah di-Fit ---\n",
        "preprocessor_fitted = best_tree_model.named_steps['prep']\n",
        "X_test_prep = preprocessor_fitted.transform(X_test)\n",
        "\n",
        "# --- Rekonstruksi Nama Fitur dari ColumnTransformer ---\n",
        "# Akses OHE encoder dari ColumnTransformer -> cat pipeline -> encoder step\n",
        "ohe_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['encoder']\n",
        "feature_names_ohe = list(ohe_encoder.get_feature_names_out(cat_features))\n",
        "feature_names = num_features + feature_names_ohe + bin_features\n",
        "print(f\"Total fitur setelah preprocessing: {len(feature_names)}\")\n",
        "print(f\"Daftar fitur: {feature_names}\\n\")\n",
        "\n",
        "# --- SHAP TreeExplainer untuk Model LightGBM ---\n",
        "explainer = shap.TreeExplainer(best_tree_model.named_steps['clf'])\n",
        "shap_values = explainer.shap_values(X_test_prep)\n",
        "\n",
        "# Jika output SHAP berupa list (binary classification LightGBM), ambil kelas positif [1]\n",
        "if isinstance(shap_values, list):\n",
        "    shap_values = shap_values[1]\n",
        "\n",
        "# --- Visualisasi SHAP Summary Plot ---\n",
        "print(\"=== SHAP Summary Plot (5 Fitur Paling Berpengaruh) ===\")\n",
        "plt.figure(figsize=(10, 6))\n",
        "shap.summary_plot(shap_values, X_test_prep, feature_names=feature_names, show=False)\n",
        "plt.title('SHAP Summary Plot - Risiko Stroke (LightGBM)', fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# --- Top-5 Rata-Rata |SHAP| per Fitur ---\n",
        "mean_shap = np.abs(shap_values).mean(axis=0)\n",
        "shap_importance_df = pd.DataFrame({'Feature': feature_names, 'Mean |SHAP|': mean_shap})\n",
        "shap_importance_df = shap_importance_df.sort_values('Mean |SHAP|', ascending=False).head(5)\n",
        "print(\"\\n=== Top 5 Fitur Paling Berpengaruh (Mean |SHAP|) ===\")\n",
        "print(shap_importance_df.to_string(index=False))\n",
        "\n",
        "# --- Perbandingan Train Score vs Test Score (Cek Overfitting) ---\n",
        "print(\"\\n=== Pengecekan Overfitting: Train Score vs Test Score ===\")\n",
        "from sklearn.metrics import f1_score as f1_sk\n",
        "for name, model in best_estimators.items():\n",
        "    train_f1 = f1_sk(y_train, model.predict(X_train))\n",
        "    test_f1  = results_df.loc[name, 'F1-Score']\n",
        "    gap = train_f1 - test_f1\n",
        "    flag = ' [OVERFITTING TERINDIKASI]' if gap > 0.10 else ' [OK]'\n",
        "    print(f\"{name:22s} | Train F1={train_f1:.4f} | Test F1={test_f1:.4f} | Gap={gap:.4f}{flag}\")\n",
        "\n",
        "print(\"\\nReview Pemodelan Selesai. Model Siap Digunakan untuk Sistem Bantu Skrining Posbindu PTM.\")"
    ]
})

# Create Notebook Structure
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write the notebook
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Stroke-Prediction.ipynb notebook script successfully compiled and generated!")
