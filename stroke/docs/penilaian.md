# Panduan Kriteria Penilaian & Hal yang Harus Dihindari (Uji Sertifikasi SKKNI Ilmuwan Data)

Dokumen ini berisi panduan kriteria penilaian, _checklist_ kelulusan per unit kompetensi SKKNI, serta daftar kesalahan fatal (_fatal errors_) yang **HARUS DIHINDARI** selama pelaksanaan uji demonstrasi sertifikasi Data Science (Studi Kasus: Skrining Risiko Stroke Posbindu PTM Semarang).

---

## FAKTOR KEGAGALAN / HAL YANG HARUS DIHINDARI (_FATAL ERRORS_)

### 1. Kebocoran Data (_Data Leakage_) — _Fatal Error Utama_

- **JANGAN** melakukan Imputasi (`SimpleImputer`), Penskalaan (`StandardScaler`), atau Oversampling (`SMOTE`) secara **global** pada seluruh dataset sebelum pembagian `train_test_split`.
- **JANGAN** memanggil `SMOTE` langsung pada data utuh $X$. SMOTE wajib terbungkus di dalam `imblearn.pipeline.Pipeline` agar hanya membangkitkan sampel sintetis pada _training fold_ CV.
- **JANGAN** memanggil `fit_transform()` pada data uji (`X_test`). Data uji **HANYA BOLEH** dipanggil menggunakan fungsi `.transform()`.

---

### 2. Salah Urutan Eksekusi Kode Program

- **JANGAN** mengeksekusi sel kode program secara mentah mengikuti nomor urut 1 s.d. 11.
- **Urutan Eksekusi Kode yang Benar**:
  $$\text{Unit 1} \rightarrow \text{2} \rightarrow \text{3} \rightarrow \text{4} \rightarrow \text{5} \rightarrow \mathbf{\text{Unit 8 (Split Data Uji Dulu!)}} \rightarrow \text{6 \& 7 (Pipeline Fit Train)} \rightarrow \text{9} \rightarrow \text{10} \rightarrow \text{11}$$
  _(Penulisan laporan tertulis tetap runtut 1–11, namun eksekusi kode program di Jupyter Notebook wajib mendahulukan Unit 8 Split)._

---

### 3. Salah Memilih Metrik Evaluasi & Terjebak "ROC Illusion"

- **JANGAN** mengandalkan **Akurasi** sebagai metrik utama pada dataset _imbalanced_ (29,78% vs 70,22%). Model tebak naif "0" memperoleh Akurasi **70,22%**, tetapi **Recall = 0%** (gagal menjaring pasien stroke).
- **JANGAN** hanya mengagungkan **ROC-AUC** tanpa mengecek **PR-AUC**. Metrik ROC-AUC rentan terhadap fenomena _ROC Illusion_ di mana dominasi sampel _True Negative_ menutupi kinerja buruk prediksi kelas minoritas stroke.
- **JANGAN** mengabaikan pengukuran **Brier Score Loss** sebagai indikator kalibrasi probabilitas klinis.

---

### 4. Bias Etis pada Pemodelan Medis Puskesmas

- **JANGAN** memasukkan fitur `SES` (_Socioeconomic Status_) ke dalam algoritma keputusan rujukan medis langsung di Puskesmas.
- **Catatan**: Memasukkan `SES` memicu _algorithmic bias_ yang diskriminatif terhadap pasien berpendapatan rendah, melanggar asas keadilan (_equity principle_) pelayanan kesehatan publik Kemenkes.

---

### 5. Kesalahan Teknis Penggunaan SHAP (Explainable AI)

- **JANGAN** mencoba menghitung _SHAP Interaction Values_ secara langsung dari model Neural Network (`MLPClassifier`) atau `LogisticRegression` menggunakan `TreeExplainer` (akan mengalami _error_ runtime).
- **Rekomendasi**: Jika model terbaik yang terpilih adalah model non-pohon (misal MLP), **WAJIB** gunakan model tree-based terbaik (seperti LightGBM atau Random Forest) sebagai substitusi untuk menghitung matriks interaksi SHAP, dan sertakan alasan tertulisnya.

---

### 6. Mengabaikan Adaptasi Ambang Batas (_Threshold Tuning_)

- **JANGAN** pasrah menggunakan _default threshold_ `0.5` pada seluruh skenario operasional medis.
- **Wajib memberikan rekomendasi penyesuaian threshold**:
    - **0.30 – 0.40**: Skrining Massal Posbindu PTM (mengejar **Recall > 85%**).
    - **0.55 – 0.65**: Rujukan Faskes Terbatas / Dokter Spesialis (mengejar **Precision > 75%**).

---

### 7. Menerima Klaim Jurnal Acuan Tanpa Critical Thinking

- **JANGAN** menelan mentah-mentah klaim F1-Score = 0,74 dari paper acuan (Tang et al., 2026).
- **Catatan**: Asesor akan memberikan **nilai tambah tinggi** jika Anda mampu membuktikan audit matematis dari _Confusion Matrix_ paper ($TP=325, FP=197, TN=1216, FN=262$) bahwa F1 sejati paper sebenarnya adalah **0,5861**.

---

## CHECKLIST KELULUSAN PER UNIT KOMPETENSI SKKNI

| No  | Kode Unit SKKNI   | Nama Unit Kompetensi                  | Poin Kunci Penilaian Asesor                                                                                  | Status Verifikasi |
| :-: | :---------------- | :------------------------------------ | :----------------------------------------------------------------------------------------------------------- | :---------------: |
|  1  | `J.62DMI00.001.1` | Menentukan Objektif Bisnis            | Merumuskan sasaran SMART Posbindu PTM Semarang sebelum sasaran teknis data science.                          |     Terpenuhi     |
|  2  | `J.62DMI00.002.1` | Menentukan Tujuan Teknis Data Science | Memetakan ke tugas klasifikasi biner, memilih Recall & PR-AUC, dan menjelaskan risiko Akurasi.               |     Terpenuhi     |
|  3  | `J.62DMI00.005.1` | Menelaah Data (EDA)                   | Mengidentifikasi _class imbalance_ (30:70), pergeseran usia stroke (>65), dan korelasi gula darah.           |     Terpenuhi     |
|  4  | `J.62DMI00.006.1` | Memvalidasi Data                      | Memverifikasi missingness BMI (<10%), 0 duplikasi, rentang fisiologis logis, & membuat kesimpulan kelayakan. |     Terpenuhi     |
|  5  | `J.62DMI00.007.1` | Menentukan Objek Data                 | Memisahkan $X$ dan $y$, serta mendokumentasikan pengeluaran fitur `SES` dari model medis demi etika.         |     Terpenuhi     |
|  6  | `J.62DMI00.008.1` | Membersihkan Data                     | Menjelaskan teknik imputasi median `BMI` terisolasi di dalam pipeline untuk mencegah _data leakage_.         |     Terpenuhi     |
|  7  | `J.62DMI00.009.1` | Mengkonstruksi Data                   | Menerapkan `StandardScaler` (Z-Score), `OneHotEncoder` (OHE), dan membungkusnya dalam `ColumnTransformer`.   |     Terpenuhi     |
|  8  | `J.62DMI00.012.1` | Membangun Skenario Model              | Menerapkan _Stratified Split 80:20_ sebelum pipeline fit, 5-Fold CV, dan menyiapkan 5 algoritma kandidat.    |     Terpenuhi     |
|  9  | `J.62DMI00.013.1` | Membangun Model                       | Mengintegrasikan SMOTE ke dalam `ImbPipeline` (fit train only) & melakukan `GridSearchCV`.                   |     Terpenuhi     |
| 10  | `J.62DMI00.014.1` | Mengevaluasi Hasil Pemodelan          | Menyajikan tabel 7 metrik evaluasi pada data uji (2.000 sampel), confusion matrix, & analisis ROC Illusion.  |     Terpenuhi     |
| 11  | `J.62DMI00.015.1` | Melakukan Review Pemodelan            | Mengukur overfitting train vs test, SHAP summary plot, interaksi Hipertensi-Jantung (0.086), & audit paper.  |     Terpenuhi     |

---

## STRATEGI MENJAWAB SAAT ASESMEN LISAN DENGAN ASESOR (DOKUMEN POTENSI KELULUSAN)

---

### Konsep Penilaian Asesmen Lisan BNSP

Dalam sistem asesmen BNSP, wawancara lisan digunakan oleh Asesor untuk memverifikasi bahwa kandidat **benar-benar memahami konsep di balik kode program** dan tidak sekadar _copy-paste_.

Kandidat yang mampu menjawab pertanyaan teknis secara **lancar, sistematis, dan berlandaskan logika medis/data science yang benar** akan secara meyakinkan dinilai **BERPOTENSI / KOMPETEN (K)**.

---

### DAFTAR PERTANYAAN POTENSIAL ASESOR & JAWABAN REKOMENDASI

#### 1. Pertanyaan Metodologi & Kebocoran Data (_Data Leakage_)

- **Asesor**: _"Mengapa Anda mendahulukan Unit 8 (Split Data 80:20) sebelum eksekusi Unit 6 (Membersihkan Data) dan Unit 7 (Mengkonstruksi Data) pada kode program?"_
    - **Jawaban Rekomendasi**: _"Secara urutan laporan tertulis, Unit 6 dan 7 berada lebih awal. Namun secara teknis kode program, data uji (test set) HARUS diisolasi terlebih dahulu menggunakan train_test_split (Unit 8) sebelum imputer, scaler, dan SMOTE di-fit. Jika imputer atau scaler di-fit sebelum split, parameter statistik data uji akan bocor ke data latih (data leakage), sehingga evaluasi model menjadi bias dan terlalu optimistis secara palsu."_

- **Asesor**: _"Mengapa teknik SMOTE dipasang di dalam `imblearn.pipeline.Pipeline`, bukan dipanggil secara terpisah pada dataset di luar pipeline?"_
    - **Jawaban Rekomendasi**: _"Jika SMOTE dipanggil langsung di luar pipeline pada seluruh dataset, SMOTE akan membangkitkan data sintetis dari sampel data uji. Dengan membungkus SMOTE di dalam ImbPipeline, SMOTE secara otomatis HANYA dieksekusi pada data latih di setiap fold cross-validation, sehingga data uji tetap 100% steril dan belum pernah dilihat oleh model."_

---

#### 2. Pertanyaan Evaluasi Metrik & Karakteristik Data

- **Asesor**: _"Mengapa Anda tidak menggunakan metrik Akurasi sebagai acuan utama menentukan model terbaik?"_
    - **Jawaban Rekomendasi**: _"Dataset ini mengalami imbalanced class (29.78% kasus stroke vs 70.22% non-stroke). Jika kita membuat model naif yang menebak seluruh pasien sebagai 'sehat', model tersebut akan mendapat Akurasi 70.22%, tetapi nilai Recall = 0% yang sangat berbahaya karena gagal mendeteksi satu pun pasien stroke. Oleh karena itu, kami mengutamakan F1-Score, Recall, dan PR-AUC."_

- **Asesor**: _"Apa yang dimaksud dengan fenomena 'ROC Illusion' dan mengapa PR-AUC lebih disarankan?"_
    - **Jawaban Rekomendasi**: _"ROC Illusion terjadi ketika metrik ROC-AUC terlihat sangat tinggi (misal 0,83) padahal performa pada kelas minoritas sebenarnya rendah. Hal ini terjadi karena True Negative Rate yang dominan meredam nilai False Positive Rate. Sebaliknya, PR-AUC berfokus langsung pada trade-off Precision dan Recall untuk kelas positif (stroke), sehingga memberikan gambaran evaluasi yang jauh lebih jujur pada data imbalanced."_

- **Asesor**: _"Bisakah Anda menjelaskan secara singkat 5 algoritma model yang Anda uji dan alasan pemilihan LightGBM sebagai model terbaik?"_
    - **Jawaban Rekomendasi**: _"Kami menguji 5 algoritma dari berbagai keluarga pemodelan: (1) **Logistic Regression** (baseline linier), (2) **MLP Classifier** (Neural Network dengan Recall tertinggi 72,40%), (3) **Random Forest** (Tree Ensemble Bagging), (4) **XGBoost** (Gradient Boosting sekuensial), dan (5) **LightGBM** (Leaf-wise Tree Boosting). LightGBM terpilih sebagai model terbaik karena secara konsisten mengungguli model lain pada metrik keseimbangan data imbalanced: **F1-Score 0,6535**, **PR-AUC 0,7120**, **ROC-AUC 0,8365**, serta kalibrasi probabilitas klinis terbaik (**Brier Loss 0,1385**)."_

- **Asesor**: _"Mengapa Anda mengukur Brier Score Loss pada evaluasi model medis?"_
    - **Jawaban Rekomendasi**: _"Brier Score mengukur kalibrasi probabilitas—seberapa dekat probabilitas prediksi model dengan kenyataan. Nilai Brier Score yang rendah (seperti 0,1385 pada LightGBM) menandakan probabilitas risiko yang dikeluarkan AI terkalibrasi secara akurat, sehingga dokter dapat mempercayai persentase risiko pasien."_

---

### RINGKASAN KARAKTERISTIK & PERFORMA 5 ALGORITMA MODEL

1. **Logistic Regression (Baseline Linear)**
    - **Keluarga**: Model Linier Probabilistik.
    - **Karakteristik**: Membutuhkan penskalaan `StandardScaler`, cepat, mudah diinterpretasikan.
    - **Hasil**: F1 = `0,6063` | PR-AUC = `0,6650` | Recall = `68,24%` _(Baseline pembanding)_.

2. **MLP Classifier (Neural Network / Multilayer Perceptron)**
    - **Keluarga**: Deep Learning / Artificial Neural Network.
    - **Karakteristik**: Sangat sensitif terhadap skala fitur, mampu menangkap pola interaksi non-linier kompleks.
    - **Hasil**: F1 = `0,6302` | PR-AUC = `0,6850` | **Recall = `72,40%`** _(Terbaik menjaring potensi kasus stroke)_.

3. **Random Forest Classifier (Tree Ensemble Bagging)**
    - **Keluarga**: Decision Tree Bagging Ensemble.
    - **Karakteristik**: Invarian terhadap skala data, robus terhadap pencilan/outlier, melatih banyak pohon secara paralel.
    - **Hasil**: F1 = `0,6315` | PR-AUC = `0,6912` | Recall = `64,20%`.

4. **XGBoost Classifier (Extreme Gradient Boosting)**
    - **Keluarga**: Gradient Boosting Framework.
    - **Karakteristik**: Membangun pohon sekuensial yang memperbaiki error pohon sebelumnya dengan regularisasi L1/L2.
    - **Hasil**: F1 = `0,6429` | PR-AUC = `0,7045` | Recall = `65,10%`.

5. **LightGBM Classifier (Optimized Pipeline - Best Model)**
    - **Keluarga**: Leaf-wise Tree Gradient Boosting.
    - **Karakteristik**: Eksekusi sangat cepat, efisiensi memori tinggi, optimal untuk _imbalanced dataset_.
    - **Hasil**: **F1 = `0,6535`** | **PR-AUC = `0,7120`** | **ROC-AUC = `0,8365`** | **Brier Loss = `0,1385`** _(Model Terbaik untuk Terpasang di Puskesmas)_.

---

#### 3. Pertanyaan Etika, Transformation, & Preprocessing Data

- **Asesor**: _"Mengapa fitur Socioeconomic Status (SES) dikeluarkan dari variabel prediktor pemodelan medis?"_
    - **Jawaban Rekomendasi**: _"Secara etis pelayanan publik di Puskesmas, keputusan rujukan klinis wajib menjunjung asas keadilan (equity principle) dan murni didasarkan pada indikator fisiologis pasien. Memasukkan SES dapat memicu algorithmic bias di mana pasien berpendapatan rendah menerima skor prioritas yang diskriminatif. Namun variabel SES tetap dipetakan untuk analisis kebijakan sosial oleh Dinas Kesehatan."_

- **Asesor**: _"Mengapa Anda memilih `SimpleImputer` dengan strategi median untuk mengimputasi BMI?"_
    - **Jawaban Rekomendasi**: _"Nilai BMI pada populasi kesehatan seringkali memiliki distribusi miring (skewed) dan rentan terhadap nilai ekstrem. Median lebih robus (tahan) terhadap pencilan dibanding rata-rata (mean), sehingga memberikan estimasi pengisian data kosong yang lebih wajar secara biologis."_

- **Asesor**: _"Mengapa Anda menggunakan `OneHotEncoder` untuk variabel kategorikal nominal (seperti gender, smoking_status) bukannya `LabelEncoder`?"_
    - **Jawaban Rekomendasi**: _"LabelEncoder mengubah kategori menjadi angka berurut (0, 1, 2). Hal ini membahayakan algoritma karena menciptakan asumsi hubungan ordinal hirarki palsu (misal 2 dianggap lebih besar dari 0). OneHotEncoder memecahnya menjadi kolom biner 0/1 yang independen tanpa memaksakan bobot urutan."_

- **Asesor**: _"Mengapa penskalaan `StandardScaler` (Z-Score) wajib digunakan pada `LogisticRegression` dan `MLPClassifier`, tetapi tidak berdampak pada `RandomForest` atau `LightGBM`?"_
    - **Jawaban Rekomendasi**: _"Logistic Regression dan Neural Network (MLP) mengandalkan perhitungan fungsi jarak dan penurunan gradien (gradient descent), sehingga perbedaan rentang angka antar-fitur (misal Usia 0-80 vs Gula Darah 50-300) akan mendistorsi pembobotan. Sebaliknya, model berbasis pohon (Random Forest, LightGBM) melakukan pemisahan (split) berdasarkan threshold independen pada tiap fitur, sehingga invarian terhadap transformasi monotonik skala."_

---

#### 4. Pertanyaan Review Pemodelan & Interpretasi SHAP (XAI)

- **Asesor**: _"Mengapa Anda memilih `StratifiedKFold` (5 Folds) untuk validasi silang (cross-validation) dibanding `KFold` acak biasa?"_
    - **Jawaban Rekomendasi**: _"Karena dataset ini imbalanced (30:70), KFold biasa berisiko menghasilkan fold validasi yang porsi kelas minoritasnya tidak seimbang (terlalu sedikit atau terlalu banyak). StratifiedKFold menjamin bahwa setiap fold memiliki proporsi kelas stroke 29,78% secara presisi."_

- **Asesor**: _"Bagaimana Anda memastikan model yang dilatih tidak mengalami overfitting?"_
    - **Jawaban Rekomendasi**: _"Kami membandingkan nilai F1-Score pada data latih (train) dan data uji (test). Pada kelima model yang diuji, selisih F1-Score (gap) berada di bawah 0.08 (misal LightGBM: Train 0.7240 vs Test 0.6535), yang mengonfirmasi bahwa model bebas dari overfitting parah dan memiliki daya generalisasi yang baik."_

- **Asesor**: _"Fitur apa yang paling berpengaruh menurut SHAP dan bagaimana pola interaksinya?"_
    - **Jawaban Rekomendasi**: _"Berdasarkan SHAP TreeExplainer pada model LightGBM, 3 fitur paling dominan adalah Usia (Age), Gula Darah Rata-Rata (Avg_Glucose), dan BMI. Selain itu, ditemukan SHAP Interaction Value yang kuat antara Hipertensi dan Penyakit Jantung (0.086), yang membuktikan risiko stroke melonjak eksponensial jika kedua komorbid diderita bersamaan."_

- **Asesor**: _"Jika model terbaik yang dipasang di Puskesmas adalah MLP (Neural Network), bagaimana cara menyajikan SHAP Interaction Values-nya?"_
    - **Jawaban Rekomendasi**: _"Karena algoritma TreeExplainer secara native hanya mendukung model berbasis pohon, jika model non-tree seperti MLP dipilih, perhitungan SHAP Interaction Values disajikan menggunakan substitusi dari model tree-based terbaik (LightGBM/Random Forest) sesuai petunjuk teknis Unit 11, karena KernelExplainer pada MLP tidak mendukung matriks interaksi pairwise secara native."_

- **Asesor**: _"Mengapa ambang batas (threshold) klasifikasi perlu disesuaikan untuk skenario Posbindu PTM?"_
    - **Jawaban Rekomendasi**: _"Secara default threshold adalah 0.5. Namun untuk skenario Skrining Massal di Posbindu PTM, threshold diturunkan ke 0.30–0.40 agar Sensitivitas/Recall meningkat (>85%), sehingga tidak ada warga berisiko stroke yang terlewat. Sedangkan untuk rujukan spesialis terbatas, threshold dinaikkan ke 0.55–0.65 untuk meningkatkan Presisi."_

---

#### 5. Pertanyaan Audit Paper Acuan (Tang et al., 2026)

- **Asesor**: _"Bagaimana hasil pemodelan Anda jika dibandingkan dengan paper acuan Tang et al. (2026)?"_
    - **Jawaban Rekomendasi**: _"Hasil pemodelan kami sangat sepadan dan presisi. Skor ROC-AUC (0,8365) dan PR-AUC (0,7120) kami identik dengan paper (0,836 & 0,71). Selain itu, melalui audit skeptis matematis pada Confusion Matrix paper (TP=325, FP=197, TN=1216, FN=262), kami menemukan F1 sejati paper adalah 0.5861 (bukan 0.74 seperti klaim teks). F1-Score model LightGBM kami (0.6535) terbukti lebih unggul dan terkalibrasi."_

---

### Ringkasan Kunci Sikap Saat Wawancara Asesmen

1. **Tenang & Percaya Diri**: Gunakan istilah ilmiah yang tepat (_imbalanced data_, _data leakage_, _PR-AUC_, _explainability_).
2. **Berorientasi Medis & Bisnis**: Setiap penjelasan teknik selalu dikaitkan dengan dampak keselamatan pasien dan efisiensi operasional Puskesmas.
3. **Paham Alur Kode**: Mampu menunjukkan sel mana yang mengeksekusi _split_, _pipeline_, _tuning_, dan _SHAP_.
