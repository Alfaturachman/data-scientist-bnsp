# Dokumentasi Proyek Berdasarkan CRISP-DM & SKKNI (Diabetes Prediction)

## 1. J.62DMI00.001.1 — Menentukan Objektif Bisnis

- **Menghemat Biaya Pengobatan:** Mendeteksi diabetes sedini mungkin agar pasien dan rumah sakit tidak perlu mengeluarkan biaya yang sangat besar untuk mengobati komplikasi penyakit yang parah di masa depan.
- **Pemeriksaan yang Lebih Praktis dan Cepat:** Tes diabetes konvensional seringkali repot karena mengharuskan pasien berpuasa, mengambil darah berkali-kali, dan butuh alat medis khusus. Penelitian ini bertujuan menciptakan cara pemeriksaan yang jauh lebih cepat, mudah, dan bisa digunakan secara luas bahkan di daerah dengan fasilitas kesehatan yang terbatas.
- **Membantu Keputusan Dokter dengan Sistem yang Handal:** Mengembangkan sistem kecerdasan buatan (AI) yang dijaga ketat agar hasil prediksinya benar-benar akurat dan terhindar dari bias atau kebocoran data (leakage-safe). Sistem ini ditujukan untuk menjadi alat bantu praktis bagi dokter dalam mengambil keputusan klinis, yang nantinya bisa diterapkan di berbagai rumah sakit dengan pasien yang beragam.

## 2. J.62DMI00.002.1 — Menentukan Tujuan Teknis Data Science

### 2.1 Membangun Alur Kerja yang Bebas Kebocoran Data (Leakage-Safe Workflow)

Tujuan teknis utama dari penelitian ini adalah merancang alur machine learning yang sangat ketat untuk mencegah kebocoran data (data leakage). Sistem ini memastikan bahwa seluruh tahapan pra-pemrosesan data—termasuk pengisian data kosong dan standarisasi skala—hanya dilakukan secara eksklusif pada data latih (training folds) melalui validasi silang, sedangkan data uji (held-out test set) dibiarkan sepenuhnya tidak tersentuh untuk pengujian akhir.

### 2.2 Pembersihan Data dan Imputasi yang Berlandaskan Logika Medis

Mengatasi masalah kualitas data pada dataset dengan cara mengidentifikasi angka "nol" yang secara klinis tidak mungkin (misalnya pada metrik Glukosa, Tekanan Darah, Insulin, Ketebalan Kulit, dan BMI) dan menganggapnya sebagai data yang hilang. Tujuan teknisnya adalah melakukan teknik imputasi (pengisian nilai) menggunakan nilai tengah berdasarkan kelas pasien (class-conditional medians) agar pengisian data tersebut tetap masuk akal secara medis.

### 2.3 Membuat Rekayasa Fitur (Feature Engineering) Tambahan

Selain mengandalkan 8 atribut data mentah bawaan, penelitian ini bertujuan merancang 16 fitur turunan (komposit) baru. Fitur ini direkayasa dalam bentuk batasan nilai (thresholds), rasio, maupun interaksi antar-variabel asli (seperti rasio glukosa atau perkalian BMI) guna memberikan sinyal data yang lebih kaya ke dalam algoritma.

Adapun 8 atribut data mentah bawaan tersebut adalah:

1. **Pregnancies**: Jumlah kehamilan.
2. **Glucose**: Konsentrasi glukosa plasma.
3. **BloodPressure**: Tekanan darah diastolik.
4. **SkinThickness**: Ketebalan lipatan kulit trisep.
5. **Insulin**: Insulin serum 2 jam.
6. **BMI**: Indeks massa tubuh.
7. **DiabetesPedigreeFunction**: Riwayat diabetes keluarga.
8. **Age**: Usia pasien.

Sedangkan 16 fitur turunan (komposit) baru yang dirancang adalah:

1. **Normal_SkinThickness**: Indikator ketebalan kulit normal (<= 20 mm) untuk memisahkan kasus lemak subkutan rendah dengan yang tinggi.
2. **Healthy_BMI**: Indikator indeks massa tubuh sehat (<= 30 kg/m²) untuk membedakan kelompok non-obesitas dengan obesitas (faktor risiko utama diabetes).
3. **Young_Low_Pregnancies**: Indikator pasien muda (<= 30 tahun) dengan kehamilan sedikit (<= 6) untuk menangkap sub-grup berisiko lebih rendah dibanding pola usia tua/kehamilan tinggi.
4. **Optimal_Glucose_BP**: Indikator glukosa (<= 105 mg/dL) dan tekanan darah diastolik (<= 80 mmHg) optimal sebagai penanda profil protektif tubuh yang sehat.
5. **Young_Normal_Glucose**: Indikator pasien muda (<= 30 tahun) dengan glukosa normal (<= 120 mg/dL) untuk mengisolasi kelompok yang secara umum berisiko sangat rendah.
6. **Healthy_BMI_SkinThickness**: Indikator kombinasi BMI sehat (<= 30 kg/m²) dan ketebalan kulit normal (<= 20 mm) untuk menandai karakteristik fisik (fenotipe) yang secara umum kurus/lean.
7. **Optimal_Glucose_BMI**: Indikator kombinasi glukosa optimal (<= 105 mg/dL) dan BMI sehat (<= 30 kg/m²) sebagai sinyal negatif kuat terhadap kecenderungan diabetes.
8. **Normal_Insulin**: Indikator insulin normal (< 200 mu U/ml) yang menunjukkan respons fisiologis insulin 2 jam yang normal setelah pembebanan glukosa.
9. **Normal_BloodPressure**: Indikator tekanan darah diastolik normal (< 80 mmHg), mengingat hipertensi sangat berkorelasi dengan risiko metabolik.
10. **Moderate_Pregnancies**: Indikator jumlah kehamilan sedang (1 sampai 3 kali) untuk memisahkan kelompok paritas menengah dengan paritas sangat rendah/tinggi.
11. **BMI_SkinThickness_Product**: Hasil kali BMI dengan SkinThickness (BMI \* SkinThickness) sebagai variabel interaksi untuk mengukur gabungan kegemukan umum (adipositas sistemik) dan distribusi lemak di bawah kulit (subkutan).
12. **Pregnancy_Age_Ratio**: Rasio jumlah kehamilan terhadap usia (Pregnancies / (Age + 1)) untuk menormalisasi jumlah kehamilan terhadap lama usia paparan risiko.
13. **Glucose_DiabetesPedigree_Ratio**: Rasio glukosa terhadap Diabetes Pedigree Function (Glucose / (DPF + 1e-6)) untuk menormalisasi kadar glukosa darah terhadap risiko keturunan/genetik pasien.
14. **Age_DiabetesPedigree_Product**: Hasil kali usia dengan Diabetes Pedigree Function (Age \* DPF) untuk memodelkan penguatan efek riwayat keluarga seiring bertambahnya usia.
15. **Age_Insulin_Ratio**: Rasio usia terhadap tingkat insulin (Age / (Insulin + 1e-6)) untuk membandingkan pertambahan usia terhadap kekuatan respons sekresi insulin.
16. **Low_BMI_SkinThickness_Product**: Indikator jika hasil kali BMI dan SkinThickness bernilai rendah (< 1034) sebagai penanda tingkat adipositas global yang sangat rendah.

## 3. J.62DMI00.005.1 — Menelaah Data (Data Understanding atau Exploratory Data Analysis)

Berikut adalah hasil penelaahan data yang dikelompokkan per poin:

- **Sumber dan Profil Subjek**: Data yang digunakan berasal dari Kaggle (UCI Machine Learning) berupa _Pima Indians Diabetes Dataset_ (PIDD) yang berisi 768 rekam medis eksklusif untuk pasien wanita berusia >= 21 tahun dari keturunan suku Indian Pima.
- **Atribut dan Variabel Target**: Dataset terdiri dari 8 fitur atribut kesehatan numerik (Pregnancies/Jumlah Kehamilan, Glucose/Glukosa, Blood Pressure/Tekanan Darah, Skin Thickness/Ketebalan Kulit, Insulin, BMI, Diabetes Pedigree Function/Fungsi Silsilah Keluarga, dan Age/Usia) serta 1 variabel target biner `Outcome` yang menentukan status pasien (1 = positif diabetes, 0 = negatif/tidak diabetes).
- **Proporsi Kelas yang Tidak Seimbang (Class Imbalance)**: Dataset ini memiliki ketidakseimbangan kelas tingkat sedang. Dari total 768 rekam medis, terdapat 500 pasien (65,1%) yang tidak menderita diabetes, dibandingkan dengan 268 pasien (34,9%) yang terdiagnosis positif diabetes.
- **Penemuan Masalah Kualitas Data (Implausible Zeros)**: Dataset ini sekilas tampak utuh karena tidak ada format nilai kosong seperti NaN secara eksplisit. Namun, telaah data menemukan banyak nilai "0" (nol) pada atribut yang secara biologis tidak mungkin bernilai nol bagi manusia hidup, seperti Tekanan Darah, Glukosa, Insulin, Ketebalan Kulit, dan BMI. Peneliti menyimpulkan bahwa nilai nol ini sebenarnya adalah indikator data yang hilang (missing data) akibat kesalahan perekaman.
- **Tingkat Keparahan Data yang Hilang**: Mayoritas kekosongan data atau nilai nol tersebut sangat terkonsentrasi pada fitur Insulin (sekitar 30% data bernilai nol) dan Ketebalan Kulit (sekitar 29% data bernilai nol), sementara fitur lainnya seperti Tekanan Darah, BMI, dan Glukosa hanya mengalami sedikit kehilangan data.
- **Validasi Statistik terhadap Logika Medis**: Dari pengecekan statistik, nilai rata-rata keseluruhan pasien untuk glukosa adalah sekitar 121, BMI sekitar 33, dan fitur Insulin menunjukkan variasi yang besar. Data eksplorasi juga memvalidasi bahwa kelompok pasien positif diabetes memiliki rata-rata kadar glukosa dan BMI yang lebih tinggi dibandingkan kelompok negatif, yang mana temuan ini sangat sejalan dan masuk akal dengan faktor risiko klinis penyakit diabetes di dunia nyata.

## 4. J.62DMI00.006.1 — Memvalidasi Data

Validasi data dilakukan dengan mengidentifikasi nilai nol (0) yang tidak masuk akal secara klinis pada fitur-fitur yang seharusnya tidak pernah bernilai nol pada individu hidup, yaitu Glucose, Blood Pressure, Skin Thickness, Insulin, dan BMI. Nilai-nilai tersebut diidentifikasi sebagai data hilang yang direkam sebagai nol. Hasil validasi menunjukkan bahwa sekitar 30% rekam memiliki nilai Insulin = 0 dan sekitar 29% memiliki nilai Skin Thickness = 0, yang mengindikasikan tingkat missingness yang signifikan dan perlu ditangani sebelum pemodelan. Dataset tidak memiliki nilai NaN eksplisit, namun nilai-nilai nol yang tidak valid tersebut harus dikoreksi agar analisis menghasilkan keluaran yang akurat.

## 5. J.62DMI00.007.1 — Menentukan Objek Data

Objek data dalam proyek ini adalah seluruh 768 rekam klinis pasien perempuan keturunan Pima Indians dengan rentang usia 21 hingga 81 tahun. Setiap rekam menjadi satu unit observasi dengan 8 atribut asli dan diperkaya menjadi 24 fitur setelah rekayasa fitur (8 asli + 16 komposit). Variabel target adalah kolom Outcome yang merupakan label biner prediksi status diabetes. Rentang nilai fitur yang relevan antara lain: Pregnancies (0–17), Glucose (0–199), Blood Pressure (0–122), Skin Thickness (0–99), Insulin (0–846), BMI (0–67,1), Diabetes Pedigree Function (0,078–2,42), dan Age (21–81).

## 6. J.62DMI00.008.1 — Membersihkan Data

Pembersihan data dilakukan dengan mengidentifikasi dan mengganti nilai nol yang tidak valid secara klinis pada fitur Glucose, Blood Pressure, Skin Thickness, Insulin, dan BMI dengan nilai NaN. Pada skenario legal (Eksperimen 1 dan 3), imputasi dilakukan menggunakan KNN Imputer (k=5) yang dilatih secara eksklusif pada data latih (training set) untuk mencegah kebocoran data. Sedangkan pada skenario bias (Eksperimen 2), digunakan ClassConditionalMedianImputer secara global pada seluruh dataset yang memicu kebocoran target. Ketidakseimbangan kelas ditangani menggunakan class-weighted loss (balanced) pada model LightGBM dan Random Forest, serta scale_pos_weight pada XGBoost. Standardisasi fitur kontinu menggunakan StandardScaler hanya diterapkan pada data latih dan diterapkan pada data uji secara terpisah.

## 7. J.62DMI00.009.1 — Mengkonstruksi Data

Konstruksi data dilakukan melalui rekayasa fitur dengan menambahkan 16 fitur komposit yang bermakna secara klinis ke dalam 8 atribut asli. Fitur-fitur baru ini mencakup indikator biner berbasis ambang batas (seperti Healthy_BMI, Normal_Insulin, Optimal_Glucose_BP) dan fitur kontinu berbasis interaksi atau rasio (seperti BMI_SkinThickness_Product, Pregnancy_Age_Ratio, Glucose_DiabetesPedigree_Ratio, Age_DiabetesPedigree_Product, dan Age_Insulin_Ratio). Setiap fitur disertai formula eksak dan justifikasi klinis yang jelas, misalnya fitur Pregnancy_Age_Ratio digunakan untuk menormalisasi jumlah kehamilan terhadap usia sebagai proksi paparan risiko. Korelasi antar fitur asli dan fitur rekayasa kemudian divisualisasikan dalam matriks korelasi untuk mendeteksi redundansi dan multikolinearitas sebelum pelatihan model.

## 8. J.62DMI00.012.1 — Membangun Skenario Model

Tiga skenario eksperimen pemodelan dibangun untuk membandingkan pendekatan pipeline yang berbeda:

- Eksperimen 1 — Split-First Pipeline (Baseline): Data dibagi terlebih dahulu secara stratifikasi (80/20), kemudian preprocessing (KNN Imputer k=5 dan StandardScaler) di-fit hanya pada data latih dan diterapkan pada data uji secara legal. Pelatihan model dilakukan menggunakan stratified 5-fold cross-validation.
- Eksperimen 2 — Preprocess-First Pipeline (Leakage): Preprocessing (Class-Conditional Median Imputation dan StandardScaler) serta penyeimbangan data menggunakan SMOTE dilakukan secara global pada seluruh dataset sebelum pemisahan data train-test dilakukan. Skenario ini mereplikasi alur bias dari jurnal acuan.
- Eksperimen 3 — Optimized Pipeline: Menggunakan alur Split-First dari Eksperimen 1, kemudian ditambahkan optimasi hyperparameter secara legal menggunakan Optuna Bayesian Search (50 trials) dengan nested cross-validation pada data latih untuk mencari kombinasi parameter terbaik.

## 9. J.62DMI00.013.1 — Membangun Model

Model dibangun menggunakan empat algoritma klasifikasi utama dalam eksperimen replikasi (LightGBM, XGBoost, Gradient Boosting, dan Random Forest), namun proyek ini juga membandingkan serta menganalisis 8 algoritma klasifikasi yang umum digunakan dalam literatur prediksi diabetes (berdasarkan jurnal acuan).

Setiap model dilatih menggunakan data pelatihan yang telah dibersihkan secara legal. Pada Eksperimen 3, hyperparameter untuk LightGBM dioptimalkan secara otomatis menggunakan Optuna Bayesian Search dengan nested cross-validation guna mencegah kebocoran data uji selama proses penyetelan hyperparameter. Seluruh proses pelatihan memastikan bahwa data uji tetap dibekukan hingga tahap evaluasi akhir.

### 9.1 Penjelasan 8 Algoritma Klasifikasi

Berikut adalah penjelasan mengenai 8 model klasifikasi yang dianalisis dalam proyek ini, mulai dari cara kerja dasar hingga kelebihan dan relevansinya terhadap kasus deteksi diabetes:

#### 1. LightGBM (Light Gradient Boosting Machine)

- **Cara Kerja:** Merupakan varian _gradient boosting_ berbasis pohon keputusan yang dikembangkan oleh Microsoft. Berbeda dengan model boosting lain yang tumbuh secara horizontal (_level-wise_), LightGBM menumbuhkan pohon secara vertikal (_leaf-wise_) dengan mencari daun dengan penurunan kehilangan (_loss reduction_) maksimum. Algoritma ini memadukan teknik GOSS (_Gradient-based One-Side Sampling_) dan EFB (_Exclusive Feature Bundling_).
- **Relevansi Klinis:** Sangat efisien dalam penggunaan memori dan memiliki kecepatan latih tinggi. Sangat andal untuk menangani dataset dengan banyak fitur komposit (rekayasa fitur) serta memiliki parameter pembobotan kelas (`class_weight='balanced'`) bawaan untuk menyeimbangkan kelas pasien diabetes.

#### 2. XGBoost (Extreme Gradient Boosting)

- **Cara Kerja:** Implementasi _gradient boosting_ yang dioptimalkan untuk performa dan skalabilitas tinggi. XGBoost meminimalkan fungsi objektif yang menggabungkan _loss function_ dengan regularisasi tingkat lanjut (L1 Lasso dan L2 Ridge) pada struktur pohon untuk menekan risiko _overfitting_.
- **Relevansi Klinis:** Menyediakan kontrol regularisasi yang ketat, membuatnya sangat kokoh terhadap gangguan (_noise_) pada data klinis. XGBoost juga mendukung parameter `scale_pos_weight` untuk mengoreksi dampak bias ketidakseimbangan kelas pasien.

#### 3. Gradient Boosting Classifier (GBM)

- **Cara Kerja:** Algoritma _ensemble learning_ berbasis _boosting_ konvensional. GBM melatih pohon keputusan secara berurutan (sekuensial). Setiap pohon baru dirancang khusus untuk memprediksi sisa kesalahan (_residual error_) yang dihasilkan oleh pohon-pohon sebelumnya menggunakan metode optimalisasi _gradient descent_.
- **Relevansi Klinis:** Berfungsi sebagai model pembanding (_baseline_) yang kuat untuk algoritma boosting modern. GBM sangat stabil, meskipun membutuhkan waktu pelatihan lebih lama dibanding XGBoost atau LightGBM karena tidak mendukung paralelisasi penuh.

#### 4. Random Forest Classifier

- **Cara Kerja:** Algoritma berbasis _bagging_ (Bootstrap Aggregating) yang membangun kumpulan (hutan) pohon keputusan independen secara paralel. Pada saat melatih setiap pohon, pemilihan fitur dilakukan secara acak pada setiap percabangan (_node split_). Hasil akhir ditentukan berdasarkan suara terbanyak (_majority voting_).
- **Relevansi Klinis:** Sangat toleran terhadap pencilan (_outliers_) dan tidak memerlukan penskalaan fitur secara khusus. Algoritma ini andal dalam memodelkan interaksi non-linear yang rumit pada dataset diabetes tanpa mudah terjebak dalam _overfitting_.

#### 5. SVM (Support Vector Machine)

- **Cara Kerja:** Algoritma yang bekerja dengan cara mencari bidang pembatas (_hyperplane_) optimal untuk memisahkan kelas-kelas data dengan margin sebesar mungkin. Untuk data yang tidak dapat dipisahkan secara linear, SVM menggunakan fungsi kernel (misalnya Radial Basis Function / RBF) untuk memetakan data ke ruang dimensi yang lebih tinggi agar dapat dipisahkan secara linear.
- **Relevansi Klinis:** Efektif untuk dataset dengan jumlah sampel sedang dan dimensi fitur menengah setelah proses rekayasa fitur. SVM membutuhkan pembersihan data dan penskalaan (Standardization) yang sangat bersih karena sangat sensitif terhadap skala fitur.

#### 6. k-NN (k-Nearest Neighbors)

- **Cara Kerja:** Model klasifikasi berbasis instansi (_lazy learner_) yang tidak membangun model parametrik eksplisit selama pelatihan. Klasifikasi data baru didasarkan pada mayoritas kelas dari $k$ tetangga terdekatnya. Kedekatan ini diukur menggunakan fungsi jarak seperti _Euclidean Distance_.
- **Relevansi Klinis:** Sangat sederhana dan intuitif (pasien dikelompokkan dengan pasien lain yang memiliki karakteristik klinis serupa). Namun, k-NN sangat sensitif terhadap skala fitur dan performanya dapat menurun jika terdapat banyak fitur yang kurang relevan (_curse of dimensionality_).

#### 7. Decision Tree (Pohon Keputusan)

- **Cara Kerja:** Model pohon keputusan tunggal yang membagi dataset secara hierarkis menjadi subset yang lebih kecil berdasarkan kriteria kemurnian informasi, seperti _Gini Impurity_ atau _Entropy_, hingga mencapai daun keputusan akhir.
- **Relevansi Klinis:** Sangat mudah dibaca dan diinterpretasikan oleh praktisi medis karena meniru alur berpikir klinis dokter saat mendiagnosis pasien (misalnya: jika Glukosa > 120, lalu jika BMI > 30, dst). Kelemahannya adalah sangat rentan terhadap variabilitas data (_high variance_) dan mudah mengalami _overfitting_.

#### 8. Logistic Regression (Regresi Logistik)

- **Cara Kerja:** Model klasifikasi linear klasik yang memperkirakan probabilitas peluang (_odds ratio_) dari kelas target biner menggunakan fungsi logistik (Sigmoid). Output dari fungsi ini dibatasi dalam rentang 0 hingga 1.
- **Relevansi Klinis:** Merupakan standar emas pemodelan klinis tradisional karena koefisien modelnya langsung merepresentasikan tingkat risiko (_Odds Ratio_) dari masing-masing variabel prediktor (misalnya pengaruh peningkatan Glukosa terhadap peluang diabetes). Sangat efisien, namun kinerjanya terbatas jika pola hubungan antar-fitur bersifat non-linear dan kompleks.

### 9.2 Optimasi Hyperparameter (Hyperparameter Tuning)

Untuk mendapatkan performa model terbaik yang valid dan dapat diandalkan, dilakukan proses penalaan hyperparameter (_hyperparameter tuning_) menggunakan framework **Optuna** pada Eksperimen 3. Berikut adalah rincian metodologi tuning yang digunakan:

#### 1. Pendekatan Bayesian Optimization (TPE Sampler)

- **Framework:** Menggunakan **Optuna**, sebuah framework optimasi hyperparameter modern yang berbasis Python.
- **Metode Pencarian:** Dibandingkan dengan _Grid Search_ (yang lambat karena mencoba semua kombinasi) atau _Random Search_ (yang tidak terarah), Optuna menggunakan **Tree-structured Parzen Estimator (TPE)**. TPE merupakan algoritma optimasi Bayesian yang memodelkan distribusi hyperparameter yang menghasilkan performa baik secara probabilitas, sehingga pencarian ruang parameter menjadi jauh lebih efisien, terarah, dan cepat.
- **Jumlah Percobaan:** Proses optimasi diatur sebanyak **50 trials** (percobaan) untuk mencari kombinasi parameter terbaik.
- **Metrik Evaluasi Utama:** Metrik yang dioptimalkan (dimaksimalkan) adalah skor rata-rata **ROC-AUC** dari validasi silang.

#### 2. Validasi Silang Bebas Kebocoran Data (Leakage-Free Cross-Validation)

Tantangan terbesar dalam hyperparameter tuning adalah risiko kebocoran data (_data leakage_). Jika preprocessing (seperti imputasi nilai kosong dengan KNN Imputer dan standardisasi dengan StandardScaler) dilakukan di luar proses validasi silang, informasi dari fold validasi akan bocor ke fold pelatihan, menghasilkan estimasi performa yang bias dan terlampau optimistis.

Untuk mencegah hal tersebut, proyek ini menerapkan alur **Nested Cross-Validation bebas leakage** di dalam fungsi objektif Optuna:

1. Data pelatihan dibagi menjadi 5 bagian (_5-Fold Stratified Cross-Validation_).
2. Di setiap fold:
    - Penggantian nilai nol anomali menjadi `NaN` dilakukan secara terpisah.
    - Imputator `KNNImputer(n_neighbors=5)` di-fit **hanya** pada fold training saat itu, kemudian diterapkan untuk mengimputasi fold training dan fold validation.
    - Penskalaan `StandardScaler` di-fit **hanya** pada fold training saat itu, kemudian diterapkan untuk fold training dan fold validation.
    - Model dilatih dengan parameter uji pada fold training yang telah diproses, lalu dievaluasi pada fold validation.
3. Skor evaluasi (ROC-AUC) rata-rata dari 5 fold tersebut menjadi feedback bagi Optuna untuk menyarankan kombinasi hyperparameter berikutnya.

#### 3. Ruang Pencarian Hyperparameter (Search Space) untuk LightGBM

Model LightGBM dipilih untuk proses optimasi dengan ruang pencarian hyperparameter sebagai berikut:

- **`n_estimators` (Rentang: 50 s.d. 150):** Mengatur jumlah pohon keputusan (_boosting rounds_) yang akan dibangun. Terlalu sedikit dapat menyebabkan _underfitting_, sedangkan terlalu banyak dapat memicu _overfitting_.
- **`max_depth` (Rentang: 2 s.d. 5):** Membatasi kedalaman maksimum setiap pohon keputusan. Pembatasan ini sangat penting pada dataset kecil untuk mencegah pohon mempelajari pola bising (_noise_) secara terlalu spesifik.
- **`learning_rate` (Rentang: 0.01 s.d. 0.1, Skala Logaritmik):** Mengontrol ukuran langkah optimasi saat memperbarui bobot model. Nilai kecil membutuhkan pohon lebih banyak tetapi menghasilkan konvergensi yang lebih halus.
- **`num_leaves` (Rentang: 4 s.d. 16):** Menentukan jumlah maksimum daun dalam satu pohon. Karena LightGBM menggunakan pertumbuhan _leaf-wise_, parameter ini harus dikontrol agar tidak melebihi $2^{\text{max\_depth}}$ untuk menghindari kompleksitas berlebih.
- **`subsample` (Rentang: 0.6 s.d. 0.9):** Menentukan proporsi acak dari sampel data yang digunakan untuk melatih setiap pohon. Teknik ini menambahkan elemen acak untuk meningkatkan generalisasi model.
- **`colsample_bytree` (Rentang: 0.6 s.d. 0.9):** Menentukan proporsi acak dari fitur yang digunakan saat membangun setiap pohon keputusan, berguna untuk meminimalkan dampak korelasi antar-fitur yang tinggi (_collinearity_).

## 10. J.62DMI00.014.1 — Mengevaluasi Hasil Pemodelan

Evaluasi dilakukan pada held-out test set yang sepenuhnya terpisah menggunakan metrik Akurasi dan ROC-AUC. Hasil evaluasi menunjukkan:

- Eksperimen 1 (Baseline Valid): LightGBM memperoleh Akurasi = 73,38% dan ROC-AUC = 0,8019; XGBoost memperoleh Akurasi = 72,08% dan ROC-AUC = 0,8076; Gradient Boosting memperoleh Akurasi = 72,08% dan ROC-AUC = 0,8106; Random Forest memperoleh Akurasi = 69,48% dan ROC-AUC = 0,7931.
- Eksperimen 2 (Leakage - Global): Terjadi lonjakan performa semu akibat kebocoran data, dengan Random Forest mencapai Akurasi = 91,00% (ROC-AUC = 0,9684) dan LightGBM mencapai Akurasi = 90,50% (ROC-AUC = 0,9644). Ini mereplikasi performa fantastis yang dilaporkan pada jurnal acuan (Akurasi ensemble = 89,61% dan ROC-AUC = 94,52%).
- Eksperimen 3 (Optimized & Valid): Melalui tuning hyperparameter berbasis Optuna secara sah, model LightGBM mencapai peningkatan performa terbaik yang valid dengan Akurasi = 75,32% dan ROC-AUC = 0,8170.

### 10.1 Penjelasan Matriks Evaluasi Klasifikasi

Untuk memahami kinerja model secara mendalam, khususnya dalam konteks medis diagnosis diabetes dengan kelas data yang tidak seimbang (_class imbalance_), berikut adalah penjelasan mengenai matriks evaluasi utama yang digunakan:

#### 1. Accuracy (Akurasi)

- **Definisi:** Proporsi dari keseluruhan prediksi yang benar (baik positif maupun negatif) yang dilakukan oleh model dibandingkan dengan total jumlah seluruh data sampel.
- **Formula:**
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
  _Di mana: TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative._
- **Interpretasi Klinis:** Persentase pasien yang berhasil diklasifikasikan dengan benar oleh AI, baik mereka yang sehat maupun yang sakit diabetes.
- **Catatan Penting:** Akurasi dapat menjadi metrik yang menyesatkan jika dataset mengalami ketidakseimbangan kelas (_class imbalance_). Sebagai contoh, jika 65% pasien dalam dataset tidak menderita diabetes, model sederhana yang memprediksi seluruh pasien sebagai "sehat" akan memiliki akurasi 65%, meskipun model tersebut gagal mendeteksi satu pun pasien diabetes (Recall = 0%).

#### 2. Precision (Presisi)

- **Definisi:** Rasio prediksi positif yang benar dibandingkan dengan seluruh data yang diprediksi positif oleh model.
- **Formula:**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Interpretasi Klinis:** Dari seluruh pasien yang oleh AI diprediksi mengidap diabetes, berapa persenkah pasien yang pada kenyataannya memang menderita diabetes.
- **Penerapan Klinis:** Presisi yang tinggi sangat penting untuk meminimalkan terjadinya kesalahan diagnosis positif (_False Positive_). Dalam dunia medis, jika presisi rendah, banyak pasien sehat akan dikategorikan sebagai sakit diabetes. Hal ini dapat menimbulkan kecemasan psikologis yang tidak perlu pada pasien serta pemborosan biaya untuk tes laboratorium lanjutan yang mahal.

#### 3. Recall (Sensitivitas)

- **Definisi:** Rasio pasien diabetes yang berhasil dideteksi dengan benar oleh model dibandingkan dengan seluruh pasien yang sebenarnya menderita diabetes.
- **Formula:**
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Interpretasi Klinis:** Dari seluruh pasien yang sebenarnya menderita diabetes di dunia nyata, berapa persenkah yang berhasil diidentifikasi dan disaring oleh model AI.
- **Penerapan Klinis:** Dalam diagnosis penyakit berbahaya seperti diabetes, **Recall sering kali lebih diutamakan daripada Presisi**. Nilai Recall yang tinggi meminimalkan kesalahan diagnosis negatif (_False Negative_)—yaitu kondisi di mana pasien yang sebenarnya sakit diabetes tetapi tidak terdeteksi oleh sistem. Keterlambatan diagnosis diabetes sangat berisiko karena pasien tidak akan mendapatkan penanganan medis segera, yang dapat memicu komplikasi kronis seperti kerusakan ginjal, kebutaan, hingga penyakit jantung.

#### 4. F1-Score

- **Definisi:** Rata-rata harmonik (_harmonic mean_) yang menyeimbangkan metrik Presisi dan Recall secara bersamaan.
- **Formula:**
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Interpretasi Klinis:** Memberikan representasi tunggal mengenai seberapa baik model mendeteksi pasien diabetes dengan meminimalkan kesalahan deteksi (_False Negative_) sekaligus meminimalkan kesalahan diagnosis salah sasaran (_False Positive_).
- **Catatan Penting:** F1-score adalah metrik yang jauh lebih objektif dan andal dibandingkan Akurasi ketika berhadapan dengan data yang tidak seimbang (_imbalanced dataset_), karena ia memberikan bobot yang setara untuk kegagalan pada sisi sensitivitas maupun kepresisian.

#### 5. ROC-AUC (Receiver Operating Characteristic - Area Under the Curve)

- **Definisi:** Area di bawah kurva ROC yang menggambarkan trade-off antara _True Positive Rate_ (Recall/Sensitivitas) dan _False Positive Rate_ (1 - Spesifisitas) pada berbagai tingkat ambang batas keputusan (_classification thresholds_). Nilainya berkisar antara 0,5 (tebakan acak) hingga 1,0 (klasifikasi sempurna).
- **Interpretasi Klinis:** Mengukur kemampuan diskriminasi model secara keseluruhan—yaitu seberapa andal model dalam membedakan antara pasien diabetes dengan pasien non-diabetes secara umum, terlepas dari batas klasifikasi (_cut-off threshold_) tertentu.
- **Penerapan Klinis:** Skor ROC-AUC sebesar 0,8170 pada Eksperimen 3 menunjukkan bahwa jika kita mengambil satu pasien diabetes dan satu pasien sehat secara acak, terdapat probabilitas sebesar 81,70% bahwa model AI akan memberikan skor probabilitas diabetes yang lebih tinggi kepada pasien yang memang menderita diabetes dibandingkan pasien yang sehat. Ini menunjukkan tingkat keandalan klasifikasi yang kuat dan stabil.

## 11. J.62DMI00.015.1 — Melakukan Proses Review Pemodelan

Review pemodelan dilakukan melalui analisis Explainable AI (XAI) menggunakan SHAP values untuk memberikan atribusi kontribusi fitur secara global (seluruh kohort) maupun lokal (per pasien). Hasil SHAP mengonfirmasi bahwa fitur dengan pengaruh terbesar adalah Glucose dan Insulin, diikuti oleh fitur rekayasa klinis seperti Pregnancy_Age_Ratio dan Glucose_DiabetesPedigree_Ratio. Review secara kritis membuktikan bahwa tingkat akurasi tinggi pada Eksperimen 2 (~90-91% Akurasi) dan klaim akurasi tinggi pada jurnal acuan (89,61%) merupakan hasil semu akibat data leakage (kebocoran data) parah karena melakukan imputasi kelas target dan penskalaan secara global sebelum pembagian data. Hal ini menegaskan pentingnya implementasi pipeline Eksperimen 1 dan 3 untuk penarikan kesimpulan klinis yang valid dan reprodusibel pada data pasien nyata.
