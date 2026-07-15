#!/usr/bin/env python
# coding: utf-8

# # Uji Kompetensi Data Science (SKKNI)
# ## Eksperimen Pipeline & Preprocessing pada Klasifikasi Diabetes (Pima Indians Diabetes)
# ### Pemetaan CRISP-DM & Standar Kompetensi Kerja Nasional Indonesia (SKKNI)
# 
# Dokumentasi proyek ini disusun untuk memenuhi standardisasi **CRISP-DM** dan mencakup unit-unit kompetensi **SKKNI** berikut:
# 1. **J.62DMI00.001.1** - Menentukan Objektif Bisnis
# 2. **J.62DMI00.002.1** - Menentukan Tujuan Teknis Data Science
# 3. **J.62DMI00.005.1** - Menelaah Data
# 4. **J.62DMI00.006.1** - Memvalidasi Data
# 5. **J.62DMI00.007.1** - Menentukan Objek Data
# 6. **J.62DMI00.008.1** - Membersihkan Data
# 7. **J.62DMI00.009.1** - Mengkonstruksi Data
# 8. **J.62DMI00.012.1** - Membangun Skenario Model
# 9. **J.62DMI00.013.1** - Membangun Model
# 10. **J.62DMI00.014.1** - Mengevaluasi Hasil Pemodelan
# 11. **J.62DMI00.015.1** - Melakukan Proses Review Pemodelan
# 
# ---
# **Role**: Senior Machine Learning Researcher  
# **Dataset**: Pima Indians Diabetes Database  
# **Model**: LightGBM, XGBoost, Gradient Boosting, dan Random Forest dengan KNN Imputation, 16 Composite Features, dan Leakage-Free Optuna Hyperparameter Tuning
# 

# In[1]:


# Pemuatan pustaka dasar dan evaluasi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Konfigurasi estetika dan peringatan
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
print("Seluruh pustaka berhasil dimuat.")


# ## 1. Menentukan Objektif Bisnis (Unit: J.62DMI00.001.1)
# 
# Objektif bisnis dari eksperimen ini adalah:
# 1. **Skrining Klinis Cepat**: Mengembangkan sistem skrining dan deteksi dini penyakit diabetes tipe 2 pada pasien berisiko tinggi secara cepat dan berbasis data.
# 2. **Intervensi Preventif Dini**: Membantu praktisi kesehatan mengidentifikasi kebutuhan pencegahan (gaya hidup/pengobatan) sebelum timbul komplikasi klinis yang lebih berat.
# 3. **Efisiensi Biaya Perawatan**: Menekan biaya jangka panjang bagi penyedia layanan kesehatan dengan melakukan pencegahan dini daripada penanganan komplikasi klinis.

# ## 2. Menentukan Tujuan Teknis Data Science (Unit: J.62DMI00.002.1)
# 
# Tujuan teknis dari pemodelan ini adalah merancang model **Klasifikasi Biner** untuk memprediksi probabilitas dan label kelas target `Outcome`:
# - `0`: Pasien sehat / negatif diabetes.
# - `1`: Pasien terindikasi diabetes / positif diabetes.
# 
# Metrik evaluasi utama yang digunakan untuk mengukur keberhasilan model adalah:
# 1. **Accuracy**: Akurasi klasifikasi keseluruhan.
# 2. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Mengukur keandalan diskriminasi kelas model tanpa terpengaruh oleh bias ambang batas (threshold).
# 
# **Batasan Integritas**: Evaluasi model harus dilakukan menggunakan pipeline yang **bebas dari kebocoran data (leakage-free)** agar performa yang dilaporkan mencerminkan performa riil saat diterapkan pada pasien baru.

# ## 3. Menelaah Data (Unit: J.62DMI00.005.1)
# 
# ### A. Deskripsi Dataset
# Eksperimen menggunakan **Pima Indians Diabetes Database** dari UCI Machine Learning Repository. Karakteristik dataset:
# - **Jumlah Sampel**: 768 baris pasien.
# - **Jumlah Fitur**: 8 fitur klinis numerik.
# - **Jumlah Target**: 1 label biner (`Outcome`).
# 
# ### B. Kamus Fitur Dataset
# 1. `Pregnancies`: Jumlah kehamilan yang pernah dialami pasien.
# 2. `Glucose`: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral (mg/dL).
# 3. `BloodPressure`: Tekanan darah diastolik (mm Hg).
# 4. `SkinThickness`: Ketebalan lipatan kulit trisep (mm).
# 5. `Insulin`: Insulin serum 2 jam (mu U/ml).
# 6. `BMI`: Indeks massa tubuh (berat dalam kg / (tinggi dalam meter)^2).
# 7. `DiabetesPedigreeFunction`: Fungsi silsilah keluarga yang menilai risiko diabetes genetik.
# 8. `Age`: Usia pasien (tahun).
# 9. `Outcome`: Label kelas target (0 = Sehat, 1 = Diabetes).

# ### A. Memuat Data dari Sumber Berkas

# In[2]:


df = pd.read_csv('diabetes.csv')
print(f"Ukuran dataset: {df.shape[0]} baris, {df.shape[1]} kolom")
df.head()


# ### B. Visualisasi Sebaran Data & Analisis Outlier / Imbalance (Unit: J.62DMI00.005.1)
# 
# Untuk memahami sebaran data dan karakteristik fitur-fiturnya:
# 1. **Class Imbalance**: Distribusi kelas target `Outcome` dianalisis untuk melihat ketidakseimbangan kelas.
# 2. **Outlier & Sebaran Fitur**: Pembuatan **Box Plot** untuk setiap fitur numerik dikelompokkan berdasarkan target `Outcome` guna mengidentifikasi pencilan (*outliers*) dan perbedaan median antar kelompok kelas.

# In[3]:


fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

# 1. Plot Distribusi Target (Outcome)
sns.countplot(x='Outcome', data=df, ax=axes[0], palette=['#007aff', '#ff9500'])
axes[0].set_title('Distribusi Kelas Target (Outcome)', fontweight='bold', fontsize=12)
axes[0].set_xticklabels(['Sehat (0)', 'Diabetes (1)'])
axes[0].set_xlabel('Outcome')
axes[0].set_ylabel('Jumlah Sampel')

# 2. Plot Box Plot untuk 8 Fitur Prediktor
features = df.columns[:-1]
for i, col in enumerate(features):
    sns.boxplot(x='Outcome', y=col, data=df, ax=axes[i+1], palette=['#007aff', '#ff3b30'])
    axes[i+1].set_title(f'Sebaran & Outlier: {col}', fontweight='bold', fontsize=12)
    axes[i+1].set_xticklabels(['Sehat (0)', 'Diabetes (1)'])
    axes[i+1].set_xlabel('Outcome')
    axes[i+1].set_ylabel(col)

plt.tight_layout()
plt.show()


# ## 4. Memvalidasi Data (Unit: J.62DMI00.006.1)
# 
# Validasi klinis dilakukan untuk memeriksa kualitas data:
# 1. **Missing Values**: Memeriksa keberadaan data kosong bertipe `NaN` bawaan.
# 2. **Anomali Nilai Nol (Implausible Zeros)**: Mengidentifikasi nilai `0` pada fitur vital (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) yang secara biologis tidak mungkin terjadi pada pasien hidup. Nilai-nilai ini dianggap sebagai data hilang (*missing values*) tersembunyi.

# ## 5. Menentukan Objek Data (Unit: J.62DMI00.007.1)
# 
# Pada tahap ini, objek data ditentukan berdasarkan relevansi klinis:
# - **Fitur Prediktor ($X$)**: 8 variabel klinis (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age).
# - **Target Label ($y$)**: `Outcome` biner (0 untuk sehat, 1 untuk terindikasi diabetes).
# 
# Berikut adalah hasil pemeriksaan kualitas data dan distribusi objek data target:

# In[4]:


# 1. Cek Missing Values bawaan
print("--- Missing Values Bawaan (NaN) ---")
print(df.isnull().sum())

# 2. Cek Anomali Nilai 0 pada Fitur Vital
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\n--- Jumlah Nilai 0 Tidak Logis per Kolom Vital ---")
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    pct = (zero_count / len(df)) * 100
    print(f"{col}: {zero_count} baris ({pct:.2f}%)")

# 3. Cek Distribusi Kelas Target (Outcome)
print("\n--- Distribusi Kelas Target (Outcome) ---")
outcome_counts = df['Outcome'].value_counts()
outcome_pcts = df['Outcome'].value_counts(normalize=True) * 100
for idx in outcome_counts.index:
    print(f"Outcome {idx}: {outcome_counts[idx]} sampel ({outcome_pcts[idx]:.2f}%)")


# ## 6. Membersihkan Data (Unit: J.62DMI00.008.1)
# 
# Langkah pembersihan data meliputi:
# 1. Mengganti nilai nol tidak logis pada fitur `['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']` menjadi `NaN`.
# 2. Melakukan imputasi nilai kosong menggunakan **KNN Imputer (k=5)** pada pipeline yang valid (Eksperimen 1 & 3), serta **Class-Conditional Median Imputer** untuk mereplikasi kebocoran data (Eksperimen 2).

# ## 7. Mengkonstruksi Data (Unit: J.62DMI00.009.1)
# 
# Langkah konstruksi data (rekayasa fitur) meliputi:
# 1. Merancang **16 Fitur Komposit Klinis** untuk memperkuat sinyal prediksi dan menangkap interaksi non-linear antar variabel klinis.
# 2. Menstandardisasi fitur menggunakan `StandardScaler`.

# In[5]:


# Class-Conditional Median Imputer (Digunakan untuk Eksperimen 2 Leakage)
class ClassConditionalMedianImputer:
    def __init__(self, cols_to_impute):
        self.cols_to_impute = cols_to_impute
        self.medians = {}
        self.overall_medians = {}
        
    def fit(self, X, y):
        for c in [0, 1]:
            self.medians[c] = {}
            for col in self.cols_to_impute:
                vals = X.loc[y == c, col]
                self.medians[c][col] = vals.median() if len(vals) > 0 else X[col].median()
        for col in self.cols_to_impute:
            self.overall_medians[col] = X[col].median()
        return self
        
    def transform(self, X, y=None):
        X_out = X.copy().reset_index(drop=True)
        if y is not None:
            y_reset = y.reset_index(drop=True)
            for c in [0, 1]:
                idx = (y_reset == c)
                if idx.any():
                    for col in self.cols_to_impute:
                        X_out.loc[idx, col] = X_out.loc[idx, col].fillna(self.medians[c][col])
        else:
            for col in self.cols_to_impute:
                X_out[col] = X_out[col].fillna(self.overall_medians[col])
        return X_out

# Rekayasa 16 Fitur Komposit Klinis
def engineer_features(df_in):
    df_out = df_in.copy().reset_index(drop=True)
    df_out['Normal_SkinThickness'] = (df_out['SkinThickness'] <= 20).astype(int)
    df_out['Healthy_BMI'] = (df_out['BMI'] <= 30).astype(int)
    df_out['Young_Low_Pregnancies'] = ((df_out['Age'] <= 30) & (df_out['Pregnancies'] <= 6)).astype(int)
    df_out['Optimal_Glucose_BP'] = ((df_out['Glucose'] <= 105) & (df_out['BloodPressure'] <= 80)).astype(int)
    df_out['Young_Normal_Glucose'] = ((df_out['Age'] <= 30) & (df_out['Glucose'] <= 120)).astype(int)
    df_out['Healthy_BMI_SkinThickness'] = ((df_out['BMI'] <= 30) & (df_out['SkinThickness'] <= 20)).astype(int)
    df_out['Optimal_Glucose_BMI'] = ((df_out['Glucose'] <= 105) & (df_out['BMI'] <= 30)).astype(int)
    df_out['Normal_Insulin'] = (df_out['Insulin'] < 200).astype(int)
    df_out['Normal_BloodPressure'] = (df_out['BloodPressure'] < 80).astype(int)
    df_out['Moderate_Pregnancies'] = ((df_out['Pregnancies'] >= 1) & (df_out['Pregnancies'] <= 3)).astype(int)
    df_out['BMI_SkinThickness_Product'] = df_out['BMI'] * df_out['SkinThickness']
    df_out['Pregnancy_Age_Ratio'] = df_out['Pregnancies'] / (df_out['Age'] + 1)
    df_out['Glucose_DiabetesPedigree_Ratio'] = df_out['Glucose'] / (df_out['DiabetesPedigreeFunction'] + 1e-6)
    df_out['Age_DiabetesPedigree_Product'] = df_out['Age'] * df_out['DiabetesPedigreeFunction']
    df_out['Age_Insulin_Ratio'] = df_out['Age'] / (df_out['Insulin'] + 1e-6)
    df_out['Low_BMI_SkinThickness_Product'] = ((df_out['BMI'] * df_out['SkinThickness']) < 1034).astype(int)
    return df_out

print("Fungsi preprocessing dan feature engineering siap digunakan.")


# ## 8. Membangun Skenario Model (Unit: J.62DMI00.012.1)
# 
# Tiga skenario eksperimen dirancang untuk membandingkan validitas metodologi:
# 1. **Eksperimen 1 — Split-First Pipeline (KNN Imputer k=5, Tanpa Leakage)**: Preprocessing dipisah secara ketat antara train dan test set.
# 2. **Eksperimen 2 — Preprocess-First Pipeline (Class-Conditional Median Imputer & Global SMOTE, Data Leakage)**: Preprocessing dilakukan secara global sebelum pemisahan data train-test untuk mereplikasi kesalahan metodologi jurnal.
# 3. **Eksperimen 3 — Optimized Pipeline (KNN Imputer k=5, Leakage-Free Optuna)**: Menggunakan alur Split-First yang aman dengan optimasi hyperparameter menggunakan Optuna pada fold training.

# ## 9. Membangun Model (Unit: J.62DMI00.013.1)
# 
# Pelatihan model menggunakan 4 algoritma klasifikasi:
# 1. **LightGBM Classifier**
# 2. **XGBoost Classifier**
# 3. **Gradient Boosting Classifier**
# 4. **Random Forest Classifier**
# 
# Berikut adalah implementasi kode untuk melatih dan mengevaluasi model pada ketiga skenario:

# In[6]:


X = df.drop(columns=['Outcome'])
y = df['Outcome']
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 1. Splitting data (80:20 Stratified)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_s1 = X_train_raw.copy()
X_test_s1 = X_test_raw.copy()

# Ganti 0 dengan NaN
for col in zero_cols:
    X_train_s1[col] = X_train_s1[col].replace(0, np.nan)
    X_test_s1[col] = X_test_s1[col].replace(0, np.nan)

# Imputasi dengan KNN Imputer (k=5) - fit di train saja
imputer1 = KNNImputer(n_neighbors=5)
X_train_imp1 = pd.DataFrame(imputer1.fit_transform(X_train_s1), columns=X.columns)
X_test_imp1 = pd.DataFrame(imputer1.transform(X_test_s1), columns=X.columns)

# Feature engineering
X_train_eng1 = engineer_features(X_train_imp1)
X_test_eng1 = engineer_features(X_test_imp1)

# Scaling
scaler1 = StandardScaler()
X_train_scaled1 = pd.DataFrame(scaler1.fit_transform(X_train_eng1), columns=X_train_eng1.columns)
X_test_scaled1 = pd.DataFrame(scaler1.transform(X_test_eng1), columns=X_test_eng1.columns)

# class_weight='balanced' menggantikan SMOTE
neg = (y_train == 0).sum(); pos = (y_train == 1).sum(); ratio = neg / pos

indiv_models_s1 = {
    'XGBoost': XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
}
results_s1 = {}
for mname, clf in indiv_models_s1.items():
    clf.fit(X_train_scaled1, y_train)
    yp = clf.predict(X_test_scaled1)
    ypr = clf.predict_proba(X_test_scaled1)[:, 1]
    results_s1[mname] = {'Accuracy': accuracy_score(y_test, yp), 'ROC-AUC': roc_auc_score(y_test, ypr),
                         'Precision': precision_score(y_test, yp), 'Recall': recall_score(y_test, yp), 'F1': f1_score(y_test, yp)}
print('Eksperimen 1 Selesai.')
pd.DataFrame(results_s1).T.round(4)


# In[7]:


X_s2 = X.copy()
for col in zero_cols:
    X_s2[col] = X_s2[col].replace(0, np.nan)

# Kebocoran Target Global (Replikasi metodologi jurnal)
imputer_global = ClassConditionalMedianImputer(zero_cols)
imputer_global.fit(X_s2, y)
X_imp_global = imputer_global.transform(X_s2, y)
X_eng_global = engineer_features(X_imp_global)

scaler_global = StandardScaler()
X_scaled_global = pd.DataFrame(scaler_global.fit_transform(X_eng_global), columns=X_eng_global.columns)

# Melakukan resampling SMOTE secara global sebelum split (leakage fatal)
smote_global = SMOTE(random_state=42)
X_res_global, y_res_global = smote_global.fit_resample(X_scaled_global, y)

# Split data setelah preprocessing global
X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(
    X_res_global, y_res_global, test_size=0.20, random_state=42, stratify=y_res_global
)

# Tanpa class_weight — replikasi bias leakage
indiv_models_s2 = {
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}
results_s2 = {}
for mname, clf in indiv_models_s2.items():
    clf.fit(X_train_s2, y_train_s2)
    yp = clf.predict(X_test_s2)
    ypr = clf.predict_proba(X_test_s2)[:, 1]
    results_s2[mname] = {'Accuracy': accuracy_score(y_test_s2, yp), 'ROC-AUC': roc_auc_score(y_test_s2, ypr),
                         'Precision': precision_score(y_test_s2, yp), 'Recall': recall_score(y_test_s2, yp), 'F1': f1_score(y_test_s2, yp)}
print('Eksperimen 2 Selesai.')
pd.DataFrame(results_s2).T.round(4)


# In[8]:


# Hyperparameter tuning LightGBM dengan nested cross-validation bebas leakage
def tune_lightgbm_leakage_free(X_tr_raw, y_tr):
    def objective(trial):
        lgbm_params = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 150),
            'max_depth': trial.suggest_int('lgbm_max_depth', 2, 5),
            'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('lgbm_leaves', 4, 16),
            'subsample': trial.suggest_float('lgbm_sub', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('lgbm_col', 0.6, 0.9),
            'random_state': 42,
            'verbose': -1
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_tr_raw, y_tr):
            X_fold_train_raw = X_tr_raw.iloc[train_idx].copy()
            X_fold_val_raw = X_tr_raw.iloc[val_idx].copy()
            y_fold_train = y_tr.iloc[train_idx]
            y_fold_val = y_tr.iloc[val_idx]
            
            for col in zero_cols:
                X_fold_train_raw[col] = X_fold_train_raw[col].replace(0, np.nan)
                X_fold_val_raw[col] = X_fold_val_raw[col].replace(0, np.nan)
                
            # Imputasi legal di dalam fold
            imputer = KNNImputer(n_neighbors=5)
            X_fold_tr_imp = pd.DataFrame(imputer.fit_transform(X_fold_train_raw), columns=X.columns)
            X_fold_val_imp = pd.DataFrame(imputer.transform(X_fold_val_raw), columns=X.columns)
            
            X_fold_tr_eng = engineer_features(X_fold_tr_imp)
            X_fold_val_eng = engineer_features(X_fold_val_imp)
            
            scaler = StandardScaler()
            X_fold_tr_sc = pd.DataFrame(scaler.fit_transform(X_fold_tr_eng), columns=X_fold_tr_eng.columns)
            X_fold_val_sc = pd.DataFrame(scaler.transform(X_fold_val_eng), columns=X_fold_val_eng.columns)
            
            model = LGBMClassifier(class_weight='balanced', **lgbm_params)
            model.fit(X_fold_tr_sc, y_fold_train)
            preds_prob = model.predict_proba(X_fold_val_sc)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds_prob))
            
        return np.mean(scores)

    from optuna.samplers import TPESampler
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50)
    return study.best_params

best_params = tune_lightgbm_leakage_free(X_train_raw, y_train)
print("Tuning selesai!")
print("Best Params:", best_params)

# Re-fit final model pada seluruh data latih dengan preprocessing legal
X_train_s3 = X_train_raw.copy()
X_test_s3 = X_test_raw.copy()

for col in zero_cols:
    X_train_s3[col] = X_train_s3[col].replace(0, np.nan)
    X_test_s3[col] = X_test_s3[col].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_s3), columns=X.columns)
X_test_imp = pd.DataFrame(imputer.transform(X_test_s3), columns=X.columns)

X_train_eng = engineer_features(X_train_imp)
X_test_eng = engineer_features(X_test_imp)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_eng), columns=X_train_eng.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_eng), columns=X_test_eng.columns)

# Tidak pakai SMOTE — class_weight ditangani di model
neg = (y_train == 0).sum(); pos = (y_train == 1).sum(); ratio = neg / pos

xgb_p = {
    'scale_pos_weight': ratio,
    'random_state': 42,
    'eval_metric': 'logloss'
}
lgbm_p = {
    'n_estimators': best_params['lgbm_n_estimators'],
    'max_depth': best_params['lgbm_max_depth'],
    'learning_rate': best_params['lgbm_lr'],
    'num_leaves': best_params['lgbm_leaves'],
    'subsample': best_params['lgbm_sub'],
    'colsample_bytree': best_params['lgbm_col'],
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1
}
rf_p = {
    'class_weight': 'balanced',
    'random_state': 42
}

opt_xgb = XGBClassifier(**xgb_p)
opt_lgbm = LGBMClassifier(**lgbm_p)
opt_rf = RandomForestClassifier(**rf_p)
opt_gbm = GradientBoostingClassifier(random_state=42)

# Evaluasi semua model individual
indiv_models_s3 = {'XGBoost': opt_xgb, 'LightGBM': opt_lgbm, 'Gradient Boosting': opt_gbm, 'Random Forest': opt_rf}
results_s3 = {}
for mname, clf in indiv_models_s3.items():
    clf.fit(X_train_scaled, y_train)
    yp = clf.predict(X_test_scaled)
    ypr = clf.predict_proba(X_test_scaled)[:, 1]
    results_s3[mname] = {'Accuracy': accuracy_score(y_test, yp), 'ROC-AUC': roc_auc_score(y_test, ypr),
                         'Precision': precision_score(y_test, yp), 'Recall': recall_score(y_test, yp), 'F1': f1_score(y_test, yp)}
print('Eksperimen 3 Selesai.')
pd.DataFrame(results_s3).T.round(4)


# ## 10. Mengevaluasi Hasil Pemodelan (Unit: J.62DMI00.014.1)
# 
# Evaluasi kinerja dilakukan dengan membandingkan metrik Accuracy, ROC-AUC, Precision, Recall, dan F1-Score dari keempat model pada ketiga skenario eksperimen:

# In[9]:


# Tabel komparasi: semua model x semua skenario
model_order = ['LightGBM', 'XGBoost', 'Gradient Boosting', 'Random Forest']

rows = []
for mname in model_order:
    row = {'Model': mname}
    for sk_label, res in [('Exp1 Split-First', results_s1), ('Exp2 Leakage', results_s2), ('Exp3 Optimized', results_s3)]:
        if mname in res:
            row[f'{sk_label} Acc'] = round(res[mname]['Accuracy'], 4)
            row[f'{sk_label} AUC'] = round(res[mname]['ROC-AUC'], 4)
        else:
            row[f'{sk_label} Acc'] = None
            row[f'{sk_label} AUC'] = None
    rows.append(row)

df_compare = pd.DataFrame(rows).set_index('Model')
print('=== Perbandingan Performa: 4 Model x 3 Skenario ===')
df_compare


# In[10]:


# Bar chart ROC-AUC Skenario 3 (valid) untuk semua model
auc_s3 = {m: results_s3[m]['ROC-AUC'] for m in model_order}
colors = ['#34c759' if m == 'LightGBM' else '#007aff' for m in model_order]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(model_order, [auc_s3[m] for m in model_order], color=colors)
ax.set_title('ROC-AUC per Model — Eksperimen 3 (Optimized, Leakage-Free)', fontweight='bold', fontsize=13)
ax.set_ylabel('ROC-AUC')
ax.set_ylim(0.6, 1.0)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{bar.get_height():.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()


# ## 11. Melakukan Proses Review Pemodelan (Unit: J.62DMI00.015.1)
# 
# ### Analisis Hasil Komparasi & Pembuktian Data Leakage
# 
# 1. **Eksperimen 1 vs Eksperimen 2 (Data Leakage Demonstration)**:
#    * **Eksperimen 1 (Aman dari Leakage)** menghasilkan performa riil model LightGBM dengan Akurasi **73.38%** dan ROC-AUC **0.8019**.
#    * **Eksperimen 2 (Terjadi Leakage parah)** menghasilkan performa LightGBM yang melonjak secara tidak wajar ke Akurasi **90.50%** dan ROC-AUC **0.9644**.
#    * **Penyebab**: Preprocessing global (imputasi bersyarat target dan SMOTE sebelum pemisahan data) membocorkan informasi label data uji ke dalam data latih. Hal ini membuktikan bahwa akurasi ~89% pada jurnal acuan disebabkan oleh kebocoran data.
# 2. **Eksperimen 3 (Optimized)**:
#    * Melalui optimasi hyperparameter dengan **Optuna** secara legal pada data latih, LightGBM mencapai performa terbaik yang valid dengan Akurasi **75.32%** dan ROC-AUC **0.8170**.
