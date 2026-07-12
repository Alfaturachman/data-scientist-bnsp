import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import optuna
import warnings

# Disable warnings and Optuna logging output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load data
df = pd.read_csv(r"d:\project sertifikasi\diabetes\diabetes.csv")
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Columns where zero is clinically implausible (treated as missing)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

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
                self.medians[c][col] = vals.median()
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

# =====================================================
# EKSPERIMEN 1: Split-First Pipeline
# =====================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train_s1 = X_train_raw.copy()
X_test_s1 = X_test_raw.copy()

for col in zero_cols:
    X_train_s1[col] = X_train_s1[col].replace(0, np.nan)
    X_test_s1[col] = X_test_s1[col].replace(0, np.nan)

imputer1 = ClassConditionalMedianImputer(zero_cols)
imputer1.fit(X_train_s1, y_train)
X_train_imp1 = imputer1.transform(X_train_s1, y_train)
X_test_imp1 = imputer1.transform(X_test_s1, y=None)

X_train_eng1 = engineer_features(X_train_imp1)
X_test_eng1 = engineer_features(X_test_imp1)

scaler1 = StandardScaler()
X_train_scaled1 = pd.DataFrame(scaler1.fit_transform(X_train_eng1), columns=X_train_eng1.columns)
X_test_scaled1 = pd.DataFrame(scaler1.transform(X_test_eng1), columns=X_test_eng1.columns)

# Resampling SMOTE for train only (to prevent leakage)
smote = SMOTE(random_state=42)
X_train_res1, y_train_res1 = smote.fit_resample(X_train_scaled1, y_train)

clf1_s1 = XGBClassifier(random_state=42, eval_metric='logloss')
clf2_s1 = LGBMClassifier(random_state=42, verbose=-1)
model_s1 = VotingClassifier(
    estimators=[('xgb', clf1_s1), ('lgbm', clf2_s1)],
    voting='soft'
)
model_s1.fit(X_train_res1, y_train_res1)

y_pred_s1 = model_s1.predict(X_test_scaled1)
y_prob_s1 = model_s1.predict_proba(X_test_scaled1)[:, 1]

# =====================================================
# EKSPERIMEN 2: Preprocess-First Pipeline (Leakage)
# =====================================================
X_s2 = X.copy()
for col in zero_cols:
    X_s2[col] = X_s2[col].replace(0, np.nan)

imputer_global = ClassConditionalMedianImputer(zero_cols)
imputer_global.fit(X_s2, y)
X_imp_global = imputer_global.transform(X_s2, y)

X_eng_global = engineer_features(X_imp_global)

scaler_global = StandardScaler()
X_scaled_global = pd.DataFrame(scaler_global.fit_transform(X_eng_global), columns=X_eng_global.columns)

# Fit SMOTE globally (this is the absolute worst leakage: SMOTE globally before splitting!)
smote_global = SMOTE(random_state=42)
X_res_global, y_res_global = smote_global.fit_resample(X_scaled_global, y)

X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(
    X_res_global, y_res_global, test_size=0.20, random_state=42, stratify=y_res_global
)

clf1_s2 = XGBClassifier(random_state=42, eval_metric='logloss')
clf2_s2 = LGBMClassifier(random_state=42, verbose=-1)
model_s2 = VotingClassifier(
    estimators=[('xgb', clf1_s2), ('lgbm', clf2_s2)],
    voting='soft'
)
model_s2.fit(X_train_s2, y_train_s2)

y_pred_s2 = model_s2.predict(X_test_s2)
y_prob_s2 = model_s2.predict_proba(X_test_s2)[:, 1]

# =====================================================
# EKSPERIMEN 3: Optimized Pipeline (SMOTE inside CV)
# =====================================================
def tune_xgb(X_tr, y_tr):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            # SMOTE only on training fold to prevent validation leakage
            sm_fold = SMOTE(random_state=42)
            X_fold_tr_res, y_fold_tr_res = sm_fold.fit_resample(X_fold_train, y_fold_train)
            
            clf = XGBClassifier(**params)
            clf.fit(X_fold_tr_res, y_fold_tr_res)
            preds = clf.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def tune_lgbm(X_tr, y_tr):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            sm_fold = SMOTE(random_state=42)
            X_fold_tr_res, y_fold_tr_res = sm_fold.fit_resample(X_fold_train, y_fold_train)
            
            clf = LGBMClassifier(**params)
            clf.fit(X_fold_tr_res, y_fold_tr_res)
            preds = clf.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study.best_params

best_xgb_params = tune_xgb(X_train_scaled1, y_train)
best_lgbm_params = tune_lgbm(X_train_scaled1, y_train)

# Final model trained on the SMOTE-resampled train split
X_train_res3, y_train_res3 = SMOTE(random_state=42).fit_resample(X_train_scaled1, y_train)

opt_xgb = XGBClassifier(**best_xgb_params, random_state=42)
opt_lgbm = LGBMClassifier(**best_lgbm_params, random_state=42, verbose=-1)

model_s3 = VotingClassifier(
    estimators=[('xgb', opt_xgb), ('lgbm', opt_lgbm)],
    voting='soft'
)
model_s3.fit(X_train_res3, y_train_res3)

y_pred_s3 = model_s3.predict(X_test_scaled1)
y_prob_s3 = model_s3.predict_proba(X_test_scaled1)[:, 1]

# Perbandingan
scenarios = ['Eksperimen 1 (Split-First)', 'Eksperimen 2 (Preprocess-First)', 'Eksperimen 3 (Optimized)']
y_tests = [y_test, y_test_s2, y_test]
y_preds = [y_pred_s1, y_pred_s2, y_pred_s3]
y_probs = [y_prob_s1, y_prob_s2, y_prob_s3]

results = []
for name, y_t, y_p, y_pr in zip(scenarios, y_tests, y_preds, y_probs):
    results.append({
        'Skenario': name,
        'Accuracy': accuracy_score(y_t, y_p),
        'Precision': precision_score(y_t, y_p),
        'Recall': recall_score(y_t, y_p),
        'F1-Score': f1_score(y_t, y_p),
        'ROC-AUC': roc_auc_score(y_t, y_pr)
    })

df_compare = pd.DataFrame(results).set_index('Skenario')
print(df_compare.round(4))
