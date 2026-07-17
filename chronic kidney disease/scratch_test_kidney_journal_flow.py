import pandas as pd
import numpy as np
import warnings
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load dataset
df = pd.read_csv('kidney_disease.csv')

# Drop id column
df_selected = df.drop(columns=['id'])

# Clean numeric columns stored as object
for col in ['pcv', 'wc', 'rc']:
    df_selected[col] = df_selected[col].astype(str).str.replace(r'\t', '', regex=True).str.replace('?', 'nan', regex=False)
    df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

# Clean string columns
for col in ['dm', 'cad', 'classification']:
    df_selected[col] = df_selected[col].astype(str).str.strip().str.replace(r'\t', '', regex=True)

# Encode target classification: ckd -> 0 (positive clinical class), notckd -> 1 (control healthy)
df_selected['classification_encoded'] = df_selected['classification'].map({'ckd': 0, 'notckd': 1})

# Encode binary columns
binary_mapping = {
    'rbc': {'normal': 1, 'abnormal': 0},
    'pc': {'normal': 1, 'abnormal': 0},
    'pcc': {'present': 1, 'notpresent': 0},
    'ba': {'present': 1, 'notpresent': 0},
    'htn': {'yes': 1, 'no': 0},
    'dm': {'yes': 1, 'no': 0},
    'cad': {'yes': 1, 'no': 0},
    'appet': {'good': 1, 'poor': 0},
    'pe': {'yes': 1, 'no': 0},
    'ane': {'yes': 1, 'no': 0}
}
for col, mapping in binary_mapping.items():
    df_selected[col] = df_selected[col].map(mapping)

X = df_selected.drop(columns=['classification', 'classification_encoded'])
y = df_selected['classification_encoded']

# Columns definitions
num_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
cat_cols = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
poly_cols = ['sc', 'bp', 'hemo', 'age']

# Imputation function
def impute_data(X_train, X_test=None):
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy() if X_test is not None else None
    
    for col in X_train.columns:
        if col in num_cols:
            mean_val = X_train[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.0
            X_train_imp[col] = X_train_imp[col].fillna(mean_val)
            if X_test_imp is not None:
                X_test_imp[col] = X_test_imp[col].fillna(mean_val)
        else:
            mode_series = X_train[col].mode()
            mode_val = mode_series[0] if not mode_series.empty else 0.0
            X_train_imp[col] = X_train_imp[col].fillna(mode_val)
            if X_test_imp is not None:
                X_test_imp[col] = X_test_imp[col].fillna(mode_val)
                
    return X_train_imp, X_test_imp

# Polynomial expansion function
def add_polynomial_features(df_in):
    df_out = df_in.copy().reset_index(drop=True)
    for i in range(len(poly_cols)):
        col1 = poly_cols[i]
        name_sq = f'{col1}_squared'
        df_out[name_sq] = df_out[col1] ** 2
        for j in range(i + 1, len(poly_cols)):
            col2 = poly_cols[j]
            name_int = f'{col1}_x_{col2}'
            df_out[name_int] = df_out[col1] * df_out[col2]
    return df_out

# Metrik evaluasi medis (TP, TN, FP, FN dihitung terhadap CKD=0 sebagai kelas positif)
def evaluate_clinical_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    # y_true: CKD=0, Not CKD=1
    # cm[0,0]: True CKD predicted CKD (TP)
    # cm[0,1]: True CKD predicted Not CKD (FN)
    # cm[1,0]: True Not CKD predicted CKD (FP)
    # cm[1,1]: True Not CKD predicted Not CKD (TN)
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return acc, roc_auc, precision, sensitivity, f1, specificity

# =====================================================
# EKSPERIMEN 1: Split-First Pipeline
# =====================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Imputasi legal
X_train_s1, X_test_s1 = impute_data(X_train_raw, X_test_raw)

# Polinomial
X_train_s1_poly = add_polynomial_features(X_train_s1)
X_test_s1_poly = add_polynomial_features(X_test_s1)

# Scaling
scaler1 = StandardScaler()
X_train_s1_scaled = pd.DataFrame(scaler1.fit_transform(X_train_s1_poly), columns=X_train_s1_poly.columns)
X_test_s1_scaled = pd.DataFrame(scaler1.transform(X_test_s1_poly), columns=X_test_s1_poly.columns)

models_def = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=10000, solver='lbfgs', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42)
}

results_s1 = {}
for mname, clf in models_def.items():
    clf.fit(X_train_s1_scaled, y_train)
    yp = clf.predict(X_test_s1_scaled)
    ypr = clf.predict_proba(X_test_s1_scaled)[:, 1]
    
    # Evaluate with CKD=0 as positive class.
    # Note: evaluate_clinical_metrics expects y_true and predictions.
    # In evaluate_clinical_metrics, tp is cm[0,0], which corresponds to target=0 predicted as 0.
    acc, roc, prec, sens, f1, spec = evaluate_clinical_metrics(y_test, yp, ypr)
    results_s1[mname] = {
        'Accuracy': acc, 'ROC-AUC': roc, 'Precision': prec, 'Recall': sens, 'F1-Score': f1, 'Specificity': spec
    }

print("Experiment 1 Done.")

# =====================================================
# EKSPERIMEN 2: Preprocess-First Pipeline (Leakage)
# =====================================================
X_imp_all, _ = impute_data(X)
X_poly_all = add_polynomial_features(X_imp_all)
scaler2 = StandardScaler()
X_scaled_all = pd.DataFrame(scaler2.fit_transform(X_poly_all), columns=X_poly_all.columns)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled_all)
strat_col = y.astype(str) + '_' + pd.Series(clusters).astype(str)

X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(
    X_scaled_all, y, test_size=0.20, stratify=strat_col, random_state=42
)

results_s2 = {}
for mname, clf in models_def.items():
    clf.fit(X_train_s2, y_train_s2)
    yp = clf.predict(X_test_s2)
    ypr = clf.predict_proba(X_test_s2)[:, 1]
    
    acc, roc, prec, sens, f1, spec = evaluate_clinical_metrics(y_test_s2, yp, ypr)
    results_s2[mname] = {
        'Accuracy': acc, 'ROC-AUC': roc, 'Precision': prec, 'Recall': sens, 'F1-Score': f1, 'Specificity': spec
    }

print("Experiment 2 Done.")

# =====================================================
# EKSPERIMEN 3: Optimized Pipeline
# =====================================================
def tune_logistic_regression(X_tr, y_tr):
    def objective(trial):
        c_param = trial.suggest_float('lr_C', 1e-4, 1e2, log=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            clf = LogisticRegression(C=c_param, max_iter=10000, solver='lbfgs', random_state=42)
            clf.fit(X_fold_train, y_fold_train)
            preds = clf.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)
    return study.best_params

def tune_random_forest(X_tr, y_tr):
    def objective(trial):
        n_est = trial.suggest_int('rf_n_estimators', 50, 200)
        max_d = trial.suggest_int('rf_max_depth', 3, 10)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_fold_train, y_fold_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
            clf.fit(X_fold_train, y_fold_train)
            preds = clf.predict_proba(X_fold_val)[:, 1]
            scores.append(roc_auc_score(y_fold_val, preds))
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)
    return study.best_params

best_lr_params = tune_logistic_regression(X_train_s1_scaled, y_train)
best_rf_params = tune_random_forest(X_train_s1_scaled, y_train)

print("Tuning Done.")
print("Best LR:", best_lr_params)
print("Best RF:", best_rf_params)

opt_models = {
    'Logistic Regression': LogisticRegression(C=best_lr_params['lr_C'], max_iter=10000, solver='lbfgs', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=best_rf_params['rf_n_estimators'], max_depth=best_rf_params['rf_max_depth'], random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42)
}

results_s3 = {}
for mname, clf in opt_models.items():
    clf.fit(X_train_s1_scaled, y_train)
    yp = clf.predict(X_test_s1_scaled)
    ypr = clf.predict_proba(X_test_s1_scaled)[:, 1]
    
    acc, roc, prec, sens, f1, spec = evaluate_clinical_metrics(y_test, yp, ypr)
    results_s3[mname] = {
        'Accuracy': acc, 'ROC-AUC': roc, 'Precision': prec, 'Recall': sens, 'F1-Score': f1, 'Specificity': spec
    }

print("Experiment 3 Done.")

# Compare Results
rows = []
for mname in models_def.keys():
    row = {'Model': mname}
    for sk_label, res in [('Exp1 Split-First', results_s1), ('Exp2 Leakage', results_s2), ('Exp3 Optimized', results_s3)]:
        row[f'{sk_label} Acc'] = round(res[mname]['Accuracy'], 4)
        row[f'{sk_label} AUC'] = round(res[mname]['ROC-AUC'], 4)
    rows.append(row)

df_compare = pd.DataFrame(rows).set_index('Model')
print(df_compare)
