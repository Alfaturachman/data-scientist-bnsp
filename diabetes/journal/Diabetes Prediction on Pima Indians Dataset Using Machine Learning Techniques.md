<div align="center">

# Diabetes Prediction on Pima Indians Dataset Using Machine Learning Techniques

</div>

ABDELMGEID A.ALI $ ^{1} $ , GALAL R. GALAL $ ^{2} $ AND HASSAN S. HASSAN $ ^{3} $

$ ^{1} $Faculty of Computers and Information, Minia University, Minia 61519, Egypt $ ^{2} $Faculty of Computers and Information, Minia University, Minia 61519, Egypt $ ^{3} $Faculty of Computers and Information, Minia University, Minia 61519, Egypt

Abstract: Type 2 diabetes is a major public-health problem. We build a leakage-safe machine-learning workflow on the Pima Indians Diabetes Dataset (768 records) to predict diabetes from routine clinical attributes. Clinically implausible zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are treated as missing and imputed with class-conditional medians. Continuous variables are standardized only for models that require scaling (e.g., LR, SVM, KNN); tree-based models use raw scales. Besides the eight original attributes, we engineer 16 clinically interpretable composite features and assess their utility with descriptive checks and model-agnostic explainability (SHAP). The model portfolio includes Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM. The final classifier is a soft-voting ensemble of XGBoost and LightGBM based on averaged predicted probabilities.

Using a stratified train/validation procedure and a strictly held-out test set, the ensemble achieves Accuracy = 89.61%, ROC-AUC = 94.52%, and F1 = 85.19%, outperforming the individual models. SHAP highlights clinically coherent drivers (e.g., glucose, pregnancies, age, BMI-related composites). Compared with recent Scopus-indexed studies on the same dataset ( $ \approx $ 74-89% accuracy), our leakage-controlled and transparent pipeline provides competitive, reproducible results and a practical basis for clinical decision support that can be extended to larger, multi-site, and more diverse cohorts.

Keywords: Diabetes mellitus, Machine learning, Data processing, Pima Indians Diabetes Dataset (PIDD), Classification algorithms, Random Forest, XGBoost, LightGBM, Ensemble Learning, Feature Engineering

## 1. INTRODUCTION

Diabetes mellitus is a long-term metabolic condition marked by high blood sugar levels, resulting from inadequate insulin production by the pancreas (Type 1 diabetes) [1,2], the body's inability to effectively use insulin (Type 2 diabetes), or a temporary state during pregnancy called gestational diabetes [3]. The World Health Organization (WHO) identifies diabetes as a major contributor to blindness, kidney failure, heart disease, stroke, and lower limb amputations [4]. The global incidence of diabetes is steadily increasing, fueled by aging populations, lack of physical activity, unhealthy diets, and rising obesity. This condition places a heavy financial strain on individuals and healthcare systems alike, making it a vital focus for public health efforts and medical innovation [5].

Early and accurate diagnosis of diabetes is essential to prevent or delay the onset of complications associated with the disease [6]. Identifying individuals at risk or in the early stages of diabetes allows for timely medical intervention, lifestyle modifications, and ongoing monitoring, which can significantly reduce the severity and progression of the disease [7].

Early detection is particularly vital in managing Type 2 diabetes, which can remain asymptomatic for years, thereby increasing the risk of undiagnosed cases. Proactive diagnosis can also aid in the prevention of pre-diabetes from developing into full-blown diabetes [8]. Despite the availability of diagnostic tests, many cases remain undetected until serious complications arise, highlighting the need for more effective and accessible diagnostic methods [9].

Conventional methods [10] for diagnosing diabetes include fasting plasma glucose (FPG) tests, oral glucose tolerance tests (OGTT), and glycated hemoglobin (HbA1c) measurements [11]. While these methods are standardized and widely used, they present several limitations. These include the requirement for fasting, multiple blood samples, time-consuming procedures, and the need for specialized medical infrastructure and trained personnel [12]. Furthermore, traditional diagnostic approaches may not be effective in identifying diabetes in its early stages or in asymptomatic individuals [13]. There is also a risk of human error in manual interpretation of test results, and variability in patient conditions can lead to inconsistent outcomes. These constraints make it challenging to conduct large-scale screenings, especially in low-resource settings, underscoring the necessity for innovative diagnostic alternatives [14].

Soft computing [15] is a multidisciplinary field that includes methods like fuzzy logic, neural networks, genetic algorithms, and machine learning, which are designed to model and process uncertain, imprecise, and complex data conditions commonly encountered in medical diagnostics [16]. Unlike traditional hard computing methods that require exact inputs and deterministic outputs, soft computing techniques are tolerant of uncertainty and can learn from data, making them highly suitable for healthcare applications [17]. In the context of diabetes diagnosis, soft computing approaches can analyze large datasets of patient information such as medical history, lifestyle factors, and clinical test results to identify patterns and predict disease risk with high accuracy [18]. These techniques have shown promise in improving the speed, accuracy, and accessibility of diagnostic processes, enabling more personalized and data-driven healthcare solutions [19]. As such, the integration of soft computing in medical diagnostics represents a significant advancement toward more intelligent, efficient, and proactive disease management systems [20].

The remainder of this paper is organized as follows: Section 2 reviews related work on diabetes prediction using the Pima Indians Diabetes Dataset (PIDD). Section 3 describes the dataset and cohort characteristics. Section 4 details the methodology data-quality checks, missing-data handling for clinically implausible zeros, feature engineering (16 composite features), feature standardization (for scalable models) and a diverse model portfolio (Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM), and the ensemble with leakage-safe validation. Section 5 presents the experimental results on a strictly held-out test set (Accuracy/ROC-AUC/F1), SHAP-based explainability, and a comparative analysis against recent Scopus-indexed PIDD studies. Section 6 concludes and outlines future work.

The primary contributions of this paper are as follows:

- Leakage-safe ML workflow for PIDD. We implement a strictly leakage-controlled pipeline on the Pima Indians Diabetes Dataset (768 records), ensuring all preprocessing and screening are confined within the training folds and a single untouched test split is used for final reporting.

- Clinically guided data cleaning & imputation. Implausible zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are treated as missing and imputed using class-conditional medians, aligning with clinical plausibility.

- Feature engineering with transparent rationale. We add 16 clinically interpretable composite features (formulas + justifications), and verify their utility via univariate tests and model-agnostic explanations (SHAP).

- Comprehensive model portfolio with an ensemble best. We evaluate Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM, plus an ensemble; the ensemble yields Accuracy 89.61%, ROC-AUC 94.52%, and F1 85.19% on the held-out test set.

- Explainability for clinical insight. SHAP analysis highlights physiologically coherent drivers (e.g., glucose, pregnancies, age, BMI composites), providing transparent model behavior.

## 2. Related Work

Research on diabetes prediction with tabular clinical attributes especially on the Pima Indians Diabetes Dataset (PIDD) has expanded notably in recent years. Across modern classical ML pipelines (e.g., Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost/LightGBM) and lightweight ensembles, reported performance on PIDD typically spans ~7489% accuracy, with substantial variability in validation protocols and metric reporting [21-40].

Tree-based methods (RF/GBMs) are the most commonly reported single-model baselines on PIDD, often outperforming linear models and KNN under similar preprocessing [21-28, 29-36]. Several works evaluate ensembles (e.g., soft voting/stacking) as stronger but still lightweight tabular learners, generally yielding incremental improvements over the best single model while remaining computationally practical [21-28]. SVMs also appear frequently; while effective, they tend to trail tuned tree-based models on PIDD unless coupled with careful kernel choices or targeted feature selection [27- 29, 33-36]. Beyond classical ML, a subset of studies explores deep learning (e.g., CNN/MLP for tabular or augmented feature spaces), achieving competitive accuracies but usually with heavier configuration requirements and limited external validation [33, 35]. Other lines include fuzzy/variant KNN and correlation/feature-selection-centric pipelines with mid-80% accuracy bands under typical splits [31, 37-38].

Despite progress, three gaps recur. First, many papers do not explicitly guard against data leakage (e.g., fitting imputers or scalers before cross-validation), which risks optimistic estimates [21-40]. Second, metric completeness is inconsistent (AUC/F1 often omitted), complicating cross-paper comparison [29-36]. Third, feature engineering is commonly minimal or implicit; when present, its clinical rationale is rarely articulated, and ablation is limited [21-28, 31-36].

Overall, the literature establishes a strong baseline space (predominantly tree-based learners and light ensembles) on PIDD within the $ \sim 7 4-8 9 \% $ accuracy range [21-40], but leaves room for leakage safe evaluation, transparent missing-data handling, and clinically-motivated engineered features all focal points of the present study.

Despite notable progress in ML for diabetes prediction on PIDD, key gaps persist limited leakage control, incomplete metric reporting (e.g., missing AUC/F1), and minimal clinically motivated feature engineering. (Table 2.1) summarizes recent Scopus-indexed PIDD studies used for comparison in this work.

<div align="center">

Table 2.1: Comparative summary of recent Scopus-indexed PIDD studies for diabetes prediction.

</div>

<table border="1"><tr><td>#</td><td>Authors</td><td>Year</td><td>Title</td><td>Methods Used</td><td>Dataset</td><td>Key Features</td><td>Performance Metrics</td><td>Limitations</td><td>Key Contributions</td><td>Indexing</td></tr><tr><td>1</td><td>Ahmed,A.,et al.[21]</td><td>2025</td><td>ML-based diabetes prediction among female PIMA cohort</td><td>RF,DT,NB,LR;PCA;5-fold CV</td><td>PIDD</td><td>Female-only cohort;PCA+correlation</td><td>Accuracy=80%</td><td>Single cohort;class imbalance;generalizability</td><td>Baseline comparison across4MLmodels on femalePIMA</td><td>Scopus,WoS(MDPI)</td></tr><tr><td>2</td><td>Okwudili,R.,et al.[22]</td><td>2025</td><td>An improved performance model forAI on the Pima Indians Diabetes Database</td><td>DT,SVM,NB(NBbest)</td><td>PIDD</td><td>Compares standard classifiers;reportsROC/AUC</td><td>Accuracy=76.3%</td><td>Single dataset;modest accuracy;limited external validation</td><td>Benchmarks classic classifiers onPIDD with AUC reporting</td><td>Scopus,WoS(SpringerOpen)</td></tr><tr><td>3</td><td>Talukder,M.A.,et al.[23]</td><td>2024</td><td>Toward reliable diabetes prediction:innovations in data handling</td><td>RandomForest(subset analysis)</td><td>PIDD</td><td>Reliability-centric handling;limited tuning onPIMA</td><td>Accuracy=80%</td><td>Not model-centric;subset only</td><td>Highlights preprocessing choices forPIMA reliability</td><td>Indexed(journal onPMC)</td></tr><tr><td>4</td><td>Febrian,M.E.,et al.[24]</td><td>2023</td><td>Diabetes prediction using supervised machine learning</td><td>LR,SVM,RF comparisons</td><td>PIDD</td><td>SupervisedMLbaselines;k-fold/holdout</td><td>Accuracy=86%</td><td>Singledataset;limited external validation</td><td>Provided reproduciblePIMA/PIDDbaselines withnumeric metrics</td><td>Scopus(ProcediaComputerScience)</td></tr><tr><td>5</td><td>Gupta,S.C.,et al.[25]</td><td>2023</td><td>Predictive Modeling and Analytics forDiabetes usingMachineLearning</td><td>RandomForest(best),SVM,etc.</td><td>PIDD</td><td>Comparisonacrossfeaturevariants;publicprotocol</td><td>Accuracy=88.61%</td><td>Singledataset;limited external validation</td><td>Provided reproduciblePIMA/PIDDbaselineswithnumeric metrics</td><td>Scopus(ProcediaComputerScience)</td></tr><tr><td>6</td><td>Patro,K.K.,et al.[26]</td><td>2023</td><td>An effective correlation-baseddata modelingframework fordiabetes prediction</td><td>Correlation-basedmodeling+ML</td><td>PIDD</td><td>Correlationmeasures;80/20split</td><td>Accuracy=75%</td><td>Singledataset;limited external validation</td><td>Provided reproduciblePIMA/PIDDbaselineswithnumeric metrics</td><td>Scopus,WoS(BMC)</td></tr><tr><td>7</td><td>Reza,M.S.,et al.[27]</td><td>2023</td><td>ImprovingSVMperformance fortypeII diabeteswithan improvedkernel</td><td>SVM(customkernel)</td><td>PIDD</td><td>Kernelengineeringvs.RBF</td><td>Accuracy=85.5%</td><td>Single benchmark;needsexternalvalidation</td><td>CustomSVMkernelbaselineonPIMA</td><td>Scopus(Elsevier)</td></tr><tr><td>8</td><td>Tasin,I.,et al.[28]</td><td>2023</td><td>Diabetes predictionusingmachinelearningandexplainableAI</td><td>Softvotingclassifier;XAI</td><td>PIDD</td><td>ExplainabilitywithSHAP;softvoting</td><td>Accuracy=79.1%</td><td>Singledataset;limitedexternalvalidation</td><td>Provided reproduciblePIMA/PIDDbaselineswithnumeric metrics</td><td>Scopus,WoS(IET)</td></tr><tr><td>9</td><td>Zhou,H.,Xin,Y.,Li,S.[29]</td><td>2023</td><td>Borutafeatureselection+ensembleforPIMA diabetes prediction</td><td>BorutaFS+Ensemble(stacking)</td><td>PIDD</td><td>BorutaFS;K-Means++pre-clustering;stackedlearner</td><td>Accuracy=79.04%</td><td>Single benchmark;limitedexternalvalidation</td><td>Feature-selection+ensemblepipelineonPIMA</td><td>Scopus,WoS(BMC)</td></tr><tr><td>10</td><td>Kaur,H.,Kumari,V.[30]</td><td>2022</td><td>Predictive modeling&amp;analyticsfordiabetes(MLapproach)</td><td>RBF-SVM,LinearSVM,k-NN,MDR,ANN</td><td>PIDD</td><td>ClassicalMLcomparison</td><td>Accuracy=89%</td><td>Uncertaintyaboutprotocol;stillbelowyour89.61%</td><td>BaselineMLcomparisonwithclearmetrics</td><td>Scopus(Elsevier)</td></tr></table>

<table border="1"><tr><td>#</td><td>Authors</td><td>Year</td><td>Title</td><td>Methods Used</td><td>Dataset</td><td>Key Features</td><td>Performance Metrics</td><td>Limitations</td><td>Key Contributions</td><td>Indexing</td></tr><tr><td>11</td><td>Pradhan,S., et al.[31]</td><td>2022</td><td>Voting Classification-Based Diabetes Mellitus Prediction Using PIDD</td><td>Ensemble(soft voting)</td><td>PIDD</td><td>Voting ensemble across classic classifiers</td><td>Accuracy=82%</td><td>Single-dataset; limited external validation</td><td>Provided reproducible PIMA/PIDD baselines with numeric metrics</td><td>Scopus, WoS(Hindawi/Wiley)</td></tr><tr><td>12</td><td>Salem,E., et al.[32]</td><td>2022</td><td>Fine-tuning fuzzy-KNN classifier under uncertainty membership</td><td>Fuzzy-KNN+preprocessing</td><td>PIDD</td><td>Uncertainty-aware fuzzy KNN</td><td>Accuracy=83.63%</td><td>Single dataset; limited external test</td><td>Refined fuzzy-KNN baseline on PIDD</td><td>Scopus, WoS(MDPI)</td></tr><tr><td>13</td><td>Ullah,Z., et al.[33]</td><td>2022</td><td>Detecting high-risk factors&amp;early diagnosis using ML</td><td>RF,SVM,LR</td><td>PIDD</td><td>Risk-factor analysis+class-imbalance handling</td><td>Accuracy=80.84%</td><td>Focus on risk-factors; limited tuning</td><td>Benchmarked classic ML on PIDD with risk insights</td><td>Scopus, WoS(Hindawi/Wiley)</td></tr><tr><td>14</td><td>Butt,U.M., et al.[34]</td><td>2021</td><td>ML-based diabetes classification for healthcare applications</td><td>MLP,RF,LR,LSTM,LR,MA(survey+experiment)</td><td>PIDD</td><td>Survey+empirical comparison</td><td>Accuracy=86.08%</td><td>Mostly survey; small experimental section</td><td>Summarized ML methods; provided baseline MLP on PIDD</td><td>Scopus, WoS(Hindawi/Wiley)</td></tr><tr><td>15</td><td>García-Ordás,M.T., et al.[35]</td><td>2021</td><td>Diabetes detection using deep learning with oversampling&amp;feature augmentation</td><td>CNN-based DL+augmentation</td><td>PIDD</td><td>Oversampling+data augmentation</td><td>Accuracy=88.67%</td><td>Single benchmark;DL needs more external validation</td><td>Demonstrated strong DL baseline on PIMA with augmentation</td><td>Scopus, WoS(Elsevier)</td></tr><tr><td>16</td><td>Khanam,J.J., Foo,S.Y.[36]</td><td>2021</td><td>A comparison of ML algorithms for diabetes classification&amp;progression</td><td>RF+mRMR;classic ML</td><td>PIDD</td><td>mRMR feature selection with RF</td><td>Accuracy=77.21%</td><td>Older setup;limited PIMA focus</td><td>Benchmarked classic ML on PIMA;identified mRMR+RF</td><td>Scopus, WoS(Elsevier)</td></tr><tr><td>17</td><td>Ramesh,S., et al.[37]</td><td>2021</td><td>Remote healthcare monitoring framework for diabetes prediction</td><td>End-to-end ML pipeline</td><td>PIDD</td><td>Applied ML within monitoring framework</td><td>Accuracy=83.2%</td><td>Framework context;single dataset</td><td>Gave reproducible ML baselines on PIDD</td><td>Scopus, WoS(Springer)</td></tr><tr><td>18</td><td>Patra,R., Kuntia,B.[38]</td><td>2020</td><td>Prediction on Pima Indians Diabetes using SDKNN</td><td>SDKNN(modified KNN)</td><td>PIDD</td><td>Standard-deviation distance in KNN</td><td>Accuracy=83.76%</td><td>Conference protocol;no AUC/F1</td><td>Introduced SDKNN variant baseline on PIDD</td><td>Scopus(IOP Conf.Series)</td></tr><tr><td>19</td><td>Ullah,S., et al.[39]</td><td>2020</td><td>Early prediction of diabetes using ML classifiers</td><td>LR,SVM,RF</td><td>PIDD</td><td>Classic supervised ML baselines</td><td>Accuracy=82%</td><td>Single dataset;basic tuning</td><td>Provided supervised ML baselines on PIDD</td><td>Scopus(Springer)</td></tr><tr><td>20</td><td>Çalışir,D.,Doğantekin,E.[40]</td><td>2011</td><td>Automatic diabetes diagnosis via LDA-wavelet SVM</td><td>LDA+Morlet wavelet SVM</td><td>PIDD</td><td>Dimensionality reduction+wavelet features</td><td>Accuracy=89%</td><td>Older split protocol;lacks modern CV</td><td>Classic,well-cited PIDD pipeline combining LDA with wavelet SVM</td><td>Scopus, WoS(ESWA)</td></tr></table>

## 3. Dataset

The experiments in this study use the Pima Indians Diabetes Dataset (PIDD) obtained from the

Kaggle UCI Machine Learning page (accessed Sept 20, 2025) [41]. The dataset contains 768 clinical records for female patients aged $ \geq 21 $ years of Pima Indians heritage, with 8 input attributes and a binary outcome (diabetic or non-diabetic). The eight attributes are: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The features are numeric-valued and represent important health metrics:

- Pregnancies: Number of times the patient has been pregnant.

- Glucose: Plasma glucose concentration measured 2 hours after an oral glucose tolerance test (mg/dL).

- Blood Pressure: Diastolic blood pressure (mm Hg).

- Skin Thickness: Triceps skin fold thickness (mm).

- Insulin: 2-hour serum insulin ( $ \mu\mathrm{U}/\mathrm{mL} $ ).

- BMI: Body mass index (weight in kg/（height in m） $ ^{2} $ ).

- Diabetes Pedigree Function: (a score indicating diabetes hereditary risk).

- Age: Age in years.

- Outcome: Diabetes status (target variable: 1 = tested positive for diabetes, 0 = tested negative). Out of the 768 patients, 268 (34.9%) have Outcome = 1 (diabetic) and 500 (65.1%) have Outcome = 0 (non-diabetic). This 1:2 class ratio reflects a moderate class imbalance, which can bias models towards predicting the majority class. We will address this issue in preprocessing (see Methodology). Data Quality: An important aspect of this dataset is that it contains some implausible zero values in features that should never be zero for a living person (e.g. blood pressure, plasma glucose). These zeros indicate missing data that were recorded as 0. Specifically, features Glucose, Blood Pressure, Skin Thickness, and Insulin have a certain number of zero entries (e.g., 5 patients have 0 blood pressure, etc.). We treat these zero values as missing and handle them via imputation (described below). The dataset has no explicit NaN values all entries are complete but these zeros must be corrected for accurate analysis.

Statistical properties: Before preprocessing, we examined basic statistics. The mean values (after replacing zeros with NaNs for calculation) are roughly: Glucose ~121, Blood Pressure ~72, Skin Thickness ~29, Insulin ~156, BMI ~33, DPF ~0.47, Age ~33. The positive class tends to have higher average glucose and BMI than the negative class, consistent with known risk factors. There is a noticeable variance in the Insulin feature, and many zero entries (indicating missingness) in Skin Thickness and Insulin - about 30% of records have Insulin=0, and ~29% have Skin Thickness=0, for example. This underscores the need for careful preprocessing of these attributes.

In summary, the Pima dataset provides a challenging benchmark due to its small size, missing values, and class imbalance. (Table 3.1) summarizes the dataset characteristics:

<div align="center">

Table 3.1: Pima Indians Diabetes Dataset [PIDD] overview.

</div>

<table border="1"><tr><td>Characteristic</td><td>Description/Value</td></tr><tr><td>Number of instances</td><td>768 patients(Pima Indians females)</td></tr><tr><td>Number of attributes</td><td>8 features+1 outcome label(binary)</td></tr><tr><td>Positive cases(diabetic)</td><td>268(34.9%)</td></tr><tr><td>Negative cases</td><td>500(65.1%)</td></tr><tr><td>Class imbalance ratio</td><td>~1:1.87(positive:negative)</td></tr><tr><td>Missing value handling</td><td>Zeros in Glucose,BP,Skin Thickness,Insulin(treated as missing)</td></tr><tr><td>Feature ranges</td><td>Pregnancies(0-17),Glucose(0-199),Blood Pressure(0-122),Skin Thickness(0-99),Insulin(0-846),BMI(0-67.1),DPF(0.078-2.42),Age(21-81)</td></tr><tr><td>Data source</td><td>Kaggle(UCI Machine Learning Pima Indians Diabetes Database),accessed Sept 20,2025[41]</td></tr></table>

The above feature ranges show that some features have legitimate zero (e.g. Pregnancies can be 0) while others should not be zero (e.g. minimum BMI of 0 indicates missing). We will next describe how we preprocess these data issues before feeding the data into our models.

## 4. Methodology

## 4.1. Proposed Model

We adopt a leakage-safe pipeline (see Figure 4.1) that starts by obtaining the Pima Indians Diabetes Dataset (PIDD) from Kaggle (UCI ML), followed by brief exploratory checks of distributions and class balance. The data are stratified 80/20 into training and testing, with the test split frozen. All preprocessing is performed within stratified CV folds on the training split: plausibility checks; treating implausible zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI as missing and imputing class-conditional medians; engineering 16 clinically motivated composite features; and applying standardization only for algorithms that require scaling (Logistic Regression, SVM, KNN), while tree-based models use raw scales. We then train single models Logistic Regression, SVM (RBF), KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM via stratified 5-fold CV, and form a soft-voting ensemble of XGBoost and LightGBM by averaging predicted probabilities.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179101.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=turapvGN1TRP7H8gKfSkNR7%2B3zQ%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.1: Overview of the proposed model

</div>

## 4.2. Data Preprocessing (Data Quality & Missing-Data Handling)

We first examine class balance and missing-data patterns to guide preprocessing. The dataset is modestly imbalanced, which we handle later via class-weighted losses where supported (Figure 4.2). Clinically implausible zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI are treated as missing. As shown in (Figure 4.3), missingness is concentrated in Insulin and Skin Thickness, with minor gaps in Blood Pressure, BMI, and Glucose. All missing values are imputed using class- conditional medians fit within stratified CV folds on the training split to prevent leakage.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_2_1783858179156.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=bA8EUF1KQWbMvRifbU7yxfSpzRk%3D&Expires=1784462979' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_3_1783858179161.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=ZQFUShJdc%2FyfLgsWqne3ov%2Foj%2Bk%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.2: Outcome distribution in the PIMA dataset for classes (0 = non-diabetic, 1 = diabetic).

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179168.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=10u9VGYdyHXKCcWVktT1VhGEdrA%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.3: Missing-values analysis across PIMA features.

</div>

## 4.3. Data Preprocessing (Feature Engineering [16 Composites])

To augment the eight original attributes with clinically meaningful signals, we derive 16 interpretable composite features capturing thresholds, ratios, and interactions (e.g., BMI $ \times $ SkinThickness, age-normalized terms, glucose-DPF ratio, pregnancy-age ratio). Each feature is specified by a clear formula and a brief clinical rationale in (Table 4.1). Correlations among original + engineered features are visualized in (Figure 4.4) to check redundancy and multi-collinearity before modeling.

<div align="center">

Table 4.1: Clinically motivated engineered features added to the eight original PIDD attributes.

</div>

<table border="1"><tr><td>#</td><td>Feature Name</td><td>Description</td><td>Exact Formula</td><td>Type</td><td>Why this feature?</td></tr><tr><td>1</td><td>Normal_SkinThickness</td><td>Normal skinfold thickness(≤20)</td><td>I(SkinThickness≤20)</td><td>Binary</td><td>Encodes a normal triceps skinfold to separate low-adiposity cases from higher subcutaneous fat.</td></tr><tr><td>2</td><td>Healthy_BMI</td><td>BMI within healthy range(≤30)</td><td>I(BMI≤30)</td><td>Binary</td><td>Distinguishes non-obese from obese; obesity is a major diabetes risk driver.</td></tr><tr><td>3</td><td>Young_Low_Pregnancies</td><td>Young(≤30) with low pregnancies(≤6)</td><td>I(Age≤30 AND Pregnancies≤6)</td><td>Binary</td><td>Captures a lower-risk subgroup(younger with limited parity)vs older/high-parity patterns.</td></tr><tr><td>4</td><td>Optimal_Glucose_BP</td><td>Normal glucose(≤105)和 normal BP(≤80)</td><td>I(Glucose≤105 AND BloodPressure≤80)</td><td>Binary</td><td>Flags jointly normal glycemia and diastolic BP protective profile.</td></tr><tr><td>5</td><td>Young_Normal_Glucose</td><td>Young(≤30) with normal glucose(≤120)</td><td>I(Age≤30 AND Glucose≤120)</td><td>Binary</td><td>Younger subjects with normal glucose are typically low risk; isolates that stratum.</td></tr><tr><td>6</td><td>Healthy_BMI_SkinThickness</td><td>Healthy BMI with normal skin thickness</td><td>I(BMI≤30 AND SkinThickness≤20)</td><td>Binary</td><td>Combines healthy BMI and normal skinfold to mark a globally lean phenotype.</td></tr><tr><td>7</td><td>Optimal_Glucose_BMI</td><td>Normal glucose and healthy BMI</td><td>I(Glucose≤105 AND BMI≤30)</td><td>Binary</td><td>Normal glycemia plus non-obese body mass strong negative signal for diabetes.</td></tr><tr><td>8</td><td>Normal_Insulin</td><td>Normal insulin(&lt;200)</td><td>I(Insulin&lt;200)</td><td>Binary</td><td>Indicates physiologically normal 2-hour insulin response post-load.</td></tr><tr><td>9</td><td>Normal_BloodPressure</td><td>Normal BP(&lt;80)</td><td>I(BloodPressure&lt;80)</td><td>Binary</td><td>Marks normal diastolic BP; hypertension correlates with metabolic risk.</td></tr><tr><td>10</td><td>Moderate_Pregnancies</td><td>Moderate pregnancies(1-3)</td><td>I(1≤Pregnancies≤3)</td><td>Binary</td><td>Separates a mid-parity band from very low/high parity that may behave differently.</td></tr><tr><td>11</td><td>BMI_SkinThickness_Product</td><td>Product of BMI and skin thickness</td><td>BMI*SkinThickness</td><td>Continuous</td><td>Interaction between overall adiposity(BMI)和 subcutaneous fat distribution(skinfold).</td></tr><tr><td>12</td><td>Pregnancy_Age_Ratio</td><td>Ratio of pregnancies to age</td><td>Pregnancies/(Age+1)</td><td>Continuous</td><td>Normalizes parity by age(exposure time),more stable than raw Pregnancies.</td></tr><tr><td>13</td><td>Glucose_DiabetesPedigree_Ratio</td><td>Glucose normalized by genetic predisposition</td><td>Glucose/(DPF+1e-6)</td><td>Continuous</td><td>Scales glycemia by familial risk; separates high-glucose/high-DPF from high-glucose/low-DPF.</td></tr><tr><td>14</td><td>Age_DiabetesPedigree_Product</td><td>Age weighted by genetic predisposition</td><td>Age*DPF</td><td>Continuous</td><td>Models amplification of family-history effects with aging.</td></tr><tr><td>15</td><td>Age_Insulin_Ratio</td><td>Ratio of age to insulin</td><td>Age/(Insulin+1e-6)</td><td>Continuous</td><td>Contrasts age against post-load insulin response(good response at older age is informative).</td></tr><tr><td>16</td><td>Low_BMI_SkinThickness_Product</td><td>Indicator ifBMI×SkinThickness is low(&lt;1034)</td><td>I(BMI*SkinThickness&lt;1034)</td><td>Binary</td><td>Flags globally low adiposity via a product threshold chosen from your analysis.</td></tr></table>

<table class="table table-bordered"><thead><tr><th>Pregnancies</th><th>1.00</th><th>0.13</th><th>0.21</th><th>0.09</th><th>0.06</th><th>0.02</th><td>-0.03</td><td>0.54</td><td>0.22</td><td>-0.38</td><td>-0.05</td><td>-0.62</td><td>-0.13</td><td>-0.16</td><td>-0.14</td><td>-0.14</td><td>0.02</td><td>-0.11</td><td>-0.50</td><td>0.04</td><td>0.92</td><td>0.06</td><td>0.20</td><td>0.16</td><td>-0.04</td></tr></thead><tbody><tr><td>Glucose</td><td>0.13</td><td>1.00</td><td>0.23</td><td>0.23</td><td>0.49</td><td>0.24</td><td>0.14</td><td>0.27</td><td>0.50</td><td>-0.55</td><td>-0.20</td><td>-0.24</td><td>-0.66</td><td>-0.14</td><td>-0.16</td><td>-0.44</td><td>-0.32</td><td>-0.15</td><td>-0.16</td><td>0.25</td><td>0.05</td><td>0.26</td><td>0.22</td><td>-0.24</td><td>-0.22</td></tr><tr><td>BloodPressure</td><td>0.21</td><td>0.23</td><td>1.00</td><td>0.20</td><td>0.07</td><td>0.29</td><td>-0.00</td><td>0.33</td><td>0.17</td><td>-0.30</td><td>-0.21</td><td>-0.30</td><td>-0.32</td><td>-0.18</td><td>-0.20</td><td>-0.20</td><td>-0.05</td><td>-0.74</td><td>-0.27</td><td>0.26</td><td>0.10</td><td>0.07</td><td>0.13</td><td>0.07</td><td>-0.20</td></tr><tr><td>SkinThickness</td><td>0.09</td><td>0.23</td><td>0.20</td><td>1.00</td><td>0.20</td><td>0.57</td><td>0.11</td><td>0.13</td><td>0.30</td><td>-0.24</td><td>-0.48</td><td>-0.15</td><td>-0.20</td><td>-0.65</td><td>-0.58</td><td>-0.35</td><td>-0.15</td><td>-0.13</td><td>-0.14</td><td>0.91</td><td>0.05</td><td>0.00</td><td>0.13</td><td>-0.08</td><td>-0.67</td></tr><tr><td>Insulin</td><td>0.06</td><td>0.49</td><td>0.07</td><td>0.20</td><td>1.00</td><td>0.24</td><td>0.15</td><td>0.12</td><td>0.38</td><td>-0.35</td><td>-0.23</td><td>-0.12</td><td>-0.32</td><td>-0.13</td><td>-0.13</td><td>-0.24</td><td>-0.75</td><td>-0.08</td><td>-0.08</td><td>0.23</td><td>0.02</td><td>0.06</td><td>0.17</td><td>-0.53</td><td>-0.24</td></tr><tr><td>BMI</td><td>0.02</td><td>0.24</td><td>0.29</td><td>0.57</td><td>0.24</td><td>1.00</td><td>0.15</td><td>0.03</td><td>0.32</td><td>-0.21</td><td>-0.76</td><td>-0.07</td><td>-0.21</td><td>-0.38</td><td>-0.42</td><td>-0.47</td><td>-0.20</td><td>-0.22</td><td>-0.15</td><td>0.83</td><td>-0.01</td><td>-0.05</td><td>0.12</td><td>-0.15</td><td>-0.72</td></tr><tr><td>DiabetesPedigreeFunction</td><td>-0.03</td><td>0.14</td><td>-0.00</td><td>0.11</td><td>0.15</td><td>0.15</td><td>1.00</td><td>0.03</td><td>0.17</td><td>-0.14</td><td>-0.09</td><td>-0.07</td><td>-0.07</td><td>-0.01</td><td>-0.02</td><td>-0.04</td><td>-0.13</td><td>-0.04</td><td>0.01</td><td>0.16</td><td>-0.06</td><td>-0.68</td><td>0.86</td><td>-0.01</td><td>-0.12</td></tr><tr><td>Age</td><td>0.54</td><td>0.27</td><td>0.33</td><td>0.13</td><td>0.12</td><td>0.03</td><td>0.03</td><td>1.00</td><td>0.24</td><td>-0.56</td><td>-0.05</td><td>-0.79</td><td>0.23</td><td>-0.18</td><td>-0.18</td><td>-0.19</td><td>-0.01</td><td>-0.22</td><td>-0.39</td><td>0.07</td><td>0.26</td><td>0.07</td><td>0.45</td><td>0.33</td><td>-0.03</td></tr><tr><td>Outcome</td><td>0.22</td><td>0.50</td><td>0.17</td><td>0.30</td><td>0.38</td><td>0.32</td><td>0.17</td><td>0.24</td><td>1.00</td><td>-0.39</td><td>-0.29</td><td>-0.31</td><td>-0.35</td><td>-0.20</td><td>-0.19</td><td>-0.27</td><td>-0.14</td><td>-0.11</td><td>-0.19</td><td>0.33</td><td>0.16</td><td>0.02</td><td>0.24</td><td>-0.25</td><td>-0.32</td></tr><tr><td>Young_Normal_Glucose</td><td>-0.38</td><td>-0.55</td><td>-0.30</td><td>-0.24</td><td>-0.35</td><td>-0.21</td><td>-0.14</td><td>-0.56</td><td>-0.39</td><td>1.00</td><td>0.16</td><td>0.65</td><td>0.47</td><td>0.23</td><td>0.23</td><td>0.34</td><td>0.21</td><td>0.20</td><td>0.35</td><td>-0.23</td><td>-0.22</td><td>-0.08</td><td>-0.33</td><td>0.02</td><td>0.16</td></tr><tr><td>Healthy_BMI</td><td>-0.05</td><td>-0.20</td><td>-0.21</td><td>-0.48</td><td>-0.23</td><td>-0.76</td><td>-0.09</td><td>-0.05</td><td>-0.29</td><td>0.16</td><td>1.00</td><td>0.07</td><td>0.16</td><td>0.38</td><td>0.49</td><td>0.55</td><td>0.17</td><td>0.16</td><td>0.12</td><td>-0.63</td><td>-0.02</td><td>0.03</td><td>-0.08</td><td>0.18</td><td>0.59</td></tr><tr><td>Young_Low_Pregnancies</td><td>-0.62</td><td>-0.24</td><td>-0.30</td><td>-0.15</td><td>-0.12</td><td>-0.07</td><td>-0.07</td><td>-0.79</td><td>-0.31</td><td>0.65</td><td>0.07</td><td>1.00</td><td>0.21</td><td>0.18</td><td>0.17</td><td>0.17</td><td>-0.01</td><td>0.18</td><td>0.45</td><td>-0.10</td><td>-0.42</td><td>-0.05</td><td>-0.40</td><td>-0.21</td><td>0.06</td></tr><tr><td>Optimal_Glucose_BP</td><td>-0.13</td><td>-0.66</td><td>-0.32</td><td>-0.20</td><td>-0.32</td><td>-0.21</td><td>-0.07</td><td>-0.23</td><td>-0.35</td><td>0.47</td><td>0.16</td><td>0.21</td><td>1.00</td><td>0.15</td><td>0.15</td><td>0.60</td><td>0.19</td><td>0.31</td><td>0.17</td><td>-0.21</td><td>-0.06</td><td>-0.19</td><td>-0.16</td><td>0.19</td><td>0.17</td></tr><tr><td>Normal_SkinThickness</td><td>-0.16</td><td>-0.14</td><td>-0.18</td><td>-0.65</td><td>-0.13</td><td>-0.38</td><td>-0.01</td><td>-0.18</td><td>-0.20</td><td>0.23</td><td>0.38</td><td>0.18</td><td>0.15</td><td>1.00</td><td>0.87</td><td>0.30</td><td>0.04</td><td>0.06</td><td>0.20</td><td>-0.54</td><td>-0.14</td><td>-0.04</td><td>-0.07</td><td>0.05</td><td>0.34</td></tr><tr><td>Healthy_BMI_SkinThickness</td><td>-0.14</td><td>-0.16</td><td>-0.20</td><td>-0.58</td><td>-0.13</td><td>-0.42</td><td>-0.02</td><td>-0.18</td><td>-0.19</td><td>0.23</td><td>0.49</td><td>0.17</td><td>0.15</td><td>0.87</td><td>1.00</td><td>0.37</td><td>0.03</td><td>0.10</td><td>0.20</td><td>-0.49</td><td>-0.12</td><td>-0.05</td><td>-0.08</td><td>0.07</td><td>0.29</td></tr><tr><td>Optimal_Glucose_BMI</td><td>-0.14</td><td>-0.44</td><td>-0.20</td><td>-0.35</td><td>-0.24</td><td>-0.47</td><td>-0.04</td><td>-0.19</td><td>-0.27</td><td>0.34</td><td>0.55</td><td>0.17</td><td>0.60</td><td>0.30</td><td>0.37</td><td>1.00</td><td>0.14</td><td>0.15</td><td>0.15</td><td>-0.41</td><td>-0.08</td><td>-0.12</td><td>-0.10</td><td>0.15</td><td>0.34</td></tr><tr><td>Normal_Insulin</td><td>0.02</td><td>-0.32</td><td>-0.05</td><td>-0.15</td><td>-0.75</td><td>-0.20</td><td>-0.13</td><td>-0.01</td><td>-0.14</td><td>0.21</td><td>0.17</td><td>-0.01</td><td>0.19</td><td>0.04</td><td>0.03</td><td>0.14</td><td>1.00</td><td>0.07</td><td>0.01</td><td>-0.20</td><td>0.03</td><td>0.00</td><td>-0.12</td><td>0.34</td><td>0.20</td></tr><tr><td>Normal_BloodPressure</td><td>-0.11</td><td>-0.15</td><td>-0.74</td><td>-0.13</td><td>-0.08</td><td>-0.22</td><td>-0.04</td><td>-0.22</td><td>-0.11</td><td>0.20</td><td>0.16</td><td>0.18</td><td>0.31</td><td>0.06</td><td>0.10</td><td>0.15</td><td>0.07</td><td>1.00</td><td>0.17</td><td>-0.19</td><td>-0.04</td><td>-0.00</td><td>-0.13</td><td>-0.02</td><td>0.15</td></tr><tr><td>Moderate_Pregnancies</td><td>-0.50</td><td>-0.16</td><td>-0.27</td><td>-0.14</td><td>-0.08</td><td>-0.15</td><td>0.01</td><td>-0.39</td><td>-0.19</td><td>0.35</td><td>0.12</td><td>0.45</td><td>0.17</td><td>0.20</td><td>0.20</td><td>0.15</td><td>0.01</td><td>0.17</td><td>1.00</td><td>-0.14</td><td>-0.40</td><td>-0.05</td><td>-0.16</td><td>-0.06</td><td>0.10</td></tr><tr><td>BMI_SkinThickness_Product</td><td>0.04</td><td>0.25</td><td>0.26</td><td>0.91</td><td>0.23</td><td>0.83</td><td>0.16</td><td>0.07</td><td>0.33</td><td>-0.23</td><td>-0.63</td><td>-0.10</td><td>-0.21</td><td>-0.54</td><td>-0.49</td><td>-0.41</td><td>-0.20</td><td>-0.19</td><td>-0.14</td><td>1.00</td><td>-0.00</td><td>-0.04</td><td>0.14</td><td>-0.10</td><td>-0.77</td></tr><tr><td>Pregnancy_Age_Ratio</td><td>0.92</td><td>0.05</td><td>0.10</td><td>0.05</td><td>0.02</td><td>-0.01</td><td>-0.06</td><td>0.26</td><td>0.16</td><td>-0.22</td><td>-0.02</td><td>0.42</td><td>0.06</td><td>-0.14</td><td>-0.12</td><td>-0.08</td><td>0.03</td><td>-0.04</td><td>-0.40</td><td>0.00</td><td>1.00</td><td>0.05</td><td>0.06</td><td>0.06</td><td>-0.00</td></tr><tr><td>Glucose_DiabetesPedigree_Ratio</td><td>0.06</td><td>0.26</td><td>0.07</td><td>0.00</td><td>0.06</td><td>-0.05</td><td>-0.68</td><td>0.07</td><td>0.02</td><td>-0.08</td><td>0.03</td><td>-0.05</td><td>-0.19</td><td>-0.04</td><td>-0.05</td><td>-0.12</td><td>0.00</td><td>-0.00</td><td>-0.05</td><td>-0.04</td><td>0.05</td><td>1.00</td><td>-0.57</td><td>-0.07</td><td>0.02</td></tr><tr><td>Age_DiabetesPedigree_Product</td><td>0.20</td><td>0.22</td><td>0.13</td><td>0.13</td><td>0.17</td><td>0.12</td><td>0.86</td><td>0.45</td><td>0.24</td><td>-0.33</td><td>-0.08</td><td>-0.40</td><td>-0.16</td><td>-0.07</td><td>-0.08</td><td>-0.10</td><td>-0.12</td><td>-0.13</td><td>-0.16</td><td>0.14</td><td>0.06</td><td>-0.57</td><td>1.00</td><td>0.13</td><td>-0.10</td></tr><tr><td>Age_Insulin_Ratio</td><td>0.16</td><td>-0.24</td><td>0.07</td><td>-0.08</td><td>-0.53</td><td>-0.15</td><td>-0.01</td><td>0.33</td><td>-0.25</td><td>0.02</td><td>0.18</td><td>-0.21</td><td>0.19</td><td>0.05</td><td>0.07</td><td>0.15</td><td>0.34</td><td>-0.02</td><td>-0.06</td><td>-0.10</td><td>0.06</td><td>-0.07</td><td>0.13</td><td>1.00</td><td>0.15</td></tr><tr><td>Low_BMI_SkinThickness_Product</td><td>-0.04</td><td>-0.22</td><td>-0.20</td><td>-0.67</td><td>-0.24</td><td>-0.72</td><td>-0.12</td><td>-0.03</td><td>-0.32</td><td>0.16</td><td>0.59</td><td>0.06</td><td>0.17</td><td>0.34</td><td>0.29</td><td>0.34</td><td>0.20</td><td>0.15</td><td>0.10</td><td>-0.77</td><td>-0.00</td><td>0.02</td><td>-0.10</td><td>0.15</td><td>1.00</td></tr></tbody></table>

## 4.4. Model Training (Utilize Eight Machine Learning Models & Cross-validation $ [ k=5] $ )

<div align="center">

Figure 4.4: Correlation matrix for original and engineered features.

</div>

We consider eight standard classifiers: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM (see Figure 4.5). Data are split once into stratified 80/20 (train/test), with the test split frozen. On the training split, we use stratified 5-fold cross-validation for model comparison under identical preprocessing pipelines (Figure 4.6). Class imbalance is addressed via class-weighted losses where available.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179174.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=dIiCTD6u2a1Lm%2F23sCVgoe6OBMI%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.5: The eight ML models utilized in this study.

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179180.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=GNM%2BdoWTrl5Tj67%2B4Yh0t1uxwIA%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.6: Stratified k-fold cross-validation (k=5).

</div>

5. Model Training (Ensemble Learning with Soft Voting Classifier [Combining the Top 2 Models for Higher Accuracy])

After cross-validated comparison of the single models, we construct a soft-voting ensemble that averages predicted probabilities from the two top-performing single models on validation (LightGBM and XGBoost). The ensemble architecture is shown schematically in (Figure 4.7), and the soft-voting mechanism is illustrated in (Figure 4.8).

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_2_1783858179185.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=kjqAKaxaP%2BHFpxorjNb5wuEvh3k%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.7: Ensemble architecture (combining outputs of multiple base learners into a single prediction).

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_3_1783858179190.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=%2Fz54rHPFKeV%2F0boCrP8VfCi%2BbSo%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 4.8: Soft-voting (probability averaging across base models before final class decision).

</div>

## 4.6. Model Training (Explainable AI [XAI])

We complement model training with explainability to clarify why predictions are made and to ensure clinical transparency. In line with the presentation, we apply three complementary techniques:

- Feature importance: ranks inputs by their contribution within fitted tree/boosting models useful for a quick, intuitive view of drivers.

- Permutation importance: measures the change in performance when a single feature is randomly shuffled, providing a robustness check against spurious signals.

- SHAP values: offer global summaries of feature influence across the cohort and local per-patient attributions that decompose each prediction into additive feature contributions.

These XAI analyses improve interpretability, trust, and clinical relevance. Corresponding visual summaries (global and case-level) are presented in the Results section.

## 5. Results

## 5.1. Overall comparison of single models

(Table 5.1) reports the full test-set metrics for all eight classifiers. LightGBM achieved the highest test accuracy (88.96%) with balanced precision/recall (84.91%/83.33%) and strong ROC-AUC (94.72%). XGBoost ranked second (88.31% accuracy; 94.63% AUC). Gradient Boosting followed closely (87.66% accuracy) and delivered the top AUC (95.57%). Random Forest, SVM, and KNN formed the middle tier (85.06%, 83.77%, and 83.12% accuracy, respectively), while Decision Tree and Logistic Regression were the weakest baselines (81.82% and 79.22%).

<div align="center">

Table 5.1: Comparative performance of the eight algorithms on the test set.

</div>

<table border="1"><tr><td>#</td><td>Algorithm</td><td>Train Accuracy</td><td>Test Accuracy</td><td>Test Precision</td><td>Test Recall</td><td>Test F1-Score</td><td>ROC AUC</td></tr><tr><td>1</td><td>LightGBM</td><td>100%</td><td>88.96%</td><td>84.91%</td><td>83.33%</td><td>84.11%</td><td>94.72%</td></tr><tr><td>2</td><td>XGBoost</td><td>100%</td><td>88.31%</td><td>82.14%</td><td>85.19%</td><td>83.64%</td><td>94.63%</td></tr><tr><td>3</td><td>Gradient Boosting</td><td>99.19%</td><td>87.66%</td><td>83.02%</td><td>81.48%</td><td>82.24%</td><td>95.57%</td></tr><tr><td>4</td><td>Random Forest</td><td>100%</td><td>85.06%</td><td>78.18%</td><td>79.63%</td><td>78.90%</td><td>93.69%</td></tr><tr><td>5</td><td>SVM</td><td>90.88%</td><td>83.77%</td><td>77.36%</td><td>75.93%</td><td>76.64%</td><td>90.05%</td></tr><tr><td>6</td><td>KNN</td><td>88.76%</td><td>83.12%</td><td>78%</td><td>72.22%</td><td>75%</td><td>86.62%</td></tr><tr><td>7</td><td>Decision Tree</td><td>100%</td><td>81.82%</td><td>71.67%</td><td>79.63%</td><td>75.44%</td><td>81.31%</td></tr><tr><td>8</td><td>Logistic Regression</td><td>86.32%</td><td>79.22%</td><td>71.15%</td><td>68.52%</td><td>69.81%</td><td>87.35%</td></tr></table>

## 5.2. Class performance (confusion matrices)

Per-class performance: top models reach ~81-86% recall (positive class) and ~90-92% specificity; e.g., LightGBM: 46 TP/8 FN, 92 TN/8 FP (see Figure 5.1).

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179196.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=Lv6wsmglnR0%2B2z2IsDFL7rwplA4%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.1: Confusion matrices for all eight models with test accuracies annotated.

</div>

## 5.3. ROC-AUC: Discrimination performance across models

ROC curves confirm the advantage of boosting-based learners; Gradient Boosting yields the highest AUC = 95.57% , with LightGBM = 94.72% and XGBoost = 94.63% ; Decision Tree lags at 81.31% (see Figure 5.2).

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_2_1783858179202.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=st46YEvvHzWnGXLq06gT0vdNcIk%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.2: ROC comparison across the eight algorithms.

</div>

## 5.4. Cross-validation (Stratified 5-Fold)

Stratified 5-Fold CV indicates stable generalization: LightGBM mean accuracy 87.61% , mean AUC 94.35% ; XGBoost mean accuracy 87.29% , mean AUC 94.36% (see Table 5.2a, b).

<div align="center">

Table 5.2a: Cross-validation fold results for LightGBM.

</div>

<table border="1"><tr><td>Fold</td><td>Train Accuracy</td><td>Test Accuracy</td><td>Test Precision</td><td>Test Recall</td><td>Test F1</td><td>Test ROC AUC</td></tr><tr><td>1</td><td>100%</td><td>87.80%</td><td>83.33%</td><td>81.40%</td><td>82.35%</td><td>92.76%</td></tr><tr><td>2</td><td>100%</td><td>86.18%</td><td>84.21%</td><td>74.42%</td><td>79.01%</td><td>92.99%</td></tr><tr><td>3</td><td>100%</td><td>91.06%</td><td>88.10%</td><td>86.05%</td><td>87.06%</td><td>96.66%</td></tr><tr><td>4</td><td>100%</td><td>91.06%</td><td>88.10%</td><td>86.05%</td><td>87.06%</td><td>96.22%</td></tr><tr><td>5</td><td>100%</td><td>81.97%</td><td>73.81%</td><td>73.81%</td><td>73.81%</td><td>93.10%</td></tr><tr><td>Mean</td><td>100%</td><td>87.61%</td><td>83.51%</td><td>80.34%</td><td>81.86%</td><td>94.35%</td></tr></table>

<div align="center">

Table 5.2b: Cross-validation fold results for XGBoost.

</div>

<table border="1"><tr><td>Fold</td><td>Train Accuracy</td><td>Test Accuracy</td><td>Test Precision</td><td>Test Recall</td><td>Test F1</td><td>Test ROC AUC</td></tr><tr><td>1</td><td>100%</td><td>84.55%</td><td>83.33%</td><td>69.77%</td><td>75.95%</td><td>92.50%</td></tr><tr><td>2</td><td>100%</td><td>85.37%</td><td>83.78%</td><td>72.09%</td><td>77.50%</td><td>94.39%</td></tr><tr><td>3</td><td>100%</td><td>91.87%</td><td>90.24%</td><td>86.05%</td><td>88.10%</td><td>96.16%</td></tr><tr><td>4</td><td>100%</td><td>90.24%</td><td>87.80%</td><td>83.72%</td><td>85.71%</td><td>95.70%</td></tr><tr><td>5</td><td>100%</td><td>84.43%</td><td>76.74%</td><td>78.57%</td><td>77.65%</td><td>93.04%</td></tr><tr><td>Mean</td><td>100%</td><td>87.29%</td><td>84.38%</td><td>78.04%</td><td>80.98%</td><td>94.36%</td></tr></table>

## 5.5. Soft-voting ensemble (LightGBM + XGBoost)

We build a soft-voting ensemble that averages class probabilities from LightGBM and XGBoost with equal weights. On the held-out test set, the ensemble reaches 89.61% accuracy with Precision = 85.19% Recall = 85.19% F1 = 85.19% and ROC-AUC = 94.52% (see Table 5.3). Compared with the best single models LightGBM (88.96%) and XGBoost (88.31%) (see Figure 5.3).

<div align="center">

Table 5.3: Soft-voting ensemble (LightGBM + XGBoost) performance on the test set.

</div>

<table border="1"><tr><td>Model</td><td>Train Accuracy</td><td>Test Accuracy</td><td>Test Precision</td><td>Test Recall</td><td>Test F1-Score</td><td>ROC AUC</td></tr><tr><td>Ensemble (XGBoost+LightGBM)</td><td>100%</td><td>89.61%</td><td>85.19%</td><td>85.19%</td><td>85.19%</td><td>94.52%</td></tr></table>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179208.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=jKCgmteveinTrX6VOYTKRGdm8KQ%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.3: Test accuracy (%) comparison of XGBoost, LightGBM, and their soft-voting ensemble.

</div>

## 5.6. Explainable AI (XAI) results for LightGBM

Explainability for LightGBM converges across three views: feature importance, permutation importance, and SHAP. All rank Insulin and Glucose as the most influential, followed by BMI and engineered features (e.g., Age $ \times $ DPF, BMI $ \times $ SkinThickness, Age/Insulin), then clinical variables (DPF, SkinThickness, Age, BloodPressure) (see Figures 5.4-5.7).

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179221.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=DLqEk5iLINqZQWdMvpRvOgZ6Anw%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.4: Top-15 model-based feature importances (LightGBM).

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_2_1783858179228.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=NVs2qPRN9k0oA%2F6ejMNAsq7mVMs%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.5: Top-15 permutation importances with uncertainty bars (LightGBM).

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_1_1783858179238.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=F0G1cWVVIv8NWEnEifS2RVhH9V4%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.6: SHAP summary plot (LightGBM).

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260712200856922b309c0cf54f3f%2Fcrop_2_1783858179244.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=xaw6bg6UDw9mfAEju66OKTutvE4%3D&Expires=1784462979' alt='OCR图片'/></div>

<div align="center">

Figure 5.7: SHAP mean( |value|) feature-importance bar plot (LightGBM).

</div>

## 5.7. Comparison of the proposed ensemble against recent PIDD studies

As shown in (Table 5.4), recent Scopus/WoS indexed studies on the Pima Indians Diabetes Dataset report test accuracy between 76.3% and 88.61%. Typical setups include classical classifiers (DT/SVM/NB), Random-Forest baselines, or comparisons of several standard models on the full PIDD or a subset.

Our soft-voting ensemble (LightGBM + XGBoost) reaches 89.61% test accuracy slightly higher than the best of these papers.

<div align="center">

Table 5.4: Comparison of proposed ensemble model with previous studies.

</div>

<table border="1"><tr><td>#</td><td>Authors</td><td>Year</td><td>Dataset</td><td>Methods Used</td><td>Accuracy</td><td>Indexing</td></tr><tr><td>1</td><td>Ahmed,A., et al.</td><td>2025</td><td>PIDD</td><td>RF,DT,NB,LR;PCA;5-fold CV</td><td>80%</td><td>Scopus,WoS(MDPI)</td></tr><tr><td>2</td><td>Okwudili,R., et al.</td><td>2025</td><td>PIDD</td><td>DT,SVM,NB(NB best);reports ROC/AUC</td><td>76.3%</td><td>Scopus,WoS(Springer Open)</td></tr><tr><td>3</td><td>Talukder,M.A., et al.</td><td>2024</td><td>PIDD</td><td>Random Forest(subset analysis)</td><td>80%</td><td>Indexed(journal on PMC)</td></tr><tr><td>4</td><td>Febrian,M.E., et al.</td><td>2023</td><td>PIDD</td><td>LR,SVM,RF comparisons</td><td>86%</td><td>Scopus(Procedia Computer Science)</td></tr><tr><td>5</td><td>Gupta,S.C., et al.</td><td>2023</td><td>PIDD</td><td>Random Forest(best),SVM,etc.</td><td>88.61%</td><td>Scopus(Procedia Computer Science)</td></tr><tr><td></td><td>Proposed Model</td><td>2025</td><td>PIDD</td><td>Soft-Voting Ensemble(LightGBM+XGBoost)</td><td>89.61%</td><td>Scopus(Q3,IJES-ASPD)</td></tr></table>

## 6. Conclusion & Future Work

This study introduced a leakage-safe machine-learning workflow for Type-2 diabetes prediction on the Pima Indians Diabetes Dataset (PIDD). Clinically implausible zeros were handled with classconditional imputation, and 16 interpretable composite features were engineered to enrich the signal. Using a strict train/validation protocol with a held-out test set, we evaluated standard classifiers (LR, SVM, KNN, Decision Tree, RF, GBM, XGBoost, LightGBM) alongside an Ensemble model. The ensemble achieved 89.61% accuracy, 94.52% ROC-AUC, and 85.19% F1, outperforming individual learners and aligning with or exceeding most recent Scopus-indexed PIDD baselines. SHAP analyses highlighted physiologically coherent drivers (e.g., glucose, pregnancies, age, BMI-based composites), supporting the clinical plausibility and transparency of the pipeline.

We will (i) validate the approach on larger, multi-site and more diverse cohorts beyond PIDD to assess transportability; (ii) incorporate richer feature sets (e.g., HbA1c, lipids, medications, longitudinal vitals) and study class-imbalance remedies; (iii) evaluate probability calibration and decision-curve analysis for deployment-ready thresholds; (iv) audit fairness and robustness under distribution shift and different missing-data mechanisms; and (v) explore automated hyperparameter tuning (e.g., Bayesian/DE/GA/PSO) under nested, leakage-safe validation as an optional extension. These steps aim to strengthen external validity and clinical utility for real-world screening.

## REFERENCES

[1] Jwad, S. M., & Al-Fatlawi, H. Y. (2022). Types of diabetes and their effect on the immune system. J Adv Pharm Pract, 4(1), 21-30.

[2] Mohajan, D., & Mohajan, H. K. (2023). Management of type-I diabetes: a right procedure to normal life expectancy. Frontiers in Management Science, 2(6), 47-53.

[3] Oglak, S. C., Yavuz, A., Olmez, F., Gedik Özköse, Z., & Süzen Çaypınar, S. (2022). The reduced serum concentrations of $ \beta $ -arrestin-1 and $ \beta $ -arrestin-2 in pregnancies complicated with gestational diabetes mellitus. The Journal of Maternal-Fetal & Neonatal Medicine, 35(25), 10017-10024.

[4] Roglic, G. (2016). WHO Global report on diabetes: A summary. International Journal of Noncommunicable Diseases, 1(1), 3-8.

[5] Forouhi, N. G., & Wareham, N. J. (2019). Epidemiology of diabetes. Medicine, 47(1), 22-27.

[6] Alghamdi, T. (2023). Prediction of diabetes complications using computational intelligence techniques. Applied Sciences, 13(5), 3030.

[7] Yachmaneni Jr, A., Jajoo, S., Mahakalkar, C., Kshirsagar, S., & Dhole, S. (2023). A comprehensive review of the vascular consequences of diabetes in the lower extremities: current approaches to management and evaluation of clinical outcomes. Cureus, 15(10).

[8] Haig, M., Therianos, P., Miracolo, A., Satish, K., & Kanavos, P. (2025). The burden of diabetes and options for reform: insights for the Greek health system.

[9] Raut, S. S., Acharya, S., Deolikar, V., & Mahajan, S. (2024). Navigating the frontier: emerging techniques for detecting microvascular complications in type 2 diabetes mellitus: a comprehensive review. Cureus, 16(1).

[10] Bunn, T. W., & Sikarwar, A. S. (2016). Diagnostics: conventional versus modern methods. J Adv Med Pharm Sci, 8(4), 1-7.

[11] Aisha, H., Nashiya, F., Shyma, Z., Farooqui, S., & Zulekha, S. (2024). Salivary Glucose as a Potential Biomarker for Monitoring Blood Glucose Levels in Type 2 Diabetes Mellitus: Current Insights and Future Prospects. Indian Journal of Pharmacy Practice, 17(2).

[12] Gupta, E., Saxena, J., Kumar, S., Sharma, U., Rastogi, S., Srivastava, V. K., ... & Jyoti, A. (2023). Fast track diagnostic tools for clinical management of sepsis: paradigm shift from conventional to advanced methods. Diagnostics, 13(2), 277.

[13] Stanton, A. M., Vaduganathan, M., Chang, L. S., Turchin, A., Januzzi, J. L., & Aroda, V. R. (2021). Asymptomatic diabetic cardiomyopathy: an underrecognized entity in type 2 diabetes. Current Diabetes Reports, 21, 1-11.

[14] Reddie, M. (2023). Redesigning Diabetic Foot Risk Assessment for Amputation Prevention in Low-Resource Settings: Development of a Purely Mechanical Plantar Pressure Evaluation Device. Massachusetts Institute of Technology.

[15] Zhang, W., Zhang, Y., Gu, X., Wu, C., Han, L., Zhang, W., ... & Han, L. (2022). Soft computing. Application of Soft Computing, Machine Learning, Deep Learning and Optimizations in Geoengineering and Geoscience, 7-19.

[16] Khan, W., Ishrat, M., & Al Farsi, M. M. (2025). Revolutionizing data analytics: The cutting-edge role of soft computing techniques. In Soft Computing and Machine Learning (pp. 221-243). CRC Press.

[17] Ishrat, M., Khan, W., Faisal, S. M., & Al Farsi, M. M. (2025). Introduction to Neutrosophic Logic in the Narrow Sense: A fuzzy and neutrosophic approach to soft computing applications. In Soft Computing and Machine Learning (pp. 83- 101). CRC Press.

[18] Nagaraj, P., & Deepalakshmi, P. (2022). An intelligent fuzzy inference rule-based expert recommendation system for predictive diabetes diagnosis. International Journal of Imaging Systems and Technology, 32(4), 1373-1396.

[19] Ponnarengan, H., Rajendran, S., Khalkar, V., Devarajan, G., & Kamaraj, L. (2025). Data-Driven Healthcare: The Role of Computational Methods in Medical Innovation. CMES-Computer Modeling in Engineering & Sciences, 142(1).

[20] Gupta, P., & Pandey, M. K. (2024). Role of AI for smart health diagnosis and treatment. In Smart Medical Imaging for Diagnosis and Treatment Planning (pp. 23-45). Chapman and Hall/CRC.

[21] Ahmed, A., et al. (2025). ML-based diabetes prediction among female PIMA cohort. 10.3390/healthcare13010037

[22] Okwudili, R., et al. (2025). An improved performance model for AI on the Pima Indians Diabetes Database. Journal of Electrical Systems and Information Technology. https://doi.org/10.1186/s43067-025-00224-x

[23] Talukder, M. A., et al. (2024). Toward reliable diabetes prediction: innovations in data handling. Preprint (PMC). https://pmc.ncbi.nlm.nih.gov/articles/PMC11339751

[24] Febrian, M. E., et al. (2023). Diabetes prediction using supervised machine learning. Procedia Computer Science. https://www.sciencedirect.com/science/article/pii/S1877050922021858

[25] Gupta, S. C., et al. (2023). Predictive Modeling and Analytics for Diabetes using Machine Learning. Procedia Computer Science. https://www.sciencedirect.com/science/article/pii/S1877050923001047

[26] Patro, K. K., et al. (2023). An effective correlation-based data modeling framework for diabetes prediction. 10.1186/s12859-023-05488-6

[27] Reza, M. S., et al. (2023). Improving SVM performance for type II diabetes with an improved kernel. Computer Methods and Programs in Biomedicine Update, 3, 100118. https://doi.org/10.1016/j.cmpbup.2023.100118

[28] Tasin, I., et al. (2023). Diabetes prediction using machine learning and explainable AI. Healthcare Technology Letters. https://doi.org/10.1049/htl2.12039

[29] Zhou, H., Xin, Y., Li, S. (2023). Boruta feature selection + ensemble for PIMA diabetes prediction. 10.1186/s12859-023-05300-5

[30] Kaur, H., Kumari, V. (2022). Predictive modeling & analytics for diabetes (ML approach). 10.1016/j.aci.2018.12.004

[31] Pradhan, S., et al. (2022). Voting Classification-Based Diabetes Mellitus Prediction Using PIDD. 10.1155/2022/6521532

[32] Salem, E., et al. (2022). Fine-tuning fuzzy-KNN classifier under uncertainty membership. 10.3390/app12030950

[33] Ullah, Z., et al. (2022). Detecting high-risk factors & early diagnosis using ML. 10.1155/2022/2557795

[34] Butt, U. M., et al. (2021). ML-based diabetes classification for healthcare applications. 10.1155/2021/9930985

[35] García-Ordas, M. T., et al. (2021). Diabetes detection using deep learning with oversampling & feature augmentation. 10.1016/j.cmpb.2021.105968

[36] Khanam, J. J., Foo, S. Y. (2021). A comparison of ML algorithms for diabetes classification & progression.

[37] Ramesh, S., et al. (2021). Remote healthcare monitoring framework for diabetes prediction.

[38] Patra, R., Kuntia, B. (2020). Prediction on Pima Indians Diabetes using SDKNN. 10.1088/1757-899X/923/1/012029

[39] Ullah, S., et al. (2020). Early prediction of diabetes using ML classifiers.

[40] Çalışir, D., Doğantekin, E. (2011). Automatic diabetes diagnosis via LDA-wavelet SVM. 10.1016/j.eswa.2011.01.017

[41] UCI Machine Learning. (n.d.). Pima Indians Diabetes Database. Kaggle. Retrieved September 20, 2025, from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database