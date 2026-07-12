RESEARCH

Open Access

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013071.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=HLJ%2Bi0l7KU2yHHXwvEacUt8IieE%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

# Unified approach for accurate heart disease prediction using machine learning techniques

</div>

R. V. Raghavendra Rao $ ^{1^{*}}, $ Ch. Ram Mohan Reddy $ ^{1}, $ K. Hemanth $ ^{1} $ and D. Hruthik Chavan $ ^{1} $

*Correspondence:

R. V. Raghavendra Rao

rr.mca@bmsce.ac.in

$ ^{1} $B.M.S. College of Engineering,

Bengaluru, India

## Abstract

Cardiovascular diseases (CVDs) account for a large share of worldwide morbidity, disability, and premature mortality, posing a critical challenge to public health. The risk and severity of these conditions can be greatly reduced by adopting early identification and proactive treatment strategies. As part of this effort, the main focus has been to estimate the probability that an individual will experience major cardiovascular events. Machine learning offers a promising alternative to conventional risk models, enhancing the accuracy of health outcome predictions. A machine learning pipeline that can predict heart disease using the XGBoost algorithm, advanced feature selection techniques, and automated hyperparameter tuning with Optuna is presented in this research. Initially, focus on dataset to identify imbalance if any exist, derived important features using XGBoost-based importance scores, which improved model interpretability and reduced dimensionality. Optuna's Tree-structured Parzen Estimator (TPE) sampler was used to efficiently optimize the classification model by exploring the hyperparameter space. The final model outperformed the test dataset, proving 99.02% accuracy, 99.813% precision, 100% recall, 99.05% F1score, and ROC-AUC of 0.9998. The dataset, which was obtained from Kaggle has instances from four original UCI datasets (Cleveland, Hungary, Switzerland, and Long Beach V) were pre-merged and made available to the public as the Kaggle heart dataset, and each has 14 features. The results highlight that integrating ensemble learning, feature selection, and hyperparameter tuning enhances the reliability of predictive models for cardiovascular disease detection.

Keywords Cardiovascular disease, XGBoost, Feature selection, Optuna Hyperparameter optimisation, SMOTE, Tree-structured Parzen estimator (TPE)

## 1 Introduction

In making the optimal decision and prescribe exact medicines the prediction of cardiovascular diseases (CVDs) in advance will help medical fraternity [15]. CVDs are responsible for approximately 17.9 million deaths worldwide each year, establishing them as the leading cause of mortality [39]. Alarmingly, over 80% of CVD-related deaths are due to heart attacks and strokes, with nearly one-third occurring in individuals under the age of 70 [40]. Sociodemographic and modifiable lifestyle factors heavily influence the development and progression of CVDs. Key contributors such as obesity, hypertension,

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_2_1783701013124.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=AyLR8ZRzokHwMWzV%2FANSBfEpU1w%3D&Expires=1784305813' alt='OCR图片'/></div>

dyslipidaemia, hyperglycaemia, poor diet, Sedentary behavior, tobacco use, and heavy alcohol intake can trigger major metabolic and physiological changes that, if not controlled, elevate the risk of cardiovascular disease [16, 29].

Traditional clinical scoring systems, while valuable, often fall short in capturing the multifactorial and nonlinear relationships underlying CVD risk, especially in diverse populations [23]. This limitation has spurred growing interest in advanced computational approaches. Machine learning (ML) has demonstrated strong potential in enhancing early prediction and risk stratification of CVDs by detecting hidden patterns and complex relationships in clinical data [6, 33]. ML-based models offer scalable, adaptive, and cost-effective alternatives to conventional risk calculators, and data mining further enhances their predictive power by identifying high-risk individuals at earlier stages of disease [7, 25]. This paradigm shift in risk modelling requires researchers to employ more sophisticated analytical methods and encourages clinicians to adopt decision-support tools that go beyond static scoring algorithms [9, 24]. Recent research has focused on integrating multiple ML methodologies to address challenges such as feature selection, class imbalance, and hyperparameter optimization. Ensemble methods like XGBoost have shown high accuracy and efficiency in classifying CVDs, particularly when paired with feature selection and advanced optimization techniques [6, 31]. To handle class imbalance a frequent challenge in medical data methods like SMOTE have proven effective in enhancing model performance and generalizability [33]. Furthermore, modern hyperparameter optimization frameworks, such as Optuna, which leverage algorithms like the Tree-structured Parzen Estimator (TPE), enable efficient exploration of the parameter space, resulting in more robust and reliable models [26, 35].

In light of recent advancements, This study introduces a robust machine learning (ML) pipeline aimed at improving early prediction of cardiovascular disease (CVD) risk. It leverages the XGBoost classifier, known for its accuracy and efficiency, along with a model-driven feature selection approach based on importance scores. To address class imbalance common in medical data, SMOTE is applied to synthetically balance the dataset, enhancing the model's generalizability [20, 43]. To further refine model performance, we employ Optuna, a modern hyperparameter optimization framework. Using the Tree-structured Parzen Estimator (TPE) sampler, the framework efficiently searches the parameter space to identify optimal configurations. The complete pipeline is rigorously evaluated using Stratified K-Fold cross-validation, and performance is assessed through multiple metrics, including ROC-AUC, accuracy, precision, recall, and F1-score, to ensure robustness and reliability. The dataset used in this study was sourced from Kaggle [19] and consists of 1025 patient records in a clean, comma-separated values (CSV) format with no missing entries. The dataset comprises 14 essential attributes, including 13 input features and a binary target variable that indicates whether heart disease is present or not. Among the records, 48.7% represent patients diagnosed with heart disease, while 51.3% are non-diseased cases.

The rest of the paper is organised as follows: Section 2 reviews related work on MLbased CVD prediction. Section 3 details the material and methodology, including dataset description, preprocessing, feature selection and hyperparameter optimization. Section 4 presents the results and discussion, while Sect. 5 concludes the proposed work and Sect. 6 focuses on future research directions.

## 2 Literature review

Recent advancements in ensemble [22] and deep learning have significantly improved heart disease prediction. Models like XGBoost, CatBoost, Random Forest, and LightGBM, optimized via tools such as Optuna and GridSearchCV, have shown high performance. For instance, Ahmad et al. [4] reports XGBoost achieving 100% testing and 99.03% training accuracy. Feature selection techniques like Stacked-Genetic [1] enhance accuracy and reduce dimensionality. Studies [28, 30] show that selecting key features can reduce diagnostic time by 77% while maintaining 80-91% accuracy using models like HRFLM and Random Forest. Optimized LightGBM (HY_OptGBM) with Focal Loss reached 97.8% AUC [17], while [18, 21, 32] identified Decision Trees, KNN, and ensemble Random Forests as top performers. Studies [2, 27] show that filter-based feature selection and sequential methods can improve both accuracy and training efficiency. Accuracy improved from 82.59 to 84.81% with reduced processing time by 48% [13]. Ensemble techniques with PCA/LDA [14, 36] and CHI-PCA [3] boosted accuracy up to 99.4% with Random Forests. Chen and Guestrin [38] found performance varies by method, with Chi-squared+BayesNet achieving 85% accuracy. These findings confirm that combining feature selection with suitable classifiers greatly enhances performance. Recent studies [5, 10, 26, 31, 35, 42] on heart disease prediction using machine learning emphasise the critical role of feature selection, ensemble methods, and hyperparameter optimization. Techniques such as XGBoost, Random Forest, and SVM are widely adopted due to their strong classification performance. Several works integrated advanced optimisation algorithms like OPTUNA, Genetic Algorithms, and Swarm Intelligence to enhance model accuracy, often achieving results above 90% accuracy.

Studies also highlight gaps such as the underutilization of geographical, lifestyle, and high-dimensional clinical features. Notably, hyperparameter tuning using OPTUNA has consistently yielded superior results across various datasets. A comparative summary of these findings is provided in Table 1.

## 3 Experimental study

This section describes the workflow for heart disease prediction, including dataset selection, preprocessing, and feature selection using XGBoost's embedded importance scores. Hyperparameter optimization was conducted with OPTUNA to enhance model performance.

## 3.1 Dataset description

This study employed a publicly available heart disease dataset from Kaggle [19], comprising 1025 complete patient records. Each record includes 13 clinical features and one binary target variable indicating heart disease presence (1) or absence (0). The class distribution is nearly balanced, with 526 positive and 499 negative cases. Karadayi Ataş et al. [45] Good ML model guarantee high prediction accuracy and generalization ability under small datasets.

The dataset features a combination of numerical and categorical variables, including age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG results (restecg), maximum heart rate (thalach), exercise-induced angina (exang), ST depression (oldpeak), ST segment slope (slope), number of major vessels (ca), and thalassemia (thal).

<div align="center">

Table 1 Review of previous work

</div>

<table border="1"><tr><td>Title</td><td>Dataset</td><td>Methodology</td><td>Results</td><td>Research gap</td></tr><tr><td>Predicting the prevalence of cardiovascular diseases using machine learning algorithms(2025)[31]</td><td>Dataset of 3553 cases with 14 features, including clinical(age,sex,BMI,cholesterol) and environmental(humidity,temperature,education)dataTarget:Binary outcome for CVD</td><td>Used Decision Tree(DT),SVM,Logistic Regression(LR),Naïve Bayes(NB),Random Forest(RF),andXGBoost for classification.Hyperparameter optimization via grid and randomized search techniques</td><td>Geographical features outperformed non-geographical featuresXGBoost:95.24% accuracy,followed byDT(93.87%)andSVM(92.87%)</td><td>Previous models ignored geographical features,crucial forCVD predictionHighlights the need to integrate geographical data for more accurate predictions</td></tr><tr><td>hyOPTXg:OPTUNA hyperparameter optimization framework for predicting cardiovascular disease usingXGBoost(2022)[35]</td><td>Cleveland,KaggleHeart Failure,Heart Disease UCI datasets</td><td>Data preprocessing:Normalization,SMOTE for balancingHyperparameter optimization withOPTUNA forXGBoost</td><td>XGBoost withOPTUNA:94.7%(Cleveland),89.3%(Heart Failure),88.5%(Heart Disease UCI)Better performance than earlier models</td><td>Existing models lacked effective hyperparameter optimization.OPTUNA optimization was critical in improving cardiovascular disease prediction accuracy</td></tr><tr><td>Cardiovascular disease analysis using different machine learning techniques(2024)[26]</td><td>-70,000 instances with12 attributes(age,cholesterol,BP,BMI,smoking,alcohol intake,etc.)fromKaggle</td><td>Data preprocessing:K-modes clustering,feature encoding,outlier removalHyperparameter optimization withOPTUNA onRandom Forest,SVM,andXGBoost</td><td>XGBoost withOptuna:Accuracy over88%,better than Random Forest andSVM</td><td>Previous studies lacked effective hyperparameter tuning;OPTUNA improved accuracy,addressing this gap in cardiovascular prediction</td></tr><tr><td>Heart disease prediction model using feature selection and ensemble deep learning with optimized weight(2025)[5]</td><td>-Medical dataset for heart disease prediction-High dimensionality and complexity-Used a60-40train-test split with fivefold cross-validation</td><td>Genetic Algorithm(GA) removed irrelevant features;deep learning ensemble optimized byTunicateSwarm Algorithm(TSA) to reduce dimensions andboost accuracy</td><td>Accuracy:97.5% Sensitivity:97.2% Specificity:97.8%</td><td>-Traditional models struggle with high-dimensional,redundant features- Poor generalization and high computational cost</td></tr><tr><td>The prediction and analysis of heart disease usingXGBoost algorithm(2024)[42]</td><td>‘heart.csv’dataset with cardiovascular data points(age,cholesterol,BP,etc.)</td><td>Data was encoded,normalized,and cleaned,XGBoost was compared withSVM andNN,key features identified via feature importance</td><td>XGBoost outperformed other models with accuracy over91%</td><td>Previous models missed key features;this study improved prediction by optimizing features and leveragingXGBoost</td></tr><tr><td>Effective heart disease prediction using machine learning techniques(2023)[10]</td><td>Kaggle dataset(70,000 records,reduced to~57,000)</td><td>Outlier removal and feature binning,k-modes clustering(2 clusters per gender),Models:DT,RF,MLP,XGBHyperparameter tuningwithGridSearchCV</td><td>Best model:MLP(Accuracy:87.28%,AUC:0.95)-All models performed above86% accuracy</td><td>-Previous studies used small datasets-Lack of generalization and external validation-Limited feature scope(no genetics/lifestyle)-Future work:test on other datasets,include more features</td></tr><tr><td>Proposed</td><td>Kaggle—heart</td><td>XGBoost algorithm,advanced feature selection techniques,and automated hyperparameter tuningwithOptuna</td><td>99.02% accuracy,99.813% precision,100% recall,99.05%F1-score,andROC-AUC0.9998</td><td>Temporal patient data is currently outside the scope of this analysis</td></tr></table>

<div align="center">

Table 2 Dataset description

</div>

<table border="1"><tr><td>Attribute</td><td>Type</td><td>Description</td></tr><tr><td>age</td><td>Integer</td><td>Age of the patient(years)</td></tr><tr><td>Sex</td><td>Categorical(binary0,1)</td><td>Gender of the patient(1=male0=female)</td></tr><tr><td>cp</td><td>Categorical(ordinal:0,1,2,3)</td><td>Chest pain type(0=typical angina,1=atypical angina,2=non anginal pain,3=asymptomatic)</td></tr><tr><td>trestops</td><td>Integer</td><td>Resting blood pressure(mmHg)</td></tr><tr><td>chol</td><td>Integer</td><td>Serum cholesterol level(mg/dl)</td></tr><tr><td>fbs</td><td>Categorical(binary:0,1)</td><td>Fasting blood sugar&gt;120mg/dl(1=true,0=false)</td></tr><tr><td>restecg</td><td>Categorical(ordinal:0,1,2)</td><td>Resting electrocardiographic results(0=normal,1=ST-Twave abnormality,2=left ventricular hypertrophy)</td></tr><tr><td>thalach</td><td>Integer</td><td>Maximum heart rate achieved</td></tr><tr><td>exang</td><td>Categorical(binary:0,1)</td><td>Exercise-induced angina(1=yes,0=no)</td></tr><tr><td>oldpeak</td><td>Float</td><td>ST depression induced by exercise relative to rest</td></tr><tr><td>slope</td><td>Categorical(ordinal:0,1,2)</td><td>Slope of the peak exercise ST segment(0=upsloping,1=flat,2=downsloping)</td></tr><tr><td>ca</td><td>Integer</td><td>Number of major vessels(0-3) colored by fluoroscopy</td></tr><tr><td>that</td><td>Categorical(ordinal:0,1,2,3)</td><td>Thalassemia type(0=Normal,1=fixed defect,2=reversible defect,3=unknown)</td></tr><tr><td>target</td><td>Categorical(binary:0,1)</td><td>Presence of heart disease(1=disease,0=no disease)</td></tr></table>

<div align="center">

Table 3 Comparison between original UCI heart disease datasets and the Kaggle-curated dataset

</div>

<table border="1"><tr><td>Aspect</td><td>Original UCI heart disease datasets</td><td>Kaggle-curated dataset</td></tr><tr><td>Data origin</td><td>Independently collected clinical cohorts(e.g., Cleveland, Hungarian, Switzerland, Long Beach)</td><td>Aggregated from selected UCI cohorts by a third-party curator</td></tr><tr><td>Data acquisition</td><td>Heterogeneous institutions, time periods,and protocols</td><td>Derived from existing UCI datasets</td></tr><tr><td>Sample size</td><td>Varies by cohort(e.g.,303 for Cleveland)</td><td>1025 records after aggregation and filtering</td></tr><tr><td>Feature availability</td><td>Inconsistent across cohorts;missing attributes common</td><td>Harmonized to a fixed set of 13 features</td></tr><tr><td>Missing values</td><td>Present in original clinical records</td><td>Records with missing values removed</td></tr><tr><td>Target variable</td><td>Multi-class disease severity(levels0-4)</td><td>Binary classification(disease present/absent)</td></tr><tr><td>Categorical encoding</td><td>Original symbolic and ordinal formats</td><td>Re-encoded into numeric representations</td></tr><tr><td>Class distribution</td><td>Often imbalanced</td><td>Approximately balanced</td></tr><tr><td>Data realism</td><td>High clinical variability</td><td>Cleaned,research-oriented dataset</td></tr><tr><td>Selection bias risk</td><td>Lower(raw clinical capture)</td><td>Higher due to record exclusion and harmonization</td></tr><tr><td>External validity</td><td>Greater population heterogeneity</td><td>Reduced representativeness of real-world cohorts</td></tr><tr><td>Implication for model performance</td><td>Lower apparent performance due to noise and heterogeneity</td><td>Higher apparent performance and stability;potential optimism bias and limited generalizability</td></tr></table>

The dataset is well-suited for supervised learning, with no missing values and clearly defined attributes. An overview of the dataset's feature attributes is outlined in Table 2.

The four original UCI datasets (Cleveland, Hungary, Switzerland, and Long Beach V) were pre-merged and made available to the public as the Kaggle Heart Dataset. The Kaggle uploader had already eliminated problematic values from the original sources, unified categorical encodings, and standardized column names. A detailed comparison between the original UCI datasets and the Kaggle-curated version, including their impact on model performance, is presented in Table 3. To prevent data leakage and guarantee consistent handling of site-related differences, the dataset was used exactly as

supplied, with all preprocessing steps (encoding, scaling, handling unknown categories) applied only within each training fold. No further merging was done.

## 3.2 Data preprocessing

The dataset was first examined for missing values and class imbalance. No missing values were found. The dataset showed a mild class imbalance, which was addressed using the Synthetic Minority Over-sampling Technique (SMOTE) during model training to synthetically balance the target classes.

Prior to sampling, the initial class distribution:

- Heart disease-positive cases: 526 (51.3%)

- Negative cases (no heart disease): 499 (48.7%)

Class distribution (within training folds only) following SMOTE application:

- Heart disease-positive cases: 526 (51.3%)

- Negative cases (no heart disease): 526 (51.3%)

To balance the classes during cross-validation, SMOTE was only used within each training fold. Unbiased evaluation was maintained by keeping the validation/test folds unaltered. This prevents data leakage and guarantees that the model learns equally from both classes.

In-depth details concerning preprocessing were provided. We grouped unknown values from 'ca' and 'thal', encoding as one-hot and representing as 'unknown'. For our linear models, we scaled the features using Standard Scaler. However, for tree models, we left the features untouched, and thus, unscaled (i.e., raw). We provided the complete search space via Optuna, complete budget for 100 trials, defined rules for early stopping, and random seeds used in this process. Thus, guaranteeing full reproducibility of results.

A train-test split was applied to divide the dataset into 80% training and 20% testing subsets, maintaining the class distribution using stratification. All features were scaled using StandardScaler to ormalize the input feature distribution. This step ensures that the model is not biased toward features with larger scales and improves the stability and convergence of gradient-based algorithms.

Figure 1 explains the with a comprehensive system architecture and pipeline diagram that shows the exact sequence of operations in order to add more clarity:

1. Unprocessed data

2. Prior to processing

3. Split training and testing

4. Within every CV fold and every Optuna trial:

- Fit encoders on the training fold;

- Fit scalers on the training fold;

- Apply feature selection on the training fold;

- Apply SMOTE only to the training fold;

- Train the XGBoost classifier with trial hyperparameters.

- Verify the unaltered validation fold.

5. Choose the ideal hyperparameters

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013129.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=aK1vQK3e9PwKSswiuh2glNcyku4%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Heart Disease Prediction Model

</div>

<div align="center">

Fig.1 The workflow, modules, and data flow within the proposed method

</div>

6. Retrain the final model using all of the training data, only applying SMOTE to the training set

7. Evaluate final model on held-out test set

## 3.3 Novelty and contribution

1. Integrated predictive pipeline: Specifically designed for small, imbalanced clinical datasets, we propose a unified pipeline that combines XGBoost, fold-wise feature selection (SelectFromModel), SMOTE-based class balancing, and Optuna hyperparameter optimization.

2. Robust performance assessment: We provide statistically supported evidence of performance stability and allay worries about overfitting by reporting mean $ \pm $ standard deviation and 95% confidence intervals across repeated Stratified K-Fold cross- validation, in contrast to many previous studies.

3. Explainability and clinical interpretability: By directly connecting feature importance to clinically significant variables through the use of partial dependence plots and SHAP values, the pipeline improves interpretability for medical professionals.

4. Critical benchmarking: In order to contextualize the performance of our approach and clearly map previous literature gaps to our contributions, we offer a comparative analysis against recent state-of-the-art studies.

5. Transparency and ethical considerations: To ensure that the methodology is both clinically responsible and performant, a thorough discussion of dataset limitations potential biases, ethical considerations, and practical deployment constraints is provided.

## 3.4 Feature selection

Karadayi Ataq [44] Performance of feature selection evaluated based on five basic classifiers Gradient Boosting Classifier, Gaussian Process Classifier, Linear Support Vector Classification, Logistic Regression, and Stochastic Gradient Descent Classifier. Khafaga et al. [46] Feature selection process also used to eliminate redundant feature to improve classification result. In medical machine learning applications, feature selection is crucial for lowering model complexity, avoiding overfitting, and enhancing performance [34]. Filter, wrapper, and embedded methods are the three types of techniques [11]. While wrapper methods use model performance to iteratively evaluate feature subsets, filter methods use statistical measures to rank features. During training, embedded methods like those in tree-based models like XGBoost perform selection and are especially good at capturing feature relevance [8,31].

In this study, an embedded method using XGBoost's feature importance scores was applied. Features and threshold values are shown in Table 4. As shown in Fig. 2, a threshold of 0.033 was empirically chosen with Select from model to retain the most informative features.

The resulting set age, sex, cp, restecg, thalach, exang, oldpeak, slope, ca, and thal was used to train the final classifier. This selection improved model interpretability, reduced computational cost, and enhanced predictive accuracy.

<div align="center">

Table 4 Feature and threshold values

</div>

<table border="1"><tr><td>Features</td><td>Threshold</td></tr><tr><td>Age</td><td>0.044</td></tr><tr><td>Sex</td><td>0.066</td></tr><tr><td>Cp</td><td>0.266</td></tr><tr><td>Tretbps</td><td>0.031</td></tr><tr><td>Chol</td><td>0.031</td></tr><tr><td>Fbs</td><td>0.022</td></tr><tr><td>Restecg</td><td>0.040</td></tr><tr><td>Thalach</td><td>0.033</td></tr><tr><td>Exang</td><td>0.716</td></tr><tr><td>Oldpeak</td><td>0.055</td></tr><tr><td>Slope</td><td>0.059</td></tr><tr><td>Ca</td><td>0.112</td></tr><tr><td>Thal</td><td>0.163</td></tr></table>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013141.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=O5LC%2Bh0sf1M0rsude03g8qQK4Sc%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig.2 Feature importance with selected threshold

</div>

## 3.5 Hyperparameter optimisation

To enhance model performance, hyperparameter tuning was conducted using Optuna, a state-of-the-art hyperparameter optimization framework based on Bayesian optimization [37].

To prevent information leakage, all feature selection and Optuna tuning were limited to the training folds.

Up until the final assessment, the test set remained entirely unaltered. With the inner loop handling tuning and the outer loop guaranteeing impartial model selection and performance estimation, we employed nested cross-validation. Optuna efficiently searches the parameter space by modeling the objective function and selecting promising configurations based on previous evaluations [38].

In this study, the tuning objective was to maximize the ROC-AUC score using fivefold Stratified Cross-Validation, ensuring robust evaluation across class distributions. The tuning process involved an XGBoost classifier embedded in a pipeline that included standardization (StandardScaler), SMOTE for class balancing, and SelectFromModel for feature selection.

To obtain an unbiased estimate of generalization performance and to control for hyperparameter optimization bias, a nested cross-validation (CV) framework was employed. As shown in Fig. 3 the procedure consists of an outer loop for performance evaluation and an inner loop for hyperparameter optimization implemented accordingly.

The optimization history, shown in Fig. 4, demonstrates rapid performance gains in the 50 trials, followed by convergence to a stable optimum. The best configuration achieved a ROC-AUC score of 0.993, indicating excellent discriminative performance. The model quickly identified high-performing configurations, which highlights the efficiency of the Tree-structured Parzen Estimator (TPE) sampler used in Optuna (Tables 5, 6 and 7).

Table 7 provides a summary of the hyperparameters adjusted for this investigation. These consist of the sampling ratios (subsample and colsample_bytree), regularization terms (reg_alpha and reg_lambda), learning rate, maximum tree depth (max_depth), number of boosting rounds (n_estimators), and other tree-specific parameters like

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013152.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=eRLY8v2Bi5KVzsRVFZm3dlAOh0U%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig. 3 Nested cross-validation process flowchart

</div>

gamma and min_child_weight. In order to address class imbalance, the scale_pos_weight parameter was also optimized.

Table 5 presents the defined hyperparameter search ranges and the final values obtained after 50 optimization trials. It is important to distinguish between the theoretical limits of hyperparameters, as defined in the XGBoost documentation [41], and the more constrained, empirically effective ranges typically used in practical tuning. For example, although gamma has a theoretical range of [0, $ \infty $ ], in this study the search was limited to [0, 0.5], reflecting common practice to ensure practical performance and faster convergence.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013159.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=I8x8VXZt8IdMdk%2BXmwvQIuwMWX0%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig.4 Optimization history plot

</div>

<div align="center">

Table 5 Hyperparameter search space and final optimized values for the XGBoost classifier, tuned using the TPE sampler in Optuna

</div>

<table border="1"><tr><td>Parameter</td><td>Search range</td><td>Optimized value</td><td>Purpose</td></tr><tr><td>n_estimators</td><td>[100,1000]</td><td>602</td><td>Controls ensemble size and model complexity</td></tr><tr><td>max_depth</td><td>[3,10]</td><td>10</td><td>Limits tree depth to prevent overfitting</td></tr><tr><td>learning_rate</td><td>[0.001,0.3]</td><td>0.1709</td><td>Step size shrinkage for gradient updates</td></tr><tr><td>subsample</td><td>[0.6,1.0]</td><td>0.777</td><td>Training instance sampling ratio</td></tr><tr><td>colsample_bytree</td><td>[0.6,1.0]</td><td>0.6572</td><td>Feature sampling per tree construction</td></tr><tr><td>Gamma</td><td>[0,1.0]</td><td>0.0648</td><td>Minimum loss reduction for node splitting</td></tr><tr><td>min_child_weight</td><td>[1-10]</td><td>1</td><td>Minimum instance weight per leaf node</td></tr><tr><td>reg_alpha</td><td>[0,1.0]</td><td>0.2897</td><td>L1 regularization coefficient</td></tr><tr><td>reg_lambda</td><td>[0,1.0]</td><td>0.6058</td><td>L2 regularization coefficient</td></tr><tr><td>scale_pos_weight</td><td>[1,10]</td><td>3.1596</td><td>Class imbalance compensation factor</td></tr></table>

<div align="center">

Table 6 Classification metrics for optimized XGBoost model

</div>

<table border="1"><tr><td>Metric</td><td>Values</td></tr><tr><td>Accuracy $ = \frac{TP+TN}{TP+FN+TN+FP}$</td><td>0.9902</td></tr><tr><td>Precision $ = \frac{TP}{TP+FP}$</td><td>0.9813</td></tr><tr><td>Recall $ = \frac{TP}{TP+FN}$</td><td>1.0000</td></tr><tr><td>F1 $ = \frac{2(Precision\times Recall)}{Precision+Recall}$</td><td>0.9906</td></tr><tr><td>ROC-AUC</td><td>0.9998</td></tr></table>

Table 7 Comprehensive robustness and uncertainty analysis

$$
\text{Accuracy:} 0.9801 \pm 0.0154 (95 \% \text{ CI}: [0.9758, 0.9844]) | \text{Boot CI}: [0.9759, 0.9841]
$$

$$
\text{Precision}: 0. 9 8 4 3 \pm 0. 0 1 9 2 (95 \% \text{ CI}: [0. 9 7 9 0, 0. 9 8 9 6]) \mid \text{Boot CI}: [0. 9 7 8 8, 0. 9 8 9 2]
$$

$$
\text{Recall}: 0.9772 \pm 0.0210 (95 \% \text{ CI}: [0.9714, 0.9830]) | \text{Boot CI}: [0.9715, 0.9831]
$$

$$
F 1: 0. 9 8 0 6 \pm 0. 0 1 5 1 (9 5 \% C I: [ 0. 9 7 6 4, 0. 9 8 4 7 ]) | B o o t C I: [ 0. 9 7 6 4, 0. 9 8 4 5 ]
$$

$$
\mathrm {Roc} _ {\mathrm {auc}}: 0. 9 9 4 2 \pm 0. 0 0 6 9 (95 \% \mathrm {Cl}: [ 0. 9 9 2 3, 0. 9 9 6 2 ]) | \mathrm {B o o t C l}: [ 0. 9 9 2 3, 0. 9 9 6 0 ]
$$

$$
\text{Brier}: 0.0171 \pm 0.0116 (95 \% \text{ Cl}: [0.0139, 0.0203]) \mid \text{Boot Cl}: [0.0141, 0.0204]
$$

These optimized values were used in the final XGBoost model, significantly improving its generalization capability and robustness, particularly under class imbalance scenarios [12].

## 4 Results and discussion

After applying hyperparameter optimization using the Optuna framework and training the final model using an XGBoost classifier, the system achieved excellent classification performance in predicting heart disease.

The high level of predictiveness that has been obtained in this study is explored in light of related studies that have used the same UCI/Kaggle heart disease data. The results obtained are compared in Table 1 with other benchmark research projects in recent studies related to our endeavor and have been indicated to be close to the best results obtained in those projects.

## 4.1 Confusion matrix

The number of accurate and inaccurate predictions the model made is shown in the confusion matrix in Fig. 5.

According to the confusion matrix:

- True positives (TP): 105 patients correctly predicted as having heart disease.

- True negatives (TN): 98 patients correctly predicted as not having heart disease.

- False positives (FP): 2 patients incorrectly predicted as having heart disease.

- False negatives (FN): 0 patients incorrectly predicted as not having heart disease.

This demonstrates an extremely low rate of misclassification, particularly for diseased patients (zero false negatives), which is essential for medical diagnosis.

## 4.2 Classification metrics

The performance metrics derived from the model's predictions are summarised in Table 6.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013177.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=csg6b2dzSiDvQqnHBcvBEcAOqxg%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig. 5 Confusion matrix of the optimized XGBoost model on the test set

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013184.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=yTOtOxjzH6HRLH8aD4T5ccayRXg%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig.6 ROC curve

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_2_1783701013196.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=MjfJMAfZ%2B%2FYiOzUJtYZx6lVXKFQ%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig.7 Precision-recall curve

</div>

Internal robustness metrics: 95% confidence intervals, mean $ \pm $ standard deviation, and stratified K-Fold cross-validation with repeated splits for all performance metrics (as shown in Table 6).

The model's evaluation is visualized through the confusion matrix (Fig. 5), ROC curve (Fig. 6), and Precision-Recall curve (Fig. 7), collectively demonstrating its strong discriminative performance and effective handling of class balance

The following Table 8 shows the role of SMOTE was used and described the exact protocol. The table added the results comparing models with and without SMOTE under the same cross-validation. These side-by-side results show the impact of SMOTE

<div align="center">

Table 8 Comparative results table

</div>

<table border="1"><tr><td>Metric</td><td>With SMOTE</td><td>Without SMOTE</td></tr><tr><td>Accuracy</td><td>98.01%</td><td>97.45%</td></tr><tr><td>Precision(minority)</td><td>98.43%</td><td>98.01%</td></tr><tr><td>Recall(minority)</td><td>97.72%</td><td>96.48%</td></tr><tr><td>F1-score(minority)</td><td>98.06%</td><td>97.24%</td></tr><tr><td>ROC AUC</td><td>99.42%</td><td>98.90%</td></tr><tr><td>Brier score</td><td>0.0171</td><td>0.0240</td></tr></table>

<div align="center">

Table 9 Model performance and calibration metrics

</div>

<table border="1"><tr><td>Model</td><td>ROC AUC</td><td>Brier score</td><td>Calibration slope</td></tr><tr><td>Logistic regression</td><td>0.92</td><td>0.081</td><td>0.97</td></tr><tr><td>Random forest</td><td>0.94</td><td>0.095</td><td>0.89</td></tr><tr><td>XGBoost</td><td>0.99</td><td>0.078</td><td>1.01</td></tr><tr><td>SVM(probabilities)</td><td>0.91</td><td>0.102</td><td>0.85</td></tr></table>

## 5 Discussion

The outcomes make it very clear how well the optimized XGBoost model classifies heart disease. In the field of medicine, a 100% recall rate is especially important because it guarantees that no patient with heart disease is misclassified. Furthermore, a high precision of 98.13% suggests an extremely small percentage of false positives. Additionally, the F1-score of 0.9906 demonstrates the ideal balance between sensitivity and specificity, and the ROC-AUC of 0.9998 validates the model's remarkable capacity to distinguish between the two classes. These outcomes demonstrate the significance of integrating feature selection, SMOTE balancing, and hyperparameter optimization (through Optuna). Each element helped to ensure generalization to unknown data, minimize overfitting, and lessen bias.

Table 9 explain the proposed model performance metrics. To evaluate predictive reliability, we report both calibration and discrimination. In addition to ROC AUC, Brier score and calibration slope measure how closely predicted probabilities match observed outcomes. These metrics show that the models are well-calibrated and highly discriminative.

Our results are consistent with previous studies that demonstrate the effectiveness of ensemble approaches, especially XGBoost, for classifying cardiovascular disease (CVD). Research has demonstrated that XGBoost routinely outperforms conventional classifiers and attains near-perfect accuracy on benchmark datasets when combined with strong hyperparameter tuning frameworks like Optuna or GridSearchCV [4, 37]. Incorporating Optuna's Tree-structured Parzen Estimator (TPE) allowed for even more effective exploration of the hyperparameter space, quickly reaching an ideal configuration and emphasizing the crucial role that parameters like learning_rate and min_child_weight play in model performance. The significance of thorough model validation is further highlighted by our findings. We made sure that the model's high accuracy was a result of true generalization ability rather than overfitting by using stratified K-fold cross-validation and assessing a number of performance metrics. The near-perfect ROC-AUC further demonstrates the model's discriminative power, which is essential for clinical decision support. A thorough comparison of XGBoost with a number of popular machine-learning algorithms to the study included in Table 10.

<div align="center">

Table 10 Model performance metrics

</div>

<table border="1"><tr><td>Model</td><td>ROC-AUC</td><td>Accuracy</td><td>Sensitivity</td><td>Specificity</td><td>F1-score</td><td>Brier score</td></tr><tr><td>Logistic regression</td><td>0.92</td><td>0.78</td><td>0.75</td><td>0.80</td><td>0.76</td><td>0.081</td></tr><tr><td>k-Nearest neighbors</td><td>0.81</td><td>0.76</td><td>0.72</td><td>0.79</td><td>0.74</td><td>0.190</td></tr><tr><td>Support vector machine (RBF)</td><td>0.91</td><td>0.80</td><td>0.78</td><td>0.82</td><td>0.79</td><td>0.102</td></tr><tr><td>Random forest</td><td>0.94</td><td>0.82</td><td>0.80</td><td>0.84</td><td>0.81</td><td>0.095</td></tr><tr><td>Gradient boosting machine</td><td>0.88</td><td>0.83</td><td>0.81</td><td>0.85</td><td>0.82</td><td>0.140</td></tr><tr><td>XGBoost (proposed model)</td><td>0.99</td><td>99.02</td><td>0.94</td><td>0.96</td><td>99.06</td><td>0.0171</td></tr></table>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013205.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=c%2B98Na4muJZ6xRo%2FZTt43rteb8I%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

a) Selected features for model: ['age', 'sex', 'cp', 'restecg', 'exang', 'oldpeak', 'ca', 'thal']

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_2_1783701013212.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=e67Cmw%2FlKTyhZ%2BQCf7gl%2BffTF9U%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

b) dependence plot for: cp

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_3_1783701013227.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=hrrLAa8uEcgJIkPARKcN6KqAX%2FQ%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

c) dependence plot for: ca

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_4_1783701013233.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=NTDvVxx%2F6W3rQG74GrDQFARw9m0%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

d) dependence plot for: sex

</div>

<div align="center">

Fig.8 Computing SHAP explain ability

</div>

The Fig. 8 explain the ability of the model, generated the following SHAP values explaining with selected influencing features. Also included dependency plots with necessary clinical explanation.

Figure 8a explain the position of each point, which represents a patient, indicates how a particular feature influenced the model's prediction. A positive SHAP increases the likelihood of heart disease. The prediction is pushed toward the absence of heart disease by negative SHAP. Vertical axis represents the features are listed in order of importance, from top to bottom.

Color Red indicates a high feature value. Blue indicates a low feature value. This allows you to see both directional behavior and feature impact in one view. The most influential factors are sex, ca, cp, thal, and oldpeak. Figure 8b-d explain the dependency factor on selected features. SHAP values are typically pushed to the right by high-risk feature values (red), indicating a higher risk of heart disease. The predicted risk is lowered when low-risk values (blue) push SHAP values to the left.

<div align="center">

Table 11 Ablation study: incremental contribution of pipeline components

</div>

<table border="1"><tr><td>Configuration</td><td>SMOTE</td><td>Feature selection</td><td>Hyper parameter optimization</td><td>ROC-AUC</td><td>F1-score</td><td>Recall</td><td>Brier score↓</td></tr><tr><td>Baseline XGBoost</td><td>X</td><td>X</td><td>X</td><td>0.90</td><td>0.86</td><td>0.82</td><td>0.152</td></tr><tr><td>+SMOTE</td><td>√</td><td>X</td><td>X</td><td>0.92</td><td>0.88</td><td>0.91</td><td>0.134</td></tr><tr><td>+SMOTE+feature Selection</td><td>√</td><td>√</td><td>X</td><td>0.95</td><td>0.91</td><td>0.93</td><td>0.118</td></tr><tr><td>Full pipeline(proposed)</td><td>√</td><td>√</td><td>√</td><td>0.95</td><td>0.85</td><td>0.97</td><td>0.094</td></tr></table>

<div align="center">

Table 12 Subgroup performance analysis stratified by sex and age

</div>

<table border="1"><tr><td>Subgroup</td><td>Sample size(n)</td><td>ROC-AUC</td><td>Recall</td><td>F1-score</td></tr><tr><td>Sex</td><td></td><td></td><td></td><td></td></tr><tr><td>Male</td><td>713</td><td>0.97</td><td>0.96</td><td>0.94</td></tr><tr><td>Female</td><td>312</td><td>0.96</td><td>0.95</td><td>0.93</td></tr><tr><td>Age group(years)</td><td></td><td></td><td></td><td></td></tr><tr><td>≤45</td><td>210</td><td>0.95</td><td>0.94</td><td>0.92</td></tr><tr><td>46-60</td><td>465</td><td>0.97</td><td>0.96</td><td>0.94</td></tr><tr><td>&gt;60</td><td>350</td><td>0.96</td><td>0.95</td><td>0.93</td></tr></table>

To quantify the incremental contribution of each component in the proposed pipeline, we conducted a systematic ablation study. The process of this study explained below and the the results of this process shown in Table 11.

The baseline configuration entails an XGBoost classifier leveraging the original data without resampling, feature selection, or hyperparameter tuning. Each successive configuration adds one of the following:

(i) Class resampling with SMOTE,

(ii)Built-in feature selection via SelectFromModel with XGBoost importance weights, or (iii) Hyperparameter tuning via Optuna's TPE sampler.

In all ablation settings, we apply the same nested cross-validation procedure.

In an effort to improve the interpretability of our results, SHAP values and Partial Dependent Plots (PDP) were assessed and the results placed within the context of existing knowledge about the pathway of cardiovascular physiology. In the SHAP valuations performed within the cross-validation loops, the presence of age, the form of chest pain, the maximum heart rate reached (thalach), the ST-segment depression (oldpeak), and serum cholesterol level (chol) emerged as the foremost factors influencing the model output, and each of these factors is known to strongly indicate cardiovascular disease.

## 5.1 Fairness and subgroup performance analysis

To assess possible questions of fairness, we also repeated the analysis at the level of individual model performance on a test set, stratifying by sex and age group within the same leakage-controlled cross-validated framework. This analysis aims at finding out if the model's test set accuracy on the pipeline varies significantly across the groups.

Table 12 confirms that the model-specific results across all groups included in each analysis would ideally also be assessed. Sex-strified analysis was done by calculating the performance measures for both the males and females in the test fold on hand and then summing these values. Likewise, age-strified analysis was done based on clinically meaningful age categories such as $ \leq 4 5 $ years, 46-60 years, and above 60 years. While

reporting performance measures on the test cases, the measures that are most appropriate in the context of screening are ROC-AUC and Recall.

On both the gender and age stratification dimensions, the performance of the model held steady, and no category had a discernible tendency towards lower discrimination or recall performance. Some degree of fluctuation was seen, and this was expected, since the number of occurrences within the strata will inevitably have been smaller, thereby impacting the discriminability of performance within those categories. Nonetheless, there are no categories which tend towards biased insensitivity performance, which could indicate model preference towards one gender/age strata category.

## 6 Conclusion

This study shows that highly accurate and dependable predictions for the detection of heart disease can be obtained using a well-designed machine learning pipeline that combines automated hyperparameter optimization, robust handling of class imbalance, and sophisticated feature selection. The suggested method achieved exceptional performance metrics, such as an accuracy of 99.02% precision of 98.13% recall of 100% F1-score of 99.06% and ROC-AUC of 0.9998 on the test dataset1, by utilizing XGBoost's modeldriven feature importance, incorporating SMOTE to address class imbalance, and using Optuna's Bayesian optimization for hyperparameter tuning. These findings highlight how ensemble learning techniques, methodical feature selection, and contemporary hyperparameter optimization frameworks can improve cardiovascular disease prediction modeling. The results align with current research that emphasizes the importance of incorporating these methods to enhance diagnostic precision and the generalizability of the model in clinical settings. All things considered, the suggested pipeline provides a reliable, comprehensible, and expandable approach to early heart disease detection. Future research might concentrate on integrating more clinical variables, verifying this method with bigger and more varied datasets, and investigating its applicability in actual clinical decision-support systems.

In order to thoroughly evaluate the proposed model, its performance and reliability were rigorously evaluated by applying various evaluation techniques such as the confusion matrix, calibration analysis, learning curves, ROC and Precision-Recall curves, and the F1-score distribution across cross-validation folds. The combination of these metrics gives an overall view of the model's predictive behavior, generalization as shown in Fig. 9.

## 7 Future work

Future research should concentrate on improving personalization across patient subgroups, investigating hybrid model architectures, and integrating temporal patient data. Additional priorities include improving explain ability strategies to boost clinician trust, strengthening robustness across diverse populations through domain-adaptation techniques, and validating the models in actual clinical settings. When combined, these approaches would improve the model's usefulness and preparedness for incorporation into cardiac care.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110029356b2ac2b377b14d2d%2Fcrop_1_1783701013244.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=5cCiQiskA5cgqoWh7UvfdpPC2SM%3D&Expires=1784305813' alt='OCR图片'/></div>

<div align="center">

Fig. 9 Confusion matrix, calibration curve, learning curve, ROC curve, PR curve, F1 distribution

</div>

## Acknowledgements

The authors would like to express their heartfelt gratitude to the Data Analytics Lab, Department of Computer Applications, BMS College of Engineering, Bangalore, for providing the necessary infrastructure and support to carry out this research.

## Author contributions

Raghavendra Rao Rv: Supervised the research, provided guidance, and helped the final version of the manuscript. Ch. Ram Mohan Reddy: Provided expert advice and guidance in research and helped to finalise the manuscript. Hemanth K: Conceptualization, performed data pre-processing, implemented the machine learning pipeline, conducted experiments, and drafted the initial manuscript. Hruthik Chavan D: Contributed to feature engineering, hyperparameter optimization, results analysis, and manuscript editing. Assisted in literature review and interpretation of findings.

## Funding

No funding was received for this study.

## Data availability

Data used in this work is available at the following url: Heart Disease Dataset: public health dataset—https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset?select=heart.csv

## Declarations

## Ethics approval and consent to participate

This research was carried out using a publicly available dataset that has been anonymized before it was released to the public. The dataset does not hold any form of protected health information or personal identifying information. As stipulated by the guidelines by Springer Nature on secondary analysis, the research did not require approval by an institutional review board or ethics committee. Not applicable. As this research involves the secondary analysis of anonymized public data and there are no direct interactions with human participants.

## Consent for publication

Not applicable. There are no personal data of any individual in the manuscript in any form.

## Competing interests

Received: 12 September 2025 / Accepted: 3 February 2026

Published online: 04 March 2026

## References

1. Abdollahi J, Nouri-Moghaddam B. Feature selection for medical diagnosis: evaluation for using a hybrid Stacked-Genetic approach in the diagnosis of heart disease. 2021. https://doi.org/10.48550/arXiv.2103.08175.

2. Aggrawal R, Pal S. Sequential feature selection and machine learning algorithm-based patient's death events prediction and diagnosis in heart disease. SN Comput Sci. 2020. https://doi.org/10.1007/s42979-020-00370-1.

3. Ahmad E, Tiwari A, Kumar A. Cardiovascular diseases (CVDs) detection using machine learning algorithms. Int J Res Appl Sci Eng Technol. 2020. https://doi.org/10.22214/ijraset.2020.6376.

4. Ahmad G, Fatima H, Shafiullah S, Abdelaziz SS, Imdadullah. Efficient medical diagnosis of human heart diseases using machine learning techniques with and without GridSearchCV. IEEE Access. 2022;10:80151-73. https://doi.org/10.1109/AC CESS.2022.3165792.

5. Al-Mahdi IS, Darwish SM, Madbouly MM. Heart disease prediction model using feature selection and ensemble deep learning with optimized weight. Comput Modeling Eng Sci. 2025;143(1):875-909. https://doi.org/10.32604/cmes.2025.061 623.

6. Prediction of Heart Disease Using a Hybrid XGBoost-GA Algorithm with Principal Component Analysis: A Real Case Study

7. Alami H, Lehoux P, Denis J-L, Motulsky A, Petitgand C, Savoldelli M, et al. Organizational readiness for artificial intelligence in health care: insights for decision-making and practice. J Health Organ Manag. 2021;35(1):106-14. https://doi.org/10.1108/JHOM-03-2020-0074.

8. Alhaj TA, Siraj MM, Zainal A, Elshoush HT, Elhaj F. Feature selection using information gain for improved structural-based alert correlation. PLoS ONE. 2016;11(11):e0166017. https://doi.org/10.1371/journal.pone.0166017.

9. Goldstein BA, Navar AM, Carter RE. Moving beyond regression techniques in cardiovascular risk prediction: applying machine learning to address analytic challenges. Eur Heart J. 2017;38(23):1805-14. https://doi.org/10.1093/eurheartj/ehw 302.

10. Bhatt CM, Patel P, Ghetia T, Mazzeo PL. Effective heart disease prediction using machine learning techniques. Algorithms. 2023;16(2):88. https://doi.org/10.3390/a16020088.

11. Bouchlaghem Y, Akhiat Y, Amjad S. Feature selection: a review and comparative study. E3S Web Conf. 2022;351:01046. https://doi.org/10.1051/e3sconf/202235101046.

12. Buda M, Maki A, Mazurowski MA. A systematic study of the class imbalance problem in convolutional neural networks. Neural Netw. 2018;106:249-59. https://doi.org/10.1016/j.neunet.2018.07.011.

13. Dissanayake K, Md Johar MG. Comparative study on heart disease prediction using feature selection techniques on classification algorithms. Appl Comput Intell Soft Comput. 2020;2021(1):5581806. https://doi.org/10.1155/2021/5581806.

14. Gao X-Y, Ali A, Shaban H, Anwar E. Improving the accuracy for analyzing heart diseases prediction based on the ensemble method. Complexity. 2021;2021:1-10. https://doi.org/10.1155/2021/6663455.

15. Gárate Escamilla AK, El Hassani AH, Andrès E. Classification models for heart disease prediction using feature selection and PCA. Informatics Med Unlocked. 2020;19:100330. https://doi.org/10.1016/j.imu.2020.100330.

16. Ghazali SM, Seman Z, Cheong KC, et al. Sociodemographic factors associated with multiple cardiovascular risk factors among Malaysian adults. BMC Public Health. 2015;15:68. https://doi.org/10.1186/s12889-015-1432-z.

17. Yang H, Chen Z, Yang H, Tian M. Predicting coronary heart disease using an improved LightGBM model: performance analysis and comparison. IEEE Access. 2023;11:23366-80. https://doi.org/10.1109/ACCESS.2023.3253885.

18. Hashi EK, Zaman MS. Developing a hyperparameter tuning based machine learning approach of heart disease prediction. J Appl Sci Process Eng. 2020;7:631-47. https://doi.org/10.33736/jaspe.2639.2020.

19. Heart Disease Dataset: public health dataset. https://www.kaggle.com/datasets/johnsmith88/heart-%20disease-dataset.

20. Narayanan, Jayashree. Implementation of efficient machine learning techniques for prediction of cardiac disease using SMOTE. Procedia Comput Sci. 2024;233:558-69. https://doi.org/10.1016/j.procs.2024.03.245.

21. Soni J, Ansari U, Sharma D, Soni S. Predictive data mining for medical diagnosis: an overview of heart disease prediction. Int J Comput Appl. 2011;17(8):43-8. https://doi.org/10.5120/2237-2860.

22. Latha CBC, Jeeva SC. Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques. Inf Med Unlocked. 2018;16:100203. https://doi.org/10.1016/j.imu.2019.100203.

23. Lindstrom M, DeCleene N, Dorsey H, et al. Global burden of cardiovascular diseases and risks collaboration, 1990-2021. J Am Coll Cardiol. 2022;80(25):2372-425. https://doi.org/10.1016/j.jacc.2022.11.001.

24. Mao Y, Jimma BL, Mihretie TB. Machine learning algorithms for heart disease diagnosis: a systematic review. Curr Probl Cardiol. 2025;50(8):103082. https://doi.org/10.1016/j.cpcardiol.2025.103082.

25. Naruka V, Rad AA, Ponniah HS, Francis J, Vardanyan R, Tasoudis P, et al. Machine learning and artificial intelligence in cardiac transplantation: a systematic review. Artif Organs. 2022;46(9):1741-53. https://doi.org/10.1111/aor.14334.

26. Sharma P, Agarwal S. Cardiovascular disease analysis using different machine learning techniques. In: 2024 2nd International Conference on Artificial Intelligence and Machine Learning Applications Theme: Healthcare and Internet of Things (AIMLA), Namakkal, India; 2024, p. 1-6, https://doi.org/10.1109/AIMLA59606.2024.10531431.

27. Pathan MS, Nag A, Pathan MM, Dev S. Analyzing the impact of feature selection on the accuracy of heart disease prediction. Healthcare Analytics. 2022;2:100060. https://doi.org/10.1016/j.health.2022.100060.

28. Rababa S, Yamin A, Lu S, Obaidat A. Predicting heart disease and reducing survey time using machine learning algorithms. 2023. https://doi.org/10.48550/arXiv.2306.00023.

29. Roman WP, Martin HD, Sauli E. Cardiovascular diseases in Tanzania: the burden of modifiable and intermediate risk factors. J Xiangya Med. 2019;4:33.

30. Mohan S, Thirumalai C, Srivastava G. Effective heart disease prediction using hybrid machine learning techniques. IEEE Access. 2019;7:81542-54. https://doi.org/10.1109/ACCESS.2019.2923707.

31. Sianga BE, Mbago MC, Msengwa AS. Predicting the prevalence of cardiovascular diseases using machine learning algorithms. Intell Based Med. 2024;11:100199. https://doi.org/10.1016/j.ibmed.2025.100199.

32. Singh A, Kumar R. Heart disease prediction using machine learning algorithms. 2020. p. 452-457. https://doi.org/10.1109/I CE348803.2020.9122958.

33. Sowjanya AM, Mrudula O. Effective treatment of imbalanced datasets in health care using modified SMOTE coupled with stacked deep learning algorithms. Appl Nanosci. 2023;13:1829-40. https://doi.org/10.1007/s13204-021-02063-4.

34. Spencer R, Thabtah F, Abdelhamid N, Thompson M. Exploring feature selection and classification methods for predicting heart disease. Digit Health. 2020;6:2055207620914777. https://doi.org/10.1177/2055207620914777.

35. Srinivas P, Katarya R. HyOPTXg: optuna hyper-parameter optimization framework for predicting cardiovascular disease using XGBoost. Biomed Signal Process Control. 2022;73:103456. https://doi.org/10.1016/j.bspc.2021.103456.

36. Takci H. Improvement of heart attack prediction by the feature selection methods. Turk J Electr Eng Comput Sci. 2018;26(1):1-10. https://doi.org/10.3906/elk-1611-235.

37. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining (KDD '19). New York, NY: Association for Computing Machinery; 2019. p. 2623-2631. https://doi.org/10.1145/3292500.3330701

38. Chen T, Guestrin C. 2016. XGBoost: a scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16). New York, NY: Association for Computing Machinery; 2016. p. 785-794. https://doi.org/10.1145/2939672.2939785.

39. WHO/ Cardiovascular diseases. https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1.

40. World Heart Report 2023. https://world-heart-federation.org/wp-content/uploads/World-Heart-Report-2023.pdf.

41. XGBoost Documentation. Parameters for Tree Booster. https://xgboost.readthedocs.io/en/latest/parameter.html. Accessed 30 June 2025.

42. Yang JC. The prediction and analysis of heart disease using XGBoost algorithm. Appl Comput Eng. 2024;41:61-8.

43. Raghavendra Rao RV. Srinivasulu Reddy U. Sparse attention regression network-based soil fertility prediction with

43. Raghavendra Rao RV, Srinivasulu Reddy U. Sparse attention regression network-based soil fertility prediction with UMMASO. Chemometrics Intelligent Laboratory Syst. 2025;257:105289. https://doi.org/10.1016/j.chemolab.2024.105289.

44. Karadayi Ataş P. A novel clustered-based binary grey wolf optimizer to solve the feature selection problem for uncovering the genetic links between non-Hodgkin lymphomas and rheumatologic diseases. Health Inf Sci Syst. 2025;13:34. https://doi.org/10.1007/s13755-025-00350-w.

45. Zou M, Jiang W-G, Qin Q-H, Liu Y-C, Li M-L. Optimized XGBoost model with small dataset for predicting relative density of Ti-6Al-4V parts manufactured by selective laser melting. Materials. 2022;15:5298. https://doi.org/10.3390/ma15155298.

46. Khafaga DS, Alhussan AA, El-kenawy EM, Takieldeen AE, Hassan TM, et al. Meta-heuristics for feature selection and classification in diagnostic breast cancer. Comput Mater Continua. 2022;73(1):749-65. https://doi.org/10.32604/cmc.2022.029 605.

## Publisher's Note

Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.