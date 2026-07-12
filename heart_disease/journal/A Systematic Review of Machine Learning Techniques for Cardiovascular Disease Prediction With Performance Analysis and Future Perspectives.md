# A Systematic Review of Machine Learning Techniques for Cardiovascular Disease Prediction With Performance Analysis and Future Perspectives

**Jurnal:** Cureus Journal of Computer Science (Scopus Q1, 2026)
**URL:** https://www.cureusjournals.com/articles/14238-a-systematic-review-of-machine-learning-techniques-for-cardiovascular-disease-prediction-with-performance-analysis-and-future-perspectives#!/

---

## Abstract

Cardiovascular disease is a chronic and significant health issue in the world, highlighting the need to detect and diagnose it in its early stages to increase the survival chances of the patient. The use of machine learning, as well as deep learning, has grown rapidly in medical research and shows great potential for improving the accuracy of heart disease prediction. They are able to identify latent connections between various clinical, demographic, and lifestyle variables and can be more accurate than traditional statistical techniques.

This review follows a systematic literature review approach based on **PRISMA guidelines**, ensuring a structured and transparent selection of relevant studies. This review discusses some of the typical supervised learning algorithms used to assess cardiovascular risk, including **Random Forests, Logistic Regression, Decision Trees, Neural Networks, Gradient Boosting, and Support Vector Machine**. The heart disease dataset, the Cleveland and Framingham Heart Study datasets, and the Kaggle datasets are the most popular datasets used in the studies.

Model performance is evaluated using metrics such as **precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC)**; however, in medical diagnosis, recall and F1-score are particularly important to minimize false negatives. Although these approaches show promising results, challenges such as **imbalanced datasets, limited interpretability, and poor generalization** across diverse populations remain. Furthermore, variations in datasets and experimental setups limit direct comparison across studies.

This review highlights current research trends, compares existing methods, and identifies key limitations. It also provides recommendations for future research to develop more reliable, interpretable, and clinically applicable heart disease prediction systems.

**Keywords:** Machine Learning, Cardiovascular Disease, Systematic Review, PRISMA, Heart Disease Prediction, Deep Learning, Interpretability

---

## 1. Introduction

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide, accounting for approximately 17.9–18 million deaths annually according to the World Health Organization (WHO). Early and accurate detection is critical to improve survival rates and enable timely clinical intervention.

Traditional statistical methods such as logistic regression scoring (e.g., Framingham Risk Score) have long been used in clinical practice. However, they often fail to capture **complex non-linear relationships** among multiple clinical, demographic, and lifestyle risk factors. Machine learning (ML) offers a powerful alternative, enabling the automated discovery of patterns in large, heterogeneous health datasets.

This systematic review synthesizes existing literature on ML-based cardiovascular disease prediction, providing:
1. A structured comparison of algorithms and their reported performance metrics.
2. An analysis of datasets commonly used in CVD prediction research.
3. Identification of current limitations and methodological gaps.
4. Recommendations for future clinically applicable research.

---

## 2. Review Methodology (PRISMA Guidelines)

This review adheres to the **Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA)** framework to ensure methodological rigor and transparency.

### 2.1 Search Strategy
- **Databases searched:** PubMed, Scopus, IEEE Xplore, Google Scholar, Web of Science
- **Search keywords:** "cardiovascular disease prediction," "heart disease machine learning," "cardiac risk classification," "deep learning heart disease"
- **Publication date range:** 2015–2025 (with focus on 2020–2025 for recent advances)

### 2.2 Inclusion & Exclusion Criteria

| Criteria | Inclusion | Exclusion |
| :--- | :--- | :--- |
| **Study type** | Original research, reviews, conference papers with ML experiments | Opinion pieces, editorials |
| **Focus** | CVD/Heart disease prediction/classification | Other diseases |
| **Method** | Supervised ML / Deep Learning approaches | Unsupervised or rule-based only |
| **Metrics** | Reports at least Accuracy or AUC | No quantitative evaluation |
| **Language** | English | Non-English |

---

## 3. Machine Learning Algorithms Reviewed

### 3.1 Classical / Traditional Supervised Learning

| Algorithm | Key Strengths | Key Weaknesses |
| :--- | :--- | :--- |
| **Logistic Regression (LR)** | Interpretable, fast, good baseline | Struggles with non-linear relationships |
| **Decision Tree (DT)** | Highly interpretable, visual | Prone to overfitting |
| **Random Forest (RF)** | Handles noise well, reduces overfitting | Less interpretable, computationally heavy |
| **k-Nearest Neighbors (kNN)** | Simple, no training required | Slow on large datasets, sensitive to scale |
| **Support Vector Machine (SVM)** | Effective in high-dimensional spaces | Computationally expensive, black-box |
| **Naïve Bayes (NB)** | Fast, handles small datasets | Assumes feature independence (unrealistic) |

### 3.2 Ensemble & Boosting Methods

| Algorithm | Best Reported Accuracy | Notes |
| :--- | :---: | :--- |
| **Random Forest** | ~88–94% | Widely used baseline |
| **Gradient Boosting (GBM)** | ~83–91% | Good generalization |
| **XGBoost** | ~91–99% | Dominant in recent studies |
| **LightGBM** | ~90–98% | Efficient for large datasets |
| **CatBoost** | ~87–93% | Handles categorical features natively |
| **AdaBoost** | ~86–93% | Sensitive to noise |

### 3.3 Deep Learning Approaches

| Model | Strengths | Reported Performance |
| :--- | :--- | :--- |
| **Feedforward NN (MLP)** | Captures non-linearity | 80–92% accuracy |
| **Convolutional Neural Network (CNN / 1D CNN)** | Auto feature extraction | Up to 98% accuracy |
| **Recurrent NN / LSTM** | Time-series pattern recognition | 85–96% accuracy |
| **CNN-BiLSTM Hybrid** | Combines spatial & temporal features | Up to 97% accuracy |
| **Transformer** | Context-aware attention mechanism | 91–99% accuracy |

---

## 4. Datasets Commonly Used

| Dataset | Source | # Instances | # Features | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Cleveland Heart Disease** | UCI ML Repository | 303 | 14 | Most widely used benchmark |
| **Hungarian** | UCI ML Repository | 294 | 14 | Often combined with Cleveland |
| **Switzerland** | UCI ML Repository | 123 | 14 | High positive diagnosis rate |
| **VA Long Beach** | UCI ML Repository | 200 | 14 | Often excluded due to high missing values |
| **Combined UCI (Kaggle)** | Kaggle / fedesoriano | 918 | 12 | Harmonized merged dataset |
| **Kaggle Heart Dataset** | Kaggle / johnsmith88 | 1,025 | 14 | Pre-merged UCI cohorts (no missing values) |
| **Framingham Heart Study** | NHLBI | 4,238 | 15 | Longitudinal study, USA |

---

## 5. Performance Benchmarking Summary

The following table synthesizes best-reported results from reviewed studies, organized by algorithm type:

| Study Category | Best Algorithm | Accuracy (%) | Recall (%) | F1-Score (%) | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Classical ML (single) | LR, SVM | 80–92 | 78–90 | 79–91 | 0.85–0.94 |
| Ensemble / Boosting | XGBoost, RF | 88–99 | 90–100 | 89–99 | 0.92–0.999 |
| Deep Learning (CNN) | 1D CNN | 92–98 | 90–96 | 91–98 | 0.96–0.99 |
| Hybrid (CNN + LSTM) | CNN-BiLSTM | 93–97 | 92–97 | 92–97 | 0.97–0.99 |
| Transformer-based | Attention CNN | 91–99 | 90–99 | 91–99 | 0.97–0.999 |

> **Key Insight:** XGBoost with hyperparameter optimization (e.g., Optuna) consistently achieves the highest accuracy (up to 99%), while deep learning models offer competitive performance with better feature extraction capabilities. However, ensemble methods remain the dominant choice in practical applications due to interpretability vs. accuracy tradeoffs.

---

## 6. Evaluation Metrics Analysis

In medical diagnostic tasks, the **choice of evaluation metric** is critical. The review emphasizes:

### 6.1 Metric Definitions

| Metric | Formula | Clinical Significance |
| :--- | :--- | :--- |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Minimizes false alarms (false positives) |
| **Recall (Sensitivity)** | TP/(TP+FN) | **Most critical** – minimizes missed diagnoses |
| **F1-Score** | 2×(P×R)/(P+R) | Balance between precision and recall |
| **AUC-ROC** | Area under ROC curve | Discriminative power across thresholds |
| **MCC** | Balanced metric for imbalanced datasets | More robust than accuracy |

### 6.2 Clinical Priority: Recall Over Precision
In cardiovascular diagnosis, **false negatives (FN)** — missing a patient who actually has heart disease — are clinically more dangerous than false positives. Therefore, **high Recall (Sensitivity) is prioritized** over precision, even at the cost of slightly lower overall accuracy.

---

## 7. Key Challenges Identified

Based on the systematic analysis, the review identifies five major challenges:

1. **Imbalanced Datasets**
   - Many real-world CVD datasets have a skewed class distribution (more negative than positive cases).
   - Techniques such as **SMOTE, oversampling, class-weighted loss functions** are essential.

2. **Limited Interpretability (Black-Box Models)**
   - Deep learning and ensemble models are difficult to explain clinically.
   - Solutions: **SHAP (SHapley Additive Explanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** are increasingly applied.

3. **Poor Generalization Across Populations**
   - Most models trained on single-center datasets (e.g., Cleveland) fail when applied to different demographics.
   - Need for **multi-center validation** and **federated learning** approaches.

4. **Dataset Heterogeneity**
   - Inconsistent feature sets, different encoding standards, and missing values across UCI cohorts.
   - Makes direct comparison between studies difficult.

5. **Lack of Prospective Clinical Validation**
   - Most studies are retrospective; real-world prospective trials are rare.
   - Models have not been tested in clinical workflow environments.

---

## 8. Future Research Directions

The review provides the following recommendations for future work:

| Direction | Description |
| :--- | :--- |
| **Explainable AI (XAI)** | Integrate SHAP/LIME into clinical workflows for decision support |
| **Federated Learning** | Enable cross-hospital model training while preserving patient privacy |
| **Multimodal Data Fusion** | Combine EHR, imaging (ECG), genomic, and wearable data |
| **Transformer Architectures** | Explore attention-based models for better contextual feature learning |
| **Real-Time Monitoring** | Integrate ML with IoT wearables (smartwatches, ECG monitors) |
| **Standardized Benchmarking** | Establish common datasets and evaluation protocols for fair comparison |
| **Prospective Clinical Trials** | Validate models in real-world healthcare settings before deployment |

---

## 9. Conclusion

This systematic review demonstrates that machine learning, particularly **ensemble methods (XGBoost, Random Forest)** and **deep learning (1D CNN, CNN-BiLSTM)**, have achieved remarkable performance in cardiovascular disease prediction, often surpassing traditional statistical models.

Key findings:
- **XGBoost** consistently delivers the best accuracy (88–99%) across diverse datasets.
- **1D CNN and deep learning** architectures provide competitive performance with automated feature extraction.
- **Interpretability (SHAP, LIME)** is increasingly adopted to align AI predictions with clinical practice.
- **PRISMA-guided systematic reviews** are essential for structured comparison of heterogeneous studies.

Despite these advances, major challenges remain — particularly in **model generalizability, clinical interpretability, and prospective validation**. Future research must focus on building robust, transparent, and equitable AI systems that can be reliably deployed in real-world healthcare environments.

---

## Keterkaitan dengan Proyek Sertifikasi

Paper ini relevan sebagai **referensi landasan** untuk menilai performa model yang dibangun dalam proyek ini. Secara spesifik:

| Aspek Proyek | Referensi dari Paper Ini |
| :--- | :--- |
| **Pemilihan Algoritma** | XGBoost dan Random Forest terbukti dominan dalam literatur → mendukung pilihan algoritma di notebook |
| **Dataset UCI** | Cleveland, Hungarian, Switzerland, VA Long Beach adalah dataset yang paling banyak digunakan dan terbukti valid secara ilmiah |
| **Metrik Evaluasi** | Recall & F1-Score diprioritaskan dalam diagnosis medis (mengurangi false negative) |
| **Penanganan Imbalanced Data** | SMOTE direkomendasikan sebagai teknik standar → diterapkan pada paper utama (Unified approach) |
| **Interpretabilitas** | SHAP & LIME menjadi standar XAI dalam CVD → landasan untuk penambahan explainability di notebook |
| **Gap Penelitian** | Generalisasi lintas populasi → menjadi justifikasi penggunaan 4 kohort UCI sekaligus |

---

*Catatan: File ini merupakan ringkasan analitik dari paper yang dapat diakses di: https://www.cureusjournals.com/articles/14238-a-systematic-review-of-machine-learning-techniques-for-cardiovascular-disease-prediction-with-performance-analysis-and-future-perspectives#!/*
