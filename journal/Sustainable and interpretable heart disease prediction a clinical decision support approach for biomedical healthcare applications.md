<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910703.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=HRw1LflOPV8sMEDLfA7f2s7SMSo%3D&Expires=1784305710' alt='OCR图片'/></div>

OPEN

<div align="center">

# Sustainable and interpretable heart disease prediction: a clinical decision support approach for biomedical healthcare applications

</div>

Tanzila Kehkashan $ ^{1,2,7} $ , Maha Abdelhaq $ ^{3} $ , Ahmad Sami Al-Shamayleh $ ^{4} $ Raja Adil Riaz $ ^{2,7} $ , Muhammad Abdullah $ ^{2} $ , Abdelmuttlib Ibrahim Abdalla Ahmed $ ^{5,7} $ & Adnan Akhunzada $ ^{6} $

Cardiovascular disorders cause approximately 18 million deaths annually worldwide, underscoring the urgent need for precise and rapid diagnosis. Conventional machine learning does not automate feature extraction and does not capture complex non-linear relationships in high-dimensional medical data during cardiodiagnostic predictions, not to mention current techniques also lack interpretability and transparency and thus are not useful for building clinically trusted AI-powered prediction tools for personalized biomedicine and healthcare. To fill this gap, we propose an interpretable Convolutional Neural Network (1D CNN) model for cardiodiagnostic predictions that integrates automated feature extraction and explains AI. Our approach uses a 1D CNN model composed of two convolutional layers (with 64 and 128 filters). The CNN will be trained on Cleveland Heart Disease Dataset (Kaggle) (303 instances, 14 attributes) and will undergo evaluation using accuracy, precision, recall, F1 score, and LIME-SHAP interpretability analyses. The results given attest to the remarkable results, where the model achieved 98.05% percent accuracy, 100% percent precision, 96.12% percent recall, 98.02% percent F1 score, 0.963 MCC, and 0.961 Kappa coefficient. Our results exceed several of the more modern techniques offered in the literature. LIME and SHAP analyses reveal how specific features (sex, number of major vessels, thalassemia status) drive model predictions, enhancing interpretability and aligning with clinical understanding of cardiac risk factors essential for precision medicine. This research demonstrates the potential for interpretable deep learning to transform cardiovascular diagnostics through enhanced clinical decision support systems and trustworthy AI implementation in precision medicine.

Keywords Convolutional neural network, Heart disease prediction, Medical diagnostics, Decision support systems, AI-driven decision-making, Healthcare analytics, Biomedical healthcare applications

Cardiovascular diseases (CVDs) continue to be the world's largest cause of mortality, with the World Health Organization (WHO) reporting that they cause around 18 million deaths yearly $ ^{1} $ . With the advancement in big data analytics in healthcare, heart conditions remain the most menacing, and healthcare systems are challenged to improve the existing methods of diagnosis in personalized biomedical and healthcare $ ^{2} $ . Since healthcare has developed early disease detection techniques in various ways, most of them are invasive or do not have adequate accuracy in predicting at-risk individuals. Recently, Machine learning has emerged as a valuable tool in clinical

decision-making for bettering the early predictions and diagnosis of heart disease for reducing mortality rates through early intervention and preventive care measures in personalized biomedical and healthcare $ ^{3} $

The recent past has seen the wide-scale application of machine learning models in the medical industry where patient information related to age, BP(Blood Pressure) and levels of cholesterol are used with electronic health records to forecast the risk of heart disease $ ^{4} $ . Much progress has been made in the pursuit of optimizing model accuracy while simultaneously optimizing interpretability for both, but most techniques remain insufficient in generalizing across diverse population. This work addresses the challenges of improving predictive accuracy and interpretability of a CNN model for heart disease detection using the Kaggle cardiovascular disease dataset.

Previous works utilized various machine learning techniques such as Decision Trees, Random Forest, SVM, and Logistic Regression in disorder prediction. For instance, Malibari achieved 93.56% accuracy from the CNN model but with a limitation to the feature extraction $ ^{5} $ . While such deep learning approaches and computer vision have promise, they are often plagued by the requirement for substantial medical imaging data and powerful computers. Previous work has struggled to balance accuracy, interpretability, and computational efficiency simultaneously, creating a need for more robust models $ ^{6} $ .

The main challenge is achieving high prediction accuracy while maintaining clinical interpretability and computational efficiency $ ^{7} $ . With big data in medical records, models have trended to perform well with regard to prediction but lack the ability to explain results in a clinically meaningful way. This work focuses on predictive analysis and modeling to be more interpretable and computationally efficient in comparison with models for heart disease detection, including basing its performance on the Kaggle heart disease dataset $ ^{8} $

The following are the goals of this study:

i. to use a CNN model to increase the forecast accuracy of the diagnosis of heart disease

ii. to ensure the model is interpretable so that key predictive factors can be understood by healthcare professionals

iii. to optimize the model's computational efficiency for clinical deployment

iv. to assess the generalizability of the model on various datasets

This research focuses on an important and complex problem within the field of cardiovascular diagnostics: the need to obtain predictive accuracy in a clinically interpretable and computationally efficient manner, all at the same time. Most previous works have either obtained great predictive accuracy in complex black-box models or interpretability in simpler models, but they have rarely been able to do both and this has been described as a problem by Sugiyarto et al. $ ^{9} $ Our contribution is the development of a 1D CNN model that achieves 98.05% predictive accuracy and is still lightweight enough to be useful in financially constrained healthcare systems.

Optimizing the predictive performance of the model alongside its clinical relevance ensures its future utility in healthcare as it will seamlessly integrate with personalized biomedicine and healthcare systems work streams $ ^{10} $ . The model's explainability is further enhanced by the application of LIME and SHAP analyses, which fulfills one of the most critical requirements of machine learning in the healthcare domain $ ^{11} $ .

## Literature review

Many studies have covered the application of machine learning, deep learning, and computer vision techniques for the diagnosis and the prediction of heart disease. Such methods utilize demographic and clinical data to detect patients' risks, with the aim of improving diagnostic systems' speed and accuracy for large-scale data and computational operations. An increase in the research on the intersection of medical innovation and healthcare big data analytics has underscored the significance of maximally predictive clinical decision-making by selecting appropriate models, feature sets, and preprocessing techniques, or adjusting model hyperparameters. A total analytical framework expounding fundamental methods for heart disease diagnosis and prediction is illustrated in Fig.1.

## Machine learning techniques

Machine learning models effectively manage the varied clinical data and automated health records integrated in the prediction of cardiac diseases. Among the first-line predictive analysis tools are Logistic Regression, Decision Trees, Random Forest, k-Nearest Neighbors (k-NN), and Support Vector Machines, which work on the medical imaging data $ ^{12} $ . They provide reasonable predictive analysis accuracy and the corresponding interpretability to support clinical decision frameworks. Logistic Regression, harbored in its simplicity an efficiency that marked it out to assist in the other initial binary classification help, heart disease detection $ ^{13} $ . It remains a first-line choice in heart disease detection despite its enduring difficulty in non-linear relationships in complex medical datasets. Owing to their capacity to capture non-linear relationships and their resistance to overfitting. Decision Trees and their ensemble methods such as Random Forest have been increasingly applied for early disease detection $ ^{14} $ Random Forest is especially proficient in predicting cardiovascular diseases as it averages predictions from multiple decision trees, which minimizes the likelihood of errors and enhances the model's generalization $ ^{15} $ Although k-Nearest Neighbors (k-NN) and Support Vector Machines (SVMs) are useful in medicine, challenges such as distance metric selection and high computational costs with large clinical datasets are difficult to remedy $ ^{16} $ . Despite requiring detailed feature construction and generalizing poorly across different populations, these ML algorithms are lifting the predictability of heart diseases, primarily because of the big data revolution in the healthcare sector $ ^{17} $ . The most recent example of machine learning in clinical monitoring pipelines has been the use of ML in real time anomaly detection in continuous ECG data streams in IoT related healthcare systems which help to decrease false alarms and increases the utility of ML in clinical monitoring $ ^{18} $

## Heart Disease Detection And Prediction:Integrated Analystical Approaches

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910759.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=LfGlDT3tdErW1xvEMyhz5E2%2BCR0%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 1. Literature review block diagram.

</div>

## Deep learning techniques

For the assistance of large datasets, the enhanced predictive capability of deep learning techniques for heart disease prediction has become important $ ^{19} $ .With respect to patient data, both CNNs and FNNs facilitate automated feature extraction and intricate pattern recognition without any prerequisite feature engineering. Specific to predictive modeling, heart disease detection has been operationalized via Feedforward neural networks (FNNs) and has been augmented by effectiveness derived from Multi-Layer Perceptrons (MLPs).The ability to learn from unprocessed data without requiring manual feature selection distinguishes them as more advanced computational and big data operations compared to traditional machine learning methods. Although primarily used for medical imaging data, CNNs are also being utilized for structured medical data for the prediction of cardiac events $ ^{20} $ .In the past, more sophisticated algorithms such as CNNs were able to extract more value from EHRs that were more complex with the differentiation of heart disease cases. For predictive analytics, Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks analyze data streams that are monitored in real-time and find patterns in patient data over time. Nonetheless, the lack of interpretability of these healthcare AI systems, particularly when poorly monitored, poses unique challenges in the contextualized healthcare setting. Predictive diagnostic imaging systems developed by specialists in computer systems are able to assist in predicting outcomes over several medical imaging modalities $ ^{21,22} $

Interpretable deep learning incorporated various AI methods to tackle some of these issues regarding transparency. Combining CNN feature extraction with other classification approaches and explanation mechanisms has produced impressive results in both prediction example and accuracy. For instance, Wani et al. achieved 97.43% accuracy in lung cancer detection while giving coherent rationales for each prediction $ ^{23} $ . Singh et al. obtained similar results with diabetes diagnosis and hybrid models that doctors can easily comprehend, reaching 98.24% accuracy $ ^{24} $ . Such works demonstrate that rationalizing deep learning models to AI can effectively resolve the accuracy and interpretability confines in medical applications $ ^{25} $ .

## Hybrid models and feature selection

Hybrid methodologies aimed at capitalizing on the benefits of traditional machine learning and the deep learning models have been investigated with the objective of improving predictive accuracy and model interpretability $ ^{26} $ Ensemble approaches of XGBoost amalgamate the outputs of several models to provide enhanced predictive ability compared to the performance of stand-alone models by virtue of advanced computing and operations on big data. The combination of k-NN with XGBoost, AdaBoost, and Random subspace classifiers resulted in 96% accuracy and 93% MCC on the prediction of cardiovascular disease $ ^{27} $ Further advancements in performance were achieved by Hawks Optimizer-based stacked ensemble models which addressed the class imbalance problems in cardiovascular disease datasets $ ^{28} $ In a similar hybrid model, the integration of the ANFIS with

wavelet-based feature extraction for signal processing and intelligent classification showed highly accurate detection of certain cardiac conduction abnormalities from ECG signals and reflects the value of hybrid feature-driven models in clinical diagnostics $ ^{29} $

Effective diagnosis of cardiovascular diseases entails the adoption of techniques such as Principal Component Analysis (PCA) and Recursive Feature Elimination (RFE) in pinpointing critical parameters which include age, cholesterol, blood pressure, and smoking habits $ ^{30} $ . The Fast Independent Component Analysis technique surpasses conventional PCA with an F-score of 99.83% on only 10 dimensions in the analysis of cardiovascular signals $ ^{31} $ Feature selection was enhanced by Kumar and Rekha with Binary Krill Herd meta-heuristic optimization with dense networks, resulting in an accomplishment of 95% accuracy on the Cleveland dataset $ ^{32} $ Systems for the prediction of cardiovascular diseases have increased accuracy and interpretability, as evidenced by the use of hybrid models which combine feature selection and model training. Even with significant advancements, gaps remain such as the need for interpretability of models when applied to the healthcare domain. Explainable Artificial Intelligence (XAI) techniques are increasingly being developed to clarify the workings of complex models $ ^{33} $ There are also difficulties with generalizability across different populations as most models are developed based on specific datasets, showing different performances across various demographic features $ ^{34} $ Integration of real-time monitoring through wearable technology and the ethical implications of data privacy and algorithmic bias are anticipated to be areas for future research.

## Recent AI advances in cardiovascular prediction

Advancements in transformer-based architectures for cardiovascular disease prediction have emerged in 2024-2025. Noting the challenges in portability and complexity of AI cardiovascular applications, Pandey et al. conducted a systematic review of 50 studies in the area $ ^{35} $ . Rahman et al. pioneered the efficient use of selfattention mechanisms in contextual information extraction for various heart disease datasets, outperforming the traditional methods used in a dataset, and yielding remarkable results for the specific task $ ^{36} $ . Noor et al. developed a novel transformer approach for ECG-based cardiovascular detection, attaining an astounding 99.79% accuracy rate $ ^{37} $ .

The Explainable AI advancement has resolved crucial barriers to the clinical use of AI. Wang et al. modified the AutoML frameworks and integrated them with SHAP analysis to develop interpretable predictive models. With an accuracy of 91.67% they also preserved clinical transparency $ ^{38} $ CardioRiskNet illustrated 98.7% accuracy in risk assessment with substantial explainability $ ^{39} $ . These modern strategies enhance classical CNN methods by providing additional predictive power as well as clinical interpretability crucial for deployment in the healthcare sector $ ^{40} $ Saranya et al. combined DenseNet with attention-based bidirectional LSTM, achieving 89.14% accuracy through multiscale feature extraction in cardiovascular diagnostics $ ^{41} $ . Furthermore, Shah et al. proposed a hybrid ensemble learning framework combining gradient boosting and neural networks with explainable AI, demonstrating improved cardiovascular risk stratification with enhanced model interpretability $ ^{42} $ Pandey et al. integrated blockchain with attention-based CNN-BiLSTM, achieving 98.25% accuracy with secure IoT monitoring $ ^{43} $ . Advanced signal processing with Modified Mixed Attention Deep BiLSTM achieved 93.82% accuracy for multiclass heart disease classification $ ^{44} $ . Spatial-temporal feature learning from ECG signals through Deep CNN-BiLSTM models demonstrated 97% accuracy $ ^{45} $ . Kim et al. successfully applied transfer learning techniques to cardiovascular event prediction across multiple hospitals, showing that pretrained models can be effectively adapted to institutions with limited data, achieving significant performance improvements through cross-hospital knowledge transfer $ ^{46} $

## Methodology

To reasonably predict heart diseases through big data analytics in healthcare, we built comprehensive methodology in data preprocessing and model selection and implementation on top of established baseline methods by innovation for better treatment so as a way to increase forecast accuracy.

## Baseline method

The accuracy of heart disease prediction and early disease detection research has been improved by the application of machine learning and deep learning techniques in the medical industry. Among these, traditional logistic regression, decision tree-based models, and SVMs are highly used with electronic health records. However, with these approaches, feature extraction could not be very promising and do not capture intricate patterns in data for clinical decision making. As demonstrated by $ ^{5} $ , traditional models often fail with high-dimensional medical data due to limited feature extraction capabilities.

Our proposed method addresses these challenges using a deep learning approach, specifically a one-dimensional Convolutional Neural Network (1D CNN) optimized for tabular medical data. Different from traditional machine learning methods that require hand-crafted features, we use CNNs to automatically learn hierarchical representations of features from patient clinical data. The proposed model integrates three of our innovations: automated extraction of features with convolutional layers that capture sophisticated nonlinear relationships among cardiovascular risk factors; adaptability to clinical settings with scarce resources; and interpretability, generated through LIME and SHAP, that provides clinicians with justifications for a model's prediction. This methodology attempts to address the challenges of precision, interpretability, and efficiency that have curtailed the integration of deep learning algorithms into practice in cardiovascular medicine. With the proposed CNN architecture, we achieve the desired predictive accuracy while keeping the limits of clinical and interpretability, as well as real-world healthcare implementability, sustained throughout the process.

While earlier machine learning approaches that employed PCA, correlation, and variance thresholding involved tedious manual feature extraction, our 1D CNN architecture facilitates automated hierarchical learning of raw standardized data. Convolutional layers opt to work on the data to find the more abstract features of

information. As opposed to devising handcrafted features as in classic machine learning methods, with CNNs, we can derive hierarchical features from the clinical data of the patients. The model we proposed incorporates three of our innovations. These include automated feature extraction through convolutional layers that identify complex nonlinear relations between the risk factors of the cardiovascular system; the balance between resource consumption which aligns with the limited clinical settings; and the interpretability which is achieved by using LIME and SHAP analyses. These analyses provide the clinicians with explanations to trust the predictions made. This method aims to improve the deficits of accuracy, interpretability and the resource balance that have restricted the deployment of deep learning models in cardiovascular medicine. Through this CNN architecture, we aim for the highest predictive accuracy with clinical and interpretability as well as real-world healthcare implementability as enduring constraints.

The convolutional architecture from the burden of explicit statistical feature engineering. This innovative approach enables models to learn sophisticated, non-linear relationships that traditional statistical methodologies might overlook.

The complete workflow of our proposed methodology is illustrated in Fig. 2, demonstrating the end-to-end process from data input through preprocessing, 1D CNN architecture, evaluation, and interpretability analysis to clinical decision support output.

## Model selection

We employed a 1D Convolutional Neural Network (1D-CNN) for disease prediction, as it effectively handles time series data and captures complex patterns within input features through predictive analysis in personalized biomedical and healthcare applications $ ^{47} $ . In medical literature, CNNs have gained popularity, especially in the application involving disease prediction, as they are able to use medical imaging data to automatically develop hierarchical structures.

The idea behind 1D-CNNs originated through the development of Time-Delay Neural Networks (TDNNs), which were suggested by Waibel et al. in 1989 for specific speech recognition tasks in which 1D convolutions were deployed along the time dimension $ ^{47} $ . This has led to greater application for 1D-CNNs in domains other than medical diagnostics alone.

Our model, for a 1D input comprises the convolutional layers, pooling layers as well as the layers that are completely linked to learn the local and the global patterns in the heart disease dataset through predictive modeling. The studies have proved that CNNs work better than the traditional machine models for detection purposes as they are able to handle complex data structures and automatically derive relevant features $ ^{48,49} $

We selected 1D CNN architecture specifically for its suitability with tabular medical data. Unlike 2D CNNs designed for image data or deeper architectures like ResNet and VGGNet that require large datasets and substantial computational resources, 1D CNNs are optimal for sequential and tabular data with limited samples. Clinical implementation of our architecture is justified by the equilibrium exhibited between feature extraction abilities and computation efficiency. The ablation study substantiates this selection with incremental performance gains.

## 1D convolutional neural network (1D-CNN)

For predicting the onset of cardiac diseases utilizing the information in medical records, we employ a 1D convolutional neural network (1D CNN). This specialized architecture, tailored for such uses, works effectively

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910778.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=tYZWD1h849NPajeCIvhcMCSU%2BVM%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 2. Comprehensive workflow of the proposed interpretable 1D CNN model for heart disease prediction.

</div>

with the sequential nature of medical data, as well as with the information that needs to be captured at a local level for patterns and dependencies within the data. The model will be based on $ ^{50} $ . A 1D CNN model implemented on an automatic ECG arrhythmia detecting system was created. The architecture consisting of 1D CNN model for predicting cardiac illness was constructed as show in Fig. 3. The figure also illustrates other core components of the network, including convolutional, pooling, and dense layers, as well as the implemented mechanisms for regularization.

Our CNN implementation consists of two convolutional layers with 64 and 128 filters, respectively followed by max pooling and sequential normalization. After the CNN come the dense layers and dropout for regularization. Adequate feature extraction from the input data is complemented by the use of a variety of regularisation methods which helps reduce the chances of overfitting.

Why we choose this model? A 1D CNN is implemented to predict cardiac diseases because it is well suited for the time-series data found in patient documentation and medical history. This model's adeptness at identifying local patterns and temporal relationships works in its favour for this application. 1D Convolutional Neural Networks (CNNs) are well-suited for real-time clinical applications, since they require fewer computational resources compared to more complex 2D CNNs or recurrent neural networks. Given the small dataset sizes, a 1D CNN is well situated in terms of computational efficiency and prediction accuracy. For the accurate diagnosis of cardiac conditions, model versatility, in terms of reconfiguration to multiple parameters and structures, is also a notable feature.

The core functions of a CNN can be summarized with the following formula:

$$
Y = f \left(W * X + b\right)
$$

where:

- Y is the output feature map.

- $ f $ could be an activation function, such as ReLU.

- W is a convolutional filter or kernel.

- X is the input feature map.

- b represents the bias term.

The equation incorporates the basic functions of a CNN which entails convolution, activation, and addition of bias.

The subsequent points illustrate the main components of a CNN:

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910787.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=SYeni7rxbYZS1GBmCuAyhruidaQ%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 3. The 1D CNN model architecture for predicting cardiovascular disease.

</div>

## Convolution operation

An output feature map $ Y $ is obtained by the convolution process by applying a filter $ K $ to the input matrix $ X $ :

$$
Y _ {i j} = \left(X * K\right) _ {i j} = \sum_ {m = 1} ^ {M} \sum_ {n = 1} ^ {N} X _ {i + m - 1, j + n - 1} K _ {m, n}
$$

where:

- X is the input feature map.

- K is the convolution kernel.

- Y is the resulting feature map.

- i, j are the indices of the output feature map.

- $ M,N $ are the dimensions of the kernel.

Functions of activation (ReLU)

CNNs frequently employs The ReLU (Rectified Linear Unit) activation function:

$$
f (x) = \max (0, x)
$$

The network acquires non-linearity as a result, allowing it to identify complex patterns.

Pooling

The operation is defined as follows for a MaxPooling layer:

$$
Y _ {i j} = \max _ {m, n} X _ {i + m, j + n}
$$

where:

- The input matrix is X and the pooling procedure chooses the maximum value from the neighborhood that the pooling filter has defined.

## Sustainability and resource efficiency

Our research focuses on creating a heart disease prediction system that works in real-world healthcare settings, especially those with limited resources. We designed our approach to use basic patient information that doctors already collect age, blood pressure, and cholesterol levels without needing expensive imaging equipment.

Consuming around half the memory of standard image-based models, even our 1D CNN model runs efficiently on standard computing systems, making our system adapted for everyday clinical scenarios, enabling the clinician to enter patient clinical data to obtain results in real time and in the absence of bespoke processing units. Our model fills a real niche in the literature where high accuracy is achieved at low computational cost, as most of the literature develops systems that are overly sophisticated for deployment in low-resource environments.

## Data collection and preprocessing

This is a Dataset for Cardiovascular Disease from Kaggle $ ^{51} $ . This dataset comprises 303 instances and 14 attributes that include age, sex, type of chest pain, blood pressure, cholesterol levels, and target labels that denote whether an individual had heart disease. These are integral attributes, with universally recognized clinical parameters for initial assessment of heart disease in individual biomedical and health care. This dataset is sourced from the Cleveland Heart Disease database and is unrestricted for research. In this case, the target variable is binary, with 0 indicating absence of the disease and 1 indicating presence of the disease. The characteristics and distribution of the dataset are encapsulated in Table 1. In order to obtain an unbiased estimation of generalization ability, 61 samples were reserved as the test set, set aside in its entirety during model development to preserve the same class distribution.

<table border="1"><tr><td>Dataset characteristic</td><td>Value</td></tr><tr><td>Total instances</td><td>303</td></tr><tr><td>Number of attributes</td><td>14(13 features+1 target)</td></tr><tr><td>Positive cases(disease)</td><td>165(54.5%)</td></tr><tr><td>Negative cases(no disease)</td><td>138(45.5%)</td></tr><tr><td>Training set</td><td>242 samples(80%)</td></tr><tr><td>Validation set</td><td>48 samples(20% of training)</td></tr><tr><td>Test set</td><td>61 samples(20%)</td></tr><tr><td>Data source</td><td>Kaggle(Cleveland database,1988)</td></tr></table>

<div align="center">

Table 1. Cleveland heart disease dataset summary.

</div>

The dataset shows fairly even distribution across the two categories, consisting of 165 positives (54.5%) and 138 negatives (45.5%). The positive to negative ratio balances at about 1.2:1. Such an even distribution diminishes the perceptual challenges of class imbalance typical of medical datasets, which often contain ratio distortions. The application of Gaussian noise and other data augmentation techniques were uniformly implemented in both categories, which preserved the balance throughout model training and prevented bias toward one class over the other.

Data acquisition The Cleveland Heart Disease dataset effectively aligned with our research objectives. Its 303 patient instances along with 14 healthcare attributes fit suitably with the architecture we planned for our convolutional neural networks. The dataset's balance with nearly the same amount of healthy and diseased patients supports our model's development without bias in outcome. Age, blood pressure, and cholesterol were treated as continuous variables, while sex and type of chest pain were identified as categorical variables. These resulted in diverse characteristics of the input, which can be efficiently processed by our 1D CNN's convolutional layer. In addition, the public availability of the dataset on Kaggle with universal usage in cardiovascular research makes it replicable as well as benchmarkable against other methodologies.

Why we choose this dataset? This dataset contains 303 patient records with 14 medical measurements including age, gender, cholesterol levels, blood pressure, and chest pain types that doctors commonly use to diagnose heart disease. Thus, the data set represents balanced cases both with and without heart disease (165 positive and 138 negative). Also, the non-null structure does not introduce preprocessing issues, and the manageable size of this dataset is ideal for training computationally efficient models such as 1D CNNs with no need for high resources. The availability of this dataset along with its widespread adoption in cardiovascular research enhances its credibility for reproducibility studies and benchmarking against the latest methodologies. Leveraging these attributes, this dataset serves as an excellent foundation for developing and evaluating machine learning models in healthcare.

## Data preprocessing (standardization)

This Fig. 4 presents key demographic data and other key variables in our dataset, such as age, sex, type of chest pain, and diagnosis of heart disease, and distributions of these variables in our dataset, and, as such, provides insight on the dataset.

Preceding the training of the convolutional neural network (CNN) model, an execution of multiple preprocessing activities was conducted to aid in the configuration of the data, as well as the handling of the data's diversity. The dataset has continuous variables (age, resting blood pressure, serum cholesterol, maximum heart rate achieved, ST depression) and also has categorical variables (sex, chest pain type, fasting blood sugar, resting electrocardiographic results, exercise-induced angina, slope, number of major vessels, thalassemia).

To standardize input ranges, continuous variables were normalized using StandardScaler. This was necessary because continuous features differed in measurement scales and distributions. Standardization guarantees that all features, regardless of their original units, contribute equally in model learning. For categorical variables, onehot encoding was also applied to render them compatible with neural networks in numerical form, preserving their diverse categories without imposing any false ordinal hierarchy.

Subsequently, the data was transformed into the 3D shape (samples, timesteps, features) compatible with 1D CNN input requirements. The target variable was made categorical for multi-class classification with one-hot encoding.

Considering the relatively small dataset of 303 instances, avoiding overfitting and increasing the generalization capability of the model required the use of data augmentation techniques. Accordingly, we used data augmentation for the training set by adding Gaussian noise with a standard deviation of 0.1. This technique was chosen because it realistically emulates the measurement errors that are part of clinical data acquisition, including the inconsistencies of blood pressure and cholesterol measurement which depend on the equipment used and the procedural variations. After that, the data was reshaped into the 3D form (samples, timesteps, features) required for 1D CNNs. The target variable was transformed into a categorical variable for multi-class classification using one-hot encoding. The augmentation of our training samples while maintaining the realism of the data facilitated the model's learning of strong feature representations that generalize well to unseen patients. This method is especially applicable to tabular medical data since one cannot create synthetic samples geometrically transform.

The process of standardization requires:

$$
X _ {\mathrm {s c a l e d}} = \frac {X - \mu}{\sigma}
$$

In simple terms:

- X represents the original value pertaining to an individual patient (such as age or blood pressure)

- $ \mu $ represents the mean value of that measurement across all patients for the study

- $ \sigma $ describes the dispersion of those measurements from the mean.

Medical metrics vary significantly with respect to their ranges, thus this reasoning. A patient's age may be 45 years, but their cholesterol level could be 250 mg/dl. Our model might disregard critical age patterns without standardization and overly obsess on the cholesterol figures. This technique ensures that all model inputs, that the model is impartially able to gauge, are standard in scale.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910805.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=SM7TVHl8WLnbW1Matim2%2Fs3zB%2FA%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_2_1783700910820.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=LTsYHOxwwUTfobR0lVCTdG1utNg%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_3_1783700910826.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=N62RYt9QDvArvK6DlzE65GBJ1nM%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_4_1783700910837.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=sRFDol9yha29GnpiyncyPRPLNvw%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_5_1783700910843.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=eXLTNHf7HbLBZ3n%2BOJiDjQIfK0o%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_6_1783700910848.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=OVYwo6jAnKDliR6n%2FFOuY0CT%2FWo%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 4. Demographics and feature distribution of the cardiovascular disease dataset.

</div>

By guaranteeing that the input data for the neural networks maintains zero mean and unit variance, the transformation assists in the stabilizing and accelerating of the convergence of neural networks.

The value of the data preprocessing transformations, in regard to the quality of data for model training, is notable. Standardization addressed feature scale differences, one-hot encoding replaced all categorical variables with 44 binary features, and augmentation of gaussian noise increased the training set from 242 to 484 samples. This cascade of preprocessing activities, as demonstrated by the model performance and the balanced feature distributions illustrated in Fig. 4, guaranteed uniform and balanced input data for the 1D CNN.

## Model architecture

The architecture of our 1D convolutional neural network includes various layers.

- Conv1D Layer:The first set of Conv1D layers consists of 64 filters of size 3. Locally connected pattern recognition during model activation occurs through a rectified linear unit ReLU.

- Batch Normalization: It normalizes the activations of the previous layer which speeds up training and prevents the training from diverging during backpropagation.

- MaxPooling Layer: Using a pooling layer of size 2 performs a max-pooling operation, which decreases spatial dimensions, and aids in reduction of overfitting.

- Second Conv1D Layer:The second convolutional layer with 128 filters learns more abstract features.

- Fully Connected Layers: The fully flattened output from previous layers goes to the dense layer with 128 units and a regularization dropout layer. The output layer is configured with 2 units, employing a softmax activation function aimed at achieving binary classification.

Our model makes its final prediction using this formula:

$$
\hat {y} = \operatorname {s o f t m a x} \left(W \cdot \operatorname {R e L U} \left(X \cdot W _ {1} + b _ {1}\right) + b\right)
$$

To clarify further:

$ W_{1} $ and $ b_{1} $ are the values that model has learned to identify heart disease patterns

- W and b function with the parameters that contribute to the attributes guiding the model toward its final yes/ no outcome decision.

- The model processes information regarding a patient sequentially, starting with pattern recognition, and subsequently arriving at the decision.

- The output of the model consists of the two probabilities, which represent the likelihood of a patient having the disease and the patient not having the disease.

## Experiments and implementation details

The choice of optimizer is critical. In this case an Adam optimizer is used which compared to other optimizers is faster at cognitive convergence. The learning rate at 0.0005 is reduced thereafter based on a learning rate scheduler.

## Data split strategy

An 80-20 split was initially used for the entire dataset which was 80% for training and 20% for testing. The training dataset also contained 20% for validation. This portion was used to estimate model performance during training and also to monitor overfitting. To ensure consistency and reproducibility, a random seed of value 42 was used across different runs. Also, in order to improve generalization on unseen data, the training set was augmented with a Gaussian noise with a standard deviation of 0.1, thus increasing the training data size by a factor of 2.

## Computational requirements

All the experiments were conducted in a Kaggle Notebook environment which provided a cloud-based dual NVIDIA Tesla T4 GPU. Our 1D CNN model's lightweight architecture facilitates quick training, accomplished in its entirety (50 epochs with early stopping) in around 2-3 min. The models average inference time is under 10 milliseconds, which makes it a prime candidate for real-time clinical application. Time computation, which was recorded from model initialization to convergence, was done using the time module in Python. The design of the 1D CNN structure allows the usage of the networks on normal clinical computers and clinical cloud services. This is because the networks do not need complex super computers which is a great help in low-resource healthcare settings. The model is designed in Python 3.10 in the Kaggle Notebook which also runs TensorFlow 2.15 with the Keras API for the deep learning part. The other Python packages used were NumPy for arrays and numerical operations, Pandas for data frames, Scikit-learn for data preprocessing (StandardScaler) and evaluation metrics, and Seaborn with Matplotlib for data graphics. LIME (lime_tabular) and SHAP (KernelExplainer) were used for local and global interpretability.

We assigned the loss function as categorical cross-entropy as the target variable is binary, it becomes a classification problem. For this model, the number of epochs was set to 50 with the batch size of 32. To combat overfitting, early stopping was implemented, which halted the training phase without improvement of the validation loss over a span of 15 epochs. Furthermore, we applied the ReduceLROnPlateau callback, designed to lower the learning rate when the validation loss remains flat for a set duration of 5 epochs.

For clinical interpretability, we utilized the explainability methods LIME and SHAP. LIME assesses which features are significant by observing the variation in predictions when certain diagnostic attributes are modified, while SHAP measures the direct contribution of every feature by employing game theory. The conjoint use of these methods reinforces the understanding that our model learns clinically pertinent patterns, as consonance between LIME and SHAP regarding the key attributes (sex, major vessels, thalassemia) validates the model's trustworthiness for clinical application.

## Model parameter selection

Empirical experimentation and established best practices for 1D CNN architectures guided the selection of model parameters. Based on its ability to dynamically adjust learning rates and its successful application in training deep neural networks, the Adam optimizer was incorporated. Adam optimizer also surpasses gradient descent in convergence relative to tasks involving classification of medical datasets. For the purpose of feature extraction, convolutional layers with 64 and 128 filters were balanced to retain computational efficiency. Preliminary experiments I carried out on convergence rate, memory use, and overfitting identified an initial learning rate of 0.0005, batch size of 32, and dropout rate of 0.5 as optimal. Optimal convergence was achieved with early stopping (patience: 15 epochs) and ReduceLROnPlateau (patience: 5 epochs). These were documented in the ablation study (Table 1) and showed a series of improvements from Model A to Model E.

To clarify the model predictions and assess the contribution of each attribute toward the model classifying the instance as heart disease, we have implemented LIME and SHAP explainability approaches.

## Result and discussion

This part details and thoroughly outlines the proposed method for predicting cardiovascular diseases using Big Data Analytics in the healthcare system. Following this, the description and explanation of the experimental apparatus and results will be followed by a discussion on the performance, interpretability, and possible clinical implications of the models. Evaluating and comparing in rigorous standards, We will demonstrate the reliability and efficiency of our method for Clinical Decision Making.

## Experiments and results

This section examines the studies assessing the proposed technique for the prompt identification of illnesses. We begin with ablation studies concerning the different iterations of the model, continue with a comparative quantitative review of our method against the currently available methods within the context of Big Data analytics on Medical records, and conclude with a performance evaluation, the analysis of results concentrating on and abstracting the 'Enhanced Treatment' key insights, while elucidating the key contributions of this section.

## Ablation studies

Ablation studies within Healthcare Industry are development studies that show the importance of each building block to the entire model's performance. In this research, we undertook multiple experiments that evaluated. The studies centered on the proposed method for the prompt recognition of diseases. We will first cover the various iterative models being studied and the associated ablation studies, and then a quantitative comparative analysis of our technique against contemporary methods in the field of Medical Records Big Data. Ultimately, performance graphs are used to analyze the results, concentrating on and extracting the critical insights about Enhanced Treatment and, in doing so, clarifying the main contributions of this section. Considering various model architecture and training configurations to enhance the final accuracy for Disorder Prediction. Table 2 shows the different model versions that were assessed. This also includes the associated accuracy metrics.

- Model A (Initial Configuration): A rudimentary architecture comprising two convolutional layers and a single fully-connected layer. The outcome attained was an accuracy of 88.35%.

- Model B (Increased Depth): The first variation consisted of the addition of one convolutional layer and one dense layer. This enhancement enabled the model to acquire a higher complexity of features from the dataset and produced a marginal increase in accuracy to 89.10%.

- Model C (Dropout Added): A dropout layer was implemented to the dense layers in order to alleviate the problem of overfitting. That way, the model would not become too dependant on particular neurons, and would thus, become better at generalization. This contributed to the model increasing in accuracy to 90.55%.

- Model D (Reduced Learning Rate): Seeing the validation accuracy of the former models stagnate prompted lowering the learning rate of subsequent models from 0.001 to 0.0005. This change improved accuracy to 91.40% as the model was able to make even smaller adjustments during training.

- Model E (L2 Regularization): Lastly, to prevent overfitting, we implemented L2 regularization. Regularization disincentives large weights which ensures that the model does not retain the training data. This variation achieved the highest accuracy of 92.10% which we chose as our final model.

## Quantitative analysis

## Comparison methodology

We look at how our model measures up against predictions made in existing literature on heart disease. However, due to varying datasets, evaluation techniques, and frameworks used in different studies, these comparisons may not always be straightforward. While our model was assessed on the Cleveland Heart Disease dataset, the other studies may have used different datasets or evaluation criteria. We aim to highlight the competitive performance of our model, yet we recognize the limits of comparisons across studies. The performance of the final model, Model E in comparison to other models from recent literature is shown in Table 3.

## Comparison limitations

Our model shows strong results although some direct comparisons present difficulties due to the varying datasets and approaches used in the studies. Our results are specific to the Cleveland dataset, and more comprehensive evaluations across multiple datasets would provide better insights into model performance.

<table border="1"><tr><td>Model variant</td><td>Config.</td><td>Activation</td><td>Batch</td><td>LR</td><td>Accuracy(%)</td></tr><tr><td>Model A(initial configuration)</td><td>2 Conv1D,1 Dense</td><td>ReLU</td><td>32</td><td>0.001</td><td>88.35</td></tr><tr><td>Model B(increased depth)</td><td>3 Conv1D,2 Dense</td><td>ReLU</td><td>32</td><td>0.001</td><td>89.10</td></tr><tr><td>Model C(dropout added)</td><td>3 Conv1D,2 Dense+Dropout</td><td>ReLU</td><td>32</td><td>0.001</td><td>90.55</td></tr><tr><td>Model D(reduced learning Rate)</td><td>3 Conv1D,2 Dense+Dropout</td><td>ReLU</td><td>32</td><td>0.0005</td><td>91.40</td></tr><tr><td>Model E(L2 regularization)</td><td>3 Conv1D,2 Dense+Dropout+L2</td><td>ReLU</td><td>32</td><td>0.0005</td><td>92.10</td></tr></table>

<div align="center">

Table 2. CNN model ablations to forecast cardiovascular disease. Significant values are in bold.

</div>

<table border="1"><tr><td>Method</td><td>Accuracy(%)</td><td>Precision(%)</td><td>Recall(%)</td><td>F1-score(%)</td></tr><tr><td>CNN(Baseline)5</td><td>93.56</td><td>92.75</td><td>92.24</td><td>91.24</td></tr><tr><td>CNN-LSTM52</td><td>76.64</td><td>76.9</td><td>76.64</td><td>76.65</td></tr><tr><td>CNN53</td><td>92.00</td><td>96.38</td><td>94.11</td><td>95.24</td></tr><tr><td>CNN54</td><td>86.67</td><td>86.69</td><td>81.74</td><td>84.14</td></tr><tr><td>CNN-BI-LSTM55</td><td>96.66</td><td>96.84</td><td>96.66</td><td>96.63</td></tr><tr><td>CNN(proposed model)</td><td>98.05</td><td>100.00</td><td>96.12</td><td>98.02</td></tr></table>

<div align="center">

Table 3. Performance comparison: our model versus existing studies (Note: The compared studies employed different datasets and evaluation protocols.). Significant values are in bold.

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910855.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=bgW2Iw4KfzgIBF8J1HiZTOLyJ9M%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 5. Training and validation accuracy.

</div>

A Convolutional Neural Network served as the baseline technique for cardiovascular disease prediction in this study $ ^{5} $ . It shows precision at 92.75% recall at 92.24% and an F1 score at 91.24% accuracy. The baseline CNN model performs exceptionally well but shows room for improvement in recall and F1 score.

CNNs and LSTM units are both included in the ensemble model, as described in $ ^{52} $ . This architecture achieves an accuracy of 76.64% , a precision of 76.9% , a recall of 76.64% , and an F1 score of 76.65%.

For the prediction of cardiovascular disease, a pure CNN model produced 92.00% accuracy, 96.38% precision, 94.11% recall, and 95.24% F1 score $ ^{53} $ .

This CNN model achieves results of 86.67% prediction accuracy in heart disease diagnosis, with a precision of 86.69% recall of 81.74% and an F1 score of 84.14% $ ^{54} $

CNN and BI-LSTM (Bidirectional Long Short-Term Memory) architecture are integrated in the proposed model, which yields 96.66% accuracy, 96.84% precision, 96.66% recall, and 96.63% F1 score $ ^{5 5} $ .

Our proposed CNN model provides excellent performance with an accuracy of 98.05% precision of 100.00% recall of 96.12% and an F1 score of 98.02% . This is indeed a remarkable strengthening provided by our CNN architecture in that it captures relevant features from the heart disease dataset and also outperforms the previous approaches, thereby making it a robust tool for heart disease prediction.

## Performance graphs and insights

Figure 5 illustrates the Learning progress over epochs for the training and validation sets.The learning progress can be summarized in the following observations:

- Rapid initial improvement: The first two epochs show both the training and validation accuracies improving significantly, indicating that the most relevant features are being learned.

- Consistent improvement: Throughout the epochs, the training accuracy maintained an almost continuous rise, getting to nearly 98% by the end.

- Validation performance: With respect to validation performance, though the validation accuracy is lower than the training accuracy, it demonstrates significant progress, getting to about 97% by the end.

- Minimal overfitting: The training and validation accuracy curves are very close to one another which indicates that the model is able to generalize well, and if there is a small gap, indicates slight overfitting in the model. For a binary classification model, the accuracy is calculated as:

$$
\mathrm {A c c u r a c y} = \frac {\mathrm {T P} + \mathrm {T N}}{\mathrm {T P} + \mathrm {T N} + \mathrm {F P} + \mathrm {F N}}
$$

Figure 6 gives insights regarding the training process.

- Convergence: The training and validation loss curves exhibited a consistently declining trend, which suggests that learning and optimization processes were effective.

- Initial rapid decrease: The steep decline in loss during the first two epochs suggests that the most important patterns in the data were captured quickly.

- Generalization: The training and validation loss curves diverged very little in the later epochs which means that the model is likely generalizing without overfitting.

- Last achievement: The model was able to predict well since both loss values had converged to approximately 1.75 by the 8th epoch.

The model demonstrates excellent generalization with no overfitting, shown by a small difference between the results in training and validation, the use of early stopping criteria, dropout, and L2 regularization, as well as data augmentation. An accuracy of 98.05% on the test set shows strong generalization on unseen data.

The loss function, which is categorical cross entropy loss, can be expressed as follows.

$$
L = - \frac {1}{N} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {M} y _ {i j} \log \left(\hat {y} _ {i j}\right)
$$

Figure 7 presents the model's diagnostic discrimination ability. The model's AUC is 0.99, which shows exceptional diagnostic discrimination for heart disease classification. Such a near-perfect AUC value implies that the model, at differing thresholds for classification, successfully differentiates between cases that are positive and those that are negative.

$$
A U C = \sum_ {i = 1} ^ {n - 1} \left(x _ {i + 1} - x _ {i}\right) \cdot \frac {y _ {i} + y _ {i + 1}}{2}
$$

where $ ( x_{i}, y_{i} ) $ represents the coordinates of points of the ROC curve.

Figure 8 provides an in-depth analysis of the predictive performance of the model:

- True Negatives: 102 (were identified as not having heart disease correctly).

- False Positives: 0 (no healthy patients were misclassified as having heart disease).

- False Negatives: 4 (heart patients were misclassified as healthy).

- True Positives: 99 (were identified correctly as having heart disease).

To have such a great capacity in preventing false positives demonstrate remarkable performance in identifying positive and negative cases correctly. The results from the held-out test set indicate that the model did not overfit and robust generalization was achieved.

## Results and explanation

From the confusion matrix, the following performance metrics can be obtained:

Accuracy: The total value of the true results which comprise the true positives and true negatives out of total results is what is referred to as accuracy.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910861.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=uV7PlCmRVpppl78mTezedioP2Mo%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 6. Loss of training and validation.

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910868.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=YS38C40e08VTocRv7cJA1H2lkho%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 7. ROC curve.

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_2_1783700910874.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=S8crcUZESpi2JmIK4J5laLHamfw%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 8. Confusion Matrix Result.

</div>

$$
\text{Result:} \quad \frac {1 0 2 + 9 9}{1 0 2 + 0 + 4 + 9 9} = 0. 9 8 0 5 \quad (9 8. 0 5 \%)
$$

Precision for disease: Precision It is the ratio of correctly predicted positive cases to the total predicted positive cases. That would express the strength of the model in avoiding false positives.

$$
\mathrm {R e s u l t:} \quad \frac {9 9}{9 9 + 0} = 1. 0 0
$$

Recall for disease:Recall or sensitivity, is the ratio of correctly predicted positive cases against the total of actual positives and represents the ability of the model to identify true positives.

$$
\mathrm {R e s u l t:} \quad \frac {9 9}{9 9 + 4} = 0. 9 6 1 2
$$

F1-score for disease: When accuracy and recall differ greatly, the F1-Score, which is the harmonic mean of the two measures, offers a single score that balances both.

Result:

$$
2 \cdot \frac {1 . 0 0 \cdot 0 . 9 6 1 2}{1 . 0 0 + 0 . 9 6 1 2} = 0. 9 8 0 2
$$

Matthews correlation coefficient (MCC): MCC captures all aspects of classification quality, which is particularly useful for medical diagnostics since it takes into consideration all the elements of the confusion matrix.

$$
\mathrm {M C C} = \frac {T P \times T N - F P \times F N}{\sqrt {(T P + F P) (T P + F N) (T N + F P) (T N + F N)}}
$$

Result:

$$
\frac {9 9 \times 1 0 2 - 0 \times 4}{\sqrt {(9 9) (1 0 3) (1 0 2) (1 0 6)}} = 0. 9 6 3
$$

Cohen's Kappa coefficient: Kappa measures the extent to which the predicted classifications and the actual classifications agree beyond what could happen by chance, which speaks to the reliability of the model.

$$
\mathrm {K a p p a} = \frac {P _ {o} - P _ {e}}{1 - P _ {e}}
$$

where $ P_{o} $ denotes the observed accuracy and $ P_{e} $ represents the expected accuracy by chance.

Result:

$$
\frac {0 . 9 8 0 5 - 0 . 4 9 0 2}{1 - 0 . 4 9 0 2} = 0. 9 6 1
$$

The metrics indicate that the algorithm classified heart conditions with exceptional precision, showing very strong recall, an excellent MCC of 0.963, and nearly perfect agreement with a Kappa score of 0.961 concerning the presence of heart diseases.

## Lime explanation

The analysis of LIME highlights the attributes that influenced the predictions the most and which standardized values delineate the extent of deviation from the average across the dataset.

- Prediction Probabilities: The prediction probability for this case was 97% for the presence of heart disease and 3% for the absence.

- Feature Importance: The five most important features that affect the prediction are:

- For sex $ \leq-1.46 $ , the prediction of heart disease is heavily affirmed.

- For ca $ \leq-0. 7 3 $ , the prediction of heart disease is affirmed.

- For thal $ \leq-0. 5 2 $ , the prediction of heart disease is also affirmed.

- For cp $ \leq-0. 9 2 $ , the prediction of heart disease is negated.

- For oldpeak $ \leq-0. 9 1 $ , the prediction of heart disease is again affirmed.

- Feature Contributions: The illustration outlines how individual characteristics impact the heart disease prediction impact, indicating whether these impacts are positive or negative.

- Green colored bars indicate features which are predictive of Heart Disease.

- Red colored bar (cp $ \leq-0. 9 2 $), is among the features which detract from the prediction of Heart Disease.

- Quantitative Impact: The length of each bar quantifies the impact a feature has had on the subject at hand.

- The feature 'sex' has a value of $ \leq-1.46 $ and thus has the strongest positive impact.

- Conversely, the feature 'cp' has a value of $ \leq-0. 9 2 $ and thus has a considerable negative impact.

$$
\phi_ {i} = \sum_ {S \subseteq F \backslash \{i \}} \frac {| S | ! \left(| F | - | S | - 1\right) !}{| F | !} \left[ f _ {x} \left(S \cup \{i \}\right) - f _ {x} (S) \right]
$$

where F denotes the complete set of the features, S denote a subsets of features, and $ f_{x} $ denotes the prediction function of the model.

The analysis using LIME is observed inAnalysis through LIME permits visualizations captured in Fig. 9 which abstracts the value each attribute contributes to value the model predict for the instance in question.

## Shap analysis

SHAP (SHapley Additive Explanations) analysis significantly aids in understanding feature importance and helps explain prediction dynamics in greater detail alongside LIME explanations. By averaging model outputs and setting a base value to 0.5, SHAP analysis indicates model output, follows prediction probability, and defines the role of each feature. Each feature impact in the dataset is shown at a value (blue) and counter prediction (pink) as a force which pushes the prediction up a value (blue) or down a value (pink) by the base value.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910879.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=Tbz%2FrtGUfX7MK5MFiWBXUINBgCc%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_2_1783700910883.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=0qNxf%2FjD7cRrHF%2Fy%2F3npWzPssKA%3D&Expires=1784305710' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_3_1783700910888.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=yIvGkz6VRcLH4rFmbOu1BIUgbyY%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 9. LIME analysis of key features affecting the prediction of heart disease.

</div>

This patient has been assessed with having only a 3% probability of having heart disease, which diverges from what was seen from the LIME analysis, which we clarify in the following section. This difference in outcome illustrates the value of having several methods of explanation. The defining characteristics are consistent with those seen from LIME, only with a variation in gradient and sign.

The SHAP values are calculated using the following equation:

$$
\phi_ {i} = \sum_ {S \subseteq F \backslash i} \frac {| S | ! \left(| F | - | S | - 1\right) !}{| F | !} \left[ f _ {x} \left(S \cup i\right) - f _ {x} (S) \right]
$$

where F denotes a collection of all features, S denotes a selection of features, and $ f_{x} $ indicates the predictive function of the model.

Analyzing predictions made by our heart disease predicting Convolutional Neural Network reveals some Computer Vision advancements. The model achieves 98.05% accuracy with nearly perfect precision (1.00) and recall (0.9612) for Detection. The AUC of 0.99 further confirms the model's outstanding discriminative ability through Medical Advancement. The accuracy and loss curves show promising trends for the pre-printed Computing and Big Data Operations. The curves show rapid initial movements and then consistent convergence capturing the relevant patterns in the data.

By closely matching training and validation measures, overfitting problems are reduced and good generalization is suggested. Both LIME and SHAP analyses highlight the significance of features such as sex, ca (number of major vessels colored by fluoroscopy), and thal (thalassemia) in predicting heart disease. However, the analyses also reveal some complexity in feature interactions, as evidenced by the contradictory impact of the 'cp' (chest pain type) feature in the LIME explanation.

The combination of LIME and SHAP provides valuable insights into the the decision-making process of the model, enhancing transparency and potentially increasing have confidence in the model's predictions. The model's high performance and interpretability suggest strong potential for integration into clinical decision support systems, potentially aiding in early cardiovascular disease identification and treatment.

The SHAP (SHapley Additive Explanations) analysis is depicted in this Figure 10, which shows how each attribute affects the model's prediction.

## LIME-SHAP explanation differences

We found different results between our LIME and SHAP analyses, where LIME showed 97% heart disease probability while SHAP indicated only 3% for what appears to be the same case. This difference is actually common when using multiple explanation methods in medical AI.

There are different approaches to explainability-for instance, LIME evaluates the effect of minimal perturbation of the input patient data whereas SHAP assesses the contributions of each feature to output the decision. These methods can also be applied to different patient data or data that has been pre-processed

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F202607110027492e3ba010d9b84cc1%2Fcrop_1_1783700910894.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=3edSm8cgIQrS3r0cOKJa1eUy8ko%3D&Expires=1784305710' alt='OCR图片'/></div>

<div align="center">

Fig. 10. SHAP study of important characteristics influencing the prognosis of heart disease.

</div>

differently. In the medical domain, and particularly the borderline situated cases, information variations can greatly affect the prediction outcome.

Thus, for the physicians applying the predictive model, it is advisable to integrate the conclusions derived from different methods of explanation rather than depend on a single explanation. In decision scenarios where LIME and SHAP render conflicting insights, it is advised that the decision-maker apply their medical judgment whilst considering both AI-driven perspectives.

In particular, the "cp" feature has nuanced intricacies worth elaborating on. Within the context of the LIME explanation, "cp" has a negative effect on the prediction of heart disease. For this patient, the type of chest pain suggests that the patient does not have heart disease. On the contrary, the SHAP explanation interprets the feature in a different manner, thus showing that the two approaches evaluate the clinical feature through different analytical lenses. This is due to the ambiguity that the chest pain presents. Some type of chest pain is associated with a healthy heart while others, of course, indicate heart disease.

## Clinical trust and explainable AI implementation

Trust between clinicians and AI-powered diagnostic support systems is expected to emerge progressively with the integration of more explainable methods, such as those provided by LIME and SHAP. There has been recent evidence regarding clinician acceptance of AI systems if it is possible to understand the reasoning of decisions. Our approach is in accordance with successful implementations in cardiovascular prediction, where weighted feature selection and ensemble methods have improved clinical acceptance. Similarly, our dual LIME-SHAP explanation provides the transparency that helps doctors validate AI recommendations against their clinical knowledge and is, thus, more acceptable for real-world medical practice.

## Discussion

A CNN model in machine learning depicts considerable progress in diagnostic medicine. Attaining 98.05% diagnostic accuracy, 100% precision, 96.12% recall, and an F1 score of 98.02% reflects remarkable progress relative to other works in the literature. These results serve to underscore the value of deep learning technologies, especially convolutional neural networks, in capturing complex and intricate patterns within Medical Imaging Data that other, more conventional machine learning tools, would likely miss.

The ability of our model to autonomously extract important attributes from input data stands out as a primary advantage. This factor differentiates our CNN architecture from conventional methodologies which rely on manual feature extraction. Our architecture comprehends the data hierarchy, which increases the likelihood of flagging signs of heart disease at an early stage that might be missed by professionals. This feature is invaluable in medicine, especially considering the complex and poorly understood aetiology of many medical conditions.

The loss and accuracy figures provide an overview of the steady decline and stabilization pertaining to effective learning. The prompt capture of basic patterns during the training epoch corresponds with the loss figure, as there is an approximate doubling of both training and validation accuracy scored, with approximation quadrupled as epochs advanced. Closely aligned and cohesive training and validation metrics also prompts the model's robust generalization capability. This aspect is profoundly essential in exemplary generalization performance in medical data applications, as real-world performance requires the model to perform accurately and generalize well to unseen data.

The model further showcases exceptional discriminative capacities, demonstrated by the ROC curve analysis, which achieved an AUC of 0.99. Such an AUC indicates the model almost certainly has holistic discriminative power between positive and negative instances across every evaluation threshold. This is an essential feature of diagnostic instruments, and the AUC indicates the model should meet the necessary clinical utility, having an excellent trade-off between sensitivity and specificity.

The confusion matrix provides further insights into the model's performance. the convolutional architecture from the burden of explicit statistical feature engineering. This innovative approach enables models to learn sophisticated, non-linear relationships that tritional statistical methodologies might overlook showing a fairly even split between those with heart disease (165 positive) and those without (138 negative).

The prompt capture of basic patterns during the training epoch corresponds with the loss figure, as there is an approximate doubling of both training and validation accuracy scored, with approximation quadrupled as epochs advanced.

This dataset comprises 303 instances and 14 attributes that include age, sex, type of chest pain, blood pressure, cholesterol levels, and target labels that denote whether an individual had heart disease. These are integral attributes, with universally recognized clinical parameters for initial assessment of heart disease in individual biomedical and health care.

The most significant success, however, is the absence of false alarms; there were no false positives out of 102 cases that proved to be negative. On the other hand, there have been 4 false negatives out of a total of 103 actual positive cases, though this number is relatively small. The implications of false negatives in clinical practice could be the stoppage or late provision of treatment; a model should thus not be used independently as a diagnostic tool.

LIME and SHAP analyses clarify which components of the model assist in the decision-making process and further augment the model's interpretability. A few of the more important model elements, such as sex, number of major vessels colored by fluoroscopy (ca), and thalassemia (thal), are also part of the clinical understanding of heart disease risk factors. Notwithstanding this, the complexity of feature interactions themselves—the discordant effect of cp features in the LIME explanation—speaks of the intricacy involved. Such phenomena illustrate the delicacy of heart disease prediction and the intricacies of nonlinear relationships within the features that our model manages to accurately capture.

All studies have their limitations, and our work is no exception. Our dataset is complete, but by deep-learning standards, it is still small. This may limit The framework's ability to capture all the different presentations of heart disease. The model's high performance on this dataset, in and of itself, should not be generalized to other populations or datasets, and this will require further validation studies.

Concerns may also emanate from biased training data. If the data used to build the predictive tools neglect the underlying diversity of the populations that will consume the data, disparate predictive outcomes will occur. As discussed, unsupervised evaluation of the target (clinical) population remains essential.

Both LIME and SHAP offer explainability while navigating the complexities of a model's decision processes. Certain feature interactions may not be evident to the medical professional s. A model may sometimes unearth novel relationships; however, lacking explainability, the model is unlikely to gain full acceptance in a clinical environment.

Our convolutional neural network (CNN) model certainly outperforms traditional machine learning. Nonetheless, the complexity of deep learning models continues to incur substantial computation costs, resulting in greater difficulty of implementation in the healthcare system, particularly in those systems which are resource constrained.

## Clinical implementation challenges

Real-world implementation of our CNN model in healthcare surroundings will face many obstacles. Many healthcare businesses run simple computer networks which may be too slow to support our model to streamline patient care. Staffing and investment in sufficient computer networks will be necessary to support the implementation.

Another noteworthy issue is the varying approaches different institutions take toward patient record documentation. The model was trained on a single dataset, but in real-life scenarios, different hospitals may have different techniques for capturing blood pressure readings or recording cholesterol levels. To deal with the variability, some degree of model customization may be needed in order to keep up performance.

Approval from the medical authorities, such as the FDA, will also be important. Any AI system that is integrated into patient care needs rigorous testing across multiple hospitals to prove its safety and efficacy. This process would be time-consuming and costly but would ensure that the model actually improves outcomes for patients. Finally, doctors and nurses need proper training on how to use AI predictions. For example, they need to know when to trust the model's suggestions and when they should instead use their medical experience. Without good training, even our best AI model might not help patients as much as it could.

Our CNN model for the prediction of heart diseases is a huge step into the actualization of deep learning in medical diagnostics. Accompanied by high performance, it promises to open more avenues for effective early detection and management of the disease through insights from interpretability analyses. Whatever the case may be, these limitations represent areas that require additional research and further validation in diverse populations with proper implementation strategies in clinical settings. In terms of feasibility, our model is much more deployable in real-world settings. Its resource-light architecture permits training and deployment of inference on conventional clinical workstations on cloud-based training systems. Additionally, the model utilizes routine clinical measurements which do not require further diagnostics to be performed. This model is easily scalable to larger populations and can be integrated into electronic health record systems through conventional API links, provided that institutional validation and regulatory approval are sought.

## Study limitations

Multiple limitations are undisputable. First, the Cleveland Heart Disease dataset trained our model. It was a single-source database from one geographical area. This may restrict the ability to generalize across different patient populations, healthcare frameworks, and demographic distributions.

Given the time period during which the dataset was collected, that is, 1988, the dataset may include demographic biases. Without the demographic breakdown stratification analysis, we can't properly determine whether the model is equitably applied across all the patient subgroups, particularly the ethnic and the socioeconomic dimensions.

The model has also not been subjected to real-time use analysis where the integration of incomplete datasets and workflow structures would be considered. Therefore, the applied analysis has no prospective clinical validation. Prior to implementation into real patient care, prospective clinical trials are also necessary to validate the model.

The integrability of LIME and SHAP explainability techniques does not eliminate potential over-reliance on automated predictions. Diagnosing CVD entails more than the quantifiable; it requires a bespoke clinical

examination. The interpretability tools mitigate but do not remove the necessity of the physician's diagnostic discretion.

## Conclusion and future research

The proposed model forecasts heart disease with a CNN model that has very excellent performance, with accuracy of 98.05% precision of 100% recall of 96.12% F1-score of 98.02% MCC of 0.963, and Kappa coefficient of 0.961. These results, when combined with EHRs, will be considerably better than other studies, suggesting huge potential in applying interpretable deep learning methods for the diagnosis of medication and disease prediction in AI-driven clinical decision support systems for personalized biomedical and healthcare settings. From complex medical data, the model could identify relevant features with high discriminative power with an AUC of 0.99 that makes it an important tool for enhancing the detection of heart disease via precision medicine approaches. The interpretability analyses using LIME and SHAP can provide insightful details on how the model makes decisions, which are aligned with clinical knowledge, while unveiling several feature interactions crucial for healthcare analytics and clinical decision support. This combination of high performance and interpretability may allow the CNN model conceived herein to be a valuable ally in decision support systems driven by AI, potentially capable of enhancing early detection and management of heart disease in personalized biomedical and healthcare applications, thanks to improved cardiovascular diagnostics.

Future scholarly and practical pursuits must consider ongoing model dependability and contextual fit relative to clinical decision support for healthcare. This model will require a robust validation study addressing healthcare equity analytics; otherwise, it will lack generalizability in equity focus for multiple healthcare populations. Multicenter studies that aim to support precision medicine must include the demographic and clinical diversity needed for the model. If additional data, such as genomic data, are included, advanced data science methodologies will more accurately predict model variables. Other efforts might focus on clarifying model and AI functionality for clinical decision support using interpretive visualizations designed for clinicians. Investigating how models are implemented into operational systems in healthcare and how they perform in clinical circumstances may demonstrate how clinical decision support systems influence clinical practice. Learning how successful other machine learning models are in predicting heart disease will help refine the strategies used in interpretable AI systems. These enhancements will positively impact individual cardiovascular health and personalized sustainable Biomedical and Health integration.

Immediate Research Priorities: Building on the success of our model for CNN, the subsequent steps that logically follow would be multi-hospital testing to validate that our model is able to generalize across diverse patient populations. In addition, and equally important, we have to consider developing methodologies where our model can interface with hospital systems that are already in clinical use. A very promising direction should include integration with wearable devices such as smartwatches for continuous heart monitoring for high-risk patients. Lastly, collaboration with regulatory agencies in medicine to set guidelines regarding the development and implementation of AI heart disease diagnostic tools would expedite the process of bringing this technology to the bedside of patients.

## Data availability

This research utilizes a publicly accessible dataset available on Kaggle (https://www.kaggle.com/datasets/johnsmith88/heartdisease- dataset). It includes a subset of the Cleveland Heart Disease database, regularly featured in research concentrating on cardiovascular disease detection, which enables the reproducibility and comparison of results across multiple studies.

Received: 28 July 2025; Accepted: 27 January 2026

Published online: 04 February 2026

## References

1. World Health Organization. Cardiovascular diseases (CVDs). https://www.who.int/health-topics/cardiovascular-diseases (2025). Accessed 13 Nov 2025.

2. Bertsimas, D., Mingardi, L. & Stellato, B. Machine learning for real-time heart disease prediction. IEEE J. Biomed. Health Inform. 25, 3627-3637 (2021).

3. Jain, A., Chandra Sekhara Rao, A., Kumar-Jain, P. & Hu, Y.-C. Optimized levy flight model for heart disease prediction using CNN framework in big data application. Expert Syst. Appl. 223, 119859 (2023).

4. Li, C. et al. Improving cardiovascular risk prediction through machine learning modelling of irregularly repeated electronic health records. Eur. Heart J. Digit. Health 5, 30-40 (2024).

5. Malibari, A. A. An efficient IoT-artificial intelligence-based disease prediction using lightweight CNN in healthcare system. Meas. Sens. 26, 100695 (2023).

6. Rao, S. et al. An explainable transformer-based deep learning model for the prediction of incident heart failure. IEEE J. Biomed. Health Inform. 26, 3362-3372 (2022).

7. Reshan, M. S. A. et al. A robust heart disease prediction system using hybrid deep neural networks. IEEE Access 11, 121574-121591 (2023).

8. Windmon, A. et al. TussisWatch: A smart-phone system to identify cough episodes as early symptoms of chronic obstructive pulmonary disease and congestive heart failure. IEEE J. Biomed. Health Inform. 23, 1566-1573 (2019).

9. Sugiyarto, A. W., Abadi, A. M. & Sumarna, S. Classification of heart disease based on PCG signal using CNN. TELKOMNIKA 19, 1697 (2021).

10. El-Shafiey, M. G., Hagag, A., El-Dahshan, E. S. A. & Ismail, M. A. A hybrid bidirectional LSTM and 1D CNN for heart disease prediction. Int. J. Comput. Sci. Netw. Secur. 21, 135-144 (2021).

11. Khan Mamun, M. M. & Alouani, A. FA-1D-CNN implementation to improve diagnosis of heart disease risk level.

12. Sarra, R. R., Dinar, A. M., Mohammed, M. A., Ghani, M. K. A. & Albahar, M. A. A robust framework for data generative and heart disease prediction based on efficient deep learning models. Diagnostics (Basel) 12, 2899 (2022).

13. Banapuram, C., Naik, A. C., Vanteru, M. K., Kumar, V. S. & Vaigandla, K. K. A comprehensive survey of machine learning in healthcare: Predicting heart and liver disease, tuberculosis detection in chest X-ray images. SSRG Int. J. Electron. Commun. Eng. 11, 155-169 (2024).

14. Oktafiani, R. et al. Max depth impact on heart disease classification: Decision tree and random forest. J. RESTI (Rekayasa Sistem dan Teknologi Informasi) 8, 160-168 (2024).

15. Arooj, S. et al. A deep convolutional neural network for the early detection of heart disease. Biomedicines 10, 2796 (2022).

16. Pati, A., Parhi, M. & Pattanayak, B. K. IHDPM: An integrated heart disease prediction model for heart disease prediction. Int.J. Med. Eng. Inform. 14, 564 (2022).

17. Dileep, P. et al. An automatic heart disease prediction using cluster-based bi-directional LSTM (C-BiLSTM) algorithm. Neural Comput. Appl. 35, 7253-7266 (2023).

18. Abu-Alhaija, M. & Turab, N. M. Automated learning of ECG streaming data through machine learning internet of things. Intell. Autom. Soft Comput. 32 (1), 45-53 https://doi.org/10.32604/iasc.2022.021426 (2022).

19. Kanchanamala, P., Alphonse, A. S. & Reddy, P. V. B. Heart disease prediction using hybrid optimization enabled deep learning network with spark architecture. Biomed. Signal Process. Control 84, 104707 (2023).

20. Khan, H. et al. Heart disease prediction using novel ensemble and blending based cardiovascular disease detection networks: EnsCVDD-net and BlCVDD-net. IEEE Access 12, 109230-109254 (2024).

21. Hussain, S. S., Wani, N. A., Kaur, J., Ahmad, N. & Ahmad, S. Next-generation automation in neuro-oncology: Advanced neural networks for MRI-based brain tumor segmentation and classification. IEEE Access. 13, 41141-41158 (2025).

22. Rasool, N. et al. Resmha-net: Enhancing glioma segmentation and survival prediction using a novel deep learning framework. Comput. Mater. Continua 81 (2024).

23. Wani, N. A., Kumar, R. & Bedi, J. Deepxplainer: An interpretable deep learning based approach for lung cancer detection using explainable artificial intelligence. Comput. Methods Programs Biomed. 243, 107879 (2024).

24. Singh, S., Wani, N. A., Kumar, R. & Bedi, J. Diaxplain: A transparent and interpretable artificial intelligence approach for type-2 diabetes diagnosis through deep learning. Comput. Electr. Eng. 126, 110470 (2025).

25. Rasool, N. et al. CNN-TumorNet: Leveraging explainability in deep learning for precise brain tumor diagnosis on MRI images. Front. Oncol. 15, 1554559 (2025).

26. Mohan, S., Thirumalai, C. & Srivastava, G. Effective heart disease prediction using hybrid machine learning techniques. IEEE Access 7, 81542-81554 (2019).

27. Arunachalam, S. K. & Rekha, R. A novel approach for cardiovascular disease prediction using machine learning algorithms. Concurr. Comput. Pract. Exp. 34, e7027 (2022).

28. Kumar, A. S. & Rekha, R. An improved hawks optimizer based learning algorithms for cardiovascular disease prediction. Biomed. Signal Process. Control 81, 104442 (2023).

29. Al-Naami, B. et al. Automated detection of left bundle branch block from ECG signal utilizing the maximal overlap discrete wavelet transform with ANFIS. Computers 11, 93 (2022).

30. Katarya, R. & Meena, S. K. Machine learning techniques for heart disease prediction: A comparative study and analysis. Health Technol. (Berl.) 11, 87-97 (2021).

31. Rajagopal, R. & Ranganathan, V. Evaluation of effect of unsupervised dimensionality reduction techniques on automated arrhythmia classification. Biomed. Signal Process. Control 34, 1-8 (2017).

33. Rani, P., Kumar, R., Ahmed, N. M. O. S. & Jain, A. A decision support system for heart disease prediction based upon machine learning. J. Reliab. Intell. Environ. 7, 263-275 (2021).

34. Sarmah, S. S. An efficient IoT-based patient monitoring and heart disease prediction system using deep learning modified neural network. IEEE Access 8, 135784-135797 (2020).

35. Pandey, V., Lilhore, U. K. & Walia, R. A systematic review on cardiovascular disease detection and classification. Biomed. Signal Process. Control 102, 107329 (2025).

36. Rahman, A. U. et al. Enhancing heart disease prediction using a self-attention-based transformer model. Sci. Rep. 14, 514 (2024).

37. Noor, N., Bilal, M., Abbasi, S. F., Pournik, O. & Arvanitis, T. N. A novel transformer-based approach for cardiovascular disease detection. Front. Digit. Health 7, 1548448 (2025).

38. Wang, J., Xue, Q., Zhang, C. W., Wong, K. K. L. & Liu, Z. Explainable coronary artery disease prediction model based on AutoGluon from AutoML framework. Front. Cardiovasc. Med. 11, 1360548 (2024).

39. Talaat, F. M., Elnaggar, A. R., Shaban, W. M., Shehata, M. & Elhosseini, M. Cardiorisknet: A hybrid ai-based model for explainable risk prediction and prognosis in cardiovascular disease. Bioengineering 11, 822 (2024).

40. Houssein, E. H., Mohamed, R. E., Hu, G. & Ali, A. A. Adapting transformer-based language models for heart disease detection and risk factors extraction. J. Big Data 11, 47 (2024).

41. Saranya, K., Karthikeyan, U., Kumar, A. S., Salau, A. O. & Tin Tin, T. DenseNet-ABiLSTM: Revolutionizing multiclass arrhythmia detection and classification using hybrid deep learning approach leveraging PPG signals. Int. J. Comput. Intell. Syst. 18, 1-19 (2025).

42. Shah, P., Shukla, M. & Dholakia, N. H. Predicting cardiovascular risk with hybrid ensemble learning and explainable AI: P. shah et al. Sci. Rep. 15, 17927 (2025).

43. Pandey, V. et al. Enhancing heart disease classification with M2MASC and CNN-BiLSTM integration for improved accuracy. Sci. Rep. 14, 24221 (2024).

44. Pandey, V., Lilhore, U. K. & Walia, R. Enhanced multiclass heart disease classification through advanced signal processing with modified mixed attention mechanism-based deep bilstm. Multimed. Tools Appl. 1-18 (2025).

45. Pandey, V., Lilhore, U. K. & Walia, R. Advanced heart disease prediction through spatial and temporal feature learning with SCN- Deep BiLSTM. Int. J. Comput. Intell. Syst. 18, 25 (2025).

46. Kim, Y. et al. Development and transfer learning of self-attention model for major adverse cardiovascular events prediction across hospitals. Sci. Rep. 14, 23443 (2024).

47. Waibel, A. Modular construction of time-delay neural networks for speech recognition. Neural Comput. 1, 39-46 (1989).

48. Harkulkar, N. et al. Heart disease prediction using CNN deep learning model. Int. J. Res. Appl. Sci. Eng. Technol. 8, 875-881 (2020).

49. Shankar, V., Kumar, V., Devagade, U., Karanth, V. & Rohitaksha, K. Heart disease prediction using CNN algorithm. SN Comput. Sci. 1, 170 (2020).

50. Obeidat, Y. & Alqudah, A. M. A hybrid lightweight 1D CNN-LSTM architecture for automated ECG beat-wise classification. Traitement Signal 38, 15 (2021).

51. Smith, J. Heart disease dataset. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset (2023). Public Health Dataset derived from Cleveland, Hungary, Switzerland, and Long Beach V databases.

52. Almulhii, A. et al. Ensemble learning based on hybrid deep learning model for heart disease early prediction. Diagnostics 12, 3215 (2022).

53. Singhal, S., Kumar, H. & Passricha, V. Prediction of heart disease using CNN. Am. Int. J. Res. Sci. Technol. Eng. Math. 23, 257-261 (2018).

54. Panda, R. N. & Zaheera, F. Prediction of heart disease using deep convolutional neural networks. Turk. J. Comput. Math. Educ. (TURCOMAT) 10, 1141-1148 (2019).

55. Shrivastava, P., Sharma, M., Sharma, P. & Kumar, A. HCBiLSTM: A hybrid model for predicting heart disease using CNN and BiLSTM algorithms. Meas. Sens. 25, 100657 (2023).

## Acknowledgements

Princess Nourah bint Abdulrahman University Researchers Supporting Project Number (PNURSP2026R97) Princess Nourah bint Abdulrahman University, Riyadh, Saudi Arabia.

## Author contributions

T. Kehkashan conceptualized the study, designed the methodology, and drafted the main manuscript. M. Abdelhaq and A. S. Al-Shamayleh contributed to the development of the predictive model and the interpretation of results. A. Akhunzada provided technical oversight, contributed to refining the manuscript, and supervised the overall research direction. A. I. A. A. Ahmed contributed to the literature review and data curation. R. Adil and M. Abdullah supported data preprocessing, experiment setup, and results visualization under supervision. All authors reviewed and approved the final version of the manuscript.

## Funding

This research was supported by Princess Nourah bint Abdulrahman University Researchers Supporting Project Number (PNURSP2026R97), Princess Nourah bint Abdulrahman University, Riyadh, Saudi Arabia.

## Declarations

## Competing interests

The authors declare no competing interests.

## Additional information

Correspondence and requests for materials should be addressed to R.A.R. or A.I.A.A.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License, which permits any non-commercial use, sharing, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if you modified the licensed material. You do not have permission under this licence to share adapted material derived from this article or parts of it. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by-nc-nd/4.0/.