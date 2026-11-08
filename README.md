# Breast Cancer Diagnosis - Classification Model Comparison

## a. Problem statement
Predict whether a breast tumour is **malignant** or **benign** from features computed on a digitised image of a fine needle aspirate (FNA) of a breast mass. Five classification models are trained and compared on the same data, and the trained models are served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic). Loaded with `sklearn.datasets.load_breast_cancer`.
- **Instances:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric, real-valued features. For each of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the dataset gives the mean, standard error and "worst" (mean of the three largest values).
- **Target:** `diagnosis`. 0 = malignant, 1 = benign (binary classification).
- **Missing values:** none.
- **Split:** 80% train / 20% test, stratified, `random_state=42`. The test split is saved as `test_data.csv` (114 rows) and is the file to upload in the app.

## c. GitHub Repository Link
<ADD YOUR REPO LINK>

Live Streamlit app: <ADD YOUR STREAMLIT APP LINK>

```
ml-assignment-2/
|-- app.py              # Streamlit app
|-- train_models.py     # trains and saves all 5 models, prints metrics
|-- requirements.txt
|-- README.md
|-- test_data.csv       # held-out test set (features + diagnosis)
|-- model/              # saved .joblib models + metrics.csv
```

Run locally:
```
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## d. Models used

Logistic Regression, KNN and SVM are wrapped in a pipeline with `StandardScaler`, so the saved model includes the scaling step. Decision Tree and Gaussian Naive Bayes use the raw features.

### Comparison table (test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Precision, Recall and F1 are for the positive class (1 = benign).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| Overall winner | |
