# Breast Cancer Diagnosis - Classification Model Comparison

## a. Problem statement
Predict whether a breast tumour is **malignant** or **benign** from features computed on a digitised image of a fine needle aspirate (FNA) of a breast mass. Five classification models are trained and compared on the same data, and the trained models are served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic). Loaded with `sklearn.datasets.load_breast_cancer`.
- **Instances:** 569 (357 benign, 212 malignant)
- **Features:** 30 numeric, real-valued features. For each of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) the dataset gives the mean, standard error and "worst" (mean of the three largest values).
- **Target:** `diagnosis`. 0 = malignant, 1 = benign (binary classification).
- **Missing values:** none.
- **Split:** 80% train / 20% test, stratified, `random_state=41`. The test split is saved as `test_data.csv` (114 rows) and is the file to upload in the app.

## c. GitHub Repository Link
https://github.com/2026ak05541-dev/ml-assignment-2

Live Streamlit app: https://ml-assignment-2-gerbbw8udepksatgqot6pm.streamlit.app/

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
| Logistic Regression | 0.9912 | 0.9888 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |
| Decision Tree | 0.9561 | 0.9420 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| KNN | 0.9912 | 0.9868 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |
| Naive Bayes | 0.9386 | 0.9917 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9912 | 0.9897 | 0.9863 | 1.0000 | 0.9931 | 0.9812 |

Precision, Recall and F1 are for the positive class (1 = benign).

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression did well, 113 out of 114 test samples correct (accuracy 0.9912, MCC 0.9812). Recall is 1.0, so it didn't miss a single benign case. The one mistake was a malignant tumour predicted as benign, which is the worse kind of error for a medical problem. I think it works this well because most of the features separate the two classes almost linearly once they are scaled. |
| Decision Tree | The Decision Tree provided the lowest AUC of all five (0.9420). It made 5 mistakes (accuracy 0.9561, MCC 0.9054). I limited the depth to 5 to stop it from overfitting, but a single tree still makes hard yes/no splits on one feature at a time, so it can't capture the smooth boundary as well as the other models. Its probability scores are also coarse, which is why its AUC is so much lower. |
| KNN | The KNN Got exactly the same results as Logistic Regression and SVM (accuracy 0.9912, recall 1.0, MCC 0.9812). Scaling was really important here, since KNN uses distances and features like area are on a much bigger scale than smoothness. With k = 7 it was stable. AUC (0.9868) was slightly lower than LR and SVM. |
| Naive Bayes | Naive Bayes has the highest AUC (0.9917), so it ranks patients by risk very well, but it has the lowest accuracy (0.9386) and MCC (0.8676) with 7 mistakes. My guess is that the "features are independent" assumption is broken in this dataset, because radius, perimeter and area are strongly correlated. That hurts the final class predictions even though the ranking is still good. |
| Support Vector Machine | SVM Tied with Logistic Regression and KNN on accuracy, precision, recall, F1 and MCC, and had the best AUC of those three (0.9897). With the RBF kernel and scaled features it found a clean boundary between the classes. Training was slower because of `probability=True`, but on 569 rows it didn't really matter. |
| Overall winner | I would choose **Logistic Regression**. LR, KNN and SVM gave identical predictions on the test set, and SVM's AUC is only 0.0009 higher, which is basically noise for 114 samples. Logistic Regression is simpler, faster, and easy to explain (we can see which features push towards malignant), which matters for a medical use case. SVM would be my close second. |
