import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

TARGET = "diagnosis"
MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Diagnosis - Model Evaluation")
st.caption("Upload test data (CSV with the 30 features + `diagnosis` column; 1 = benign, 0 = malignant).")

uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
model_name = st.sidebar.selectbox("Choose a model", list(MODELS))

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. You can use `test_data.csv` from the repo.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"CSV must contain the target column `{TARGET}`.")
    st.stop()

model = joblib.load(MODELS[model_name])
X, y = df.drop(columns=TARGET), df[TARGET]
missing = set(model.feature_names_in_) - set(X.columns)
if missing:
    st.error(f"Missing feature columns: {sorted(missing)}")
    st.stop()
X = X[model.feature_names_in_]

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

st.subheader(f"Evaluation metrics - {model_name}")
metrics = {
    "Accuracy": accuracy_score(y, pred),
    "AUC": roc_auc_score(y, proba),
    "Precision": precision_score(y, pred),
    "Recall": recall_score(y, pred),
    "F1": f1_score(y, pred),
    "MCC": matthews_corrcoef(y, pred),
}
for col, (k, v) in zip(st.columns(6), metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(4))
