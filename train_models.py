"""Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
save them to model/, write test_data.csv and print a metrics table."""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "diagnosis"  # 1 = benign, 0 = malignant

data = load_breast_cancer(as_frame=True)
df = data.frame.rename(columns={"target": TARGET})
df.to_csv("breast_cancer_full.csv", index=False)

X, y = df.drop(columns=TARGET), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=41)
X_test.assign(**{TARGET: y_test}).to_csv("test_data.csv", index=False)

# Scaling lives inside each pipeline, so the saved model handles raw features.
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=41),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=41)),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })

results = pd.DataFrame(rows).round(4)
results.to_csv("model/metrics.csv", index=False)
print(results.to_string(index=False))
