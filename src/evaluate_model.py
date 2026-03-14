import pandas as pd
import os 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ===============================
# Load dataset
# ===============================


data_path = os.path.join(os.path.dirname(__file__), '..', 'data' , 'data_clean.csv')

df = pd.read_csv(data_path)

target = df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]


# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# Models
# ===============================

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier()
}


# ===============================
# Evaluation
# ===============================

for name, model in models.items():

    print("\n===========================")
    print("Model:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)