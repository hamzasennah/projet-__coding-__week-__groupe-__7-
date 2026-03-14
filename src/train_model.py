import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ===============================
# 1 Load dataset
# ===============================

data_path = os.path.join(os.path.dirname(__file__), '..', 'data' , 'data_clean.csv')

df = pd.read_csv(data_path)

print("Dataset loaded")
print("Columns:", df.columns)


# ===============================
# 2 Target (last column)
# ===============================

target = df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]


# ===============================
# 3 Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# 4 Models
# ===============================

models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier()
}


best_model = None
best_score = 0
best_name = ""


# ===============================
# 5 Training models
# ===============================

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr"
    )

    print("ROC-AUC:", roc_auc)

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model
        best_name = name


# ===============================
# 6 Save best model
# ===============================

joblib.dump(best_model, "best_model.pkl")

print("\nBest model:", best_name)
print("Best ROC-AUC:", best_score)
print("Model saved as best_model.pkl")