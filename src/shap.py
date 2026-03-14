import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── Chemins (même logique que train_model.py et evaluate_model.py) ────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data_clean.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
SAVE_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Chargement (même logique que train_model.py) ──────────────────────────────
def load_data_and_model():
    """Charge le dataset et le modèle sauvegardé par train_model.py"""

    # Dataset — même façon que train_model.py
    df = pd.read_csv(DATA_PATH)
    target = df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]

    feature_names = X.columns.tolist()

    # Split — même paramètres que train_model.py
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèle sauvegardé par train_model.py
    assert os.path.exists(MODEL_PATH), \
        "❌ best_model.pkl introuvable. Lance d'abord : python src/train_model.py"
    model = joblib.load(MODEL_PATH)

    return model, X_test.values, y_test.values, feature_names, y.unique()
# ── Plot 1 : SHAP Summary (beeswarm) ─────────────────────────────────────────
def generate_shap_plots():
    """
    Génère et sauvegarde :
    - models/shap_summary.png  (beeswarm)
    - models/shap_bar.png      (importance globale)
    """
    print("📦 Chargement du modèle et des données...")
    model, X_test, y_test, feature_names, classes = load_data_and_model()

    print("🔍 Calcul des SHAP values (200 samples)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])
# ── Beeswarm ──────────────────────────────────────────────────────────
    print("📊 Génération du SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test[:200],
        feature_names=feature_names,
        show=False,
        max_display=10
    )
    path1 = os.path.join(SAVE_DIR, 'shap_summary.png')
    plt.savefig(path1, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Sauvegardé → {path1}")


