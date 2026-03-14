"""
SHAP Explainability Module
Compatible avec la structure réelle du projet Groupe 7
- best_model.pkl  → à la RACINE du projet
- data_clean.csv  → à la RACINE du projet
- shap.py         → dans src/
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ── Chemins corrigés selon la structure réelle ────────────────────────────────
# src/shap.py  →  '..' remonte à la racine du projet
ROOT       = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH  = os.path.join(ROOT, 'data_clean.csv')
MODEL_PATH = os.path.join(ROOT, 'best_model.pkl')
SAVE_DIR   = os.path.join(ROOT, 'models')

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Chargement ────────────────────────────────────────────────────────────────
def load_data_and_model():
    """Charge le dataset et best_model.pkl depuis la racine."""

    # Vérification explicite
    assert os.path.exists(DATA_PATH),  f"❌ Fichier introuvable : {DATA_PATH}"
    assert os.path.exists(MODEL_PATH), f"❌ Fichier introuvable : {MODEL_PATH}"

    # Dataset
    df     = pd.read_csv(DATA_PATH)
    target = df.columns[-1]
    X      = df.drop(target, axis=1)
    y      = df[target]

    feature_names = X.columns.tolist()

    # Split — mêmes paramètres que train_model.py
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèle
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé : {type(model).__name__}")

    return model, X_test.values, y_test.values, feature_names, y.unique()


# ── Génération des deux graphiques SHAP ──────────────────────────────────────
def generate_shap_plots():
    """
    Génère et sauvegarde dans models/ :
    - shap_bar.png      → Feature Importance (graphique 1)
    - shap_summary.png  → SHAP Summary Plot  (graphique 2)
    """
    print("📦 Chargement du modèle et des données...")
    model, X_test, y_test, feature_names, classes = load_data_and_model()

    print("🔍 Calcul des SHAP values sur 200 échantillons...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:200])

    # ── Graphique 1 : Bar plot (Feature Importance globale) ───────────────
    print("📊 Génération du graphique 1 : Feature Importance (bar)...")
    plt.figure(figsize=(10, 6))

    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values)

    shap.summary_plot(
        mean_shap,
        X_test[:200],
        feature_names=feature_names,
        plot_type='bar',
        show=False,
        max_display=15
    )
    plt.title('Feature Importance — Valeurs SHAP Moyennes', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance SHAP moyenne (impact sur la prédiction)', fontsize=11)
    plt.tight_layout()

    path1 = os.path.join(SAVE_DIR, 'shap_bar.png')
    plt.savefig(path1, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"✅ Sauvegardé → {path1}")

    # ── Graphique 2 : Beeswarm (Summary Plot) ─────────────────────────────
    print("📊 Génération du graphique 2 : SHAP Summary Plot (beeswarm)...")
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_test[:200],
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title('SHAP Summary Plot — Impact par Individu', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    path2 = os.path.join(SAVE_DIR, 'shap_summary.png')
    plt.savefig(path2, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close()
    print(f"✅ Sauvegardé → {path2}")

    print("\n🎉 Les deux graphiques sont générés avec succès !")
    print(f"   📁 Dossier : {os.path.abspath(SAVE_DIR)}")


# ── Explication par patient (appelé depuis app.py) ────────────────────────────
def explain_patient(X_patient, feature_names):
    """
    Calcule les SHAP values pour UN patient.

    Args:
        X_patient     : array (1, n_features)
        feature_names : list — noms des features

    Returns:
        pred_label : int — classe prédite
        shap_df    : DataFrame trié par importance SHAP
    """
    model = joblib.load(MODEL_PATH)

    pred_label = model.predict(X_patient)[0]

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_patient)

    if isinstance(shap_vals, list):
        sv = shap_vals[pred_label][0]
    else:
        sv = shap_vals[0]

    shap_df = (
        pd.DataFrame({'Feature': feature_names, 'SHAP Value': sv})
        .assign(Abs=lambda x: x['SHAP Value'].abs())
        .sort_values('Abs', ascending=False)
        .drop(columns='Abs')
        .reset_index(drop=True)
    )

    return pred_label, shap_df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    generate_shap_plots()