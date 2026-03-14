code d'interface 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ═══════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="MediObesity — Analyse Clinique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════
#  CSS PERSONNALISÉ
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg: #f5f7fa;
    --card: #ffffff;
    --primary: #2E86AB;
    --secondary: #A23B72;
    --success: #06a77d;
    --warning: #f4a261;
    --danger: #e63946;
    --text: #1a2238;
    --text-light: #6b7a99;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2a3a 0%, #2c3e50 100%) !important;
}
section[data-testid="stSidebar"] * { color: #ecf0f1 !important; }

/* Cards */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border: 1px solid #eaeef2;
    transition: transform 0.2s;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.1); }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    border-left: 4px solid var(--primary);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--primary);
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Model cards */
.model-grid {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.model-card {
    flex: 1;
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    border: 2px solid #eaeef2;
    transition: all 0.2s;
}
.model-card.best {
    border-color: #06a77d;
    background: #f0fdf9;
}
.model-icon { font-size: 2rem; }
.model-name { font-weight: 700; margin: 0.5rem 0; }
.badge {
    background: #06a77d;
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
}

/* Results */
.result-box {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
}
.result-box.success { background: #e6f7f0; border: 2px solid #06a77d; }
.result-box.warning { background: #fff4e5; border: 2px solid #f4a261; }
.result-box.danger { background: #fee9e9; border: 2px solid #e63946; }

/* Buttons */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════
CLASS_NAMES = {
    0: "Poids Insuffisant",
    1: "Poids Normal",
    2: "Obésité Type I",
    3: "Obésité Type II",
    4: "Obésité Type III",
    5: "Surpoids Niveau I",
    6: "Surpoids Niveau II",
}

CLASS_INFO = {
    0: ("Poids Insuffisant",  "success", "IMC < 18.5"),
    1: ("Poids Normal",       "success", "18.5 ≤ IMC < 25"),
    2: ("Obésité Type I",     "danger",  "30 ≤ IMC < 35"),
    3: ("Obésité Type II",    "danger",  "35 ≤ IMC < 40"),
    4: ("Obésité Type III",   "danger",  "IMC ≥ 40"),
    5: ("Surpoids Niveau I",  "warning", "25 ≤ IMC < 27.5"),
    6: ("Surpoids Niveau II", "warning", "27.5 ≤ IMC < 30"),
}

CLASS_COLORS = {
    0: "#06a77d", 1: "#2E86AB", 2: "#f4a261",
    3: "#e63946", 4: "#A23B72", 5: "#f9c74f", 6: "#f3722c"
}

# Modèles disponibles (uniquement les 3 demandés)
ALGO_LIST = [
    "LightGBM Classifier",
    "Random Forest Classifier",
    "XGBoost Classifier"
]

ALGO_COLORS = {
    "LightGBM Classifier": "#06a77d",
    "Random Forest Classifier": "#2E86AB",
    "XGBoost Classifier": "#f4a261"
}

ALGO_ICONS = {
    "LightGBM Classifier": "⚡",
    "Random Forest Classifier": "🌲",
    "XGBoost Classifier": "🚀"
}

ALGO_DESC = {
    "LightGBM Classifier": "Gradient Boosting ultra-rapide, optimal sur données médicales",
    "Random Forest Classifier": "Forêt d'arbres, robuste et interprétable",
    "XGBoost Classifier": "Extreme Gradient Boosting, haute performance"
}

# Mappings pour les variables catégorielles
GENDER_MAP = {"Féminin": 0, "Masculin": 1}
BINARY_MAP = {"Non": 0, "Oui": 1}
CAEC_MAP = {"Jamais": 3, "Parfois": 2, "Fréquemment": 1, "Toujours": 0}
CALC_MAP = {"Jamais": 3, "Parfois": 2, "Fréquemment": 1, "Toujours": 0}
MTRANS_MAP = {"Automobile": 0, "Vélo": 1, "Moto": 2, "Transport en commun": 3, "Marche": 4}

# ═══════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════
@st.cache_data
def load_data():
    """Charge le dataset"""
    paths = ["data_clean.csv", "data/data_clean.csv", "../data_clean.csv"]
    for path in paths:
        try:
            df = pd.read_csv(path)
            return df
        except:
            continue
    st.error("❌ Fichier data_clean.csv introuvable")
    st.stop()

@st.cache_resource
def train_model(algo_name="LightGBM Classifier"):
    """Entraîne un modèle spécifique"""
    df = load_data()
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "LightGBM Classifier": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbose=-1
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0
        )
    }
    
    model = models[algo_name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average="weighted"),
        'precision': precision_score(y_test, y_pred, average="weighted"),
        'recall': recall_score(y_test, y_pred, average="weighted"),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        'feature_names': X.columns.tolist()
    }
    
    return model, scaler, metrics, X_test_scaled, y_test, y_pred

@st.cache_data
def compare_all_models():
    """Compare les 3 modèles"""
    results = {}
    for algo in ALGO_LIST:
        _, _, metrics, _, _, _ = train_model(algo)
        results[algo] = {
            "Accuracy": round(metrics['accuracy'] * 100, 2),
            "F1-Score": round(metrics['f1'] * 100, 2),
            "Précision": round(metrics['precision'] * 100, 2),
            "Rappel": round(metrics['recall'] * 100, 2)
        }
    return pd.DataFrame(results).T

# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div style='font-size: 3rem;'>🏥</div>
        <h2 style='color: white; margin: 0;'>MediObesity</h2>
        <p style='color: rgba(255,255,255,0.7); font-size: 0.8rem;'>Système de Diagnostic IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    pages = [
        "🏠 Tableau de bord",
        "📊 Exploration",
        "📈 Analyse statistique",
        "🤖 Entraînement",
        "⚖️ Comparaison",
        "🩺 Diagnostic individuel"
    ]
    
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    
    st.divider()
    
    st.markdown("### 🤖 Algorithme ML")
    algo = st.radio(
        "Sélectionner un modèle",
        ALGO_LIST,
        index=0,  # LightGBM par défaut
        format_func=lambda x: f"{ALGO_ICONS[x]} {x}"
    )
    
    # Badge pour LightGBM (meilleur modèle)
    if algo == "LightGBM Classifier":
        st.markdown("""
        <div style='background: rgba(6,167,125,0.2); padding: 0.5rem; border-radius: 8px; text-align: center;'>
            ⭐ <strong>Meilleur modèle</strong> sur ce dataset
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"""
    <div style='font-size: 0.8rem; color: rgba(255,255,255,0.7);'>
        <strong style='color: {ALGO_COLORS[algo]}'>{ALGO_ICONS[algo]} {algo}</strong><br>
        {ALGO_DESC[algo]}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════
df = load_data()

# ═══════════════════════════════════════════
#  PAGE 1: TABLEAU DE BORD
# ═══════════════════════════════════════════
if page == pages[0]:
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🏥 Tableau de Bord Clinique</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Analyse descriptive et visualisation des données patients
    </p>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-label'>Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        obese_count = df[df["NObeyesdad"].isin([2,3,4])].shape[0]
        obese_pct = round(obese_count / len(df) * 100, 1)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color: #e63946;'>{obese_count}</div>
            <div class='metric-label'>Cas d'obésité ({obese_pct}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        imc_mean = round((df["Weight"] / (df["Height"]**2)).mean(), 1)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{imc_mean}</div>
            <div class='metric-label'>IMC moyen</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        age_mean = round(df["Age"].mean(), 1)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{age_mean}</div>
            <div class='metric-label'>Âge moyen</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution des classes")
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = df["NObeyesdad"].value_counts().sort_index()
        colors = [CLASS_COLORS[i] for i in counts.index]
        bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=colors)
        ax.set_ylabel("Nombre de patients")
        ax.tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"{val} ({round(val/len(df)*100,1)}%)",
                   ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 👤 Répartition par genre")
        fig, ax = plt.subplots(figsize=(8, 6))
        gender_counts = df["Gender"].value_counts()
        ax.pie(gender_counts.values,
               labels=["Homme" if i==1 else "Femme" for i in gender_counts.index],
               autopct='%1.1f%%',
               colors=["#2E86AB", "#A23B72"])
        st.pyplot(fig)
    
    # Distribution IMC
    st.markdown("### 📈 Distribution de l'IMC par classe")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(7):
        imc_vals = df[df["NObeyesdad"] == i]["Weight"] / (df[df["NObeyesdad"] == i]["Height"]**2)
        ax.hist(imc_vals, bins=30, alpha=0.5, color=CLASS_COLORS[i], label=CLASS_NAMES[i])
    ax.set_xlabel("IMC (kg/m²)")
    ax.set_ylabel("Fréquence")
    ax.legend()
    st.pyplot(fig)
    
    # Aperçu des données
    st.markdown("### 📋 Aperçu du dataset")
    st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════
#  PAGE 2: EXPLORATION
# ═══════════════════════════════════════════
elif page == pages[1]:
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📊 Exploration Clinique</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Analyse détaillée des variables cliniques
    </p>
    """, unsafe_allow_html=True)
    
    # Sélection de variable
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove("NObeyesdad")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Variable 1", numeric_cols, index=2)
    with col2:
        var2 = st.selectbox("Variable 2", numeric_cols, index=3)
    
    # Scatter plot
    st.markdown(f"### Relation {var1} vs {var2}")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(7):
        subset = df[df["NObeyesdad"] == i]
        ax.scatter(subset[var1], subset[var2],
                  color=CLASS_COLORS[i], label=CLASS_NAMES[i], alpha=0.6)
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.legend()
    st.pyplot(fig)
    
    # Boxplots
    st.markdown("### Distribution par classe")
    var_box = st.selectbox("Variable à analyser", numeric_cols, key="box_var")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data_to_plot = [df[df["NObeyesdad"] == i][var_box].dropna() for i in range(7)]
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], [CLASS_COLORS[i] for i in range(7)]):
        patch.set_facecolor(color)
    
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(7)], rotation=45)
    ax.set_ylabel(var_box)
    plt.tight_layout()
    st.pyplot(fig)

# ═══════════════════════════════════════════
#  PAGE 3: ANALYSE STATISTIQUE
# ═══════════════════════════════════════════
elif page == pages[2]:
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📈 Analyse Statistique</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Corrélations et statistiques descriptives
    </p>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Corrélations", "📋 Statistiques"])
    
    with tab1:
        st.markdown("### Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Corrélation avec le diagnostic")
        target_corr = corr["NObeyesdad"].drop("NObeyesdad").sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        target_corr.plot(kind='barh', color=['#06a77d' if x>0 else '#e63946' for x in target_corr.values])
        ax.set_xlabel("Coefficient de corrélation")
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### Statistiques descriptives")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'),
                    use_container_width=True)

# ═══════════════════════════════════════════
#  PAGE 4: ENTRAÎNEMENT
# ═══════════════════════════════════════════
elif page == pages[3]:
    st.markdown(f"""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🤖 Entraînement du Modèle</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Modèle sélectionné : <strong style='color: {ALGO_COLORS[algo]}'>{ALGO_ICONS[algo]} {algo}</strong>
        { '⭐ (Meilleur modèle)' if algo == "LightGBM Classifier" else '' }
    </p>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Lancer l'entraînement", use_container_width=True):
        with st.spinner("Entraînement en cours..."):
            model, scaler, metrics, X_test, y_test, y_pred = train_model(algo)
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
        col3.metric("Précision", f"{metrics['precision']*100:.2f}%")
        col4.metric("Rappel", f"{metrics['recall']*100:.2f}%")
        
        # Matrice de confusion
        st.markdown("### Matrice de confusion")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                   xticklabels=[CLASS_NAMES[i] for i in range(7)],
                   yticklabels=[CLASS_NAMES[i] for i in range(7)],
                   cmap='Blues', ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Importance des features
        if metrics['feature_importance'] is not None:
            st.markdown("### Importance des variables")
            feat_imp = pd.DataFrame({
                'Variable': metrics['feature_names'],
                'Importance': metrics['feature_importance']
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(feat_imp['Variable'], feat_imp['Importance'],
                   color=ALGO_COLORS[algo])
            ax.set_xlabel("Importance relative")
            plt.tight_layout()
            st.pyplot(fig)

# ═══════════════════════════════════════════
#  PAGE 5: COMPARAISON
# ═══════════════════════════════════════════
elif page == pages[4]:
    st.markdown("""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>⚖️ Comparaison des Modèles</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Random Forest vs XGBoost vs LightGBM
    </p>
    """, unsafe_allow_html=True)
    
    with st.spinner("Comparaison des modèles en cours..."):
        comparison_df = compare_all_models()
    
    # Tableau comparatif
    st.markdown("### Tableau des performances")
    st.dataframe(
        comparison_df.style
            .highlight_max(color='#e6f7f0', axis=0)
            .format("{:.2f}%"),
        use_container_width=True
    )
    
    # Graphique comparatif
    st.markdown("### Comparaison visuelle")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    metrics_names = ["Accuracy", "F1-Score", "Précision", "Rappel"]
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = [comparison_df.loc[algo, metric] for algo in ALGO_LIST]
        bars = ax.bar(range(3), values, color=[ALGO_COLORS[algo] for algo in ALGO_LIST])
        ax.set_xticks(range(3))
        ax.set_xticklabels([a.replace(" Classifier", "") for a in ALGO_LIST])
        ax.set_title(metric)
        ax.set_ylim(85, 100)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Cartes des modèles
    st.markdown("### Résultats détaillés")
    cols = st.columns(3)
    best_model = comparison_df["Accuracy"].idxmax()
    
    for col, algo in zip(cols, ALGO_LIST):
        with col:
            is_best = (algo == best_model)
            st.markdown(f"""
            <div class='card' style='text-align: center; {"border: 2px solid #06a77d;" if is_best else ""}'>
                <div style='font-size: 2rem;'>{ALGO_ICONS[algo]}</div>
                <h3>{algo.replace(" Classifier", "")}</h3>
                <div style='font-size: 2rem; color: {ALGO_COLORS[algo]};'>
                    {comparison_df.loc[algo, "Accuracy"]:.1f}%
                </div>
                <div style='color: #6b7a99;'>Accuracy</div>
                <div style='margin-top: 0.5rem;'>
                    F1: {comparison_df.loc[algo, "F1-Score"]:.1f}%
                </div>
                { '<div class="badge">⭐ Meilleur modèle</div>' if is_best else '' }
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  PAGE 6: DIAGNOSTIC INDIVIDUEL
# ═══════════════════════════════════════════
elif page == pages[5]:
    st.markdown(f"""
    <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🩺 Diagnostic Individuel</h1>
    <p style='color: #6b7a99; margin-bottom: 2rem;'>
        Modèle utilisé : <strong style='color: {ALGO_COLORS[algo]}'>{ALGO_ICONS[algo]} {algo}</strong>
    </p>
    """, unsafe_allow_html=True)
    
    # Initialisation du modèle
    with st.spinner("Chargement du modèle..."):
        model, scaler, metrics, _, _, _ = train_model(algo)
    
    # Formulaire patient
    st.markdown("### 📋 Formulaire patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Données biométriques**")
        gender = st.selectbox("Genre", ["Féminin", "Masculin"])
        age = st.slider("Âge", 10, 80, 30)
        height = st.slider("Taille (m)", 1.40, 2.10, 1.70, 0.01)
        weight = st.slider("Poids (kg)", 30.0, 170.0, 70.0, 0.5)
        family = st.selectbox("Antécédents familiaux d'obésité", ["Non", "Oui"])
        
        # IMC en temps réel
        imc = weight / (height ** 2)
        if imc < 18.5:
            imc_color, imc_status = "#06a77d", "Insuffisant"
        elif imc < 25:
            imc_color, imc_status = "#2E86AB", "Normal"
        elif imc < 30:
            imc_color, imc_status = "#f4a261", "Surpoids"
        else:
            imc_color, imc_status = "#e63946", "Obésité"
        
        st.markdown(f"""
        <div style='background: white; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <div style='color: #6b7a99;'>IMC calculé</div>
            <div style='font-size: 2rem; color: {imc_color};'>{imc:.1f}</div>
            <div style='color: {imc_color};'>{imc_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Habitudes alimentaires**")
        favc = st.selectbox("Aliments caloriques (FAVC)", ["Non", "Oui"])
        fcvc = st.slider("Fréquence légumes (FCVC)", 1.0, 3.0, 2.0, 0.1)
        ncp = st.slider("Repas principaux/jour", 1.0, 4.0, 3.0, 0.5)
        caec = st.selectbox("Grignotage (CAEC)", ["Jamais", "Parfois", "Fréquemment", "Toujours"])
        calc = st.selectbox("Alcool (CALC)", ["Jamais", "Parfois", "Fréquemment", "Toujours"])
    
    with col3:
        st.markdown("**Mode de vie**")
        smoke = st.selectbox("Tabagisme", ["Non", "Oui"])
        ch2o = st.slider("Eau (L/jour)", 1.0, 3.0, 2.0, 0.1)
        scc = st.selectbox("Surveillance calories", ["Non", "Oui"])
        faf = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, 1.0, 0.1)
        tue = st.slider("Temps écran (heures/jour)", 0.0, 2.0, 1.0, 0.1)
        mtrans = st.selectbox("Transport", list(MTRANS_MAP.keys()))
    
    # Diagnostic
    if st.button("🔍 Lancer le diagnostic", use_container_width=True):
        # Préparer les données
        input_data = pd.DataFrame([{
            "Gender": GENDER_MAP[gender],
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history_with_overweight": BINARY_MAP[family],
            "FAVC": BINARY_MAP[favc],
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": CAEC_MAP[caec],
            "SMOKE": BINARY_MAP[smoke],
            "CH2O": ch2o,
            "SCC": BINARY_MAP[scc],
            "FAF": faf,
            "TUE": tue,
            "CALC": CALC_MAP[calc],
            "MTRANS": MTRANS_MAP[mtrans]
        }])
        
        # Standardiser et prédire
        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
        else:
            probabilities = None
        
        # Afficher le résultat
        class_name, class_type, imc_range = CLASS_INFO[prediction]
        
        emoji = "✅" if class_type == "success" else "⚠️" if class_type == "warning" else "🚨"
        
        st.markdown(f"""
        <div class='result-box {class_type}'>
            <div style='font-size: 3rem;'>{emoji}</div>
            <h2>{class_name}</h2>
            <p><strong>IMC : {imc:.1f}</strong> | {imc_range}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilités
        if probabilities is not None:
            st.markdown("### Probabilités par classe")
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = [CLASS_COLORS[i] for i in range(7)]
            bars = ax.bar([CLASS_NAMES[i] for i in range(7)], probabilities, color=colors)
            
            #