import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
#  CONFIG
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="MediObesity — Analyse Clinique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════
#  CSS MÉDICAL PREMIUM
# ═══════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    --bg:         #eef2f7;
    --card:       #ffffff;
    --blue:       #0b5394;
    --blue-mid:   #1565c0;
    --blue-lt:    #1976d2;
    --teal:       #00838f;
    --teal-lt:    #4dd0e1;
    --green:      #2e7d32;
    --green-lt:   #43a047;
    --orange:     #e65100;
    --orange-lt:  #fb8c00;
    --red:        #b71c1c;
    --red-lt:     #e53935;
    --text:       #0d1b2a;
    --text-muted: #546e7a;
    --border:     #cfd8dc;
    --shadow:     rgba(11,83,148,.12);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071e3d 0%, #0b3566 55%, #0b5394 100%) !important;
    border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] * { color: #dceefb !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: .9rem !important;
    padding: .45rem .6rem !important;
    border-radius: 6px;
    transition: background .2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.08) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }

/* ── Main area ── */
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Banner ── */
.med-banner {
    background: linear-gradient(135deg, #071e3d 0%, #0b5394 55%, #1565c0 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(7,30,61,.35);
}
.med-banner::before {
    content: '⚕';
    position: absolute; right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 9rem; opacity: .06;
}
.med-banner::after {
    content: '';
    position: absolute; top: -40px; left: -40px;
    width: 240px; height: 240px;
    background: rgba(255,255,255,.04);
    border-radius: 50%;
}
.banner-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 .5rem;
    letter-spacing: -.02em;
    text-shadow: 0 2px 12px rgba(0,0,0,.2);
}
.banner-sub {
    font-size: 1rem;
    opacity: .8;
    margin: 0;
    font-weight: 400;
    letter-spacing: .01em;
}
.banner-badge {
    display: inline-block;
    background: rgba(255,255,255,.15);
    border: 1px solid rgba(255,255,255,.3);
    border-radius: 20px;
    padding: .28rem .85rem;
    font-size: .76rem;
    font-weight: 600;
    margin-top: .85rem;
    margin-right: .45rem;
    letter-spacing: .03em;
    backdrop-filter: blur(4px);
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 1.1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 130px;
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border-top: 5px solid var(--blue);
    box-shadow: 0 3px 16px var(--shadow);
    transition: transform .22s, box-shadow .22s;
    position: relative; overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute; bottom: -20px; right: -20px;
    width: 80px; height: 80px;
    background: rgba(11,83,148,.04);
    border-radius: 50%;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 8px 28px rgba(11,83,148,.2); }
.kpi-card.green  { border-top-color: var(--green-lt); }
.kpi-card.orange { border-top-color: var(--orange-lt); }
.kpi-card.red    { border-top-color: var(--red-lt); }
.kpi-card.teal   { border-top-color: var(--teal); }
.kpi-num  {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem; font-weight: 800; color: var(--blue);
    line-height: 1;
}
.kpi-card.green  .kpi-num  { color: var(--green-lt); }
.kpi-card.orange .kpi-num  { color: var(--orange-lt); }
.kpi-card.red    .kpi-num  { color: var(--red-lt); }
.kpi-card.teal   .kpi-num  { color: var(--teal); }
.kpi-lbl {
    font-size: .76rem; color: var(--text-muted);
    margin-top: 6px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .06em;
}

/* ── Section header ── */
.sec-head {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem; font-weight: 700;
    color: var(--blue-mid);
    border-bottom: 2px solid var(--teal-lt);
    padding-bottom: .5rem;
    margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: .5rem;
}

/* ── Info box ── */
.info-panel {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 5px solid var(--blue);
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    margin-bottom: 1rem;
}
.info-panel.green  { border-left-color: var(--green-lt); }
.info-panel.orange { border-left-color: var(--orange-lt); }
.info-panel.red    { border-left-color: var(--red-lt); }
.info-panel.teal   { border-left-color: var(--teal); }
.info-panel-title { font-weight: 700; color: var(--blue-mid); font-size: .95rem; margin-bottom: .35rem; }
.info-panel.green  .info-panel-title { color: var(--green); }
.info-panel.orange .info-panel-title { color: var(--orange); }
.info-panel.red    .info-panel-title { color: var(--red); }

/* ── Model cards ── */
.model-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem;
    border: 2px solid var(--border);
    text-align: center;
    transition: border-color .2s, transform .2s, box-shadow .2s;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    height: 100%;
}
.model-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px var(--shadow); }
.model-card.best  { border-color: var(--green-lt); background: linear-gradient(135deg,#f1f8f1,#e8f5e9); }
.model-icon { font-size: 2.2rem; margin-bottom: .5rem; }
.model-name { font-weight: 800; font-size: .95rem; color: var(--text); }
.model-desc { font-size: .79rem; color: var(--text-muted); margin-top: .35rem; line-height: 1.5; }
.best-pill {
    display: inline-block;
    background: var(--green-lt);
    color: white; border-radius: 20px;
    padding: .18rem .75rem;
    font-size: .72rem; font-weight: 700; margin-top: .5rem;
}

/* ── Result box ── */
.result-box {
    border-radius: 16px; padding: 2.2rem;
    text-align: center; margin: 1.2rem 0;
    box-shadow: 0 6px 28px rgba(0,0,0,.1);
}
.result-box.green  { background: linear-gradient(135deg,#e8f5e9,#c8e6c9); border: 2px solid var(--green-lt); }
.result-box.orange { background: linear-gradient(135deg,#fff3e0,#ffe0b2); border: 2px solid var(--orange-lt); }
.result-box.red    { background: linear-gradient(135deg,#ffebee,#ffcdd2); border: 2px solid var(--red-lt); }
.result-title { font-family:'Playfair Display',serif; font-size: 1.75rem; font-weight: 800; }
.result-box.green  .result-title { color: var(--green); }
.result-box.orange .result-title { color: var(--orange); }
.result-box.red    .result-title { color: var(--red); }
.result-imc { font-size: 1rem; color: var(--text-muted); margin-top: .5rem; }
.result-desc { font-size: .9rem; color: var(--text); margin-top: .7rem; font-weight: 500; }

/* ── Rec card ── */
.rec-card {
    background: white; border-radius: 12px;
    padding: 1.1rem 1.4rem; margin: .6rem 0;
    display: flex; align-items: flex-start; gap: 1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
    border-left: 5px solid var(--blue);
    transition: transform .18s;
}
.rec-card:hover { transform: translateX(4px); }
.rec-card.green  { border-left-color: var(--green-lt); }
.rec-card.orange { border-left-color: var(--orange-lt); }
.rec-card.red    { border-left-color: var(--red-lt); }
.rec-icon { font-size: 1.5rem; flex-shrink: 0; margin-top: .1rem; }
.rec-title { font-weight: 700; font-size: .92rem; color: var(--text); }
.rec-text  { font-size: .83rem; color: var(--text-muted); margin-top: .25rem; line-height: 1.5; }

/* ── Chip ── */
.chip {
    display: inline-block;
    background: #e3f2fd; color: var(--blue-mid);
    border-radius: 20px; padding: .28rem .9rem;
    font-size: .79rem; font-weight: 700; margin: .2rem;
    border: 1px solid #bbdefb;
    letter-spacing: .02em;
}

/* ── IMC live card ── */
.imc-card {
    border-radius: 12px; padding: 1.1rem 1rem;
    text-align: center; margin-top: .7rem;
    box-shadow: 0 3px 14px rgba(0,0,0,.07);
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #0b5394, #1565c0) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: .65rem 2.2rem !important;
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
    letter-spacing: .04em !important; font-size: .92rem !important;
    box-shadow: 0 5px 18px rgba(11,83,148,.35) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    opacity: .9 !important; transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(11,83,148,.45) !important;
}
div[data-testid="metric-container"] {
    background: white; border-radius: 12px;
    padding: 1.1rem; box-shadow: 0 2px 12px rgba(0,0,0,.06);
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600; letter-spacing: .02em;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  CONSTANTS — 3 ALGORITHMS ONLY
# ═══════════════════════════════════════════
MED_PALETTE = ["#0b5394","#1565c0","#00838f","#2e7d32","#e65100","#b71c1c","#4a148c","#00695c"]

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
    0: ("Poids Insuffisant",  "green",  "IMC < 18.5",     "Risque de carences nutritionnelles. Consultez un médecin."),
    1: ("Poids Normal",       "green",  "18.5 ≤ IMC < 25","Profil clinique sain. Maintenir les bonnes habitudes."),
    2: ("Obésité Type I",     "red",    "30 ≤ IMC < 35",  "Risque cardiovasculaire modéré. Suivi médical recommandé."),
    3: ("Obésité Type II",    "red",    "35 ≤ IMC < 40",  "Risque cardiovasculaire élevé. Consultation spécialiste."),
    4: ("Obésité Type III",   "red",    "IMC ≥ 40",       "Obésité morbide. Prise en charge médicale urgente."),
    5: ("Surpoids Niveau I",  "orange", "25 ≤ IMC < 27.5","Surveiller l'alimentation. Augmenter l'activité physique."),
    6: ("Surpoids Niveau II", "orange", "27.5 ≤ IMC < 30","Consulter un diététicien. Bilan lipidique conseillé."),
}
CLASS_COLORS_HEX = {
    0:"#2e7d32", 1:"#0b5394", 2:"#fb8c00", 3:"#e53935",
    4:"#6a1a9a", 5:"#f9c74f", 6:"#ef6c00"
}

# ── 3 MODELS ──
ALGO_LIST   = ["LightGBM Classifier", "Random Forest Classifier", "XGBoost Classifier"]
ALGO_COLORS = {
    "LightGBM Classifier":      "#2e7d32",
    "Random Forest Classifier": "#0b5394",
    "XGBoost Classifier":       "#e65100",
}
ALGO_ICONS  = {
    "LightGBM Classifier":      "⚡",
    "Random Forest Classifier": "🌲",
    "XGBoost Classifier":       "🚀",
}
ALGO_DESC   = {
    "LightGBM Classifier":      "Gradient Boosting ultra-rapide, optimal sur données médicales tabulaires. Meilleure précision diagnostique.",
    "Random Forest Classifier": "Forêt d'arbres de décision, robuste et interprétable cliniquement. Idéal pour l'explicabilité.",
    "XGBoost Classifier":       "Extreme Gradient Boosting, très performant sur données structurées. Excellent équilibre vitesse/précision.",
}

GENDER_MAP = {"Féminin": 0, "Masculin": 1}
BINARY_MAP = {"Non": 0, "Oui": 1}
CAEC_MAP   = {"Jamais": 3, "Parfois": 2, "Fréquemment": 1, "Toujours": 0}
CALC_MAP   = {"Jamais": 3, "Parfois": 2, "Fréquemment": 1, "Toujours": 0}
MTRANS_MAP = {"Automobile": 0, "Vélo": 1, "Moto": 2, "Transport en commun": 3, "Marche": 4}

PAGES = [
    "🏥  Tableau de bord",
    "📊  Exploration clinique",
    "📈  Analyse statistique",
    "🤖  Entraînement & Évaluation",
    "⚖️  Comparaison des modèles",
    "🩺  Diagnostic individuel",
]

# ═══════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════
def med_fig():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cfd8dc",  "axes.labelcolor": "#0d1b2a",
        "xtick.color": "#546e7a",     "ytick.color": "#546e7a",
        "text.color": "#0d1b2a",      "grid.color": "#eceff1",
        "legend.facecolor": "white",  "legend.edgecolor": "#cfd8dc",
        "font.family": "DejaVu Sans",
    })

@st.cache_data
def load_data():
    for p in ["data_clean.csv", "data/data_clean.csv", "../data_clean.csv"]:
        try:
            df = pd.read_csv(p)
            return df
        except:
            pass
    st.error("❌ Fichier data_clean.csv introuvable. Placez-le dans le même dossier que app.py.")
    st.stop()

@st.cache_resource
def train_model(algo="LightGBM Classifier"):
    df  = load_data()
    X   = df.drop("NObeyesdad", axis=1)
    y   = df["NObeyesdad"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    sc  = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    models = {
        "LightGBM Classifier":      LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=63, random_state=42, verbose=-1),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost Classifier":       XGBClassifier(n_estimators=200, learning_rate=.05, max_depth=6, random_state=42, eval_metric="mlogloss", verbosity=0),
    }
    clf  = models[algo]
    clf.fit(Xtrs, ytr)
    yp   = clf.predict(Xtes)
    return (clf, sc, X.columns.tolist(),
            accuracy_score(yte, yp), f1_score(yte, yp, average="weighted"),
            precision_score(yte, yp, average="weighted"), recall_score(yte, yp, average="weighted"),
            confusion_matrix(yte, yp), classification_report(yte, yp, output_dict=True),
            Xtes, yte, yp)

@st.cache_data
def compare_models():
    rows = {}
    for a in ALGO_LIST:
        r = train_model(a)
        rows[a] = {"Accuracy": round(r[3]*100, 2), "F1-Score": round(r[4]*100, 2),
                   "Précision": round(r[5]*100, 2), "Rappel": round(r[6]*100, 2)}
    return pd.DataFrame(rows).T

# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.6rem 0 1rem'>
        <div style='font-size:3.2rem'>🏥</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.4rem;
                    font-weight:800;color:white;margin-top:.4rem;letter-spacing:-.01em'>
            MediObesity
        </div>
        <div style='font-size:.76rem;color:rgba(255,255,255,.55);margin-top:5px;
                    font-weight:500;letter-spacing:.08em;text-transform:uppercase'>
            Système de Diagnostic IA
        </div>
        <div style='font-size:.7rem;color:rgba(255,255,255,.35);margin-top:3px'>
            Groupe 7 · Projet 2
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()
    st.markdown("""<div style='font-size:.76rem;font-weight:700;
        color:rgba(255,255,255,.6);margin-bottom:.6rem;letter-spacing:.08em;
        text-transform:uppercase'>Algorithme ML</div>""", unsafe_allow_html=True)

    algo = st.radio("algo", ALGO_LIST, index=0,
        label_visibility="collapsed",
        format_func=lambda x: ("⭐ " if x == "LightGBM Classifier" else ALGO_ICONS[x]+" ") + x
    )

    st.markdown(f"""
    <div style='background:rgba(46,125,50,.18);border:1px solid rgba(46,125,50,.45);
                border-radius:10px;padding:.7rem .9rem;margin-top:.7rem;font-size:.79rem;
                color:#a5d6a7;line-height:1.5'>
        ⭐ <strong>LightGBM</strong> est le modèle le plus performant sur ce dataset médical.
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
    <div style='font-size:.79rem;color:rgba(255,255,255,.5);line-height:1.7;padding-bottom:1rem'>
        <span style='color:{ALGO_COLORS[algo]};font-weight:700;font-size:.85rem'>
            {ALGO_ICONS[algo]} {algo}
        </span><br>
        {ALGO_DESC[algo]}
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════
df = load_data()

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.markdown("""
    <div class='med-banner'>
        <div class='banner-title'>🏥 Tableau de Bord Clinique</div>
        <div class='banner-sub'>Système de prédiction du niveau d'obésité par apprentissage automatique</div>
        <div>
            <span class='banner-badge'>🧬 Machine Learning</span>
            <span class='banner-badge'>📋 2 087 patients</span>
            <span class='banner-badge'>🎯 7 classes d'obésité</span>
            <span class='banner-badge'>3 algorithmes comparés</span>
        </div>
    </div>""", unsafe_allow_html=True)

    imc_vals = df["Weight"] / (df["Height"]**2)
    obese    = df[df["NObeyesdad"].isin([2, 3, 4])].shape[0]

    st.markdown(f"""
    <div class='kpi-row'>
        <div class='kpi-card'>
            <div class='kpi-num'>{len(df):,}</div>
            <div class='kpi-lbl'>👤 Patients</div>
        </div>
        <div class='kpi-card teal'>
            <div class='kpi-num' style='color:#00838f'>{df.shape[1]-1}</div>
            <div class='kpi-lbl'>🔬 Variables cliniques</div>
        </div>
        <div class='kpi-card green'>
            <div class='kpi-num' style='color:#43a047'>{df['NObeyesdad'].nunique()}</div>
            <div class='kpi-lbl'>📊 Classes IMC</div>
        </div>
        <div class='kpi-card red'>
            <div class='kpi-num' style='color:#e53935'>{obese:,}</div>
            <div class='kpi-lbl'>⚠️ Cas d'obésité</div>
        </div>
        <div class='kpi-card orange'>
            <div class='kpi-num' style='color:#fb8c00'>{round(imc_vals.mean(),1)}</div>
            <div class='kpi-lbl'>📏 IMC moyen</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-num'>{round(df["Age"].mean(),1)}</div>
            <div class='kpi-lbl'>🗓️ Âge moyen</div>
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.65, 1])

    with col1:
        st.markdown("<div class='sec-head'>📊 Répartition des classes d'obésité</div>", unsafe_allow_html=True)
        med_fig()
        counts = df["NObeyesdad"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        colors_b = [CLASS_COLORS_HEX[i] for i in counts.index]
        bars = ax.barh([CLASS_NAMES[i] for i in counts.index], counts.values,
                       color=colors_b, edgecolor="none", height=.62)
        for bar, val in zip(bars, counts.values):
            pct = round(val/len(df)*100, 1)
            ax.text(val+8, bar.get_y()+bar.get_height()/2,
                    f"{val}  ({pct}%)", va="center", fontsize=9, color="#546e7a", fontweight="600")
        ax.set_xlabel("Nombre de patients", fontsize=10)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="x", alpha=.35, linestyle="--")
        ax.set_xlim(0, counts.max()*1.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec-head'>👥 Genre des patients</div>", unsafe_allow_html=True)
        med_fig()
        g = df["Gender"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 4.8))
        wedges, texts, autos = ax2.pie(
            g.values,
            labels=["Homme" if i==1 else "Femme" for i in g.index],
            autopct="%1.1f%%",
            colors=["#0b5394","#1976d2"],
            startangle=90, pctdistance=.75,
            wedgeprops={"edgecolor":"white","linewidth":3, "shadow": True},
        )
        for t in texts:  t.set_color("#0d1b2a"); t.set_fontsize(12); t.set_fontweight("600")
        for a in autos:  a.set_color("white"); a.set_fontsize(11); a.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.markdown("<div class='sec-head'>📈 Distribution de l'IMC par classe clinique</div>", unsafe_allow_html=True)
    med_fig()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    for i in range(7):
        vals = (df[df["NObeyesdad"]==i]["Weight"] / df[df["NObeyesdad"]==i]["Height"]**2).dropna()
        if len(vals) > 3:
            vals.plot.kde(ax=ax3, color=CLASS_COLORS_HEX[i], linewidth=2.5, label=CLASS_NAMES[i])
            ax3.fill_between(
                np.linspace(vals.min(), vals.max(), 200), 0,
                [ax3.lines[-1].get_ydata()[j]
                 for j in np.linspace(0, len(ax3.lines[-1].get_ydata())-1, 200, dtype=int)],
                alpha=.07, color=CLASS_COLORS_HEX[i]
            )
    ax3.set_xlabel("IMC (kg/m²)", fontsize=10)
    ax3.set_ylabel("Densité", fontsize=10)
    ax3.set_title("Distribution de l'IMC selon la classe clinique", fontsize=12, pad=10)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(alpha=.35, linestyle="--")
    ax3.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    # ── 3 Model cards ──
    st.markdown("<div class='sec-head'>🤖 Algorithmes de classification</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, a in zip([c1, c2, c3], ALGO_LIST):
        is_best = a == "LightGBM Classifier"
        col.markdown(f"""
        <div class='model-card {"best" if is_best else ""}'>
            <div class='model-icon'>{ALGO_ICONS[a]}</div>
            <div class='model-name' style='color:{ALGO_COLORS[a]}'>{a}</div>
            <div class='model-desc'>{ALGO_DESC[a]}</div>
            {"<div class='best-pill'>⭐ Meilleur modèle</div>" if is_best else ""}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>📋 Aperçu du dataset clinique</div>", unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True, height=320)

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATION CLINIQUE
# ══════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.markdown("""
    <div class='med-banner'>
        <div class='banner-title'>📊 Exploration Clinique</div>
        <div class='banner-sub'>Analyse exploratoire des variables biométriques et comportementales</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "📦 Boxplots cliniques",
                                       "🔵 Relations biométriques", "🗂️ Dataset complet"])
    num_cols = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()

    with tab1:
        st.markdown("<div class='sec-head'>📊 Distribution des variables cliniques</div>", unsafe_allow_html=True)
        chosen = st.selectbox("Sélectionner une variable clinique", num_cols)
        c1, c2 = st.columns(2)
        with c1:
            med_fig()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[chosen], bins=40, color="#0b5394", edgecolor="white", alpha=.85, linewidth=.4)
            ax.axvline(df[chosen].mean(),   color="#e53935", linestyle="--", lw=2, label=f"Moyenne : {df[chosen].mean():.2f}")
            ax.axvline(df[chosen].median(), color="#43a047", linestyle="--", lw=2, label=f"Médiane : {df[chosen].median():.2f}")
            ax.set_xlabel(chosen, fontsize=10); ax.set_ylabel("Fréquence", fontsize=10)
            ax.set_title(f"Distribution — {chosen}", fontsize=12, pad=10)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=.35, linestyle="--")
            ax.legend(fontsize=8.5)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        with c2:
            med_fig()
            fig, ax = plt.subplots(figsize=(6, 4))
            for i in range(7):
                vals = df[df["NObeyesdad"]==i][chosen].dropna()
                if len(vals) > 5:
                    vals.plot.kde(ax=ax, color=CLASS_COLORS_HEX[i], label=CLASS_NAMES[i], linewidth=2.2)
            ax.set_xlabel(chosen, fontsize=10)
            ax.set_title(f"Densité de {chosen} par classe", fontsize=12, pad=10)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", alpha=.35, linestyle="--")
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True)

        s = df[chosen]
        cc1,cc2,cc3,cc4,cc5 = st.columns(5)
        cc1.metric("Moyenne",     f"{s.mean():.3f}")
        cc2.metric("Médiane",     f"{s.median():.3f}")
        cc3.metric("Écart-type",  f"{s.std():.3f}")
        cc4.metric("Min",         f"{s.min():.3f}")
        cc5.metric("Max",         f"{s.max():.3f}")

        st.markdown("<div class='sec-head'>📋 Vue d'ensemble — toutes les variables</div>", unsafe_allow_html=True)
        med_fig()
        n_c = 4; n_r = int(np.ceil(len(num_cols)/n_c))
        fig_all, axes = plt.subplots(n_r, n_c, figsize=(14, n_r*3))
        axes = axes.flatten()
        for idx, col_n in enumerate(num_cols):
            axes[idx].hist(df[col_n], bins=30, color=MED_PALETTE[idx % len(MED_PALETTE)],
                           edgecolor="white", alpha=.85, linewidth=.4)
            axes[idx].set_title(col_n, fontsize=9, pad=4)
            axes[idx].spines[["top","right"]].set_visible(False)
            axes[idx].tick_params(labelsize=7)
            axes[idx].grid(axis="y", alpha=.3, linestyle="--")
        for j in range(len(num_cols), len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_all, use_container_width=True)

    with tab2:
        st.markdown("<div class='sec-head'>📦 Boxplots cliniques par niveau d'obésité</div>", unsafe_allow_html=True)
        bx_var = st.selectbox("Variable clinique", num_cols, key="bxv")
        med_fig()
        fig, ax = plt.subplots(figsize=(12, 5))
        for i in range(7):
            vals = df[df["NObeyesdad"]==i][bx_var]
            ax.boxplot(vals, positions=[i], widths=.62, patch_artist=True,
                       boxprops=dict(facecolor=CLASS_COLORS_HEX[i], alpha=.72),
                       medianprops=dict(color="white", linewidth=2.8),
                       whiskerprops=dict(color="#546e7a", linewidth=1.3),
                       capprops=dict(color="#546e7a", linewidth=1.3),
                       flierprops=dict(marker="o", color="#546e7a", markersize=3, alpha=.5))
        ax.set_xticks(range(7))
        ax.set_xticklabels([CLASS_NAMES[i].replace(" ", "\n") for i in range(7)], fontsize=9)
        ax.set_ylabel(bx_var, fontsize=10)
        ax.set_title(f"Distribution clinique de {bx_var} par niveau d'obésité", fontsize=12, pad=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", alpha=.35, linestyle="--")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

        st.markdown("<div class='sec-head'>📍 Taille vs Poids — cartographie clinique</div>", unsafe_allow_html=True)
        med_fig()
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax2.scatter(sub["Height"], sub["Weight"], s=28, alpha=.55,
                       color=CLASS_COLORS_HEX[i], label=CLASS_NAMES[i], edgecolors="none")
        ax2.set_xlabel("Taille (m)", fontsize=10); ax2.set_ylabel("Poids (kg)", fontsize=10)
        ax2.set_title("Cartographie biométrique — Taille vs Poids", fontsize=12, pad=10)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(alpha=.28, linestyle="--")
        ax2.legend(fontsize=7.5, ncol=2)
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

    with tab3:
        st.markdown("<div class='sec-head'>🔵 Analyse des relations entre variables</div>", unsafe_allow_html=True)
        sx1, sx2 = st.columns(2)
        xv = sx1.selectbox("Axe X", num_cols, index=2)
        yv = sx2.selectbox("Axe Y", num_cols, index=3)
        med_fig()
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax3.scatter(sub[xv], sub[yv], s=24, alpha=.55,
                       color=CLASS_COLORS_HEX[i], label=CLASS_NAMES[i], edgecolors="none")
        ax3.set_xlabel(xv, fontsize=10); ax3.set_ylabel(yv, fontsize=10)
        ax3.set_title(f"Relation clinique : {xv} vs {yv}", fontsize=12, pad=10)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(alpha=.28, linestyle="--")
        ax3.legend(fontsize=7.5, ncol=2)
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

        st.markdown("<div class='sec-head'>📊 Âge vs Poids — profil épidémiologique</div>", unsafe_allow_html=True)
        med_fig()
        fig4, ax4 = plt.subplots(figsize=(11, 5))
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax4.scatter(sub["Age"], sub["Weight"], s=22, alpha=.5,
                       color=CLASS_COLORS_HEX[i], label=CLASS_NAMES[i], edgecolors="none")
        ax4.set_xlabel("Âge (années)", fontsize=10); ax4.set_ylabel("Poids (kg)", fontsize=10)
        ax4.set_title("Profil épidémiologique — Âge vs Poids", fontsize=12, pad=10)
        ax4.spines[["top","right"]].set_visible(False)
        ax4.grid(alpha=.25, linestyle="--")
        ax4.legend(fontsize=7.5, ncol=4)
        plt.tight_layout(); st.pyplot(fig4, use_container_width=True)

    with tab4:
        st.markdown("<div class='sec-head'>🗂️ Dataset clinique complet</div>", unsafe_allow_html=True)
        cols_sel = st.multiselect("Colonnes à afficher", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[cols_sel], use_container_width=True, height=500)
        st.markdown(f"**{len(df):,} patients · {len(cols_sel)} colonnes sélectionnées**")

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — ANALYSE STATISTIQUE
# ══════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.markdown("""
    <div class='med-banner'>
        <div class='banner-title'>📈 Analyse Statistique</div>
        <div class='banner-sub'>Corrélations cliniques, statistiques descriptives et détection d'anomalies</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔗 Corrélations", "📋 Statistiques descriptives", "⚠️ Détection d'outliers"])

    with tab1:
        st.markdown("<div class='sec-head'>🔗 Matrice de corrélation clinique</div>", unsafe_allow_html=True)
        med_fig()
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask,
                    cmap=sns.diverging_palette(240, 10, as_cmap=True),
                    center=0, annot=True, fmt=".2f", annot_kws={"size": 8},
                    linewidths=.5, linecolor="white", cbar_kws={"shrink": .75})
        ax.set_title("Corrélations entre variables cliniques", fontsize=13, pad=12)
        plt.xticks(fontsize=8, rotation=45, ha="right")
        plt.yticks(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

        st.markdown("<div class='sec-head'>🎯 Corrélations avec le diagnostic</div>", unsafe_allow_html=True)
        target_corr = corr["NObeyesdad"].drop("NObeyesdad").sort_values(key=abs, ascending=False)
        med_fig()
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        cols_bar = ["#43a047" if v > 0 else "#e53935" for v in target_corr.values]
        ax2.barh(target_corr.index, target_corr.values, color=cols_bar, edgecolor="none", height=.62)
        ax2.axvline(0, color="#cfd8dc", linewidth=1.5)
        ax2.set_xlabel("Coefficient de corrélation de Pearson", fontsize=10)
        ax2.set_title("Impact de chaque variable sur le diagnostic d'obésité", fontsize=12, pad=10)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="x", alpha=.35, linestyle="--")
        for i, (val, name) in enumerate(zip(target_corr.values, target_corr.index)):
            ax2.text(val + (.005 if val >= 0 else -.005), i,
                     f"{val:.3f}", va="center",
                     ha="left" if val >= 0 else "right",
                     fontsize=8.5, color="#546e7a", fontweight="600")
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

    with tab2:
        st.markdown("<div class='sec-head'>📋 Statistiques descriptives globales</div>", unsafe_allow_html=True)
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"),
                     use_container_width=True, height=400)
        st.markdown("<div class='sec-head'>📊 Statistiques par classe clinique</div>", unsafe_allow_html=True)
        sv = st.selectbox("Variable", df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist())
        sbc = df.groupby("NObeyesdad")[sv].describe().round(3)
        sbc.index = [CLASS_NAMES[i] for i in sbc.index]
        st.dataframe(sbc.style.background_gradient(cmap="Blues"), use_container_width=True)

    with tab3:
        st.markdown("<div class='sec-head'>⚠️ Détection d'anomalies — Méthode IQR</div>", unsafe_allow_html=True)
        nc_out = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()
        out_rows = []
        for c in nc_out:
            Q1, Q3 = df[c].quantile(.25), df[c].quantile(.75); IQR = Q3-Q1
            n = ((df[c] < Q1-1.5*IQR) | (df[c] > Q3+1.5*IQR)).sum()
            out_rows.append({"Variable": c, "Q1": round(Q1,3), "Q3": round(Q3,3),
                             "IQR": round(IQR,3), "Outliers": n, "% Outliers": round(n/len(df)*100,2)})
        out_df = pd.DataFrame(out_rows).sort_values("Outliers", ascending=False)
        st.dataframe(out_df.style.background_gradient(subset=["Outliers","% Outliers"], cmap="Reds"),
                     use_container_width=True)
        med_fig()
        fig3, ax3 = plt.subplots(figsize=(9, 4.5))
        colors_out = ["#e53935" if v>5 else "#fb8c00" if v>2 else "#43a047" for v in out_df["% Outliers"]]
        ax3.bar(out_df["Variable"], out_df["% Outliers"], color=colors_out, edgecolor="none", width=.65)
        ax3.set_ylabel("% Outliers", fontsize=10)
        ax3.set_title("Taux d'anomalies par variable clinique", fontsize=12, pad=10)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(axis="y", alpha=.35, linestyle="--")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.markdown(f"""
    <div class='med-banner'>
        <div class='banner-title'>🤖 Entraînement & Évaluation</div>
        <div class='banner-sub'>
            Algorithme sélectionné :
            <strong>{ALGO_ICONS[algo]} {algo}</strong>
            {'&nbsp;⭐ Meilleur modèle' if algo == "LightGBM Classifier" else ""}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-panel teal'>
        <div class='info-panel-title'>ℹ️ Instructions</div>
        <div style='font-size:.88rem;color:#546e7a'>
            Algorithme actif : <strong style='color:{ALGO_COLORS[algo]}'>{ALGO_ICONS[algo]} {algo}</strong>
            — Cliquez sur le bouton ci-dessous pour démarrer l'entraînement et visualiser les résultats.
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button(f"🚀  Lancer l'entraînement — {algo}"):
        with st.spinner("⏳ Entraînement du modèle en cours…"):
            clf, sc, fc, acc, f1, prec, rec, cm, cr, _, yte, yp = train_model(algo)

        st.markdown("<div class='sec-head'>📊 Métriques de performance clinique</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Accuracy",  f"{acc*100:.2f}%")
        c2.metric("📊 F1-Score",  f"{f1*100:.2f}%")
        c3.metric("🎯 Précision", f"{prec*100:.2f}%")
        c4.metric("📡 Rappel",    f"{rec*100:.2f}%")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔲 Matrice de confusion", "📋 Rapport clinique",
            "📊 Importance des features", "📉 Performance par classe"
        ])

        with tab1:
            st.markdown("<div class='sec-head'>🔲 Matrice de confusion</div>", unsafe_allow_html=True)
            med_fig()
            fig, ax = plt.subplots(figsize=(9, 7))
            lbl = [CLASS_NAMES[i].replace(" ", "\n") for i in range(7)]
            sns.heatmap(cm, ax=ax, annot=True, fmt="d",
                        cmap=sns.light_palette("#0b5394", as_cmap=True),
                        xticklabels=lbl, yticklabels=lbl,
                        linewidths=.5, linecolor="white", cbar_kws={"shrink": .75})
            ax.set_xlabel("Diagnostic prédit", fontsize=11)
            ax.set_ylabel("Diagnostic réel", fontsize=11)
            ax.set_title(f"Matrice de confusion — {algo}", fontsize=13, pad=12)
            plt.xticks(fontsize=8); plt.yticks(fontsize=8)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True)

            st.markdown("<div class='sec-head'>📊 Matrice normalisée (%)</div>", unsafe_allow_html=True)
            med_fig()
            cm_n = cm.astype(float)/cm.sum(axis=1)[:, np.newaxis]*100
            fig2, ax2 = plt.subplots(figsize=(9, 7))
            sns.heatmap(cm_n, ax=ax2, annot=True, fmt=".1f",
                        cmap=sns.light_palette("#2e7d32", as_cmap=True),
                        xticklabels=lbl, yticklabels=lbl,
                        linewidths=.5, linecolor="white", cbar_kws={"shrink": .75, "label": "%"})
            ax2.set_xlabel("Diagnostic prédit"); ax2.set_ylabel("Diagnostic réel")
            ax2.set_title("Matrice de confusion normalisée (%)", fontsize=13, pad=12)
            plt.xticks(fontsize=8); plt.yticks(fontsize=8)
            plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

        with tab2:
            st.markdown("<div class='sec-head'>📋 Rapport de classification clinique</div>", unsafe_allow_html=True)
            cr_df = pd.DataFrame(cr).T
            st.dataframe(
                cr_df.style
                    .background_gradient(cmap="Blues", subset=["precision","recall","f1-score"])
                    .format("{:.3f}", subset=["precision","recall","f1-score"])
                    .format("{:.0f}", subset=["support"]),
                use_container_width=True)

        with tab3:
            if hasattr(clf, "feature_importances_"):
                fi_df = pd.DataFrame({"Variable": fc, "Importance": clf.feature_importances_}).sort_values("Importance")
                med_fig()
                fig3, ax3 = plt.subplots(figsize=(9, 6))
                med_c = fi_df["Importance"].median()
                cols_fi = [ALGO_COLORS[algo] if v > med_c else "#e0e0e0" for v in fi_df["Importance"]]
                ax3.barh(fi_df["Variable"], fi_df["Importance"], color=cols_fi, edgecolor="none", height=.65)
                for i, (f_, v) in enumerate(zip(fi_df["Variable"], fi_df["Importance"])):
                    ax3.text(v+.001, i, f"{v:.4f}", va="center", fontsize=8, color="#546e7a", fontweight="600")
                ax3.spines[["top","right","left"]].set_visible(False)
                ax3.grid(axis="x", alpha=.35, linestyle="--")
                ax3.set_title(f"Importance des variables cliniques — {algo}", fontsize=13, pad=12)
                ax3.set_xlabel("Importance relative", fontsize=10)
                plt.tight_layout(); st.pyplot(fig3, use_container_width=True)
            else:
                st.info("Importance des features non disponible pour ce modèle.")

        with tab4:
            st.markdown("<div class='sec-head'>📉 Métriques par classe clinique</div>", unsafe_allow_html=True)
            med_fig()
            f1_c   = [cr.get(str(i),{}).get("f1-score",0) for i in range(7)]
            prec_c = [cr.get(str(i),{}).get("precision",0) for i in range(7)]
            rec_c  = [cr.get(str(i),{}).get("recall",0) for i in range(7)]
            fig4, ax4 = plt.subplots(figsize=(11, 5))
            x_p = np.arange(7); w=.27
            ax4.bar(x_p-w,  prec_c, w, label="Précision", color="#0b5394", alpha=.88, edgecolor="none")
            ax4.bar(x_p,    rec_c,  w, label="Rappel",    color="#00838f", alpha=.88, edgecolor="none")
            ax4.bar(x_p+w,  f1_c,   w, label="F1-Score",  color="#2e7d32", alpha=.88, edgecolor="none")
            ax4.set_xticks(x_p)
            ax4.set_xticklabels([CLASS_NAMES[i].replace(" ","\n") for i in range(7)], fontsize=8.5)
            ax4.set_ylim(0, 1.12); ax4.set_ylabel("Score", fontsize=10)
            ax4.set_title("Précision, Rappel et F1-Score par classe", fontsize=13, pad=12)
            ax4.spines[["top","right"]].set_visible(False)
            ax4.grid(axis="y", alpha=.35, linestyle="--")
            ax4.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 5 — COMPARAISON (3 modèles)
# ══════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.markdown("""
    <div class='med-banner'>
        <div class='banner-title'>⚖️ Comparaison des Modèles</div>
        <div class='banner-sub'>Mise en compétition des 3 algorithmes de classification clinique</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("⏳ Évaluation des 3 modèles en cours…"):
        sc_df = compare_models()

    st.markdown("<div class='sec-head'>📋 Tableau comparatif des performances</div>", unsafe_allow_html=True)
    st.dataframe(
        sc_df.style
            .highlight_max(color="#c8e6c9", axis=0)
            .highlight_min(color="#ffcdd2", axis=0)
            .format("{:.2f}%"),
        use_container_width=True)

    st.markdown("<div class='sec-head'>📊 Comparaison visuelle — 4 métriques</div>", unsafe_allow_html=True)
    med_fig()
    metrics_c = ["Accuracy","F1-Score","Précision","Rappel"]
    fig_c, axes_c = plt.subplots(1, 4, figsize=(16, 5))
    for idx, metric in enumerate(metrics_c):
        vals   = sc_df[metric]
        colors = [ALGO_COLORS[a] for a in ALGO_LIST]
        bars   = axes_c[idx].bar(range(3), vals.values, color=colors, edgecolor="none", width=.55)
        axes_c[idx].set_xticks(range(3))
        axes_c[idx].set_xticklabels(
            [a.replace(" Classifier","") for a in ALGO_LIST],
            fontsize=8, rotation=20, ha="right"
        )
        axes_c[idx].set_ylim(vals.min()-3, 100)
        axes_c[idx].set_title(f"{metric} (%)", fontsize=10, pad=8, fontweight="600")
        axes_c[idx].spines[["top","right"]].set_visible(False)
        axes_c[idx].grid(axis="y", alpha=.35, linestyle="--")
        for bar, v in zip(bars, vals.values):
            axes_c[idx].text(bar.get_x()+bar.get_width()/2, v+.2,
                             f"{v:.1f}%", ha="center", va="bottom",
                             fontsize=8, color="#0d1b2a", fontweight="700")
    plt.tight_layout(); st.pyplot(fig_c, use_container_width=True)

    st.markdown("<div class='sec-head'>🕸️ Radar chart — vision globale</div>", unsafe_allow_html=True)
    med_fig()
    cats = ["Accuracy","F1-Score","Précision","Rappel"]; N = len(cats)
    angles = [n/float(N)*2*np.pi for n in range(N)]; angles += angles[:1]
    fig_r, ax_r = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax_r.set_facecolor("#f8fafc"); fig_r.patch.set_facecolor("white")
    ax_r.grid(color="#cfd8dc", linestyle="--", linewidth=.8)
    ax_r.spines["polar"].set_color("#cfd8dc")
    for a in ALGO_LIST:
        vals_r = [sc_df.loc[a, m] for m in cats]; vals_r += vals_r[:1]
        ax_r.plot(angles, vals_r, linewidth=2.5, color=ALGO_COLORS[a],
                  label=a.replace(" Classifier",""))
        ax_r.fill(angles, vals_r, alpha=.12, color=ALGO_COLORS[a])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(cats, fontsize=11, color="#0d1b2a", fontweight="600")
    ax_r.set_ylim(80, 100)
    ax_r.tick_params(axis="y", colors="#546e7a", labelsize=8)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=10)
    ax_r.set_title("Performance comparative des 3 modèles", fontsize=12, color="#0d1b2a", pad=22, fontweight="700")
    plt.tight_layout(); st.pyplot(fig_r, use_container_width=True)

    # ── Podium final 3 modèles ──
    st.markdown("<div class='sec-head'>🏆 Résultats finaux</div>", unsafe_allow_html=True)
    best = sc_df["Accuracy"].idxmax()
    c1, c2, c3 = st.columns(3)
    for col, a in zip([c1, c2, c3], ALGO_LIST):
        is_best = a == best
        col.markdown(f"""
        <div class='model-card {"best" if is_best else ""}' style='text-align:center'>
            <div class='model-icon'>{ALGO_ICONS[a]}</div>
            <div class='model-name' style='color:{ALGO_COLORS[a]};font-size:1rem'>{a.replace(" Classifier","")}</div>
            <div style='font-family:"Playfair Display",serif;font-size:1.8rem;
                        color:{ALGO_COLORS[a]};margin:.6rem 0;font-weight:800'>
                {sc_df.loc[a,"Accuracy"]:.2f}%
            </div>
            <div class='model-desc'>Accuracy</div>
            <div style='color:#546e7a;font-size:.86rem;margin-top:.35rem;font-weight:600'>
                F1 : {sc_df.loc[a,"F1-Score"]:.2f}%
            </div>
            {"<div class='best-pill'>⭐ Meilleur modèle</div>" if is_best else ""}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE 6 — DIAGNOSTIC INDIVIDUEL
# ══════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.markdown(f"""
    <div class='med-banner'>
        <div class='banner-title'>🩺 Diagnostic Individuel</div>
        <div class='banner-sub'>
            Évaluation personnalisée du risque d'obésité — Modèle :
            <strong>{ALGO_ICONS[algo]} {algo}</strong>
            {'&nbsp;· ⭐ Meilleur modèle' if algo == "LightGBM Classifier" else ""}
        </div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("⏳ Initialisation du modèle clinique…"):
        clf, sc_m, fc, acc, f1, prec, rec, *_ = train_model(algo)

    st.markdown(f"""
    <div style='margin-bottom:1.5rem'>
        <span class='chip'>{ALGO_ICONS[algo]} {algo}</span>
        <span class='chip'>✅ Accuracy : {acc*100:.1f}%</span>
        <span class='chip'>📊 F1-Score : {f1*100:.1f}%</span>
        <span class='chip'>🎯 Précision : {prec*100:.1f}%</span>
        <span class='chip'>📡 Rappel : {rec*100:.1f}%</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'>📋 Dossier Patient</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='info-panel'>
            <div class='info-panel-title'>👤 Données biométriques</div>
        </div>""", unsafe_allow_html=True)
        gender = st.selectbox("Genre", ["Féminin","Masculin"])
        age    = st.slider("Âge (années)", 10, 80, 25)
        height = st.slider("Taille (m)", 1.40, 2.10, 1.70, 0.01)
        weight = st.slider("Poids (kg)", 30.0, 170.0, 70.0, 0.5)
        family = st.selectbox("Antécédents familiaux d'obésité", ["Non","Oui"])

        imc_rt = round(weight/(height**2), 1)
        if imc_rt < 18.5:   imc_c, imc_t, imc_bg = "#0b5394", "Insuffisant", "#e3f2fd"
        elif imc_rt < 25:   imc_c, imc_t, imc_bg = "#2e7d32", "Normal ✓",   "#e8f5e9"
        elif imc_rt < 30:   imc_c, imc_t, imc_bg = "#e65100", "Surpoids",   "#fff3e0"
        else:               imc_c, imc_t, imc_bg = "#b71c1c", "Obésité ⚠️", "#ffebee"

        st.markdown(f"""
        <div style='background:{imc_bg};border-radius:12px;padding:1.1rem 1rem;
                    text-align:center;margin-top:.7rem;
                    border:2px solid {imc_c};box-shadow:0 3px 14px rgba(0,0,0,.08)'>
            <div style='font-size:.73rem;color:#546e7a;font-weight:700;text-transform:uppercase;
                        letter-spacing:.07em'>IMC en temps réel</div>
            <div style='font-family:"Playfair Display",serif;font-size:2.4rem;
                        font-weight:800;color:{imc_c};line-height:1.1'>{imc_rt}</div>
            <div style='font-size:.85rem;color:{imc_c};font-weight:700;margin-top:2px'>{imc_t}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-panel'>
            <div class='info-panel-title'>🍽️ Habitudes alimentaires</div>
        </div>""", unsafe_allow_html=True)
        favc = st.selectbox("Aliments très caloriques (FAVC)", ["Non","Oui"])
        fcvc = st.slider("Fréquence légumes (FCVC)", 1.0, 3.0, 2.0, 0.1,
                         help="1=Jamais · 2=Parfois · 3=Toujours")
        ncp  = st.slider("Repas principaux / jour (NCP)", 1.0, 4.0, 3.0, 0.5)
        caec = st.selectbox("Alimentation entre repas (CAEC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"])
        calc = st.selectbox("Consommation d'alcool (CALC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"])
        st.markdown("""
        <div class='info-panel green' style='margin-top:.8rem'>
            <div class='info-panel-title'>💡 Recommandation OMS</div>
            <div style='font-size:.82rem;color:#546e7a'>
                5 portions de fruits/légumes par jour et 3 repas équilibrés
                réduisent le risque d'obésité de <strong>35%</strong>.
            </div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='info-panel'>
            <div class='info-panel-title'>🏃 Mode de vie & activité</div>
        </div>""", unsafe_allow_html=True)
        smoke  = st.selectbox("Tabagisme (SMOKE)", ["Non","Oui"])
        ch2o   = st.slider("Eau / jour (litres)", 1.0, 3.0, 2.0, 0.1,
                           help="Recommandation : 2L minimum/jour")
        scc    = st.selectbox("Surveillance calorique (SCC)", ["Non","Oui"])
        faf    = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, 1.0, 0.1,
                           help="OMS : 150 min d'activité modérée/semaine")
        tue    = st.slider("Temps écran / jour (heures)", 0.0, 2.0, 1.0, 0.1)
        mtrans = st.selectbox("Transport principal", list(MTRANS_MAP.keys()))

    st.markdown("")
    bcol, _ = st.columns([1,3])
    with bcol:
        pred_btn = st.button("🩺  Lancer le diagnostic", use_container_width=True)

    if pred_btn:
        row = {
            "Gender": GENDER_MAP[gender], "Age": float(age),
            "Height": height, "Weight": weight,
            "family_history_with_overweight": BINARY_MAP[family],
            "FAVC": BINARY_MAP[favc], "FCVC": fcvc, "NCP": ncp,
            "CAEC": CAEC_MAP[caec], "SMOKE": BINARY_MAP[smoke],
            "CH2O": ch2o, "SCC": BINARY_MAP[scc],
            "FAF": faf, "TUE": tue,
            "CALC": CALC_MAP[calc], "MTRANS": MTRANS_MAP[mtrans],
        }
        Xn  = pd.DataFrame([row])[fc]
        Xns = sc_m.transform(Xn)
        pred  = int(clf.predict(Xns)[0])
        proba = clf.predict_proba(Xns)[0] if hasattr(clf, "predict_proba") else None
        info  = CLASS_INFO[pred]
        imc_v = round(weight/(height**2), 1)

        st.markdown("---")
        st.markdown("<div class='sec-head'>🔬 Résultat du Diagnostic</div>", unsafe_allow_html=True)

        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            emoji = "✅" if info[1]=="green" else "⚠️" if info[1]=="orange" else "🚨"
            st.markdown(f"""
            <div class='result-box {info[1]}'>
                <div style='font-size:2.8rem;margin-bottom:.5rem'>{emoji}</div>
                <div class='result-title'>{info[0]}</div>
                <div class='result-imc'>
                    <strong>IMC : {imc_v}</strong> &nbsp;|&nbsp; {info[2]}
                </div>
                <div class='result-desc'>{info[3]}</div>
            </div>""", unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""
            <div class='info-panel' style='height:100%'>
                <div class='info-panel-title'>📋 Résumé du dossier patient</div>
                <div style='font-size:.86rem;line-height:2.1;color:#0d1b2a'>
                    <b>Genre :</b> {"Homme" if gender=="Masculin" else "Femme"}<br>
                    <b>Âge :</b> {age} ans<br>
                    <b>Taille / Poids :</b> {height} m · {weight} kg<br>
                    <b>IMC :</b> <span style='color:{"#2e7d32" if imc_v<25 else "#e65100" if imc_v<30 else "#b71c1c"};
                        font-weight:800;font-size:1rem'>{imc_v}</span><br>
                    <b>Activité physique :</b> {faf} j/sem.<br>
                    <b>Hydratation :</b> {ch2o} L/jour<br>
                    <b>Tabagisme :</b> {smoke}<br>
                    <b>Antécédents familiaux :</b> {family}
                </div>
            </div>""", unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<div class='sec-head'>📊 Probabilités diagnostiques par classe</div>", unsafe_allow_html=True)
            med_fig()
            fig_p, ax_p = plt.subplots(figsize=(11, 4.5))
            cols_p  = [CLASS_COLORS_HEX[i] for i in range(7)]
            alpha_p = [1.0 if i==pred else .4 for i in range(7)]
            bars_p  = ax_p.bar([CLASS_NAMES[i] for i in range(7)], proba,
                               color=cols_p, edgecolor="none", width=.65)
            for bar, a_val in zip(bars_p, alpha_p): bar.set_alpha(a_val)
            ax_p.set_ylim(0, 1.15)
            ax_p.set_ylabel("Probabilité", fontsize=10)
            ax_p.set_title("Distribution des probabilités diagnostiques", fontsize=12, pad=10)
            ax_p.spines[["top","right"]].set_visible(False)
            ax_p.grid(axis="y", alpha=.35, linestyle="--")
            plt.xticks(rotation=22, ha="right", fontsize=9)
            for bar, p in zip(bars_p, proba):
                if p > .01:
                    ax_p.text(bar.get_x()+bar.get_width()/2, p+.016,
                              f"{p*100:.1f}%", ha="center", va="bottom",
                              fontsize=9, color="#0d1b2a", fontweight="700")
            plt.tight_layout(); st.pyplot(fig_p, use_container_width=True)

        st.markdown("<div class='sec-head'>💊 Recommandations médicales personnalisées</div>", unsafe_allow_html=True)

        recs = []
        if faf < 1.0:
            recs.append(("red","🏃","Activité physique insuffisante",
                         "Pratiquer au moins 150 min d'activité modérée par semaine (OMS 2023). Commencer par 20 min/jour."))
        elif faf >= 2.5:
            recs.append(("green","🏃","Excellente activité physique",
                         f"Votre niveau d'activité ({faf} j/semaine) est optimal. Continuez ainsi — cela réduit le risque cardiovasculaire de 30%."))
        else:
            recs.append(("orange","🏃","Activité physique à augmenter",
                         "Progresser vers 3-4 séances par semaine pour atteindre les recommandations OMS."))

        if ch2o < 1.5:
            recs.append(("red","💧","Hydratation critique",
                         f"Seulement {ch2o}L/j — risque de déshydratation. Objectif minimum : 2L/jour, idéalement 2.5L."))
        elif ch2o >= 2.0:
            recs.append(("green","💧","Bonne hydratation",
                         f"Consommation de {ch2o}L/jour. Conforme aux recommandations médicales."))

        if caec in ["Fréquemment","Toujours"]:
            recs.append(("red","🍪","Grignotage excessif détecté",
                         "Le grignotage fréquent augmente l'apport calorique journalier de 20-30%. Privilégier des collations saines."))

        if smoke == "Oui":
            recs.append(("red","🚬","Tabagisme — facteur de risque métabolique",
                         "Le tabac perturbe le métabolisme et favorise la prise de poids abdominale. Consultation sevrage recommandée."))

        if family == "Oui":
            recs.append(("orange","🧬","Prédisposition génétique identifiée",
                         "Les antécédents familiaux multiplient le risque par 2-3. Suivi médical annuel et bilan métabolique conseillés."))

        if imc_v >= 30:
            recs.append(("red","⚕️","Consultation médicale urgente recommandée",
                         "IMC ≥ 30 : bilan lipidique, glycémie à jeun et tension artérielle indispensables. Orientation endocrinologue."))
        elif 25 <= imc_v < 30:
            recs.append(("orange","⚕️","Suivi médical préventif conseillé",
                         "IMC en zone surpoids. Consultation diététicien-nutritionniste et bilan lipidique recommandés."))
        else:
            recs.append(("green","⚕️","Profil clinique satisfaisant",
                         "IMC dans la norme. Maintenir les habitudes alimentaires équilibrées et l'activité physique régulière."))

        for color, icon, title, text in recs:
            st.markdown(f"""
            <div class='rec-card {color}'>
                <div class='rec-icon'>{icon}</div>
                <div>
                    <div class='rec-title'>{title}</div>
                    <div class='rec-text'>{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#f8fafc;border-radius:12px;padding:1rem 1.5rem;
                    margin-top:1.5rem;border:1px solid #cfd8dc;
                    font-size:.8rem;color:#546e7a;text-align:center'>
            ⚠️ <strong>Avertissement :</strong> Ce diagnostic est généré par un modèle d'IA à des fins éducatives.
            Il ne remplace en aucun cas un avis médical professionnel.
            Consultez un médecin pour tout problème de santé.
        </div>""", unsafe_allow_html=True)