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
    classification_report, precision_score, recall_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ═══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ObesoScan — Système Clinique IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
#  GLOBAL CSS — DARK MEDICAL PREMIUM (thème verrouillé)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ══════════════════════════════════════════
   VERROUILLAGE THÈME SOMBRE — NE PAS RETIRER
══════════════════════════════════════════ */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
header[data-testid="stHeader"],
footer { display:none !important; visibility:hidden !important; }

html,body,#root,.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stBottom"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],
[data-testid="element-container"],
.main,.block-container,
.reportview-container,
[class^="css-"],[class*=" css-"] {
    background-color:#07101f !important;
    color:#dde6f0 !important;
}
:root,[data-theme="light"],[data-theme="dark"],
.stApp[data-theme="light"],.stApp[data-theme="dark"] {
    --background-color:#07101f !important;
    --secondary-background-color:#0d1a2e !important;
    --text-color:#dde6f0 !important;
    --primary-color:#0ea5e9 !important;
    color-scheme:dark !important;
}

/* ── Variables design system ── */
:root {
    --bg:        #07101f;
    --surface:   #0d1a2e;
    --surface2:  #112039;
    --surface3:  #172847;
    --border:    rgba(14,165,233,.18);
    --border-sm: rgba(255,255,255,.07);
    --sky:       #0ea5e9;
    --sky-dim:   rgba(14,165,233,.12);
    --indigo:    #6366f1;
    --violet:    #8b5cf6;
    --teal:      #14b8a6;
    --emerald:   #10b981;
    --amber:     #f59e0b;
    --rose:      #f43f5e;
    --text:      #dde6f0;
    --muted:     #4a6080;
    --dim:       #7a9ab8;
    --nurse:     #0ea5e9;
    --doctor:    #8b5cf6;
    --lgbm:      #10b981;
    --rf:        #0ea5e9;
    --xgb:       #f59e0b;
    --font-ui:   'Inter', sans-serif;
    --font-disp: 'Playfair Display', serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html,body,[class*="css"] {
    font-family: var(--font-ui) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#030c1a 0%,#071630 55%,#0a1e40 100%) !important;
    border-right: 1px solid rgba(14,165,233,.15) !important;
    width: 275px !important;
}
section[data-testid="stSidebar"] * { color: #c8daf0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(14,165,233,.15) !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: .875rem !important;
    padding: .5rem .7rem !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: all .18s;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(14,165,233,.1) !important;
    border-color: rgba(14,165,233,.25) !important;
}

/* ── Main ── */
.main .block-container { padding: 1.75rem 2.2rem 3rem; max-width: 1500px; }
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius:3px; }

/* ── Inputs ── */
[data-baseweb="select"]>div,[data-baseweb="input"]>div,
.stTextInput>div>div,.stNumberInput>div>div {
    background: var(--surface2) !important;
    border-color: var(--border-sm) !important;
    color: var(--text) !important;
}
[data-baseweb="popover"],[role="listbox"],[data-baseweb="menu"] {
    background: var(--surface2) !important;
}
[role="option"] { color: var(--text) !important; }
[role="option"]:hover { background: var(--surface3) !important; }
label,.stSelectbox label,.stSlider label,
.stNumberInput label,[data-testid="stWidgetLabel"] {
    color: var(--dim) !important; font-size:.84rem !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="thumb"] { background:var(--sky) !important; border-color:var(--sky) !important; }
.stSlider [data-baseweb="track-fill"] { background:var(--sky) !important; }
.stSlider [data-baseweb="track"] { background:var(--surface3) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sm) !important;
    border-radius: 10px !important;
    padding: .3rem !important; gap: .2rem; margin-bottom:1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    font-weight: 600 !important; font-size:.84rem !important;
    border-radius: 8px !important; border: none !important;
}
.stTabs [aria-selected="true"] { background:var(--surface2) !important; color:var(--sky) !important; }
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] { display:none !important; }

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-sm) !important;
    border-radius: 12px !important; padding: 1rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--muted) !important; font-size:.73rem !important;
    text-transform:uppercase; letter-spacing:.06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important; font-family:var(--font-disp) !important; font-size:1.5rem !important;
}

/* ── Button ── */
.stButton>button {
    background: linear-gradient(135deg,#0369a1,#0ea5e9) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: .65rem 2rem !important;
    font-family: var(--font-ui) !important; font-weight: 700 !important;
    letter-spacing: .04em !important; font-size: .88rem !important;
    box-shadow: 0 4px 20px rgba(14,165,233,.3) !important;
    transition: all .2s !important;
}
.stButton>button:hover {
    opacity:.9 !important; transform:translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(14,165,233,.4) !important;
}

/* ── Alerts ── */
.stAlert,[data-testid="stNotification"] {
    background: var(--surface) !important;
    border-color: var(--border-sm) !important; color: var(--text) !important;
}
.streamlit-expanderHeader { background: var(--surface) !important; color: var(--text) !important; }
.stSpinner>div { border-top-color: var(--sky) !important; }
hr { border-color: var(--border-sm) !important; }
[data-baseweb="tag"] { background: var(--surface3) !important; }
[data-baseweb="tag"] span { color: var(--text) !important; }

/* ══════════════════════════════════════════
   COMPOSANTS PERSONNALISÉS
══════════════════════════════════════════ */

/* Banner */
.banner {
    border-radius: 18px; padding: 2.2rem 2.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 12px 40px rgba(0,0,0,.5);
}
.banner-nurse  { background: linear-gradient(135deg,#032040 0%,#063a6e 55%,#0a4f8a 100%); }
.banner-doctor { background: linear-gradient(135deg,#1a0a38 0%,#2d1065 55%,#3b1485 100%); }
.banner::before {
    content: '⚕';
    position: absolute; right:2.5rem; top:50%; transform:translateY(-50%);
    font-size:9rem; opacity:.06; pointer-events:none;
}
.banner::after {
    content:''; position:absolute; top:-50px; left:-50px;
    width:250px; height:250px;
    background:rgba(255,255,255,.03); border-radius:50%; pointer-events:none;
}
.banner-pre {
    font-size:.7rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
    margin-bottom:.5rem; opacity:.7;
}
.banner-title {
    font-family: var(--font-disp);
    font-size: 2.1rem; font-weight:700; color:#f0f7ff;
    margin:0 0 .4rem; line-height:1.15;
}
.banner-sub { font-size:.9rem; opacity:.7; margin:0; font-weight:400; line-height:1.6; }
.banner-badge {
    display:inline-block;
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2);
    border-radius:20px; padding:.25rem .8rem; font-size:.72rem; font-weight:600;
    margin:.65rem .3rem 0 0; letter-spacing:.03em; backdrop-filter:blur(4px);
}

/* KPI Grid */
.kpi-grid {
    display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
    gap:1rem; margin:1.5rem 0;
}
.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border-sm);
    border-top: 4px solid var(--sky);
    border-radius: 14px; padding: 1.35rem 1.3rem;
    position: relative; overflow: hidden;
    transition: transform .22s, box-shadow .22s;
    box-shadow: 0 4px 20px rgba(0,0,0,.35);
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 10px 32px rgba(0,0,0,.5); }
.kpi-card::after {
    content:''; position:absolute; bottom:-24px; right:-24px;
    width:80px; height:80px; border-radius:50%;
    background:rgba(255,255,255,.03); pointer-events:none;
}
.kc-sky    { border-top-color:var(--sky); }
.kc-indigo { border-top-color:var(--indigo); }
.kc-violet { border-top-color:var(--violet); }
.kc-teal   { border-top-color:var(--teal); }
.kc-emerald{ border-top-color:var(--emerald); }
.kc-amber  { border-top-color:var(--amber); }
.kc-rose   { border-top-color:var(--rose); }
.kpi-num {
    font-family: var(--font-disp);
    font-size:2rem; color:var(--sky); line-height:1; margin-bottom:.35rem;
}
.kc-indigo  .kpi-num { color:var(--indigo); }
.kc-violet  .kpi-num { color:var(--violet); }
.kc-teal    .kpi-num { color:var(--teal); }
.kc-emerald .kpi-num { color:var(--emerald); }
.kc-amber   .kpi-num { color:var(--amber); }
.kc-rose    .kpi-num { color:var(--rose); }
.kpi-lbl {
    font-size:.69rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted);
}

/* Waiting room widget */
.wait-card {
    background: linear-gradient(135deg,#071a38,#0d2548);
    border: 1px solid rgba(14,165,233,.25);
    border-radius: 16px; padding: 1.5rem 1.8rem;
    box-shadow: 0 6px 28px rgba(0,0,0,.45);
}
.wait-title {
    font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
    color:var(--sky); margin-bottom:.8rem; display:flex; align-items:center; gap:.5rem;
}
.wait-num {
    font-family:var(--font-disp);
    font-size:3.5rem; font-weight:700; color:#f0f7ff; line-height:1;
    text-align:center; margin:.5rem 0;
}
.wait-sub { font-size:.78rem; color:var(--muted); text-align:center; }
.wait-status {
    display:inline-flex; align-items:center; gap:.4rem;
    border-radius:20px; padding:.25rem .75rem;
    font-size:.72rem; font-weight:700;
}
.ws-low  { background:rgba(16,185,129,.15); color:#34d399; border:1px solid rgba(16,185,129,.3); }
.ws-mid  { background:rgba(245,158,11,.15);  color:#fbbf24; border:1px solid rgba(245,158,11,.3); }
.ws-high { background:rgba(244,63,94,.15);   color:#fb7185; border:1px solid rgba(244,63,94,.3); }

/* Section head */
.sec-head {
    font-family: var(--font-disp);
    font-size:1.1rem; color:#e8f1fa;
    margin:2rem 0 1rem;
    display:flex; align-items:center; gap:.6rem;
    padding-bottom:.5rem;
    border-bottom:1px solid var(--border-sm);
}
.dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-sky    { background:var(--sky);     box-shadow:0 0 8px var(--sky); }
.dot-violet { background:var(--violet);  box-shadow:0 0 8px var(--violet); }
.dot-nurse  { background:var(--nurse);   box-shadow:0 0 8px var(--nurse); }
.dot-doctor { background:var(--doctor);  box-shadow:0 0 8px var(--doctor); }
.dot-emerald{ background:var(--emerald); box-shadow:0 0 8px var(--emerald); }

/* Panel */
.panel {
    background: var(--surface);
    border: 1px solid var(--border-sm);
    border-left: 4px solid var(--sky);
    border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom:1rem;
}
.p-nurse   { border-left-color:var(--nurse); }
.p-doctor  { border-left-color:var(--doctor); }
.p-sky     { border-left-color:var(--sky); }
.p-emerald { border-left-color:var(--emerald); }
.p-amber   { border-left-color:var(--amber); }
.p-rose    { border-left-color:var(--rose); }
.panel-title {
    font-weight:700; font-size:.87rem; color:var(--sky);
    margin-bottom:.3rem; letter-spacing:.03em;
}
.p-emerald .panel-title { color:var(--emerald); }
.p-amber   .panel-title { color:var(--amber); }
.p-rose    .panel-title { color:var(--rose); }
.p-nurse   .panel-title { color:var(--nurse); }
.p-doctor  .panel-title { color:#a78bfa; }
.panel-body { font-size:.84rem; color:var(--dim); line-height:1.6; }

/* Model card */
.model-card {
    background: var(--surface);
    border: 1px solid var(--border-sm);
    border-radius: 14px; padding: 1.6rem 1.4rem;
    text-align:center; transition:all .22s;
    position:relative; overflow:hidden;
}
.model-card:hover { border-color:rgba(14,165,233,.3); transform:translateY(-3px); box-shadow:0 10px 32px rgba(0,0,0,.5); }
.model-card.best {
    border-color:rgba(16,185,129,.4);
    background:linear-gradient(160deg,rgba(16,185,129,.06),var(--surface));
}
.model-card.best::before {
    content:'⭐ Meilleur Modèle';
    position:absolute; top:10px; right:10px;
    background:rgba(16,185,129,.15); color:#34d399;
    border:1px solid rgba(16,185,129,.35); border-radius:20px;
    padding:.18rem .7rem; font-size:.64rem; font-weight:700; letter-spacing:.05em;
}
.model-icon  { font-size:2.2rem; margin-bottom:.6rem; }
.model-name  { font-weight:700; font-size:.95rem; color:var(--text); margin-bottom:.35rem; }
.model-score { font-family:var(--font-disp); font-size:1.8rem; margin:.5rem 0; }
.model-desc  { font-size:.77rem; color:var(--muted); line-height:1.5; }

/* Result box */
.result-box {
    border-radius:16px; padding:2.2rem; text-align:center; margin:1rem 0;
    border:1px solid var(--border-sm); background:var(--surface);
    position:relative; overflow:hidden;
}
.rb-green  { border-color:rgba(16,185,129,.35); background:linear-gradient(160deg,rgba(16,185,129,.07),var(--surface)); }
.rb-amber  { border-color:rgba(245,158,11,.35);  background:linear-gradient(160deg,rgba(245,158,11,.07),var(--surface)); }
.rb-rose   { border-color:rgba(244,63,94,.35);   background:linear-gradient(160deg,rgba(244,63,94,.07),var(--surface)); }
.result-emoji { font-size:3rem; display:block; margin-bottom:.8rem; }
.result-title { font-family:var(--font-disp); font-size:1.85rem; font-weight:700; }
.rb-green .result-title { color:#6ee7b7; }
.rb-amber .result-title { color:#fcd34d; }
.rb-rose  .result-title { color:#fda4af; }
.result-imc  { font-family:var(--font-mono); font-size:.87rem; color:var(--muted); margin:.5rem 0; }
.result-desc { font-size:.87rem; color:var(--dim); margin-top:.5rem; }

/* Rec card */
.rec-card {
    background:var(--surface); border:1px solid var(--border-sm);
    border-left:4px solid var(--sky); border-radius:10px;
    padding:1rem 1.3rem; margin:.5rem 0;
    display:flex; align-items:flex-start; gap:1rem; transition:transform .18s;
}
.rec-card:hover { transform:translateX(4px); }
.rc-emerald { border-left-color:var(--emerald); }
.rc-amber   { border-left-color:var(--amber); }
.rc-rose    { border-left-color:var(--rose); }
.rec-icon  { font-size:1.35rem; flex-shrink:0; margin-top:2px; }
.rec-title { font-weight:700; font-size:.87rem; color:var(--text); margin-bottom:.2rem; }
.rec-text  { font-size:.79rem; color:var(--muted); line-height:1.55; }

/* IMC live */
.imc-live {
    border-radius:12px; padding:1.2rem; text-align:center; margin-top:.8rem;
    border:1px solid var(--border-sm); background:var(--surface2);
}
.imc-label { font-size:.66rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
.imc-value { font-family:var(--font-disp); font-size:2.6rem; line-height:1.1; margin:.2rem 0; }
.imc-cat   { font-size:.82rem; font-weight:700; margin-top:2px; }

/* Chip */
.chip {
    display:inline-flex; align-items:center; gap:.3rem;
    background:var(--surface2); color:var(--dim);
    border:1px solid var(--border-sm); border-radius:6px;
    padding:.24rem .72rem; font-size:.73rem; font-weight:600;
    margin:.2rem; font-family:var(--font-mono);
}
.chip-sky { background:var(--sky-dim); color:var(--sky); border-color:rgba(14,165,233,.3); }

/* Form section */
.form-section {
    background:var(--surface); border:1px solid var(--border-sm);
    border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1.2rem;
}
.form-title {
    font-size:.77rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
    margin-bottom:1rem; padding-bottom:.5rem; border-bottom:1px solid var(--border-sm);
}
.ft-nurse  { color:var(--nurse); }
.ft-doctor { color:var(--doctor); }

/* Stat row */
.stat-row {
    display:flex; justify-content:space-between;
    padding:.4rem 0; border-bottom:1px solid rgba(255,255,255,.04); font-size:.84rem;
}
.stat-row:last-child { border-bottom:none; }
.sk { color:var(--muted); font-weight:500; }
.sv { color:var(--text); font-weight:700; font-family:var(--font-mono); font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════
CLASS_NAMES = {
    0:"Poids Insuffisant", 1:"Poids Normal",
    2:"Obésité Type I",    3:"Obésité Type II",
    4:"Obésité Type III",  5:"Surpoids Niveau I",
    6:"Surpoids Niveau II",
}
CLASS_INFO = {
    0:("Poids Insuffisant",  "green",  "IMC < 18.5",       "Risque de carences nutritionnelles. Suivi médical recommandé."),
    1:("Poids Normal",       "green",  "18.5 ≤ IMC < 25",  "Profil clinique sain. Maintenir les habitudes actuelles."),
    2:("Obésité Type I",     "red",    "30 ≤ IMC < 35",    "Risque cardiovasculaire modéré. Suivi médical requis."),
    3:("Obésité Type II",    "red",    "35 ≤ IMC < 40",    "Risque cardiovasculaire élevé. Consultation spécialiste."),
    4:("Obésité Type III",   "red",    "IMC ≥ 40",         "Obésité morbide. Prise en charge médicale urgente."),
    5:("Surpoids Niveau I",  "amber",  "25 ≤ IMC < 27.5",  "Surveiller l'alimentation. Augmenter l'activité physique."),
    6:("Surpoids Niveau II", "amber",  "27.5 ≤ IMC < 30",  "Bilan lipidique conseillé. Consultation diététicien."),
}
CLASS_HEX = {
    0:"#10b981",1:"#0ea5e9",2:"#f59e0b",3:"#f43f5e",
    4:"#8b5cf6",5:"#fbbf24",6:"#f97316",
}
CLASS_RB = {0:"rb-green",1:"rb-green",2:"rb-rose",3:"rb-rose",4:"rb-rose",5:"rb-amber",6:"rb-amber"}

ALGO_LIST   = ["LightGBM Classifier","Random Forest Classifier","XGBoost Classifier"]
ALGO_ICONS  = {"LightGBM Classifier":"⚡","Random Forest Classifier":"🌲","XGBoost Classifier":"🚀"}
ALGO_COLORS = {"LightGBM Classifier":"#10b981","Random Forest Classifier":"#0ea5e9","XGBoost Classifier":"#f59e0b"}
ALGO_DESC   = {
    "LightGBM Classifier":      "Gradient Boosting ultra-rapide. Optimal sur données médicales tabulaires. Meilleure précision diagnostique.",
    "Random Forest Classifier": "Forêt d'arbres de décision. Robuste, interprétable et stable cliniquement.",
    "XGBoost Classifier":       "Extreme Gradient Boosting. Excellent équilibre vitesse / précision sur données structurées.",
}
BEST_ALGO = "LightGBM Classifier"

GENDER_MAP = {"Féminin":0,"Masculin":1}
BINARY_MAP = {"Non":0,"Oui":1}
CAEC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
CALC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
MTRANS_MAP = {"Automobile":0,"Vélo":1,"Moto":2,"Transport en commun":3,"Marche":4}

ROLES       = ["👩‍⚕️  Infirmière — Saisie Patient","👨‍⚕️  Médecin — Analyse & Diagnostic"]
NURSE_PAGES = ["📋  Dossier Patient","📏  Questionnaire Clinique"]
DOC_PAGES   = ["🏥  Tableau de Bord","📊  Exploration Clinique",
               "📈  Analyse Statistique",
               "⚖️  Comparaison des Modèles","🩺  Diagnostic IA"]
PALETTE     = ["#0ea5e9","#6366f1","#10b981","#f59e0b","#f43f5e","#8b5cf6","#f97316","#14b8a6"]


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def dark_fig(w=10, h=5, ncols=1, nrows=1):
    plt.rcParams.update({
        "figure.facecolor":"#0d1a2e","axes.facecolor":"#112039",
        "axes.edgecolor":"#1e3456",  "axes.labelcolor":"#7a9ab8",
        "xtick.color":"#4a6080",     "ytick.color":"#4a6080",
        "text.color":"#dde6f0",      "grid.color":"#172847",
        "legend.facecolor":"#0d1a2e","legend.edgecolor":"#1e3456",
        "font.family":"DejaVu Sans", "figure.dpi":110,
    })
    if ncols==1 and nrows==1:
        return plt.subplots(figsize=(w,h))
    return plt.subplots(nrows,ncols,figsize=(w,h))


@st.cache_data
def load_data():
    for p in ["data_clean.csv","data/data_clean.csv","../data/data_clean.csv",
              "../data_clean.csv","/mnt/user-data/uploads/data_clean__1_.csv"]:
        try: return pd.read_csv(p)
        except: pass
    st.error("❌ data_clean.csv introuvable."); st.stop()


@st.cache_data
def decode_df(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["Gender"]  = d["Gender"].map({0:"Féminin",1:"Masculin"})
    d["family_history_with_overweight"] = d["family_history_with_overweight"].map({0:"Non",1:"Oui"})
    d["FAVC"]    = d["FAVC"].map({0:"Non",1:"Oui"})
    d["SMOKE"]   = d["SMOKE"].map({0:"Non",1:"Oui"})
    d["SCC"]     = d["SCC"].map({0:"Non",1:"Oui"})
    d["CAEC"]    = d["CAEC"].map({0:"Toujours",1:"Fréquemment",2:"Parfois",3:"Jamais"})
    d["CALC"]    = d["CALC"].map({0:"Toujours",1:"Fréquemment",2:"Parfois",3:"Jamais"})
    d["MTRANS"]  = d["MTRANS"].map({0:"Automobile",1:"Vélo",2:"Moto",3:"Transport en commun",4:"Marche"})
    d["NObeyesdad"] = d["NObeyesdad"].map(CLASS_NAMES)
    d.columns = ["Genre","Âge","Taille (m)","Poids (kg)","Ant. Familiaux",
                 "FAVC","FCVC","NCP","CAEC","Tabac","Eau/j","SCC",
                 "Activité","Écran","Alcool","Transport","Diagnostic"]
    return d


def render_html_table(data: pd.DataFrame, max_rows: int = 12, height: int = 400):
    """Table HTML hardcodée — indépendante du thème Streamlit."""
    DIAG_C = {
        "Poids Normal":       ("#06301a","#10b981"),
        "Poids Insuffisant":  ("#062040","#38bdf8"),
        "Surpoids Niveau I":  ("#2a1c04","#fbbf24"),
        "Surpoids Niveau II": ("#2a1204","#f97316"),
        "Obésité Type I":     ("#2a1a00","#f59e0b"),
        "Obésité Type II":    ("#2a0814","#f43f5e"),
        "Obésité Type III":   ("#1a0838","#a78bfa"),
    }
    TH = ("background:#0a1628;color:#3a5a7a;font-size:.7rem;font-weight:700;"
          "letter-spacing:.07em;text-transform:uppercase;padding:.6rem .9rem;"
          "text-align:left;border-bottom:2px solid #1e3456;white-space:nowrap;"
          "position:sticky;top:0;z-index:1;")
    S_IDX = "background:{bg};color:#1e3456;font-size:.72rem;font-family:'JetBrains Mono',monospace;padding:.58rem .9rem;border-bottom:1px solid #0e1a28;"
    S_NUM = "background:{bg};color:#5a80a0;font-size:.82rem;font-family:'JetBrains Mono',monospace;padding:.58rem .9rem;border-bottom:1px solid #0e1a28;text-align:right;"
    S_TXT = "background:{bg};color:#c8d8e8;font-size:.82rem;padding:.58rem .9rem;border-bottom:1px solid #0e1a28;"

    subset = data.head(max_rows)
    header = (f"<th style='{TH}'>#</th>"
              + "".join([f"<th style='{TH}'>{c}</th>" for c in subset.columns]))
    rows = ""
    for ri,(_, row) in enumerate(subset.iterrows()):
        bg = "#0d1a2e" if ri%2==0 else "#0a1422"
        cells = f"<td style='{S_IDX.format(bg=bg)}'>{ri}</td>"
        for col in subset.columns:
            val = row[col]
            if col == "Diagnostic":
                bg2,fc = DIAG_C.get(str(val),("#0d1a2e","#7a9ab8"))
                cells += (f"<td style='{S_TXT.format(bg=bg)}'>"
                          f"<span style='background:{bg2};color:{fc};border:1px solid {fc}30;"
                          f"border-radius:5px;padding:.16rem .55rem;font-size:.75rem;font-weight:700;"
                          f"white-space:nowrap'>{val}</span></td>")
            elif isinstance(val, float):
                cells += f"<td style='{S_NUM.format(bg=bg)}'>{val:.2f}</td>"
            elif isinstance(val, (int, np.integer)):
                cells += f"<td style='{S_NUM.format(bg=bg)}'>{val}</td>"
            else:
                cells += f"<td style='{S_TXT.format(bg=bg)}'>{val}</td>"
        rows += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:12px;
                overflow:auto;max-height:{height}px;box-shadow:0 4px 24px rgba(0,0,0,.5)'>
        <table style='width:100%;border-collapse:collapse;min-width:900px'>
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style='padding:.4rem 1rem;font-size:.69rem;color:#1e3456;
                    border-top:1px solid #0e1a28;background:#08111e'>
            {min(max_rows,len(data))} / {len(data)} lignes affichées
        </div>
    </div>""", unsafe_allow_html=True)


def render_cmp_table(sc_df):
    """Tableau de comparaison 100% inline — jamais influencé par le thème."""
    metrics    = ["Accuracy","F1-Score","Précision","Rappel"]
    best_v     = {m: sc_df[m].max() for m in metrics}
    worst_v    = {m: sc_df[m].min() for m in metrics}

    S_TH = ("background:#0a1628;color:#3a5a7a;font-size:.71rem;font-weight:700;"
            "letter-spacing:.09em;text-transform:uppercase;padding:.75rem 1.2rem;"
            "text-align:left;border-bottom:2px solid #1e3456;")
    S_TD = ("background:#0d1a2e;color:#c8d8e8;font-size:.9rem;"
            "font-family:'JetBrains Mono',monospace;padding:.82rem 1.2rem;"
            "border-bottom:1px solid #0e1a28;text-align:right;")
    S_AL = ("background:#0d1a2e;color:#c8d8e8;font-size:.87rem;font-weight:600;"
            "padding:.82rem 1.2rem;border-bottom:1px solid #0e1a28;")

    header = (f"<th style='{S_TH}'>Algorithme</th>"
              + "".join([f"<th style='{S_TH}text-align:right'>{m}</th>" for m in metrics]))
    rows   = ""
    for algo in ALGO_LIST:
        cells = (f"<td style='{S_AL}'>"
                 f"<span style='color:{ALGO_COLORS[algo]};margin-right:.4rem'>{ALGO_ICONS[algo]}</span>"
                 f"{algo}</td>")
        for m in metrics:
            v = sc_df.loc[algo,m]
            if v == best_v[m]:
                sty = f"{S_TD}color:#34d399;font-weight:800;"
                suf = (" <span style='font-size:.6rem;background:#062e1a;color:#34d399;"
                       "border:1px solid #34d39930;border-radius:3px;padding:.05rem .3rem;"
                       "font-family:sans-serif;vertical-align:middle'>▲</span>")
            elif v == worst_v[m]:
                sty = f"{S_TD}color:#fb7185;font-weight:600;"
                suf = ""
            else:
                sty,suf = S_TD,""
            cells += f"<td style='{sty}'>{v:.2f}%{suf}</td>"
        rows += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:12px;
                overflow:hidden;margin-top:.5rem;box-shadow:0 6px 30px rgba(0,0,0,.5)'>
        <table style='width:100%;border-collapse:collapse'>
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <div style='padding:.45rem 1.2rem;font-size:.7rem;color:#1e3456;
                    border-top:1px solid #0e1a28;background:#08111e'>
            <span style='color:#34d399'>▲</span> Meilleur score &nbsp;·&nbsp;
            <span style='color:#fb7185'>Rouge</span> = score le plus bas
        </div>
    </div>""", unsafe_allow_html=True)


# ── Train model — cache par algo ──────────────────────────────
# Chaque algorithme est entraîné et mis en cache séparément.
# Le cache est invalidé si les données ou le code changent.
@st.cache_resource
def train_model(algo: str = BEST_ALGO):
    df   = load_data()
    X    = df.drop("NObeyesdad", axis=1)
    y    = df["NObeyesdad"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    sc   = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    clfs = {
        "LightGBM Classifier":      LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=63,random_state=42,verbose=-1),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
        "XGBoost Classifier":       XGBClassifier(n_estimators=200,learning_rate=.05,max_depth=6,random_state=42,eval_metric="mlogloss",verbosity=0),
    }
    clf = clfs[algo]
    clf.fit(Xtrs, ytr)
    yp  = clf.predict(Xtes)
    return (clf, sc, X.columns.tolist(),
            accuracy_score(yte,yp), f1_score(yte,yp,average="weighted"),
            precision_score(yte,yp,average="weighted"), recall_score(yte,yp,average="weighted"),
            confusion_matrix(yte,yp), classification_report(yte,yp,output_dict=True),
            Xtes, yte, yp)


@st.cache_data
def compare_models():
    rows = {}
    for a in ALGO_LIST:
        r = train_model(a)
        rows[a] = {"Accuracy":round(r[3]*100,2),"F1-Score":round(r[4]*100,2),
                   "Précision":round(r[5]*100,2),"Rappel":round(r[6]*100,2)}
    return pd.DataFrame(rows).T


# ═══════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════
for key,val in [("patient",{}),("waiting",0),("trained_algos",set())]:
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.6rem 0 1rem'>
        <div style='font-size:2.8rem'>🏥</div>
        <div style='font-family:"Playfair Display",serif;font-size:1.45rem;
                    font-weight:700;color:#e8f4ff;margin-top:.4rem'>ObesoScan</div>
        <div style='font-size:.67rem;color:rgba(255,255,255,.3);margin-top:4px;
                    font-weight:600;letter-spacing:.12em;text-transform:uppercase'>
            Système Clinique IA · Groupe 7
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                "text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:.5rem'>Rôle</div>",
                unsafe_allow_html=True)
    role     = st.radio("role", ROLES, label_visibility="collapsed")
    is_nurse = role == ROLES[0]
    is_doc   = not is_nurse
    st.divider()

    st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                "text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:.5rem'>Navigation</div>",
                unsafe_allow_html=True)
    page = st.radio("nav", NURSE_PAGES if is_nurse else DOC_PAGES, label_visibility="collapsed")
    st.divider()

    if is_doc:
        st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                    "text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:.5rem'>Algorithme ML</div>",
                    unsafe_allow_html=True)
        algo = st.radio("algo", ALGO_LIST, index=0, label_visibility="collapsed",
                        format_func=lambda x: ALGO_ICONS[x]+" "+x)
        is_cached = algo in st.session_state["trained_algos"]
        st.markdown(f"""
        <div style='background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);
                    border-radius:10px;padding:.7rem .9rem;margin-top:.6rem;font-size:.77rem;
                    color:#6ee7b7;line-height:1.55'>
            <strong style='color:#34d399'>⚡ LightGBM</strong> — modèle le plus performant.
            {"<br><span style='color:#059669;font-size:.68rem'>✓ Déjà entraîné (cache)</span>" if is_cached else ""}
        </div>""", unsafe_allow_html=True)
    else:
        algo = BEST_ALGO

    pat = st.session_state.get("patient", {})
    if pat:
        imc_sb = round(pat.get("weight",70)/(pat.get("height",1.70)**2),1)
        st.markdown(f"""
        <div style='background:rgba(14,165,233,.08);border:1px solid rgba(14,165,233,.18);
                    border-radius:10px;padding:.7rem .9rem;margin-top:.5rem;font-size:.76rem;
                    color:#7dd3fc;line-height:1.6'>
            👤 <strong style='color:#38bdf8'>Dossier actif</strong><br>
            {pat.get("gender","—")} · {pat.get("age","—")} ans · IMC {imc_sb}
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════
df = load_data()


# ╔═══════════════════════════════════════════════════════════╗
#  NURSE PAGE 1 — Dossier Patient
# ╚═══════════════════════════════════════════════════════════╝
if is_nurse and page == NURSE_PAGES[0]:
    st.markdown("""
    <div class='banner banner-nurse'>
        <div class='banner-pre'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-title'>Dossier Patient</div>
        <div class='banner-sub'>Saisie des données biométriques et administratives du patient</div>
        <span class='banner-badge'>saisie-initiale</span>
        <span class='banner-badge'>données-patient</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='panel p-nurse'>
        <div class='panel-title'>ℹ️ Instructions</div>
        <div class='panel-body'>
            Remplissez tous les champs. Les données seront transmises au médecin pour le
            diagnostic IA. Les champs <span style='color:#f43f5e;font-weight:700'>*</span>
            sont obligatoires.
        </div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🪪 Identité & Biométrie</div>",
                    unsafe_allow_html=True)
        gender = st.selectbox("Genre *", ["Féminin","Masculin"], key="n_gender")
        age    = st.number_input("Âge *", min_value=10, max_value=90, value=28, step=1, key="n_age")
        c1,c2  = st.columns(2)
        height = c1.number_input("Taille (m) *", min_value=1.40, max_value=2.15,
                                  value=1.70, step=0.01, format="%.2f", key="n_height")
        weight = c2.number_input("Poids (kg) *", min_value=30.0, max_value=200.0,
                                  value=70.0, step=0.5, key="n_weight")
        st.markdown("</div>", unsafe_allow_html=True)

        imc = round(weight/(height**2),1)
        if imc<18.5:   ic,it="#38bdf8","Poids Insuffisant"
        elif imc<25:   ic,it="#34d399","Poids Normal ✓"
        elif imc<30:   ic,it="#fbbf24","Surpoids"
        else:          ic,it="#f43f5e","Obésité ⚠️"
        st.markdown(f"""
        <div class='imc-live'>
            <div class='imc-label'>Indice de Masse Corporelle</div>
            <div class='imc-value' style='color:{ic}'>{imc}</div>
            <div class='imc-cat' style='color:{ic}'>{it}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='form-section' style='margin-top:1.2rem'>"
                    "<div class='form-title ft-nurse'>🩺 Antécédents & Statut</div>",
                    unsafe_allow_html=True)
        family = st.selectbox("Antécédents familiaux *", ["Non","Oui"], key="n_family")
        smoke  = st.selectbox("Tabagisme actif", ["Non","Oui"], key="n_smoke")
        scc    = st.selectbox("Surveillance calorique (SCC)", ["Non","Oui"], key="n_scc")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🍽️ Habitudes Alimentaires</div>",
                    unsafe_allow_html=True)
        favc = st.selectbox("Aliments très caloriques (FAVC)", ["Non","Oui"], key="n_favc")
        fcvc = st.slider("Fréquence légumes (FCVC)", 1.0, 3.0, 2.0, 0.1, key="n_fcvc",
                         help="1=Jamais · 2=Parfois · 3=Toujours")
        ncp  = st.slider("Repas principaux / jour (NCP)", 1.0, 4.0, 3.0, 0.5, key="n_ncp")
        caec = st.selectbox("Grignotage entre repas (CAEC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"], key="n_caec")
        calc = st.selectbox("Consommation alcool (CALC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"], key="n_calc")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🏃 Activité & Mode de Vie</div>",
                    unsafe_allow_html=True)
        ch2o   = st.slider("Eau / jour (L)", 1.0, 3.0, 2.0, 0.1, key="n_ch2o")
        faf    = st.slider("Activité physique (j/sem)", 0.0, 3.0, 1.0, 0.1, key="n_faf")
        tue    = st.slider("Temps écran (h/j)", 0.0, 2.0, 1.0, 0.1, key="n_tue")
        mtrans = st.selectbox("Transport principal", list(MTRANS_MAP.keys()), key="n_mtrans")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Récapitulatif</div>",
                unsafe_allow_html=True)
    cc1,cc2,cc3,cc4 = st.columns(4)
    cc1.metric("🧍 Patient", f"{['F','M'][GENDER_MAP[gender]]} · {age} ans")
    cc2.metric("📏 Taille / Poids", f"{height}m · {weight}kg")
    cc3.metric("📊 IMC", f"{imc}")
    cc4.metric("🏃 Activité", f"{faf}j/sem")

    st.markdown("""
    <div class='panel p-emerald' style='margin-top:1rem'>
        <div class='panel-title'>✅ Dossier prêt</div>
        <div class='panel-body'>
            Passez au <strong>Questionnaire Clinique</strong> pour compléter le dossier,
            puis transmettez au médecin.
        </div>
    </div>""", unsafe_allow_html=True)

    st.session_state["patient"] = {
        "gender":gender,"age":age,"height":height,"weight":weight,
        "family":family,"smoke":smoke,"scc":scc,"favc":favc,
        "fcvc":fcvc,"ncp":ncp,"caec":caec,"calc":calc,
        "ch2o":ch2o,"faf":faf,"tue":tue,"mtrans":mtrans,
    }


# ── NURSE PAGE 2 — Questionnaire Clinique ───────────────────
elif is_nurse and page == NURSE_PAGES[1]:
    st.markdown("""
    <div class='banner banner-nurse'>
        <div class='banner-pre'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-title'>Questionnaire Clinique</div>
        <div class='banner-sub'>Évaluation des facteurs de risque comportementaux</div>
        <span class='banner-badge'>questionnaire</span><span class='banner-badge'>facteurs-risque</span>
    </div>""", unsafe_allow_html=True)

    pat = st.session_state.get("patient",{})
    if not pat:
        st.markdown("""<div class='panel p-amber'>
            <div class='panel-title'>⚠️ Dossier manquant</div>
            <div class='panel-body'>Remplissez d'abord le <strong>Dossier Patient</strong>.</div>
        </div>""", unsafe_allow_html=True)
    else:
        c1,c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='form-section'><div class='form-title ft-nurse'>📋 Récapitulatif biométrique</div>",
                        unsafe_allow_html=True)
            imc_q = round(pat["weight"]/(pat["height"]**2),1)
            rows_s = [("Genre","Femme" if pat["gender"]=="Féminin" else "Homme"),
                      ("Âge",f'{pat["age"]} ans'),("Taille",f'{pat["height"]} m'),
                      ("Poids",f'{pat["weight"]} kg'),("IMC",f'{imc_q}'),
                      ("Ant. familiaux",pat["family"]),("Tabagisme",pat["smoke"])]
            html_r = "".join([f"<div class='stat-row'><span class='sk'>{k}</span>"
                              f"<span class='sv'>{v}</span></div>" for k,v in rows_s])
            st.markdown(f"{html_r}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🔍 Score de risque</div>",
                        unsafe_allow_html=True)
            score,flags = 0,[]
            if imc_q>=30:   score+=3; flags.append(("rose","IMC ≥ 30 — Obésité clinique"))
            elif imc_q>=25: score+=2; flags.append(("amber","IMC 25–30 — Zone Surpoids"))
            else:           flags.append(("emerald","IMC dans la norme"))
            if pat.get("family")=="Oui": score+=2; flags.append(("amber","Antécédents familiaux"))
            if pat.get("faf",1)<1.0:     score+=1; flags.append(("amber","Activité physique insuffisante"))
            if pat.get("smoke")=="Oui":  score+=1; flags.append(("amber","Tabagisme actif"))
            if pat.get("caec") in ["Fréquemment","Toujours"]: score+=1; flags.append(("amber","Grignotage fréquent"))
            if pat.get("calc") in ["Fréquemment","Toujours"]: score+=1; flags.append(("amber","Alcool fréquent"))
            if pat.get("ch2o",2)<1.5:    score+=1; flags.append(("rose","Hydratation insuffisante"))

            level = "Risque Faible" if score<=2 else "Risque Modéré" if score<=4 else "Risque Élevé"
            lc    = "#34d399" if score<=2 else "#fbbf24" if score<=4 else "#f43f5e"
            st.markdown(f"""
            <div style='background:rgba(0,0,0,.2);border:1px solid {lc}30;
                        border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem'>
                <div style='font-size:.66rem;font-weight:700;letter-spacing:.1em;
                            text-transform:uppercase;color:#4a6080;margin-bottom:.3rem'>Score de risque</div>
                <div style='font-family:"Playfair Display",serif;font-size:2rem;color:{lc}'>{score}/10</div>
                <div style='font-size:.8rem;font-weight:700;color:{lc};margin-top:.2rem'>{level}</div>
            </div>""", unsafe_allow_html=True)

            C_MAP = {"emerald":"#34d399","amber":"#fbbf24","rose":"#f43f5e"}
            I_MAP = {"emerald":"✅","amber":"⚠️","rose":"🚨"}
            for col,msg in flags:
                c = C_MAP[col]; icon = I_MAP[col]
                st.markdown(f"""
                <div style='background:rgba(0,0,0,.15);border-left:3px solid {c};
                            border-radius:0 8px 8px 0;padding:.44rem .9rem;
                            margin:.3rem 0;font-size:.8rem;color:#7a9ab8'>
                    {icon} {msg}
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='panel p-nurse' style='margin-top:1.5rem'>
            <div class='panel-title'>📨 Dossier prêt à transmettre</div>
            <div class='panel-body'>
                Le médecin peut accéder au <strong>Diagnostic IA</strong> pour la prédiction personnalisée.
            </div>
        </div>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════╗
#  DOCTOR PAGE 1 — Tableau de Bord
# ╚═══════════════════════════════════════════════════════════╝
elif is_doc and page == DOC_PAGES[0]:
    st.markdown("""
    <div class='banner banner-doctor'>
        <div class='banner-pre'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-title'>Tableau de Bord — Salle d'Attente</div>
        <div class='banner-sub'>Gestion en temps réel des patients présents dans la clinique</div>
        <span class='banner-badge'>gestion-flux</span>
        <span class='banner-badge'>temps-réel</span>
        <span class='banner-badge'>groupe-7</span>
    </div>""", unsafe_allow_html=True)

    n_wait = st.session_state["waiting"]
    if n_wait == 0:      ws_cls,ws_txt,ws_icon = "ws-low",  "Salle vide",       "🟢"
    elif n_wait <= 3:    ws_cls,ws_txt,ws_icon = "ws-low",  "Flux normal",       "🟢"
    elif n_wait <= 8:    ws_cls,ws_txt,ws_icon = "ws-mid",  "Attente modérée",   "🟡"
    else:                ws_cls,ws_txt,ws_icon = "ws-high", "Salle chargée",     "🔴"

    # ── Compteur principal centré ──
    st.markdown(f"""
    <div style='display:flex;justify-content:center;margin:2rem 0 1.5rem'>
        <div style='background:linear-gradient(160deg,#071a38,#0d2548);
                    border:1px solid rgba(14,165,233,.3);border-radius:24px;
                    padding:3rem 5rem;text-align:center;
                    box-shadow:0 12px 48px rgba(0,0,0,.6);min-width:320px'>
            <div style='font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;
                        color:#3a5a7a;margin-bottom:.8rem'>Patients en salle d'attente</div>
            <div style='font-family:"Playfair Display",serif;font-size:6rem;
                        color:#e8f4ff;line-height:1;font-weight:700'>{n_wait}</div>
            <div style='margin-top:1rem'>
                <span class='wait-status {ws_cls}'>{ws_icon} {ws_txt}</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Boutons d'action ──
    st.markdown("<div class='sec-head'><div class='dot dot-sky'></div>Actions</div>",
                unsafe_allow_html=True)
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1, 1, 1, 1], gap="medium")

    with btn_c1:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid rgba(14,165,233,.2);border-radius:14px;
                    padding:1.4rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2rem;margin-bottom:.4rem'>➕</div>
            <div style='font-size:.75rem;color:#3a5a7a;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Nouveau patient</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Patient arrivé (+1)", use_container_width=True, key="btn_add"):
            st.session_state["waiting"] += 1
            st.rerun()

    with btn_c2:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid rgba(16,185,129,.2);border-radius:14px;
                    padding:1.4rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2rem;margin-bottom:.4rem'>✅</div>
            <div style='font-size:.75rem;color:#3a5a7a;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Patient traité</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Patient sorti (−1)", use_container_width=True, key="btn_sub"):
            if st.session_state["waiting"] > 0:
                st.session_state["waiting"] -= 1
            st.rerun()

    with btn_c3:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid rgba(245,158,11,.2);border-radius:14px;
                    padding:1.4rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2rem;margin-bottom:.4rem'>🔢</div>
            <div style='font-size:.75rem;color:#3a5a7a;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Définir le total</div>
        </div>""", unsafe_allow_html=True)
        manual_n = st.number_input("Nombre exact", min_value=0, max_value=100,
                                    value=st.session_state["waiting"], step=1, key="wait_manual",
                                    label_visibility="collapsed")
        if manual_n != st.session_state["waiting"]:
            st.session_state["waiting"] = manual_n
            st.rerun()

    with btn_c4:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid rgba(244,63,94,.2);border-radius:14px;
                    padding:1.4rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2rem;margin-bottom:.4rem'>🗑️</div>
            <div style='font-size:.75rem;color:#3a5a7a;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Réinitialiser</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Remettre à zéro", use_container_width=True, key="btn_reset"):
            st.session_state["waiting"] = 0
            st.rerun()

    # ── Jauge visuelle ──
    st.markdown("<div class='sec-head'><div class='dot dot-sky'></div>Visualisation de la capacité (20 places)</div>",
                unsafe_allow_html=True)
    n_cur     = st.session_state["waiting"]
    max_slots = 20
    if n_cur <= 3:    slot_color = "#34d399"
    elif n_cur <= 8:  slot_color = "#fbbf24"
    else:             slot_color = "#f43f5e"

    dots = ""
    for i in range(max_slots):
        if i < min(n_cur, max_slots):
            dots += (f"<span title='Patient {i+1}' style='display:inline-flex;align-items:center;"
                     f"justify-content:center;width:44px;height:44px;border-radius:10px;"
                     f"background:{slot_color}20;border:1.5px solid {slot_color}60;"
                     f"margin:4px;font-size:.75rem;font-weight:700;color:{slot_color};"
                     f"font-family:\"JetBrains Mono\",monospace'>{i+1}</span>")
        else:
            dots += (f"<span style='display:inline-flex;align-items:center;justify-content:center;"
                     f"width:44px;height:44px;border-radius:10px;background:#0a1422;"
                     f"border:1.5px solid #1e3456;margin:4px;font-size:.75rem;color:#1e3456;"
                     f"font-family:\"JetBrains Mono\",monospace'>{i+1}</span>")

    overflow = ""
    if n_cur > max_slots:
        overflow = (f"<div style='margin-top:.8rem;font-size:.8rem;color:#f43f5e;font-weight:700'>"
                    f"+ {n_cur - max_slots} patients hors capacité affichée</div>")

    pct = min(int(n_cur / max_slots * 100), 100)
    bar_c = "#34d399" if pct <= 40 else "#fbbf24" if pct <= 70 else "#f43f5e"
    alert = (f"<div style='margin-top:1.2rem;background:rgba(244,63,94,.08);"
             f"border:1px solid rgba(244,63,94,.25);border-radius:8px;padding:.6rem 1rem;"
             f"font-size:.79rem;color:#fb7185;font-weight:600'>"
             f"⚠️ Capacité dépassée — prévoir un renfort médical</div>") if n_cur > max_slots//2 else ""

    st.markdown(f"""
    <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:14px;
                padding:1.6rem 1.8rem'>
        <div style='display:flex;justify-content:space-between;align-items:center;
                    margin-bottom:1rem'>
            <div style='font-size:.68rem;font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;color:#3a5a7a'>Occupation</div>
            <div style='font-family:"JetBrains Mono",monospace;font-size:.9rem;
                        color:{bar_c};font-weight:700'>{n_cur}/{max_slots} — {pct}%</div>
        </div>
        <div style='background:#172847;border-radius:6px;height:8px;overflow:hidden;margin-bottom:1.2rem'>
            <div style='width:{pct}%;height:100%;background:{bar_c};
                        border-radius:6px;transition:width .4s'></div>
        </div>
        <div style='line-height:1.6'>{dots}</div>
        {overflow}{alert}
    </div>""", unsafe_allow_html=True)


# ── DOCTOR PAGE 2 — Exploration Clinique ────────────────────
elif is_doc and page == DOC_PAGES[1]:
    st.markdown("""
    <div class='banner banner-doctor'>
        <div class='banner-pre'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-title'>Exploration Clinique</div>
        <div class='banner-sub'>Analyse exploratoire des variables biométriques et comportementales</div>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distributions","📦 Boxplots","🔵 Relations","🗂️ Dataset"])
    num_cols = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()

    with tab1:
        chosen = st.selectbox("Variable clinique", num_cols)
        c1,c2  = st.columns(2)
        with c1:
            fig,ax = dark_fig(6,4)
            ax.hist(df[chosen],bins=40,color="#0ea5e9",edgecolor="none",alpha=.85)
            ax.axvline(df[chosen].mean(),  color="#f43f5e",linestyle="--",lw=1.8,
                       label=f"Moy : {df[chosen].mean():.2f}")
            ax.axvline(df[chosen].median(),color="#34d399",linestyle="--",lw=1.8,
                       label=f"Méd : {df[chosen].median():.2f}")
            ax.set_xlabel(chosen,fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.18,linestyle="--")
            ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True)
        with c2:
            fig,ax = dark_fig(6,4)
            for i in range(7):
                v = df[df["NObeyesdad"]==i][chosen].dropna()
                if len(v)>5: v.plot.kde(ax=ax,color=CLASS_HEX[i],label=CLASS_NAMES[i],lw=2)
            ax.set_xlabel(chosen,fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y",alpha=.15,linestyle="--")
            ax.legend(fontsize=7,ncol=2)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True)

        s = df[chosen]
        cc = st.columns(5)
        for met,val in zip(["Moyenne","Médiane","Écart-type","Min","Max"],
                            [s.mean(),s.median(),s.std(),s.min(),s.max()]):
            cc[["Moyenne","Médiane","Écart-type","Min","Max"].index(met)].metric(met,f"{val:.3f}")

        nr = int(np.ceil(len(num_cols)/4))
        fig_a,axes = dark_fig(14,nr*3,ncols=4,nrows=nr)
        axes = axes.flatten()
        for idx,cn in enumerate(num_cols):
            axes[idx].hist(df[cn],bins=25,color=PALETTE[idx%len(PALETTE)],edgecolor="none",alpha=.85)
            axes[idx].set_title(cn,fontsize=8.5,color="#7a9ab8")
            axes[idx].spines[["top","right"]].set_visible(False)
            axes[idx].tick_params(labelsize=7)
        for j in range(len(num_cols),len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_a,use_container_width=True)

    with tab2:
        bxv = st.selectbox("Variable",num_cols,key="bxv")
        fig,ax = dark_fig(12,5)
        for i in range(7):
            vals = df[df["NObeyesdad"]==i][bxv]
            ax.boxplot(vals,positions=[i],widths=.58,patch_artist=True,
                       boxprops=dict(facecolor=CLASS_HEX[i],alpha=.65),
                       medianprops=dict(color="white",linewidth=2.5),
                       whiskerprops=dict(color="#2d4a6a",lw=1.2),
                       capprops=dict(color="#2d4a6a",lw=1.2),
                       flierprops=dict(marker="o",color="#2d4a6a",markersize=2.5,alpha=.4))
        ax.set_xticks(range(7))
        ax.set_xticklabels([CLASS_NAMES[i].replace(" ","\n") for i in range(7)],fontsize=8.5,color="#7a9ab8")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18,linestyle="--")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True)

        fig2,ax2 = dark_fig(10,5)
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax2.scatter(sub["Height"],sub["Weight"],s=20,alpha=.45,
                        color=CLASS_HEX[i],label=CLASS_NAMES[i],edgecolors="none")
        ax2.set_xlabel("Taille (m)",fontsize=9); ax2.set_ylabel("Poids (kg)",fontsize=9)
        ax2.set_title("Taille vs Poids — cartographie biométrique",fontsize=11,pad=10,color="#dde6f0")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(alpha=.15,linestyle="--")
        ax2.legend(fontsize=7,ncol=2)
        plt.tight_layout(); st.pyplot(fig2,use_container_width=True)

    with tab3:
        sx1,sx2 = st.columns(2)
        xv = sx1.selectbox("Axe X",num_cols,index=2)
        yv = sx2.selectbox("Axe Y",num_cols,index=3)
        fig3,ax3 = dark_fig(10,5)
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax3.scatter(sub[xv],sub[yv],s=20,alpha=.45,
                        color=CLASS_HEX[i],label=CLASS_NAMES[i],edgecolors="none")
        ax3.set_xlabel(xv,fontsize=9); ax3.set_ylabel(yv,fontsize=9)
        ax3.set_title(f"{xv} vs {yv}",fontsize=11,pad=10,color="#dde6f0")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(alpha=.15,linestyle="--")
        ax3.legend(fontsize=7.5,ncol=2)
        plt.tight_layout(); st.pyplot(fig3,use_container_width=True)

    with tab4:
        df_dec   = decode_df(df)
        cols_sel = st.multiselect("Colonnes",df_dec.columns.tolist(),default=df_dec.columns.tolist())
        st.markdown(f"<span class='chip chip-sky'>{len(df):,} patients</span>"
                    f"<span class='chip'>{len(cols_sel)} colonnes</span>",
                    unsafe_allow_html=True)
        render_html_table(df_dec[cols_sel], max_rows=20, height=520)


# ── DOCTOR PAGE 3 — Analyse Statistique ─────────────────────
elif is_doc and page == DOC_PAGES[2]:
    st.markdown("""
    <div class='banner banner-doctor'>
        <div class='banner-pre'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-title'>Analyse Statistique</div>
        <div class='banner-sub'>Corrélations cliniques, statistiques descriptives et détection d'anomalies</div>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🔗 Corrélations","📋 Statistiques","⚠️ Outliers"])

    with tab1:
        corr = df.corr()
        fig,ax = dark_fig(10,8)
        mask   = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,ax=ax,mask=mask,
                    cmap=sns.diverging_palette(220,20,as_cmap=True),
                    center=0,annot=True,fmt=".2f",annot_kws={"size":7.5},
                    linewidths=.4,linecolor="#07101f",cbar_kws={"shrink":.7})
        ax.set_title("Corrélations inter-variables",fontsize=12,pad=10,color="#dde6f0")
        plt.xticks(fontsize=7.5,rotation=45,ha="right",color="#7a9ab8")
        plt.yticks(fontsize=7.5,color="#7a9ab8")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True)

        tc = corr["NObeyesdad"].drop("NObeyesdad").sort_values(key=abs,ascending=False)
        fig2,ax2 = dark_fig(9,4.5)
        ax2.barh(tc.index,tc.values,
                 color=["#34d399" if v>0 else "#f43f5e" for v in tc.values],
                 edgecolor="none",height=.58)
        ax2.axvline(0,color="#1e3456",lw=1.5)
        ax2.set_xlabel("Coefficient de Pearson",fontsize=9)
        ax2.set_title("Impact sur le diagnostic",fontsize=11,pad=10,color="#dde6f0")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="x",alpha=.18,linestyle="--")
        for i,(v,n) in enumerate(zip(tc.values,tc.index)):
            ax2.text(v+(.004 if v>=0 else -.004),i,f"{v:.3f}",va="center",
                     ha="left" if v>=0 else "right",fontsize=8,color="#4a6080",fontweight="600")
        plt.tight_layout(); st.pyplot(fig2,use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T, use_container_width=True, height=400)
        sv  = st.selectbox("Variable par classe",
                            df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist())
        sbc = df.groupby("NObeyesdad")[sv].describe().round(3)
        sbc.index = [CLASS_NAMES[i] for i in sbc.index]
        st.dataframe(sbc, use_container_width=True)

    with tab3:
        nc   = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()
        rows = []
        for c in nc:
            Q1,Q3 = df[c].quantile(.25),df[c].quantile(.75); IQR=Q3-Q1
            n = ((df[c]<Q1-1.5*IQR)|(df[c]>Q3+1.5*IQR)).sum()
            rows.append({"Variable":c,"Q1":round(Q1,3),"Q3":round(Q3,3),
                         "IQR":round(IQR,3),"Outliers":n,"% Outliers":round(n/len(df)*100,2)})
        out = pd.DataFrame(rows).sort_values("Outliers",ascending=False)
        st.dataframe(out, use_container_width=True)
        fig3,ax3 = dark_fig(9,4)
        ax3.bar(out["Variable"],out["% Outliers"],
                color=["#f43f5e" if v>5 else "#f59e0b" if v>2 else "#10b981" for v in out["% Outliers"]],
                edgecolor="none",width=.6)
        ax3.set_ylabel("% Outliers",fontsize=9)
        ax3.set_title("Taux d'anomalies par variable",fontsize=11,pad=10,color="#dde6f0")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(axis="y",alpha=.18,linestyle="--")
        plt.xticks(rotation=30,ha="right",fontsize=8.5)
        plt.tight_layout(); st.pyplot(fig3,use_container_width=True)


# ── DOCTOR PAGE 4 — Comparaison des Modèles ─────────────────
elif is_doc and page == DOC_PAGES[3]:
    st.markdown("""
    <div class='banner banner-doctor'>
        <div class='banner-pre'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-title'>Comparaison des Modèles</div>
        <div class='banner-sub'>Mise en compétition · Random Forest · XGBoost · LightGBM</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("⏳ Évaluation des 3 modèles…"):
        sc_df = compare_models()
    # Marquer les 3 comme entraînés
    for a in ALGO_LIST:
        st.session_state["trained_algos"].add(a)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Tableau comparatif</div>",
                unsafe_allow_html=True)
    render_cmp_table(sc_df)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Comparaison visuelle</div>",
                unsafe_allow_html=True)
    metrics_c = ["Accuracy","F1-Score","Précision","Rappel"]
    fig_c,axes_c = dark_fig(16,5,ncols=4,nrows=1)
    for idx,metric in enumerate(metrics_c):
        vals  = sc_df[metric]
        bars  = axes_c[idx].bar(range(3),vals.values,
                                color=[ALGO_COLORS[a] for a in ALGO_LIST],
                                edgecolor="none",width=.52)
        axes_c[idx].set_xticks(range(3))
        axes_c[idx].set_xticklabels(
            [a.replace(" Classifier","") for a in ALGO_LIST],
            fontsize=7.5,rotation=18,ha="right",color="#7a9ab8")
        axes_c[idx].set_ylim(vals.min()-3,100)
        axes_c[idx].set_title(f"{metric} (%)",fontsize=9.5,pad=8,fontweight="600",color="#dde6f0")
        axes_c[idx].spines[["top","right"]].set_visible(False)
        axes_c[idx].grid(axis="y",alpha=.18,linestyle="--")
        for bar,v in zip(bars,vals.values):
            axes_c[idx].text(bar.get_x()+bar.get_width()/2,v+.15,
                             f"{v:.1f}%",ha="center",va="bottom",
                             fontsize=7.5,color="#dde6f0",fontweight="700")
    plt.tight_layout(); st.pyplot(fig_c,use_container_width=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Radar Chart</div>",
                unsafe_allow_html=True)
    cats   = ["Accuracy","F1-Score","Précision","Rappel"]; N=len(cats)
    angles = [n/float(N)*2*np.pi for n in range(N)]; angles += angles[:1]
    plt.rcParams.update({"figure.facecolor":"#0d1a2e","axes.facecolor":"#0d1a2e","text.color":"#dde6f0"})
    fig_r,ax_r = plt.subplots(figsize=(6.5,6.5),subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor("#0d1a2e"); ax_r.set_facecolor("#112039")
    ax_r.grid(color="#1e3456",linestyle="--",lw=.8)
    ax_r.spines["polar"].set_color("#1e3456")
    for a in ALGO_LIST:
        vr = [sc_df.loc[a,m] for m in cats]; vr += vr[:1]
        ax_r.plot(angles,vr,lw=2.5,color=ALGO_COLORS[a],label=a.replace(" Classifier",""))
        ax_r.fill(angles,vr,alpha=.1,color=ALGO_COLORS[a])
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(cats,fontsize=10.5,color="#7a9ab8",fontweight="600")
    ax_r.set_ylim(80,100)
    ax_r.tick_params(axis="y",colors="#3a5a7a",labelsize=7.5)
    ax_r.legend(loc="upper right",bbox_to_anchor=(1.45,1.1),fontsize=9.5,labelcolor="#dde6f0")
    ax_r.set_title("Performance comparative",fontsize=11,color="#dde6f0",pad=20)
    plt.tight_layout(); st.pyplot(fig_r,use_container_width=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>🏆 Podium</div>",
                unsafe_allow_html=True)
    best = sc_df["Accuracy"].idxmax()
    c1,c2,c3 = st.columns(3,gap="medium")
    for col,a in zip([c1,c2,c3],ALGO_LIST):
        col.markdown(f"""
        <div class='model-card {"best" if a==best else ""}' style='text-align:center'>
            <div class='model-icon'>{ALGO_ICONS[a]}</div>
            <div class='model-name' style='color:{ALGO_COLORS[a]}'>{a.replace(" Classifier","")}</div>
            <div class='model-score' style='color:{ALGO_COLORS[a]}'>{sc_df.loc[a,"Accuracy"]:.2f}%</div>
            <div style='color:#4a6080;font-size:.82rem;font-weight:600'>Accuracy</div>
            <div style='color:#3a5a7a;font-size:.79rem;margin-top:.3rem'>F1 : {sc_df.loc[a,"F1-Score"]:.2f}%</div>
        </div>""", unsafe_allow_html=True)


# ── DOCTOR PAGE 6 — Diagnostic IA ───────────────────────────
elif is_doc and page == DOC_PAGES[4]:
    st.markdown(f"""
    <div class='banner banner-doctor'>
        <div class='banner-pre'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-title'>Diagnostic Individuel IA</div>
        <div class='banner-sub'>Prédiction personnalisée · {ALGO_ICONS[algo]} <strong>{algo}</strong>
            {"&ensp;· ⭐ Meilleur modèle" if algo==BEST_ALGO else ""}</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Initialisation du modèle…"):
        clf,sc_m,fc,acc,f1,prec,rec,*_ = train_model(algo)
    st.session_state["trained_algos"].add(algo)

    st.markdown(
        f"<span class='chip chip-sky'>{ALGO_ICONS[algo]} {algo}</span>"
        f"<span class='chip'>✅ Acc {acc*100:.1f}%</span>"
        f"<span class='chip'>F1 {f1*100:.1f}%</span>"
        f"<span class='chip'>Prec {prec*100:.1f}%</span>"
        f"<span class='chip'>Rapp {rec*100:.1f}%</span>",
        unsafe_allow_html=True)

    pat = st.session_state.get("patient",{})
    if pat:
        st.markdown("""
        <div class='panel p-nurse'>
            <div class='panel-title'>🔗 Dossier infirmière importé</div>
            <div class='panel-body'>Données pré-chargées. Ajustez si nécessaire avant le diagnostic.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Paramètres Patient</div>",
                unsafe_allow_html=True)

    def pidx(lst,key,default):
        v = pat.get(key,default)
        try: return lst.index(v)
        except: return lst.index(default)

    col1,col2,col3 = st.columns(3,gap="large")

    with col1:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🪪 Biométrie</div>",
                    unsafe_allow_html=True)
        gender = st.selectbox("Genre",["Féminin","Masculin"],
                              index=pidx(["Féminin","Masculin"],"gender","Féminin"),key="d_gender")
        age    = st.slider("Âge",10,80,pat.get("age",26),key="d_age")
        height = st.slider("Taille (m)",1.40,2.10,float(pat.get("height",1.70)),0.01,key="d_height")
        weight = st.slider("Poids (kg)",30.0,170.0,float(pat.get("weight",70.0)),0.5,key="d_weight")
        family = st.selectbox("Antécédents familiaux",["Non","Oui"],
                              index=pidx(["Non","Oui"],"family","Non"),key="d_family")
        st.markdown("</div>",unsafe_allow_html=True)

        imc_d = round(weight/(height**2),1)
        if imc_d<18.5:   idc,idt="#38bdf8","Poids Insuffisant"
        elif imc_d<25:   idc,idt="#34d399","Poids Normal ✓"
        elif imc_d<30:   idc,idt="#fbbf24","Surpoids"
        else:            idc,idt="#f43f5e","Obésité ⚠️"
        st.markdown(f"""
        <div class='imc-live'>
            <div class='imc-label'>IMC Calculé</div>
            <div class='imc-value' style='color:{idc}'>{imc_d}</div>
            <div class='imc-cat' style='color:{idc}'>{idt}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🍽️ Alimentation</div>",
                    unsafe_allow_html=True)
        favc = st.selectbox("Aliments caloriques",["Non","Oui"],
                            index=pidx(["Non","Oui"],"favc","Non"),key="d_favc")
        fcvc = st.slider("Légumes (FCVC)",1.0,3.0,float(pat.get("fcvc",2.0)),0.1,key="d_fcvc")
        ncp  = st.slider("Repas / jour",1.0,4.0,float(pat.get("ncp",3.0)),0.5,key="d_ncp")
        caec = st.selectbox("Grignotage (CAEC)",["Jamais","Parfois","Fréquemment","Toujours"],
                            index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"caec","Parfois"),key="d_caec")
        calc = st.selectbox("Alcool (CALC)",["Jamais","Parfois","Fréquemment","Toujours"],
                            index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"calc","Jamais"),key="d_calc")
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("""
        <div class='panel p-emerald' style='margin-top:.8rem'>
            <div class='panel-title'>💡 Référence OMS</div>
            <div class='panel-body'>5 fruits/légumes/jour et 3 repas équilibrés réduisent le risque d'obésité de 35%.</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🏃 Mode de Vie</div>",
                    unsafe_allow_html=True)
        smoke  = st.selectbox("Tabagisme",["Non","Oui"],
                              index=pidx(["Non","Oui"],"smoke","Non"),key="d_smoke")
        ch2o   = st.slider("Eau / jour (L)",1.0,3.0,float(pat.get("ch2o",2.0)),0.1,key="d_ch2o")
        scc    = st.selectbox("Surveillance calorique",["Non","Oui"],
                              index=pidx(["Non","Oui"],"scc","Non"),key="d_scc")
        faf    = st.slider("Activité (j/sem)",0.0,3.0,float(pat.get("faf",1.0)),0.1,key="d_faf")
        tue    = st.slider("Temps écran (h/j)",0.0,2.0,float(pat.get("tue",1.0)),0.1,key="d_tue")
        mtrans = st.selectbox("Transport",list(MTRANS_MAP.keys()),
                              index=pidx(list(MTRANS_MAP.keys()),"mtrans","Automobile"),key="d_mtrans")
        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("")
    bcol,_ = st.columns([1,3])
    with bcol:
        diag_btn = st.button("🩺  Lancer le Diagnostic",use_container_width=True)

    if diag_btn:
        row = {
            "Gender":GENDER_MAP[gender],"Age":float(age),
            "Height":height,"Weight":weight,
            "family_history_with_overweight":BINARY_MAP[family],
            "FAVC":BINARY_MAP[favc],"FCVC":fcvc,"NCP":ncp,
            "CAEC":CAEC_MAP[caec],"SMOKE":BINARY_MAP[smoke],
            "CH2O":ch2o,"SCC":BINARY_MAP[scc],
            "FAF":faf,"TUE":tue,
            "CALC":CALC_MAP[calc],"MTRANS":MTRANS_MAP[mtrans],
        }
        Xn   = pd.DataFrame([row])[fc]
        Xns  = sc_m.transform(Xn)
        pred  = int(clf.predict(Xns)[0])
        proba = clf.predict_proba(Xns)[0] if hasattr(clf,"predict_proba") else None
        info  = CLASS_INFO[pred]
        rb    = CLASS_RB[pred]
        emoji = "✅" if info[1]=="green" else "⚠️" if info[1]=="amber" else "🚨"

        # Décrémenter salle d'attente si possible
        if st.session_state["waiting"] > 0:
            st.session_state["waiting"] -= 1

        st.markdown("---")
        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Résultat du Diagnostic</div>",
                    unsafe_allow_html=True)
        rc1,rc2 = st.columns([1.2,1],gap="large")
        with rc1:
            st.markdown(f"""
            <div class='result-box {rb}'>
                <span class='result-emoji'>{emoji}</span>
                <div class='result-title'>{info[0]}</div>
                <div class='result-imc'>IMC : {imc_d} &nbsp;·&nbsp; {info[2]}</div>
                <div class='result-desc'>{info[3]}</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            col_imc = idc
            st.markdown(f"""
            <div class='panel p-doctor' style='height:100%'>
                <div class='panel-title'>📋 Résumé Patient</div>
                <div style='font-size:.85rem;line-height:2.1;color:#7a9ab8'>
                    <b style='color:#dde6f0'>Genre :</b> {"Homme" if gender=="Masculin" else "Femme"}<br>
                    <b style='color:#dde6f0'>Âge :</b> {age} ans<br>
                    <b style='color:#dde6f0'>Taille / Poids :</b> {height} m · {weight} kg<br>
                    <b style='color:#dde6f0'>IMC :</b>
                    <span style='color:{col_imc};font-family:"JetBrains Mono",monospace;
                                 font-weight:800;font-size:1rem'>{imc_d}</span><br>
                    <b style='color:#dde6f0'>Activité :</b> {faf} j/sem<br>
                    <b style='color:#dde6f0'>Hydratation :</b> {ch2o} L/j<br>
                    <b style='color:#dde6f0'>Tabagisme :</b> {smoke}<br>
                    <b style='color:#dde6f0'>Ant. familiaux :</b> {family}
                </div>
            </div>""", unsafe_allow_html=True)

        if proba is not None:
            st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Probabilités diagnostiques</div>",
                        unsafe_allow_html=True)
            fig_p,ax_p = dark_fig(11,4.5)
            alpha_p = [1.0 if i==pred else .38 for i in range(7)]
            bars_p  = ax_p.bar([CLASS_NAMES[i] for i in range(7)],proba,
                               color=[CLASS_HEX[i] for i in range(7)],edgecolor="none",width=.58)
            for bar,av in zip(bars_p,alpha_p): bar.set_alpha(av)
            ax_p.set_ylim(0,1.15)
            ax_p.set_ylabel("Probabilité",fontsize=9,color="#4a6080")
            ax_p.spines[["top","right"]].set_visible(False)
            ax_p.grid(axis="y",alpha=.18,linestyle="--")
            plt.xticks(rotation=22,ha="right",fontsize=8.5,color="#7a9ab8")
            for bar,p in zip(bars_p,proba):
                if p>.015:
                    ax_p.text(bar.get_x()+bar.get_width()/2,p+.015,
                              f"{p*100:.1f}%",ha="center",va="bottom",
                              fontsize=8.5,color="#dde6f0",fontweight="700")
            plt.tight_layout(); st.pyplot(fig_p,use_container_width=True)

        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Recommandations Médicales</div>",
                    unsafe_allow_html=True)
        recs = []
        if faf<1.0:
            recs.append(("rose","🏃","Activité physique insuffisante",
                         "≥ 150 min d'activité modérée/sem (OMS). Débuter par 20 min/j de marche rapide."))
        elif faf>=2.5:
            recs.append(("emerald","🏃","Activité physique optimale",
                         f"Niveau excellent ({faf} j/sem). Réduction du risque cardiovasculaire de 30%."))
        else:
            recs.append(("amber","🏃","Activité physique à renforcer",
                         "Progresser vers 3–4 séances/semaine (recommandations OMS 2024)."))
        if ch2o<1.5:
            recs.append(("rose","💧","Hydratation critique",
                         f"{ch2o} L/j. Objectif minimum : 2 L/j (2.5 L en période chaude)."))
        elif ch2o>=2.0:
            recs.append(("emerald","💧","Hydratation satisfaisante",
                         f"{ch2o} L/jour — conforme aux recommandations EFSA."))
        if caec in ["Fréquemment","Toujours"]:
            recs.append(("rose","🍪","Grignotage excessif",
                         "+20–30% d'apport calorique. Orienter vers un diététicien."))
        if smoke=="Oui":
            recs.append(("rose","🚬","Tabagisme actif",
                         "Perturbe le métabolisme lipidique. Consultation sevrage tabagique."))
        if family=="Oui":
            recs.append(("amber","🧬","Prédisposition génétique",
                         "Risque ×2–3. Suivi médical annuel et bilan métabolique complet."))
        if imc_d>=30:
            recs.append(("rose","⚕️","Consultation spécialiste urgente",
                         "Bilan lipidique, glycémie à jeun, TA. Orientation endocrinologue / nutritionniste."))
        elif 25<=imc_d<30:
            recs.append(("amber","⚕️","Suivi préventif recommandé",
                         "Consultation diététicien et bilan cardiovasculaire préventif."))
        else:
            recs.append(("emerald","⚕️","Profil clinique satisfaisant",
                         "IMC OMS normal. Maintenir les habitudes. Prochain bilan dans 12 mois."))

        RC_MAP = {"emerald":"rc-emerald","amber":"rc-amber","rose":"rc-rose"}
        for color,icon,title,text in recs:
            st.markdown(f"""
            <div class='rec-card {RC_MAP[color]}'>
                <div class='rec-icon'>{icon}</div>
                <div>
                    <div class='rec-title'>{title}</div>
                    <div class='rec-text'>{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#0d1a2e;border-radius:10px;padding:.85rem 1.3rem;
                    margin-top:1.5rem;border:1px solid rgba(255,255,255,.05);
                    font-size:.76rem;color:#1e3456;text-align:center;line-height:1.7'>
            ⚠️ <strong style='color:#2d4a6a'>Avertissement légal :</strong>
            Ce diagnostic IA est à des fins éducatives uniquement.
            Il ne se substitue pas à l'évaluation clinique d'un professionnel de santé qualifié.
        </div>""", unsafe_allow_html=True)
