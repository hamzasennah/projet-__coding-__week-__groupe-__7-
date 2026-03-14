import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, precision_score, recall_score,
)

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
#  CSS — DARK PREMIUM MEDICAL
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #0a0f1e !important;
    --surface:   #111827 !important;
    --surface2:  #1a2235 !important;
    --surface3:  #222d42 !important;
    --border:    rgba(255,255,255,.08);
    --glow:      rgba(0,212,180,.25);
    --teal:      #00d4b4;
    --teal-dim:  rgba(0,212,180,.12);
    --blue:      #3b82f6;
    --green:     #22c55e;
    --green-dim: rgba(34,197,94,.12);
    --amber:     #f59e0b;
    --amber-dim: rgba(245,158,11,.12);
    --red:       #ef4444;
    --red-dim:   rgba(239,68,68,.12);
    --violet:    #8b5cf6;
    --violet-dim:rgba(139,92,246,.12);
    --cyan:      #06b6d4;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --dim:       #94a3b8;
    --shadow:    0 0 30px rgba(0,212,180,.12);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main,
.block-container,
[class*="css"] {
    background-color: #0a0f1e !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid rgba(255,255,255,.08) !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.08) !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: .87rem !important;
    padding: .55rem .75rem !important;
    border-radius: 8px;
    margin: 2px 0 !important;
    transition: all .18s;
    border: 1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1a2235 !important;
    border-color: rgba(0,212,180,.25) !important;
}

.main .block-container { padding: 1.8rem 2.2rem 3rem; max-width: 1500px; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #222d42; border-radius: 3px; }

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
    background: #1a2235 !important;
    border-color: rgba(255,255,255,.1) !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] option,
[data-baseweb="menu"] {
    background: #111827 !important;
    color: #e2e8f0 !important;
}
[data-baseweb="popover"] { background: #111827 !important; }
[role="listbox"] { background: #111827 !important; }
[role="option"] { color: #e2e8f0 !important; }
[role="option"]:hover { background: #1a2235 !important; }
label, .stSelectbox label, .stSlider label,
.stNumberInput label, [data-testid="stWidgetLabel"] {
    color: #94a3b8 !important;
    font-size: .84rem !important;
}

.stSlider [data-baseweb="thumb"] { background: #00d4b4 !important; border-color: #00d4b4 !important; }
.stSlider [data-baseweb="track-fill"] { background: #00d4b4 !important; }
.stSlider [data-baseweb="track"] { background: #222d42 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    padding: .3rem !important;
    gap: .2rem;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    border-radius: 8px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1a2235 !important;
    color: #00d4b4 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

div[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: .75rem !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.6rem !important;
}

.stButton > button {
    background: linear-gradient(135deg,#00a896,#00d4b4) !important;
    color: #0a0f1e !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 2rem !important;
    font-family: 'DM Sans',sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    font-size: .88rem !important;
    box-shadow: 0 4px 20px rgba(0,212,180,.25) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    opacity: .9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,212,180,.35) !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
    background: #111827 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,.08) !important;
}
.stDataFrame thead th {
    background: #1a2235 !important;
    color: #94a3b8 !important;
    font-size: .78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: .06em;
    border-bottom: 1px solid rgba(255,255,255,.08) !important;
}
.stDataFrame tbody tr { background: #111827 !important; }
.stDataFrame tbody tr:nth-child(even) { background: #141c2e !important; }
.stDataFrame tbody td {
    color: #e2e8f0 !important;
    font-size: .83rem !important;
    border-bottom: 1px solid rgba(255,255,255,.04) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stDataFrame tbody td:first-child {
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stAlert, [data-testid="stNotification"] {
    background: #111827 !important;
    border-color: rgba(255,255,255,.1) !important;
    color: #e2e8f0 !important;
}
.streamlit-expanderHeader {
    background: #111827 !important;
    color: #e2e8f0 !important;
}

.stSpinner > div { border-top-color: #00d4b4 !important; }

.stNumberInput button {
    background: #222d42 !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,.1) !important;
}

[data-baseweb="tag"] { background: #222d42 !important; }
[data-baseweb="tag"] span { color: #e2e8f0 !important; }

hr { border-color: rgba(255,255,255,.08) !important; }

/* ══════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════ */

.page-banner {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg,rgba(0,212,180,.05) 0%,transparent 60%);
    pointer-events: none;
}
.banner-nurse::before  { background: linear-gradient(135deg,rgba(6,182,212,.07) 0%,transparent 60%); }
.banner-doctor::before { background: linear-gradient(135deg,rgba(124,58,237,.07) 0%,transparent 60%); }
.banner-eyebrow {
    font-size: .7rem; font-weight: 700; letter-spacing: .13em;
    text-transform: uppercase; margin-bottom: .55rem;
}
.ey-nurse  { color: #06b6d4; }
.ey-doctor { color: #a78bfa; }
.banner-h1 {
    font-family: 'DM Serif Display',serif;
    font-size: 2rem; font-weight: 400; color: #f8fafc;
    margin: 0 0 .4rem; line-height: 1.15;
}
.banner-sub { font-size: .9rem; color: #94a3b8; line-height: 1.6; margin: 0; }
.banner-tag {
    display: inline-block;
    background: #222d42; border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px; padding: .2rem .65rem;
    font-size: .71rem; font-weight: 600; color: #64748b;
    margin: .6rem .3rem 0 0;
    font-family: 'JetBrains Mono',monospace;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit,minmax(140px,1fr));
    gap: .9rem; margin: 1.5rem 0;
}
.kpi-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px; padding: 1.4rem 1.3rem;
    position: relative; overflow: hidden;
    transition: border-color .2s,transform .2s;
}
.kpi-card:hover { border-color: rgba(0,212,180,.25); transform: translateY(-2px); }
.kpi-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: #00d4b4; border-radius: 0 0 14px 14px;
}
.c-blue::after   { background: #3b82f6; }
.c-green::after  { background: #22c55e; }
.c-amber::after  { background: #f59e0b; }
.c-red::after    { background: #ef4444; }
.c-violet::after { background: #8b5cf6; }
.kpi-num {
    font-family: 'DM Serif Display',serif;
    font-size: 2rem; color: #00d4b4; line-height: 1; margin-bottom: .35rem;
}
.c-blue   .kpi-num { color: #3b82f6; }
.c-green  .kpi-num { color: #22c55e; }
.c-amber  .kpi-num { color: #f59e0b; }
.c-red    .kpi-num { color: #ef4444; }
.c-violet .kpi-num { color: #8b5cf6; }
.kpi-lbl {
    font-size: .7rem; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: #64748b;
}

.sec-head {
    font-family: 'DM Serif Display',serif;
    font-size: 1.12rem; color: #f8fafc;
    margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: .6rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid rgba(255,255,255,.08);
}
.dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-teal   { background:#00d4b4; box-shadow:0 0 8px #00d4b4; }
.dot-nurse  { background:#06b6d4; box-shadow:0 0 8px #06b6d4; }
.dot-doctor { background:#8b5cf6; box-shadow:0 0 8px #8b5cf6; }
.dot-violet { background:#8b5cf6; box-shadow:0 0 8px #8b5cf6; }

.panel {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-left: 4px solid #00d4b4;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
}
.p-teal   { border-left-color: #00d4b4; }
.p-blue   { border-left-color: #3b82f6; }
.p-green  { border-left-color: #22c55e; }
.p-amber  { border-left-color: #f59e0b; }
.p-red    { border-left-color: #ef4444; }
.p-violet { border-left-color: #8b5cf6; }
.p-nurse  { border-left-color: #06b6d4; }
.p-doctor { border-left-color: #8b5cf6; }
.panel-title {
    font-weight: 700; font-size: .87rem; color: #00d4b4;
    margin-bottom: .3rem; letter-spacing: .03em;
}
.p-green  .panel-title { color: #22c55e; }
.p-amber  .panel-title { color: #f59e0b; }
.p-red    .panel-title { color: #ef4444; }
.p-violet .panel-title { color: #a78bfa; }
.p-nurse  .panel-title { color: #06b6d4; }
.p-doctor .panel-title { color: #a78bfa; }
.panel-body { font-size: .84rem; color: #94a3b8; line-height: 1.6; }

.model-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px; padding: 1.6rem 1.4rem;
    text-align: center; transition: all .22s;
    position: relative; overflow: hidden;
    height: 100%;
}
.model-card:hover { border-color: rgba(0,212,180,.25); transform: translateY(-3px); }
.model-card.best {
    border-color: rgba(0,212,180,.35);
    background: linear-gradient(160deg,rgba(0,212,180,.06),#111827);
}
.model-card.best::before {
    content: '⭐ Meilleur Modèle';
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,212,180,.15); color: #00d4b4;
    border: 1px solid rgba(0,212,180,.3);
    border-radius: 20px; padding: .18rem .7rem;
    font-size: .65rem; font-weight: 700; letter-spacing: .05em;
}
.model-icon  { font-size: 2.2rem; margin-bottom: .6rem; }
.model-name  { font-weight: 700; font-size: .95rem; color: #e2e8f0; margin-bottom: .35rem; }
.model-score { font-family:'DM Serif Display',serif; font-size:1.8rem; margin:.5rem 0; }
.model-desc  { font-size: .77rem; color: #64748b; line-height: 1.5; }

.result-box {
    border-radius: 16px; padding: 2.2rem;
    text-align: center; margin: 1rem 0;
    border: 1px solid rgba(255,255,255,.08);
    background: #111827;
    position: relative; overflow: hidden;
}
.rb-green { border-color:rgba(34,197,94,.3);  background:linear-gradient(160deg,rgba(34,197,94,.06),#111827); }
.rb-amber { border-color:rgba(245,158,11,.3); background:linear-gradient(160deg,rgba(245,158,11,.06),#111827); }
.rb-red   { border-color:rgba(239,68,68,.3);  background:linear-gradient(160deg,rgba(239,68,68,.06),#111827); }
.result-emoji { font-size:3rem; display:block; margin-bottom:.8rem; }
.result-title {
    font-family:'DM Serif Display',serif;
    font-size:1.9rem; font-weight:400;
}
.rb-green .result-title { color:#86efac; }
.rb-amber .result-title { color:#fcd34d; }
.rb-red   .result-title { color:#fca5a5; }
.result-imc  { font-family:'JetBrains Mono',monospace; font-size:.87rem; color:#64748b; margin:.5rem 0; }
.result-desc { font-size:.87rem; color:#94a3b8; margin-top:.5rem; }

.rec-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-left: 4px solid #00d4b4;
    border-radius: 10px; padding: 1rem 1.3rem; margin: .5rem 0;
    display: flex; align-items: flex-start; gap: 1rem;
    transition: transform .18s, border-color .18s;
}
.rec-card:hover { transform: translateX(4px); border-color: rgba(0,212,180,.3); }
.rc-green { border-left-color: #22c55e; }
.rc-amber { border-left-color: #f59e0b; }
.rc-red   { border-left-color: #ef4444; }
.rec-icon  { font-size:1.35rem; flex-shrink:0; margin-top:2px; }
.rec-title { font-weight:700; font-size:.87rem; color:#e2e8f0; margin-bottom:.2rem; }
.rec-text  { font-size:.79rem; color:#64748b; line-height:1.55; }

.imc-live {
    border-radius: 12px; padding: 1.2rem;
    text-align: center; margin-top: .8rem;
    border: 1px solid rgba(255,255,255,.08);
    background: #1a2235;
}
.imc-label { font-size:.67rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#64748b; }
.imc-value { font-family:'DM Serif Display',serif; font-size:2.6rem; line-height:1.1; margin:.2rem 0; }
.imc-cat   { font-size:.82rem; font-weight:700; margin-top:2px; }

.chip {
    display: inline-flex; align-items: center; gap: .3rem;
    background: #1a2235; color: #94a3b8;
    border: 1px solid rgba(255,255,255,.08); border-radius: 6px;
    padding: .24rem .72rem; font-size: .74rem; font-weight: 600;
    margin: .2rem; font-family: 'JetBrains Mono',monospace;
}
.chip-teal { background:rgba(0,212,180,.1); color:#00d4b4; border-color:rgba(0,212,180,.25); }
.chip-violet { background:rgba(139,92,246,.1); color:#a78bfa; border-color:rgba(139,92,246,.25); }

.form-section {
    background: #111827;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.form-title {
    font-size: .78rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: 1rem;
    padding-bottom: .5rem; border-bottom: 1px solid rgba(255,255,255,.08);
}
.ft-nurse  { color: #06b6d4; }
.ft-doctor { color: #8b5cf6; }

.stat-row {
    display: flex; justify-content: space-between;
    padding: .4rem 0; border-bottom: 1px solid rgba(255,255,255,.05);
    font-size: .84rem;
}
.stat-row:last-child { border-bottom: none; }
.sk { color: #64748b; font-weight: 500; }
.sv { color: #e2e8f0; font-weight: 700; font-family:'JetBrains Mono',monospace; font-size:.81rem; }

.cmp-table { width:100%; border-collapse:collapse; }
.cmp-table th {
    background: #1a2235; color: #64748b; font-size:.75rem;
    font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:.7rem 1rem; text-align:left; border-bottom:1px solid rgba(255,255,255,.08);
}
.cmp-table td {
    padding:.75rem 1rem; border-bottom:1px solid rgba(255,255,255,.05);
    color:#e2e8f0; font-size:.88rem; font-family:'JetBrains Mono',monospace;
}
.cmp-table tr:hover td { background: #141c2e; }
.cmp-best  { color:#00d4b4 !important; font-weight:700 !important; }
.cmp-worst { color:#ef4444 !important; }

/* SHAP specific */
.shap-legend-item {
    display: inline-flex; align-items: center; gap: .5rem;
    margin: .2rem .5rem; font-size: .8rem; color: #94a3b8;
}
.shap-dot {
    width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0;
}
.shap-insight-card {
    background: #0f1928;
    border: 1px solid rgba(139,92,246,.2);
    border-radius: 12px; padding: 1.2rem 1.4rem; margin: .4rem 0;
    display: flex; align-items: flex-start; gap: .9rem;
}
.shap-insight-icon { font-size: 1.4rem; flex-shrink: 0; }
.shap-insight-title { font-weight: 700; font-size: .85rem; color: #c4b5fd; margin-bottom: .15rem; }
.shap-insight-text  { font-size: .79rem; color: #64748b; line-height: 1.55; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
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
    0: ("Poids Insuffisant",  "green", "IMC < 18.5",      "Risque de carences nutritionnelles. Suivi médical recommandé."),
    1: ("Poids Normal",       "green", "18.5 ≤ IMC < 25", "Profil clinique sain. Maintenir les habitudes actuelles."),
    2: ("Obésité Type I",     "red",   "30 ≤ IMC < 35",   "Risque cardiovasculaire modéré. Suivi médical requis."),
    3: ("Obésité Type II",    "red",   "35 ≤ IMC < 40",   "Risque cardiovasculaire élevé. Consultation spécialiste."),
    4: ("Obésité Type III",   "red",   "IMC ≥ 40",        "Obésité morbide. Prise en charge médicale urgente."),
    5: ("Surpoids Niveau I",  "amber", "25 ≤ IMC < 27.5", "Surveiller l'alimentation. Augmenter l'activité physique."),
    6: ("Surpoids Niveau II", "amber", "27.5 ≤ IMC < 30", "Bilan lipidique conseillé. Consultation diététicien."),
}
CLASS_HEX = {
    0:"#22c55e",1:"#3b82f6",2:"#f59e0b",3:"#ef4444",
    4:"#8b5cf6",5:"#fbbf24",6:"#f97316",
}

ALGO_LIST   = ["LightGBM Classifier","Random Forest Classifier","XGBoost Classifier"]
ALGO_ICONS  = {"LightGBM Classifier":"⚡","Random Forest Classifier":"🌲","XGBoost Classifier":"🚀"}
ALGO_COLORS = {"LightGBM Classifier":"#00d4b4","Random Forest Classifier":"#3b82f6","XGBoost Classifier":"#f97316"}
ALGO_DESC   = {
    "LightGBM Classifier":      "Gradient Boosting ultra-rapide. Optimal sur données médicales tabulaires. Meilleure précision diagnostique.",
    "Random Forest Classifier": "Ensemble d'arbres de décision. Robuste et interprétable cliniquement.",
    "XGBoost Classifier":       "Extreme Gradient Boosting. Excellent équilibre vitesse / précision sur données structurées.",
}
BEST_ALGO = "LightGBM Classifier"

GENDER_MAP = {"Féminin":0,"Masculin":1}
BINARY_MAP = {"Non":0,"Oui":1}
CAEC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
CALC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
MTRANS_MAP = {"Automobile":0,"Vélo":1,"Moto":2,"Transport en commun":3,"Marche":4}

ROLES       = ["👩‍⚕️  Infirmière — Saisie Patient","👨‍⚕️  Médecin — Analyse & Diagnostic"]
NURSE_PAGES = ["📋  Dossier Patient","📏  Questionnaire Clinique","🏥  Tableau de Bord"]
DOC_PAGES   = ["📊  Exploration Clinique","📈  Analyse Statistique",
               "⚖️  Comparaison des Modèles","🩺  Diagnostic IA"]
PALETTE     = ["#00d4b4","#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#f97316","#06b6d4"]

# Labels lisibles pour les features SHAP
FEATURE_LABELS = {
    "Gender":                          "Genre",
    "Age":                             "Âge",
    "Height":                          "Taille (m)",
    "Weight":                          "Poids (kg)",
    "family_history_with_overweight":  "Ant. familiaux obésité",
    "FAVC":                            "Aliments caloriques (FAVC)",
    "FCVC":                            "Fréquence légumes (FCVC)",
    "NCP":                             "Repas/jour (NCP)",
    "CAEC":                            "Grignotage (CAEC)",
    "SMOKE":                           "Tabagisme",
    "CH2O":                            "Eau/jour (L)",
    "SCC":                             "Surveillance cal. (SCC)",
    "FAF":                             "Activité physique (j/sem)",
    "TUE":                             "Temps écran (h/j)",
    "CALC":                            "Alcool (CALC)",
    "MTRANS":                          "Transport",
}


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def dark_fig(w=10, h=5, ncols=1, nrows=1):
    plt.rcParams.update({
        "figure.facecolor":"#111827","axes.facecolor":"#1a2235",
        "axes.edgecolor":"#2d3a52",  "axes.labelcolor":"#94a3b8",
        "xtick.color":"#64748b",     "ytick.color":"#64748b",
        "text.color":"#e2e8f0",      "grid.color":"#1e293b",
        "legend.facecolor":"#111827","legend.edgecolor":"#2d3a52",
        "font.family":"DejaVu Sans", "figure.dpi":110,
    })
    if ncols==1 and nrows==1:
        return plt.subplots(figsize=(w,h))
    return plt.subplots(nrows,ncols,figsize=(w,h))


def set_dark_matplotlib():
    """Apply dark theme to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":"#111827","axes.facecolor":"#1a2235",
        "axes.edgecolor":"#2d3a52",  "axes.labelcolor":"#94a3b8",
        "xtick.color":"#64748b",     "ytick.color":"#64748b",
        "text.color":"#e2e8f0",      "grid.color":"#1e293b",
        "legend.facecolor":"#111827","legend.edgecolor":"#2d3a52",
        "font.family":"DejaVu Sans", "figure.dpi":110,
    })


@st.cache_data
def load_data():
    for p in ["data_clean.csv","data/data_clean.csv","../data/data_clean.csv",
              "../data_clean.csv","/mnt/user-data/uploads/data_clean__2_.csv"]:
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


@st.cache_resource
def train_model(algo=BEST_ALGO):
    df   = load_data()
    X    = df.drop("NObeyesdad", axis=1)
    y    = df["NObeyesdad"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    sc   = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)
    clfs = {
        "LightGBM Classifier":      LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=63, random_state=42, verbose=-1),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost Classifier":       XGBClassifier(n_estimators=200, learning_rate=.05, max_depth=6, random_state=42, eval_metric="mlogloss", verbosity=0),
    }
    clf  = clfs[algo]
    clf.fit(Xtrs, ytr)
    yp   = clf.predict(Xtes)

    # Explainer SHAP — TreeExplainer pour modèles d'arbres
    explainer = shap.TreeExplainer(clf)

    # Échantillon test pour les visualisations globales (max 200 pour la perf)
    shap_sample_size = min(200, len(Xtes))
    Xtes_sample = Xtes[:shap_sample_size]

    # Calcul des valeurs SHAP sur l'échantillon (liste de tableaux, un par classe)
    shap_values = explainer.shap_values(Xtes_sample)

    # Feature names depuis les colonnes originales
    feature_names = X.columns.tolist()

    return (
        clf, sc, feature_names,
        accuracy_score(yte, yp),
        f1_score(yte, yp, average="weighted"),
        precision_score(yte, yp, average="weighted"),
        recall_score(yte, yp, average="weighted"),
        confusion_matrix(yte, yp),
        classification_report(yte, yp, output_dict=True),
        Xtes, yte, yp,
        explainer, shap_values, Xtes_sample,
    )


@st.cache_data
def compare_models():
    rows = {}
    for a in ALGO_LIST:
        r = train_model(a)
        rows[a] = {"Accuracy":round(r[3]*100,2),"F1-Score":round(r[4]*100,2),
                   "Précision":round(r[5]*100,2),"Rappel":round(r[6]*100,2)}
    return pd.DataFrame(rows).T


def render_cmp_table(sc_df):
    metrics = ["Accuracy","F1-Score","Précision","Rappel"]
    best_vals  = {m: sc_df[m].max() for m in metrics}
    worst_vals = {m: sc_df[m].min() for m in metrics}
    rows_html = ""
    for algo in ALGO_LIST:
        icon = ALGO_ICONS[algo]; color = ALGO_COLORS[algo]
        cells = f"<td style='color:{color};font-weight:700;font-size:.87rem'>{icon} {algo}</td>"
        for m in metrics:
            v = sc_df.loc[algo, m]
            cls = "cmp-best" if v==best_vals[m] else ("cmp-worst" if v==worst_vals[m] else "")
            cells += f"<td class='{cls}'>{v:.2f}%</td>"
        rows_html += f"<tr>{cells}</tr>"
    header = "".join([f"<th>{h}</th>" for h in ["Algorithme"]+metrics])
    st.markdown(f"""
    <div style='background:#111827;border:1px solid rgba(255,255,255,.08);
                border-radius:12px;overflow:hidden;margin-top:.5rem'>
        <table class='cmp-table'>
            <thead><tr>{header}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div style='padding:.55rem 1rem;font-size:.72rem;color:#334155;
                    border-top:1px solid rgba(255,255,255,.05)'>
            <span style='color:#00d4b4;font-weight:700'>■</span> Meilleur &nbsp;
            <span style='color:#ef4444;font-weight:700'>■</span> Moins bon
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  SHAP UTILITIES
# ═══════════════════════════════════════════════════════════

def get_shap_for_class(shap_values, class_idx):
    """
    Extrait les valeurs SHAP pour une classe donnée.
    shap_values peut être :
      - une liste de tableaux (shape: [n_classes][n_samples, n_features]) → TreeExplainer standard
      - un tableau 3D (shape: [n_samples, n_features, n_classes])         → certaines versions
    """
    if isinstance(shap_values, list):
        return shap_values[class_idx]          # shape (n_samples, n_features)
    elif shap_values.ndim == 3:
        return shap_values[:, :, class_idx]    # shape (n_samples, n_features)
    else:
        return shap_values                     # binaire ou déjà mono-classe


def get_expected_value(explainer, class_idx):
    """Extrait la valeur de base (expected_value) pour la classe donnée."""
    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        return float(ev[class_idx])
    return float(ev)


def compute_global_shap_importance(shap_values, n_classes):
    """
    Calcule l'importance globale des features = moyenne des |SHAP| sur toutes les classes.
    Retourne un tableau de shape (n_features,).
    """
    importances = []
    for c in range(n_classes):
        sv_c = get_shap_for_class(shap_values, c)
        importances.append(np.abs(sv_c).mean(axis=0))
    return np.mean(importances, axis=0)


def plot_shap_waterfall_patient(explainer, patient_shap_values, pred_class,
                                 patient_data, feature_names, class_name, class_color):
    """
    Waterfall SHAP individuel pour le patient → classe prédite.
    Retourne la figure matplotlib.
    """
    set_dark_matplotlib()

    sv_patient = get_shap_for_class(patient_shap_values, pred_class)[0]
    base_val   = get_expected_value(explainer, pred_class)

    # Trier par |SHAP| décroissant
    n_display  = 10
    order      = np.argsort(np.abs(sv_patient))[::-1][:n_display]

    sv_top    = sv_patient[order]
    feat_top  = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    data_top  = patient_data[order]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1a2235")

    colors = [class_color if v > 0 else "#3b82f6" for v in sv_top]
    bars   = ax.barh(range(n_display), sv_top[::-1],
                     color=colors[::-1], edgecolor="none", height=0.62)

    labels_disp = [f"{feat_top[::-1][i]}  = {data_top[::-1][i]:.2f}" for i in range(n_display)]
    ax.set_yticks(range(n_display))
    ax.set_yticklabels(labels_disp, fontsize=8.5, color="#e2e8f0")
    ax.axvline(0, color="#475569", lw=1.2, linestyle="--")
    ax.set_xlabel("Contribution SHAP (impact sur la prédiction)", fontsize=9, color="#94a3b8")
    ax.set_title(f"Explication individuelle — {class_name}",
                 fontsize=11, color="#e2e8f0", pad=12, fontweight="600")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linestyle="--")

    # Annotations valeurs
    for bar, v in zip(bars[::-1], sv_top[::-1]):
        if abs(v) > 0.005:
            ax.text(v + (0.003 if v >= 0 else -0.003),
                    bar.get_y() + bar.get_height()/2,
                    f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.5, color="#e2e8f0", fontweight="700")

    plt.tight_layout()
    return fig


def plot_shap_global_importance(shap_values, feature_names, n_classes, class_color):
    """
    Graphique d'importance globale SHAP (mean |SHAP| toutes classes).
    """
    set_dark_matplotlib()

    importance = compute_global_shap_importance(shap_values, n_classes)
    sorted_idx = np.argsort(importance)
    labels     = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1a2235")

    bar_colors = [class_color if i == sorted_idx[-1] else
                  ("#a78bfa" if i == sorted_idx[-2] else "#00d4b4")
                  for i in sorted_idx]

    ax.barh(range(len(sorted_idx)), importance[sorted_idx],
            color=bar_colors, edgecolor="none", height=0.62)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(labels, fontsize=8.5, color="#e2e8f0")
    ax.set_xlabel("Importance SHAP moyenne |SHAP|", fontsize=9, color="#94a3b8")
    ax.set_title("Importance Globale des Variables", fontsize=11,
                 color="#e2e8f0", pad=12, fontweight="600")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linestyle="--")

    for i, v in enumerate(importance[sorted_idx]):
        ax.text(v + importance.max() * 0.01, i, f"{v:.4f}",
                va="center", fontsize=7.5, color="#64748b", fontweight="600")

    plt.tight_layout()
    return fig


def plot_shap_beeswarm(shap_values, Xtes_sample, feature_names, n_classes):
    """
    SHAP Summary Beeswarm plot (toutes classes, style médical sombre).
    """
    set_dark_matplotlib()

    # Agréger les valeurs SHAP : moyenne absolue sur les classes pour le beeswarm
    # On utilise la classe la plus fréquente prédite (ou on empile)
    importance = compute_global_shap_importance(shap_values, n_classes)
    order      = np.argsort(importance)[::-1][:12]  # top 12 features

    # Construire un tableau (n_samples, n_features) = moyenne toutes classes
    sv_mean = np.mean([get_shap_for_class(shap_values, c)
                       for c in range(n_classes)], axis=0)

    n_feats = len(order)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1a2235")

    for rank, feat_idx in enumerate(order[::-1]):
        sv_f  = sv_mean[:, feat_idx]
        raw_f = Xtes_sample[:, feat_idx]

        # Normaliser les valeurs brutes → couleur (bleu=bas, rouge=haut)
        vmin, vmax = raw_f.min(), raw_f.max()
        if vmax > vmin:
            norm = (raw_f - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(raw_f)

        colors = plt.cm.RdBu_r(norm)

        # Jitter vertical pour lisibilité
        jitter = np.random.normal(0, 0.08, size=len(sv_f))
        ax.scatter(sv_f, rank + jitter, c=colors, s=14,
                   alpha=0.65, linewidths=0, zorder=2)

    feat_labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
                   for i in order[::-1]]
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(feat_labels, fontsize=9, color="#e2e8f0")
    ax.axvline(0, color="#475569", lw=1.2, linestyle="--")
    ax.set_xlabel("Valeur SHAP (impact moyen sur la prédiction)", fontsize=9, color="#94a3b8")
    ax.set_title("Distribution SHAP — Impact de chaque variable sur les patients",
                 fontsize=11, color="#e2e8f0", pad=12, fontweight="600")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle="--")

    # Légende couleur (valeur haute/basse)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Valeur de la variable\n(bleu=bas  rouge=haut)", fontsize=7.5,
                   color="#64748b")
    cbar.ax.yaxis.set_tick_params(color="#64748b", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#64748b")
    cbar.outline.set_edgecolor("#2d3a52")

    plt.tight_layout()
    return fig


def generate_shap_insights(sv_patient, feature_names, pred_class, class_name):
    """
    Génère des insights médicaux automatiques basés sur les valeurs SHAP individuelles.
    Retourne une liste de tuples (icon, title, text).
    """
    insights = []
    order = np.argsort(np.abs(sv_patient))[::-1]

    top3_idx  = order[:3]
    top3_feat = [feature_names[i] for i in top3_idx]
    top3_sv   = [sv_patient[i]    for i in top3_idx]

    FEAT_CLINICAL = {
        "Weight":                         ("⚖️", "Le poids est le facteur n°{rank} dans ce diagnostic."),
        "Height":                         ("📏", "La taille est le facteur n°{rank} (influence via l'IMC)."),
        "Age":                            ("🎂", "L'âge joue un rôle n°{rank} dans cette prédiction."),
        "Gender":                         ("👤", "Le genre est le {rank}e facteur le plus influent."),
        "FAF":                            ("🏃", "L'activité physique est le facteur n°{rank} (±{val:+.3f})."),
        "CH2O":                           ("💧", "L'hydratation est le facteur n°{rank} dans ce diagnostic."),
        "FCVC":                           ("🥦", "La consommation de légumes influence la prédiction (rang {rank})."),
        "family_history_with_overweight": ("🧬", "Les antécédents familiaux constituent le facteur n°{rank}."),
        "FAVC":                           ("🍔", "La consommation d'aliments caloriques impacte le diagnostic (rang {rank})."),
        "CAEC":                           ("🍪", "Le grignotage est le facteur n°{rank} (impact {val:+.3f})."),
        "SMOKE":                          ("🚬", "Le tabagisme est le {rank}e facteur le plus influent."),
        "MTRANS":                         ("🚗", "Le mode de transport impacte le diagnostic (rang {rank})."),
    }

    for rank, (feat, sv) in enumerate(zip(top3_feat, top3_sv), 1):
        direction = "augmente" if sv > 0 else "réduit"
        label     = FEATURE_LABELS.get(feat, feat)
        if feat in FEAT_CLINICAL:
            icon, tpl = FEAT_CLINICAL[feat]
            title = tpl.format(rank=rank, val=sv)
        else:
            icon  = "📌"
            title = f"{label} est le facteur n°{rank} (impact {sv:+.3f})"
        text = (f"Cette variable {direction} la probabilité de '{class_name}' "
                f"de {abs(sv):.3f} point SHAP. "
                f"{'Valeur haute → risque accru.' if sv > 0 else 'Valeur basse → effet protecteur.'}")
        insights.append((icon, title, text))

    return insights


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;text-align:center'>
        <div style='font-size:2.6rem;margin-bottom:.4rem'>🏥</div>
        <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;
                    color:#f8fafc;font-weight:400'>ObesoScan</div>
        <div style='font-size:.67rem;color:#334155;font-weight:700;
                    letter-spacing:.12em;text-transform:uppercase;margin-top:3px'>
            Système Clinique IA · Groupe 7
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                "text-transform:uppercase;color:#334155;margin-bottom:.5rem'>Rôle</div>",
                unsafe_allow_html=True)
    role      = st.radio("role", ROLES, label_visibility="collapsed")
    is_nurse  = role == ROLES[0]
    is_doctor = not is_nurse
    st.divider()

    st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                "text-transform:uppercase;color:#334155;margin-bottom:.5rem'>Navigation</div>",
                unsafe_allow_html=True)
    page = st.radio("nav", NURSE_PAGES if is_nurse else DOC_PAGES, label_visibility="collapsed")
    st.divider()

    if is_doctor:
        st.markdown("<div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;"
                    "text-transform:uppercase;color:#334155;margin-bottom:.5rem'>Algorithme ML</div>",
                    unsafe_allow_html=True)
        algo = st.radio("algo", ALGO_LIST, index=0, label_visibility="collapsed",
                        format_func=lambda x: ALGO_ICONS[x]+" "+x)
        st.markdown("""
        <div style='background:rgba(0,212,180,.07);border:1px solid rgba(0,212,180,.2);
                    border-radius:10px;padding:.7rem .9rem;margin-top:.6rem;font-size:.77rem;
                    color:#5eead4;line-height:1.55'>
            <strong style='color:#00d4b4'>⚡ LightGBM</strong> — modèle le plus performant sur ce dataset médical.
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"""
        <div style='font-size:.72rem;color:#334155;line-height:1.8;padding-bottom:.5rem'>
            <span style='color:#a78bfa;font-weight:700'>👨‍⚕️ Mode Médecin</span><br>
            {ALGO_ICONS[algo]} {algo}
        </div>""", unsafe_allow_html=True)
    else:
        algo = BEST_ALGO
        st.markdown("""
        <div style='font-size:.72rem;color:#334155;line-height:1.8;padding-bottom:.5rem'>
            <span style='color:#06b6d4;font-weight:700'>👩‍⚕️ Mode Infirmière</span><br>
            Saisie & collecte des données patient.
        </div>""", unsafe_allow_html=True)

# ── Load data ──
df = load_data()


# ╔═══════════════════════════════════════════════════════════╗
#  NURSE — Page 1 : Dossier Patient
# ╚═══════════════════════════════════════════════════════════╝
if is_nurse and page == NURSE_PAGES[0]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Dossier Patient</div>
        <div class='banner-sub'>Saisie des données biométriques et administratives du patient</div>
        <span class='banner-tag'>saisie-initiale</span>
        <span class='banner-tag'>données-patient</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='panel p-nurse'>
        <div class='panel-title'>ℹ️ Instructions</div>
        <div class='panel-body'>
            Remplissez tous les champs. Les données collectées seront transmises au médecin
            pour le diagnostic IA. Les champs marqués <span style='color:#ef4444;font-weight:700'>*</span>
            sont obligatoires.
        </div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🪪 Identité & Biométrie</div>",
                    unsafe_allow_html=True)
        gender = st.selectbox("Genre *", ["Féminin","Masculin"], key="n_gender")
        age    = st.number_input("Âge (années) *", min_value=10, max_value=90, value=28, step=1, key="n_age")
        c1, c2 = st.columns(2)
        height = c1.number_input("Taille (m) *", min_value=1.40, max_value=2.15,
                                  value=1.70, step=0.01, format="%.2f", key="n_height")
        weight = c2.number_input("Poids (kg) *", min_value=30.0, max_value=200.0,
                                  value=70.0, step=0.5, key="n_weight")
        st.markdown("</div>", unsafe_allow_html=True)

        imc = round(weight/(height**2), 1)
        if imc<18.5:   imc_c,imc_t="#60a5fa","Poids Insuffisant"
        elif imc<25:   imc_c,imc_t="#22c55e","Poids Normal ✓"
        elif imc<30:   imc_c,imc_t="#f59e0b","Surpoids"
        else:          imc_c,imc_t="#ef4444","Obésité ⚠️"
        st.markdown(f"""
        <div class='imc-live'>
            <div class='imc-label'>Indice de Masse Corporelle</div>
            <div class='imc-value' style='color:{imc_c}'>{imc}</div>
            <div class='imc-cat' style='color:{imc_c}'>{imc_t}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='form-section' style='margin-top:1.2rem'>"
                    "<div class='form-title ft-nurse'>🩺 Antécédents & Statut</div>",
                    unsafe_allow_html=True)
        family = st.selectbox("Antécédents familiaux d'obésité *", ["Non","Oui"], key="n_family")
        smoke  = st.selectbox("Tabagisme actif", ["Non","Oui"], key="n_smoke")
        scc    = st.selectbox("Surveillance calorique (SCC)", ["Non","Oui"], key="n_scc")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🍽️ Habitudes Alimentaires</div>",
                    unsafe_allow_html=True)
        favc = st.selectbox("Aliments très caloriques (FAVC)", ["Non","Oui"], key="n_favc")
        fcvc = st.slider("Fréquence légumes (FCVC)", 1.0, 3.0, 2.0, 0.1, key="n_fcvc")
        ncp  = st.slider("Repas principaux / jour (NCP)", 1.0, 4.0, 3.0, 0.5, key="n_ncp")
        caec = st.selectbox("Alimentation entre les repas (CAEC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"], key="n_caec")
        calc = st.selectbox("Consommation d'alcool (CALC)",
                            ["Jamais","Parfois","Fréquemment","Toujours"], key="n_calc")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🏃 Activité & Mode de Vie</div>",
                    unsafe_allow_html=True)
        ch2o   = st.slider("Eau / jour (litres)", 1.0, 3.0, 2.0, 0.1, key="n_ch2o")
        faf    = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, 1.0, 0.1, key="n_faf")
        tue    = st.slider("Temps écran quotidien (heures)", 0.0, 2.0, 1.0, 0.1, key="n_tue")
        mtrans = st.selectbox("Transport principal", list(MTRANS_MAP.keys()), key="n_mtrans")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Récapitulatif du dossier</div>",
                unsafe_allow_html=True)
    cc1,cc2,cc3,cc4 = st.columns(4)
    cc1.metric("🧍 Patient",       f"{['F','M'][GENDER_MAP[gender]]} · {age} ans")
    cc2.metric("📏 Taille / Poids", f"{height}m · {weight}kg")
    cc3.metric("📊 IMC",           f"{imc}")
    cc4.metric("🏃 Activité",      f"{faf}j/sem")

    st.markdown("""
    <div class='panel p-green' style='margin-top:1rem'>
        <div class='panel-title'>✅ Dossier prêt pour le médecin</div>
        <div class='panel-body'>Données collectées. Passez au <strong>Questionnaire Clinique</strong> pour compléter, puis transmettez au médecin.</div>
    </div>""", unsafe_allow_html=True)

    st.session_state["patient"] = {
        "gender":gender,"age":age,"height":height,"weight":weight,
        "family":family,"smoke":smoke,"scc":scc,"favc":favc,
        "fcvc":fcvc,"ncp":ncp,"caec":caec,"calc":calc,
        "ch2o":ch2o,"faf":faf,"tue":tue,"mtrans":mtrans,
    }


# ── NURSE Page 2 : Questionnaire Clinique ───────────────────
elif is_nurse and page == NURSE_PAGES[1]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Questionnaire Clinique</div>
        <div class='banner-sub'>Évaluation complémentaire des facteurs de risque comportementaux</div>
        <span class='banner-tag'>questionnaire</span><span class='banner-tag'>facteurs-risque</span>
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
            rows  = [("Genre","Femme" if pat["gender"]=="Féminin" else "Homme"),
                     ("Âge",f'{pat["age"]} ans'),("Taille",f'{pat["height"]} m'),
                     ("Poids",f'{pat["weight"]} kg'),("IMC",f'{imc_q}'),
                     ("Ant. familiaux",pat["family"]),("Tabagisme",pat["smoke"])]
            html  = "".join([f"<div class='stat-row'><span class='sk'>{k}</span><span class='sv'>{v}</span></div>"
                             for k,v in rows])
            st.markdown(f"{html}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🔍 Évaluation des risques</div>",
                        unsafe_allow_html=True)
            score,flags = 0,[]
            if imc_q>=30:   score+=3; flags.append(("red","IMC ≥ 30 — Obésité clinique"))
            elif imc_q>=25: score+=2; flags.append(("amber","IMC 25–30 — Zone Surpoids"))
            else:           flags.append(("green","IMC dans la norme"))
            if pat.get("family")=="Oui": score+=2; flags.append(("amber","Antécédents familiaux d'obésité"))
            if pat.get("faf",1)<1.0:     score+=1; flags.append(("amber","Activité physique insuffisante"))
            if pat.get("smoke")=="Oui":  score+=1; flags.append(("amber","Tabagisme actif"))
            if pat.get("caec") in ["Fréquemment","Toujours"]: score+=1; flags.append(("amber","Grignotage fréquent"))
            if pat.get("calc") in ["Fréquemment","Toujours"]: score+=1; flags.append(("amber","Alcool fréquent"))
            if pat.get("ch2o",2)<1.5:    score+=1; flags.append(("red","Hydratation insuffisante"))

            level = "Risque Faible" if score<=2 else "Risque Modéré" if score<=4 else "Risque Élevé"
            lc    = "#22c55e" if score<=2 else "#f59e0b" if score<=4 else "#ef4444"
            st.markdown(f"""
            <div style='background:rgba(0,0,0,.2);border:1px solid {lc}33;
                        border-radius:12px;padding:1rem;text-align:center;margin-bottom:1rem'>
                <div style='font-size:.67rem;font-weight:700;letter-spacing:.1em;
                            text-transform:uppercase;color:#475569;margin-bottom:.3rem'>Score de risque</div>
                <div style='font-family:"DM Serif Display",serif;font-size:2rem;color:{lc}'>{score}/10</div>
                <div style='font-size:.81rem;font-weight:700;color:{lc};margin-top:.2rem'>{level}</div>
            </div>""", unsafe_allow_html=True)
            for col,msg in flags:
                icon = "✅" if col=="green" else "⚠️" if col=="amber" else "🚨"
                c    = "#22c55e" if col=="green" else "#f59e0b" if col=="amber" else "#ef4444"
                st.markdown(f"""
                <div style='background:rgba(0,0,0,.15);border-left:3px solid {c};
                            border-radius:0 8px 8px 0;padding:.44rem .9rem;
                            margin:.3rem 0;font-size:.8rem;color:#94a3b8'>
                    {icon} {msg}
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='panel p-nurse' style='margin-top:1.5rem'>
            <div class='panel-title'>📨 Transmission au médecin</div>
            <div class='panel-body'>Dossier complet. Le médecin peut accéder au module <strong>Diagnostic IA</strong> pour la prédiction et les recommandations personnalisées.</div>
        </div>""", unsafe_allow_html=True)


# ── NURSE Page 3 : Tableau de Bord ──────────────────────────
elif is_nurse and page == NURSE_PAGES[2]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Tableau de Bord - Compteur Patients</div>
        <div class='banner-sub'>Gestion simple du flux de patients dans la clinique</div>
        <span class='banner-tag'>compteur</span>
        <span class='banner-tag'>flux-patients</span>
    </div>""", unsafe_allow_html=True)

    # Initialisation du compteur
    if 'patient_counter' not in st.session_state:
        st.session_state.patient_counter = 0

    # Affichage du compteur principal
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div style='background:linear-gradient(160deg,#0d1a2e,#132237);
                    border:2px solid #0ea5e9;
                    border-radius:30px;
                    padding:3rem 2rem;
                    text-align:center;
                    margin:1rem 0 2rem 0;
                    box-shadow:0 20px 40px rgba(14,165,233,0.2)'>
            <div style='font-size:.8rem;font-weight:700;letter-spacing:.15em;
                        text-transform:uppercase;color:#4a6080;margin-bottom:1rem'>
                Patients en consultation
            </div>
            <div style='font-family:"DM Serif Display",serif;font-size:7rem;
                        font-weight:800;color:#e8f4ff;line-height:1;
                        text-shadow:0 0 30px rgba(14,165,233,0.5)'>
                {st.session_state.patient_counter}
            </div>
            <div style='margin-top:1rem'>
                <span style='display:inline-block;background:rgba(14,165,233,0.1);
                           border:1px solid #0ea5e9;border-radius:20px;
                           padding:.3rem 1rem;font-size:.8rem;color:#7dd3fc'>
                    ⏱️ Dernière mise à jour
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Boutons de contrôle
    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Contrôle du flux</div>",
                unsafe_allow_html=True)

    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4, gap="medium")

    with btn_c1:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #0ea5e9;border-radius:16px;
                    padding:1.5rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2.2rem;margin-bottom:.5rem'>➕</div>
            <div style='font-size:.75rem;color:#4a6080;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Nouveau patient</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Patient arrivé (+1)", use_container_width=True, key="btn_add_nurse"):
            st.session_state.patient_counter += 1
            st.session_state.setdefault('patient_log', []).append({
                "heure":   datetime.datetime.now().strftime("%H:%M:%S"),
                "type":    "arrivée",
                "message": "1 patient admis en clinique",
                "total":   st.session_state.patient_counter,
            })
            st.rerun()

    with btn_c2:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #10b981;border-radius:16px;
                    padding:1.5rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2.2rem;margin-bottom:.5rem'>✅</div>
            <div style='font-size:.75rem;color:#4a6080;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Patient traité</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Patient sorti (-1)", use_container_width=True, key="btn_sub_nurse"):
            if st.session_state.patient_counter > 0:
                st.session_state.patient_counter -= 1
                st.session_state.setdefault('patient_log', []).append({
                    "heure":   datetime.datetime.now().strftime("%H:%M:%S"),
                    "type":    "sortie",
                    "message": "1 patient a quitté la clinique",
                    "total":   st.session_state.patient_counter,
                })
            st.rerun()

    with btn_c3:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #f59e0b;border-radius:16px;
                    padding:1.5rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2.2rem;margin-bottom:.5rem'>🔄</div>
            <div style='font-size:.75rem;color:#4a6080;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Réinitialiser</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Remettre à zéro", use_container_width=True, key="btn_reset_nurse"):
            st.session_state.setdefault('patient_log', []).append({
                "heure":   datetime.datetime.now().strftime("%H:%M:%S"),
                "type":    "reset",
                "message": f"Remise à zéro ({st.session_state.patient_counter} patients sortis)",
                "total":   0,
            })
            st.session_state.patient_counter = 0
            st.rerun()

    with btn_c4:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #f43f5e;border-radius:16px;
                    padding:1.5rem;text-align:center;margin-bottom:.5rem'>
            <div style='font-size:2.2rem;margin-bottom:.5rem'>⚡</div>
            <div style='font-size:.75rem;color:#4a6080;font-weight:700;
                        letter-spacing:.06em;text-transform:uppercase'>Action rapide</div>
        </div>""", unsafe_allow_html=True)
        quick_add = st.number_input("Ajouter plusieurs", min_value=1, max_value=20,
                                     value=1, step=1, key="quick_add_nurse",
                                     label_visibility="collapsed")
        if st.button(f"Ajouter {quick_add}", use_container_width=True, key="btn_quick_nurse"):
            st.session_state.patient_counter += quick_add
            st.session_state.setdefault('patient_log', []).append({
                "heure":   datetime.datetime.now().strftime("%H:%M:%S"),
                "type":    "arrivée",
                "message": f"{quick_add} patient(s) admis en groupe",
                "total":   st.session_state.patient_counter,
            })
            st.rerun()

    # ── Calculs capacité ────────────────────────────────────
    CAPACITE_MAX  = 20
    nb_patients   = st.session_state.patient_counter
    places_libres = max(0, CAPACITE_MAX - nb_patients)
    capacity_pct  = min(int(nb_patients / CAPACITE_MAX * 100), 100)
    progress      = min(nb_patients / CAPACITE_MAX, 1.0)

    if capacity_pct == 0:
        statut_txt, statut_color, statut_icon = "Clinique vide", "#3b82f6", "🔵"
    elif capacity_pct < 50:
        statut_txt, statut_color, statut_icon = "Disponible", "#22c55e", "🟢"
    elif capacity_pct < 80:
        statut_txt, statut_color, statut_icon = "Affluence modérée", "#f59e0b", "🟡"
    elif capacity_pct < 100:
        statut_txt, statut_color, statut_icon = "Quasi complet", "#f97316", "🟠"
    else:
        statut_txt, statut_color, statut_icon = "COMPLET", "#ef4444", "🔴"

    bar_color = "#34d399" if progress < 0.3 else "#fbbf24" if progress < 0.7 else "#f43f5e"
    kpi_cls_places = "c-green" if places_libres > 10 else "c-amber" if places_libres > 4 else "c-red"

    # ── Alerte statut ────────────────────────────────────────
    if capacity_pct >= 100:
        st.markdown(f"""
        <div style='background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.4);
                    border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;
                    display:flex;align-items:center;gap:.8rem'>
            <span style='font-size:1.4rem'>🚨</span>
            <div>
                <div style='font-weight:700;color:#fca5a5;font-size:.9rem'>Capacité maximale atteinte</div>
                <div style='color:#64748b;font-size:.79rem'>Aucune nouvelle admission possible. Orientez les patients.</div>
            </div>
        </div>""", unsafe_allow_html=True)
    elif capacity_pct >= 80:
        st.markdown(f"""
        <div style='background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.35);
                    border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;
                    display:flex;align-items:center;gap:.8rem'>
            <span style='font-size:1.4rem'>⚠️</span>
            <div>
                <div style='font-weight:700;color:#fdba74;font-size:.9rem'>Capacité quasi atteinte</div>
                <div style='color:#64748b;font-size:.79rem'>Plus que {places_libres} place(s) disponible(s). Vigilance requise.</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── KPI row : 4 cartes ──────────────────────────────────
    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Tableau de capacité</div>",
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.markdown(f"""
        <div class='kpi-card c-blue'>
            <div class='kpi-num'>{nb_patients}</div>
            <div class='kpi-lbl'>Patients présents</div>
        </div>""", unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class='kpi-card {kpi_cls_places}'>
            <div class='kpi-num'>{places_libres}</div>
            <div class='kpi-lbl'>Places disponibles / {CAPACITE_MAX}</div>
        </div>""", unsafe_allow_html=True)

    with col_s3:
        kpi_cls_pct = "c-green" if capacity_pct < 50 else "c-amber" if capacity_pct < 80 else "c-red"
        st.markdown(f"""
        <div class='kpi-card {kpi_cls_pct}'>
            <div class='kpi-num'>{capacity_pct}%</div>
            <div class='kpi-lbl'>Taux d'occupation</div>
        </div>""", unsafe_allow_html=True)

    with col_s4:
        st.markdown(f"""
        <div class='kpi-card' style='border-bottom:3px solid {statut_color}'>
            <div style='font-size:1.6rem;margin-bottom:.35rem'>{statut_icon}</div>
            <div style='font-family:"DM Serif Display",serif;font-size:1.1rem;
                        color:{statut_color};line-height:1.2;font-weight:700'>
                {statut_txt}
            </div>
            <div class='kpi-lbl' style='margin-top:.35rem'>Statut clinique</div>
        </div>""", unsafe_allow_html=True)

    # ── Barre de progression avancée ────────────────────────
    segments = []
    for i in range(CAPACITE_MAX):
        if i < nb_patients:
            if nb_patients / CAPACITE_MAX < 0.5:
                seg_col = "#34d399"
            elif nb_patients / CAPACITE_MAX < 0.8:
                seg_col = "#fbbf24"
            else:
                seg_col = "#f43f5e"
        else:
            seg_col = "#172847"
        segments.append(f"""<div style='flex:1;height:28px;background:{seg_col};
            border-radius:4px;margin:0 1px;transition:background .3s'></div>""")

    st.markdown(f"""
    <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:16px;
                padding:1.6rem 1.8rem;margin-top:1rem'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem'>
            <span style='color:#4a6080;font-size:.82rem;font-weight:600'>
                📊 Occupation — {nb_patients} sur {CAPACITE_MAX} places
            </span>
            <span style='background:rgba(14,165,233,.1);border:1px solid #0ea5e9;
                         border-radius:20px;padding:.2rem .8rem;
                         font-size:.75rem;color:#7dd3fc;font-weight:700'>
                {places_libres} libre(s)
            </span>
        </div>
        <div style='display:flex;gap:2px;margin-bottom:.8rem'>
            {''.join(segments)}
        </div>
        <div style='display:flex;justify-content:space-between;font-size:.72rem;color:#334155'>
            <span>0</span>
            <span style='color:{statut_color};font-weight:700'>{statut_icon} {statut_txt}</span>
            <span>{CAPACITE_MAX}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Journal des mouvements ───────────────────────────────
    if 'patient_log' not in st.session_state:
        st.session_state.patient_log = []

    if 'last_counter' not in st.session_state:
        st.session_state.last_counter = nb_patients

    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Journal des mouvements</div>",
                unsafe_allow_html=True)

    if not st.session_state.patient_log:
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:12px;
                    padding:1.8rem;text-align:center;color:#334155;font-size:.83rem;
                    letter-spacing:.04em'>
            📋 &nbsp; Aucun mouvement enregistré pour le moment.
        </div>""", unsafe_allow_html=True)
    else:
        # En-tête du tableau
        st.markdown("""
        <div style='background:#0d1a2e;border:1px solid #1e3456;border-radius:12px;overflow:hidden'>
            <div style='display:grid;grid-template-columns:32px 80px 1fr 80px;
                        gap:0;padding:.5rem 1rem;
                        background:#111827;border-bottom:1px solid #1e3456'>
                <span style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:#334155'></span>
                <span style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:#334155'>Heure</span>
                <span style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:#334155'>Mouvement</span>
                <span style='font-size:.65rem;font-weight:700;letter-spacing:.08em;
                             text-transform:uppercase;color:#334155;text-align:right'>Total</span>
            </div>
        </div>""", unsafe_allow_html=True)

        entries = list(reversed(st.session_state.patient_log[-10:]))
        for idx, entry in enumerate(entries):
            if entry["type"] == "arrivée":
                e_color, e_icon, e_bg = "#34d399", "➕", "rgba(52,211,153,.07)"
            elif entry["type"] == "sortie":
                e_color, e_icon, e_bg = "#f43f5e", "➖", "rgba(244,63,94,.07)"
            else:
                e_color, e_icon, e_bg = "#94a3b8", "🔄", "rgba(148,163,184,.05)"

            border_bottom = "border-bottom:1px solid rgba(255,255,255,.04);" if idx < len(entries)-1 else ""
            badge_bg = e_color + "22"

            st.markdown(
                f"<div style='background:{e_bg};{border_bottom}"
                f"display:grid;grid-template-columns:32px 80px 1fr 80px;"
                f"align-items:center;gap:0;padding:.6rem 1rem;margin-top:-1px'>"
                f"<span style='color:{e_color};font-size:.95rem;text-align:center'>{e_icon}</span>"
                f"<span style='color:#475569;font-family:\"JetBrains Mono\",monospace;"
                f"font-size:.72rem;letter-spacing:.02em'>{entry['heure']}</span>"
                f"<span style='color:#cbd5e1;font-size:.82rem'>{entry['message']}</span>"
                f"<span style='background:{badge_bg};color:{e_color};"
                f"border-radius:20px;padding:.15rem .6rem;font-size:.7rem;"
                f"font-weight:700;text-align:center;display:block'>{entry['total']} pat.</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Bouton vider le journal
    if st.session_state.patient_log:
        st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Vider le journal", key="btn_clear_log"):
            st.session_state.patient_log = []
            st.rerun()


# ── DOCTOR Page 1 : Exploration Clinique ────────────────────
elif is_doctor and page == DOC_PAGES[0]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Exploration Clinique</div>
        <div class='banner-sub'>Analyse exploratoire des variables biométriques et comportementales</div>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Distributions","📦 Boxplots","🔵 Relations","🗂️ Dataset"])
    num_cols = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()

    with tab1:
        chosen = st.selectbox("Variable clinique", num_cols)
        c1,c2  = st.columns(2)
        with c1:
            fig,ax = dark_fig(6,4)
            ax.hist(df[chosen],bins=40,color="#00d4b4",edgecolor="none",alpha=.8)
            ax.axvline(df[chosen].mean(),color="#ef4444",linestyle="--",lw=1.8,label=f"Moy : {df[chosen].mean():.2f}")
            ax.axvline(df[chosen].median(),color="#22c55e",linestyle="--",lw=1.8,label=f"Méd : {df[chosen].median():.2f}")
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

        s = df[chosen]; cc = st.columns(5)
        for met,val in zip(["Moyenne","Médiane","Écart-type","Min","Max"],
                            [s.mean(),s.median(),s.std(),s.min(),s.max()]):
            cc[["Moyenne","Médiane","Écart-type","Min","Max"].index(met)].metric(met,f"{val:.3f}")

        nr = int(np.ceil(len(num_cols)/4))
        fig_all,axes = dark_fig(14,nr*3,ncols=4,nrows=nr)
        axes = axes.flatten()
        for idx,cn in enumerate(num_cols):
            axes[idx].hist(df[cn],bins=25,color=PALETTE[idx%len(PALETTE)],edgecolor="none",alpha=.85)
            axes[idx].set_title(cn,fontsize=8.5,color="#94a3b8")
            axes[idx].spines[["top","right"]].set_visible(False)
            axes[idx].tick_params(labelsize=7)
        for j in range(len(num_cols),len(axes)): axes[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_all,use_container_width=True)

    with tab2:
        bxv = st.selectbox("Variable",num_cols,key="bxv")
        fig,ax = dark_fig(12,5)
        for i in range(7):
            vals = df[df["NObeyesdad"]==i][bxv]
            ax.boxplot(vals,positions=[i],widths=.58,patch_artist=True,
                       boxprops=dict(facecolor=CLASS_HEX[i],alpha=.65),
                       medianprops=dict(color="white",linewidth=2.5),
                       whiskerprops=dict(color="#475569",lw=1.2),
                       capprops=dict(color="#475569",lw=1.2),
                       flierprops=dict(marker="o",color="#475569",markersize=2.5,alpha=.4))
        ax.set_xticks(range(7))
        ax.set_xticklabels([CLASS_NAMES[i].replace(" ","\n") for i in range(7)],fontsize=8.5,color="#94a3b8")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18,linestyle="--")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True)

    with tab3:
        sx1,sx2 = st.columns(2)
        xv = sx1.selectbox("Axe X",num_cols,index=2)
        yv = sx2.selectbox("Axe Y",num_cols,index=3)
        fig3,ax3 = dark_fig(10,5)
        for i in range(7):
            sub = df[df["NObeyesdad"]==i]
            ax3.scatter(sub[xv],sub[yv],s=20,alpha=.45,color=CLASS_HEX[i],label=CLASS_NAMES[i],edgecolors="none")
        ax3.set_xlabel(xv,fontsize=9); ax3.set_ylabel(yv,fontsize=9)
        ax3.set_title(f"{xv} vs {yv}",fontsize=11,pad=10,color="#e2e8f0")
        ax3.spines[["top","right"]].set_visible(False)
        ax3.grid(alpha=.15,linestyle="--")
        ax3.legend(fontsize=7.5,ncol=2)
        plt.tight_layout(); st.pyplot(fig3,use_container_width=True)

    with tab4:
        df_dec   = decode_df(df)
        cols_sel = st.multiselect("Colonnes",df_dec.columns.tolist(),default=df_dec.columns.tolist())
        st.dataframe(df_dec[cols_sel],use_container_width=True,height=500)
        st.markdown(f"<span class='chip chip-teal'>{len(df):,} patients</span>"
                    f"<span class='chip'>{len(cols_sel)} colonnes</span>",
                    unsafe_allow_html=True)


# ── DOCTOR Page 2 : Analyse Statistique ─────────────────────
elif is_doctor and page == DOC_PAGES[1]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Analyse Statistique</div>
        <div class='banner-sub'>Corrélations cliniques, statistiques descriptives et détection d'anomalies</div>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3 = st.tabs(["🔗 Corrélations","📋 Statistiques","⚠️ Outliers"])

    with tab1:
        corr = df.corr()
        fig,ax = dark_fig(10,8)
        mask   = np.triu(np.ones_like(corr,dtype=bool))
        sns.heatmap(corr,ax=ax,mask=mask,
                    cmap=sns.diverging_palette(200,10,as_cmap=True),
                    center=0,annot=True,fmt=".2f",annot_kws={"size":7.5},
                    linewidths=.4,linecolor="#0a0f1e",cbar_kws={"shrink":.7})
        ax.set_title("Corrélations inter-variables",fontsize=12,pad=10,color="#e2e8f0")
        plt.xticks(fontsize=7.5,rotation=45,ha="right",color="#94a3b8")
        plt.yticks(fontsize=7.5,color="#94a3b8")
        plt.tight_layout(); st.pyplot(fig,use_container_width=True)

        tc = corr["NObeyesdad"].drop("NObeyesdad").sort_values(key=abs,ascending=False)
        fig2,ax2 = dark_fig(9,4.5)
        ax2.barh(tc.index,tc.values,
                 color=["#22c55e" if v>0 else "#ef4444" for v in tc.values],
                 edgecolor="none",height=.58)
        ax2.axvline(0,color="#334155",lw=1.5)
        ax2.set_xlabel("Coefficient de Pearson",fontsize=9)
        ax2.set_title("Impact sur le diagnostic d'obésité",fontsize=11,pad=10,color="#e2e8f0")
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="x",alpha=.18,linestyle="--")
        plt.tight_layout(); st.pyplot(fig2,use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"),
                     use_container_width=True,height=400)
        sv  = st.selectbox("Variable par classe",
                            df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist())
        sbc = df.groupby("NObeyesdad")[sv].describe().round(3)
        sbc.index = [CLASS_NAMES[i] for i in sbc.index]
        st.dataframe(sbc.style.background_gradient(cmap="Blues"),use_container_width=True)

    with tab3:
        nc   = df.select_dtypes(include=np.number).columns.drop("NObeyesdad").tolist()
        rows = []
        for c in nc:
            Q1,Q3 = df[c].quantile(.25),df[c].quantile(.75); IQR=Q3-Q1
            n = ((df[c]<Q1-1.5*IQR)|(df[c]>Q3+1.5*IQR)).sum()
            rows.append({"Variable":c,"Q1":round(Q1,3),"Q3":round(Q3,3),
                         "IQR":round(IQR,3),"Outliers":n,"% Outliers":round(n/len(df)*100,2)})
        out = pd.DataFrame(rows).sort_values("Outliers",ascending=False)
        st.dataframe(out.style.background_gradient(subset=["Outliers","% Outliers"],cmap="Reds"),
                     use_container_width=True)


# ── DOCTOR Page 3 : Comparaison des Modèles ─────────────────
elif is_doctor and page == DOC_PAGES[2]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Comparaison des Modèles</div>
        <div class='banner-sub'>Mise en compétition · Random Forest · XGBoost · LightGBM</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("⏳ Évaluation des 3 modèles…"):
        sc_df = compare_models()

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Tableau comparatif</div>",
                unsafe_allow_html=True)
    render_cmp_table(sc_df)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Comparaison visuelle</div>",
                unsafe_allow_html=True)
    metrics_c = ["Accuracy","F1-Score","Précision","Rappel"]
    fig_c,axes_c = dark_fig(16,5,ncols=4,nrows=1)
    for idx,metric in enumerate(metrics_c):
        vals = sc_df[metric]
        bars = axes_c[idx].bar(range(3),vals.values,
                               color=[ALGO_COLORS[a] for a in ALGO_LIST],
                               edgecolor="none",width=.52)
        axes_c[idx].set_xticks(range(3))
        axes_c[idx].set_xticklabels(
            [a.replace(" Classifier","") for a in ALGO_LIST],
            fontsize=7.5,rotation=18,ha="right",color="#94a3b8")
        axes_c[idx].set_ylim(vals.min()-3,100)
        axes_c[idx].set_title(f"{metric} (%)",fontsize=9.5,pad=8,fontweight="600",color="#e2e8f0")
        axes_c[idx].spines[["top","right"]].set_visible(False)
        axes_c[idx].grid(axis="y",alpha=.18,linestyle="--")
        for bar,v in zip(bars,vals.values):
            axes_c[idx].text(bar.get_x()+bar.get_width()/2,v+.15,
                             f"{v:.1f}%",ha="center",va="bottom",
                             fontsize=7.5,color="#e2e8f0",fontweight="700")
    plt.tight_layout(); st.pyplot(fig_c,use_container_width=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>🏆 Résultats finaux</div>",
                unsafe_allow_html=True)
    best = sc_df["Accuracy"].idxmax()
    c1,c2,c3 = st.columns(3,gap="medium")
    for col,a in zip([c1,c2,c3],ALGO_LIST):
        col.markdown(f"""
        <div class='model-card {"best" if a==best else ""}' style='text-align:center'>
            <div class='model-icon'>{ALGO_ICONS[a]}</div>
            <div class='model-name' style='color:{ALGO_COLORS[a]}'>{a.replace(" Classifier","")}</div>
            <div class='model-score' style='color:{ALGO_COLORS[a]}'>{sc_df.loc[a,"Accuracy"]:.2f}%</div>
            <div style='color:#64748b;font-size:.82rem;font-weight:600'>Accuracy</div>
            <div style='color:#475569;font-size:.79rem;margin-top:.3rem'>F1 : {sc_df.loc[a,"F1-Score"]:.2f}%</div>
        </div>""", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════════════════════╗
#  DOCTOR Page 4 : Diagnostic IA  ██ avec SHAP ██
# ╚═══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[3]:
    st.markdown(f"""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Diagnostic Individuel IA</div>
        <div class='banner-sub'>Prédiction personnalisée · {ALGO_ICONS[algo]} <strong>{algo}</strong>
            {"&ensp;· ⭐ Meilleur modèle" if algo==BEST_ALGO else ""}</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Initialisation du modèle et des valeurs SHAP…"):
        (clf, sc_m, fc,
         acc, f1, prec, rec,
         cm, cr,
         Xtes, yte, yp,
         explainer, shap_values, Xtes_sample) = train_model(algo)

    n_classes = len(CLASS_NAMES)

    st.markdown(
        f"<span class='chip chip-teal'>{ALGO_ICONS[algo]} {algo}</span>"
        f"<span class='chip'>✅ Acc {acc*100:.1f}%</span>"
        f"<span class='chip'>F1 {f1*100:.1f}%</span>"
        f"<span class='chip'>Prec {prec*100:.1f}%</span>"
        f"<span class='chip'>Rapp {rec*100:.1f}%</span>"
        f"<span class='chip chip-violet'>🔍 SHAP activé</span>",
        unsafe_allow_html=True)

    pat = st.session_state.get("patient",{})
    if pat:
        st.markdown("""
        <div class='panel p-nurse'>
            <div class='panel-title'>🔗 Dossier infirmière importé</div>
            <div class='panel-body'>Données pré-chargées depuis l'interface infirmière. Ajustez si nécessaire.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Paramètres Patient</div>",
                unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3,gap="large")

    def pidx(lst, key, default):
        v = pat.get(key, default)
        try: return lst.index(v)
        except: return lst.index(default)

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
        if imc_d<18.5:   imc_dc,imc_dt="#60a5fa","Poids Insuffisant"
        elif imc_d<25:   imc_dc,imc_dt="#22c55e","Poids Normal ✓"
        elif imc_d<30:   imc_dc,imc_dt="#f59e0b","Surpoids"
        else:            imc_dc,imc_dt="#ef4444","Obésité ⚠️"
        st.markdown(f"""
        <div class='imc-live'>
            <div class='imc-label'>IMC Calculé</div>
            <div class='imc-value' style='color:{imc_dc}'>{imc_d}</div>
            <div class='imc-cat' style='color:{imc_dc}'>{imc_dt}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🍽️ Alimentation</div>",
                    unsafe_allow_html=True)
        favc = st.selectbox("Aliments caloriques (FAVC)",["Non","Oui"],
                            index=pidx(["Non","Oui"],"favc","Non"),key="d_favc")
        fcvc = st.slider("Légumes (FCVC)",1.0,3.0,float(pat.get("fcvc",2.0)),0.1,key="d_fcvc")
        ncp  = st.slider("Repas / jour (NCP)",1.0,4.0,float(pat.get("ncp",3.0)),0.5,key="d_ncp")
        caec = st.selectbox("Grignotage (CAEC)",["Jamais","Parfois","Fréquemment","Toujours"],
                            index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"caec","Parfois"),key="d_caec")
        calc = st.selectbox("Alcool (CALC)",["Jamais","Parfois","Fréquemment","Toujours"],
                            index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"calc","Jamais"),key="d_calc")
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("""
        <div class='panel p-green' style='margin-top:.8rem'>
            <div class='panel-title'>💡 Référence OMS</div>
            <div class='panel-body'>5 fruits/légumes/jour et 3 repas équilibrés réduisent le risque d'obésité de 35%.</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🏃 Mode de Vie</div>",
                    unsafe_allow_html=True)
        smoke  = st.selectbox("Tabagisme",["Non","Oui"],
                              index=pidx(["Non","Oui"],"smoke","Non"),key="d_smoke")
        ch2o   = st.slider("Eau / jour (L)",1.0,3.0,float(pat.get("ch2o",2.0)),0.1,key="d_ch2o")
        scc    = st.selectbox("Surveillance cal. (SCC)",["Non","Oui"],
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

    # ════════════════════════════════════════════════════════
    #  RÉSULTAT DU DIAGNOSTIC + SHAP
    # ════════════════════════════════════════════════════════
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

        rb_class  = {"green":"rb-green","amber":"rb-amber","red":"rb-red"}[info[1]]
        emoji     = "✅" if info[1]=="green" else "⚠️" if info[1]=="amber" else "🚨"
        pred_color = CLASS_HEX[pred]

        # ── Résultat principal ──────────────────────────────
        st.markdown("---")
        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Résultat du Diagnostic</div>",
                    unsafe_allow_html=True)
        rc1,rc2 = st.columns([1.2,1],gap="large")
        with rc1:
            st.markdown(f"""
            <div class='result-box {rb_class}'>
                <span class='result-emoji'>{emoji}</span>
                <div class='result-title'>{info[0]}</div>
                <div class='result-imc'>IMC : {imc_d} &nbsp;·&nbsp; {info[2]}</div>
                <div class='result-desc'>{info[3]}</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class='panel p-doctor' style='height:100%'>
                <div class='panel-title'>📋 Résumé Patient</div>
                <div style='font-size:.85rem;line-height:2.1;color:#94a3b8'>
                    <b style='color:#e2e8f0'>Genre :</b> {"Homme" if gender=="Masculin" else "Femme"}<br>
                    <b style='color:#e2e8f0'>Âge :</b> {age} ans<br>
                    <b style='color:#e2e8f0'>Taille / Poids :</b> {height} m · {weight} kg<br>
                    <b style='color:#e2e8f0'>IMC :</b>
                    <span style='color:{imc_dc};font-family:"JetBrains Mono",monospace;
                                 font-weight:800;font-size:1rem'>{imc_d}</span><br>
                    <b style='color:#e2e8f0'>Activité :</b> {faf} j/sem<br>
                    <b style='color:#e2e8f0'>Hydratation :</b> {ch2o} L/j<br>
                    <b style='color:#e2e8f0'>Tabagisme :</b> {smoke}<br>
                    <b style='color:#e2e8f0'>Ant. familiaux :</b> {family}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Probabilités ────────────────────────────────────
        if proba is not None:
            st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Probabilités diagnostiques</div>",
                        unsafe_allow_html=True)
            fig_p,ax_p = dark_fig(11,4.5)
            bars_p = ax_p.bar([CLASS_NAMES[i] for i in range(7)],proba,
                              color=[CLASS_HEX[i] for i in range(7)],edgecolor="none",width=.58)
            for bar,av in zip(bars_p,[1.0 if i==pred else .38 for i in range(7)]):
                bar.set_alpha(av)
            ax_p.set_ylim(0,1.15)
            ax_p.set_ylabel("Probabilité",fontsize=9,color="#64748b")
            ax_p.spines[["top","right"]].set_visible(False)
            ax_p.grid(axis="y",alpha=.18,linestyle="--")
            plt.xticks(rotation=22,ha="right",fontsize=8.5,color="#94a3b8")
            for bar,p_v in zip(bars_p,proba):
                if p_v>.015:
                    ax_p.text(bar.get_x()+bar.get_width()/2,p_v+.015,
                              f"{p_v*100:.1f}%",ha="center",va="bottom",
                              fontsize=8.5,color="#e2e8f0",fontweight="700")
            plt.tight_layout(); st.pyplot(fig_p,use_container_width=True)

        # ── Recommandations ─────────────────────────────────
        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Recommandations Médicales Personnalisées</div>",
                    unsafe_allow_html=True)
        recs = []
        if faf<1.0:
            recs.append(("red","🏃","Activité physique insuffisante",
                         "Prescrire ≥ 150 min d'activité modérée/sem (OMS). Débuter par 20 min/j de marche rapide."))
        elif faf>=2.5:
            recs.append(("green","🏃","Activité physique optimale",
                         f"Niveau excellent ({faf} j/sem). Réduction du risque cardiovasculaire de 30%."))
        else:
            recs.append(("amber","🏃","Activité physique à renforcer",
                         "Progresser vers 3–4 séances/semaine (recommandations OMS 2024)."))
        if ch2o<1.5:
            recs.append(("red","💧","Hydratation critique",
                         f"{ch2o} L/j. Objectif minimum : 2 L/j (2.5 L en période chaude)."))
        elif ch2o>=2.0:
            recs.append(("green","💧","Hydratation satisfaisante",
                         f"{ch2o} L/jour — conforme aux recommandations EFSA."))
        if caec in ["Fréquemment","Toujours"]:
            recs.append(("red","🍪","Grignotage excessif",
                         "+20–30% d'apport calorique. Orienter vers un diététicien."))
        if smoke=="Oui":
            recs.append(("red","🚬","Tabagisme actif",
                         "Perturbe le métabolisme lipidique. Consultation sevrage tabagique."))
        if family=="Oui":
            recs.append(("amber","🧬","Prédisposition génétique",
                         "Risque ×2–3. Suivi médical annuel et bilan métabolique complet."))
        if imc_d>=30:
            recs.append(("red","⚕️","Consultation spécialiste urgente",
                         "Bilan lipidique, glycémie à jeun, TA. Orientation endocrinologue / nutritionniste."))
        elif 25<=imc_d<30:
            recs.append(("amber","⚕️","Suivi préventif recommandé",
                         "Consultation diététicien et bilan cardiovasculaire préventif."))
        else:
            recs.append(("green","⚕️","Profil clinique satisfaisant",
                         "IMC OMS normal. Maintenir les habitudes. Prochain bilan dans 12 mois."))
        rc_map = {"green":"rc-green","amber":"rc-amber","red":"rc-red"}
        for color,icon,title,text in recs:
            st.markdown(f"""
            <div class='rec-card {rc_map[color]}'>
                <div class='rec-icon'>{icon}</div>
                <div>
                    <div class='rec-title'>{title}</div>
                    <div class='rec-text'>{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)


        # ════════════════════════════════════════════════════
        #  ██  SECTION SHAP — EXPLICABILITÉ IA  ██
        # ════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("""
        <div class='sec-head'>
            <div class='dot dot-violet'></div>
            🔍 Explicabilité SHAP — Pourquoi cette prédiction ?
        </div>""", unsafe_allow_html=True)

        # Panel de présentation SHAP
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(139,92,246,.07),rgba(0,0,0,0));
                    border:1px solid rgba(139,92,246,.25);border-radius:14px;
                    padding:1.4rem 1.8rem;margin-bottom:1.5rem'>
            <div style='font-size:.67rem;font-weight:700;letter-spacing:.12em;
                        text-transform:uppercase;color:#7c3aed;margin-bottom:.5rem'>
                SHAP · SHapley Additive exPlanations
            </div>
            <div style='font-size:.92rem;color:#e2e8f0;font-weight:600;margin-bottom:.4rem'>
                L'IA explique son raisonnement pour ce patient
            </div>
            <div style='font-size:.83rem;color:#64748b;line-height:1.7'>
                SHAP décompose la prédiction <strong style='color:{pred_color}'>{info[0]}</strong>
                variable par variable. Chaque barre indique <em>dans quelle mesure</em> et
                <em>dans quel sens</em> cette variable a influencé le diagnostic —
                rendant l'IA <strong style='color:#a78bfa'>totalement transparente</strong>
                et auditée cliniquement.
            </div>
            <div style='margin-top:.9rem;display:flex;gap:1.5rem;flex-wrap:wrap'>
                <span><span style='display:inline-block;width:14px;height:14px;
                    background:{pred_color};border-radius:3px;vertical-align:middle;
                    margin-right:.4rem'></span>
                    <span style='font-size:.78rem;color:#94a3b8'>Augmente le risque</span></span>
                <span><span style='display:inline-block;width:14px;height:14px;
                    background:#3b82f6;border-radius:3px;vertical-align:middle;
                    margin-right:.4rem'></span>
                    <span style='font-size:.78rem;color:#94a3b8'>Diminue le risque</span></span>
                <span><span style='font-size:.78rem;color:#64748b'>
                    Valeur de base : E[f(X)] =
                    {get_expected_value(explainer, pred):.3f}</span></span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Calcul SHAP pour ce patient spécifique ──────────
        with st.spinner("🔬 Calcul des valeurs SHAP individuelles…"):
            patient_shap = explainer.shap_values(Xns)

        sv_patient_pred = get_shap_for_class(patient_shap, pred)[0]
        patient_data_arr = Xn.values[0]

        # ── Ligne 1 : Waterfall + Importance globale ────────
        shap_col1, shap_col2 = st.columns([1.35, 1], gap="large")

        with shap_col1:
            st.markdown(f"""
            <div style='font-size:.72rem;font-weight:700;letter-spacing:.09em;
                        text-transform:uppercase;color:#8b5cf6;margin-bottom:.6rem'>
                📊 Explication individuelle — Classe prédite : {info[0]}
            </div>""", unsafe_allow_html=True)

            fig_wf = plot_shap_waterfall_patient(
                explainer, patient_shap, pred,
                patient_data_arr, fc, info[0], pred_color
            )
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.markdown("""
            <div style='font-size:.75rem;color:#475569;margin-top:.3rem;line-height:1.6;
                        padding:.6rem .9rem;background:#0d1523;border-radius:8px'>
                <strong style='color:#64748b'>Comment lire :</strong>
                Chaque barre = contribution d'une variable à la prédiction finale.
                La valeur entre parenthèses est la valeur réelle du patient pour cette variable.
                Le total des barres s'accumule de la valeur de base vers la prédiction finale.
            </div>""", unsafe_allow_html=True)

        with shap_col2:
            st.markdown(f"""
            <div style='font-size:.72rem;font-weight:700;letter-spacing:.09em;
                        text-transform:uppercase;color:#8b5cf6;margin-bottom:.6rem'>
                🏆 Importance Globale des Variables (SHAP)
            </div>""", unsafe_allow_html=True)

            fig_imp = plot_shap_global_importance(shap_values, fc, n_classes, pred_color)
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)

            st.markdown("""
            <div style='font-size:.75rem;color:#475569;margin-top:.3rem;line-height:1.6;
                        padding:.6rem .9rem;background:#0d1523;border-radius:8px'>
                <strong style='color:#64748b'>Comment lire :</strong>
                Importance moyenne sur l'ensemble du jeu de test (200 patients).
                Variables surlignées = les plus déterminantes pour tous les diagnostics.
            </div>""", unsafe_allow_html=True)

        # ── Insights médicaux automatiques ──────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;letter-spacing:.09em;
                    text-transform:uppercase;color:#8b5cf6;margin:.8rem 0 .6rem'>
            🧠 Insights Médicaux Générés par SHAP
        </div>""", unsafe_allow_html=True)

        insights = generate_shap_insights(sv_patient_pred, fc, pred, info[0])
        insight_cols = st.columns(len(insights), gap="medium")
        for col_i, (icon, title, text) in zip(insight_cols, insights):
            with col_i:
                st.markdown(f"""
                <div class='shap-insight-card'>
                    <div class='shap-insight-icon'>{icon}</div>
                    <div>
                        <div class='shap-insight-title'>{title}</div>
                        <div class='shap-insight-text'>{text}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── SHAP Summary Beeswarm (population) ─────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;letter-spacing:.09em;
                    text-transform:uppercase;color:#8b5cf6;margin:1.5rem 0 .6rem'>
            🌡️ Distribution SHAP — Vue Population (jeu de test)
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='panel p-violet' style='margin-bottom:1rem'>
            <div class='panel-title'>📖 Interprétation du graphique population</div>
            <div class='panel-body'>
                Chaque point représente <strong>un patient du jeu de test</strong>.
                La couleur indique la valeur de la variable pour ce patient
                (<span style='color:#ef4444;font-weight:700'>rouge = élevée</span>,
                <span style='color:#3b82f6;font-weight:700'>bleu = faible</span>).
                La position horizontale montre si la variable pousse la prédiction vers
                une classe supérieure (droite) ou inférieure (gauche).
                Un nuage rouge à droite signifie que les valeurs hautes de cette variable
                augmentent fortement le risque d'obésité.
            </div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("📡 Génération du SHAP Summary Plot…"):
            fig_bee = plot_shap_beeswarm(shap_values, Xtes_sample, fc, n_classes)
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)

        # ── Tableau SHAP du patient ─────────────────────────
        st.markdown("""
        <div style='font-size:.72rem;font-weight:700;letter-spacing:.09em;
                    text-transform:uppercase;color:#8b5cf6;margin:1.5rem 0 .6rem'>
            📋 Tableau Détaillé des Valeurs SHAP — Ce Patient
        </div>""", unsafe_allow_html=True)

        shap_df = pd.DataFrame({
            "Variable":       [FEATURE_LABELS.get(f, f) for f in fc],
            "Valeur Patient": [f"{v:.3f}" for v in patient_data_arr],
            "SHAP":           sv_patient_pred,
            "|SHAP|":         np.abs(sv_patient_pred),
            "Sens":           ["↑ Augmente" if v > 0 else "↓ Diminue" for v in sv_patient_pred],
        }).sort_values("|SHAP|", ascending=False).reset_index(drop=True)
        shap_df["SHAP"]  = shap_df["SHAP"].round(4)
        shap_df["|SHAP|"] = shap_df["|SHAP|"].round(4)

        # Colorier la colonne Sens
        def color_sens(val):
            if "↑" in str(val): return "color: #ef4444; font-weight: 700"
            if "↓" in str(val): return "color: #3b82f6; font-weight: 700"
            return ""
        def color_shap(val):
            try:
                v = float(val)
                if v > 0.01:  return "color: #fca5a5; font-weight: 700"
                if v < -0.01: return "color: #93c5fd; font-weight: 700"
            except: pass
            return "color: #64748b"

        styled = (shap_df.style
                  .applymap(color_sens, subset=["Sens"])
                  .applymap(color_shap, subset=["SHAP"])
                  .background_gradient(subset=["|SHAP|"], cmap="Purples"))
        st.dataframe(styled, use_container_width=True, height=420)

        # ── Note de clôture SHAP ────────────────────────────
        st.markdown(f"""
        <div style='background:#0d1523;border:1px solid rgba(139,92,246,.18);
                    border-radius:12px;padding:1.2rem 1.6rem;margin-top:1rem;
                    display:flex;align-items:flex-start;gap:1rem'>
            <div style='font-size:1.6rem;flex-shrink:0'>🏥</div>
            <div>
                <div style='font-size:.85rem;font-weight:700;color:#c4b5fd;margin-bottom:.3rem'>
                    Interprétabilité clinique certifiée — SHAP + {algo}
                </div>
                <div style='font-size:.79rem;color:#475569;line-height:1.65'>
                    Ce diagnostic s'appuie sur un modèle {ALGO_ICONS[algo]} <strong style='color:#94a3b8'>{algo}</strong>
                    atteignant <strong style='color:#00d4b4'>{acc*100:.1f}% d'accuracy</strong> (F1 = {f1*100:.1f}%).
                    Les valeurs SHAP garantissent la traçabilité de chaque décision —
                    conformément aux exigences de l'IA médicale explicable (XAI).
                    Les 3 facteurs les plus décisifs pour ce patient sont :
                    <strong style='color:#e2e8f0'>
                        {", ".join([FEATURE_LABELS.get(fc[i], fc[i])
                                    for i in np.argsort(np.abs(sv_patient_pred))[::-1][:3]])}
                    </strong>.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
