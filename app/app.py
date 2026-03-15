import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#  COUNTRY CODES
# ═══════════════════════════════════════════════════════════
COUNTRY_CODES = [
    ("🇲🇦 Maroc",           "+212"),
    ("🇫🇷 France",          "+33"),
    ("🇩🇿 Algérie",         "+213"),
    ("🇹🇳 Tunisie",         "+216"),
    ("🇸🇦 Arabie Saoudite", "+966"),
    ("🇦🇪 Émirats Arabes",  "+971"),
    ("🇺🇸 États-Unis",      "+1"),
    ("🇨🇦 Canada",          "+1"),
    ("🇬🇧 Royaume-Uni",     "+44"),
    ("🇩🇪 Allemagne",       "+49"),
    ("🇧🇪 Belgique",        "+32"),
    ("🇪🇸 Espagne",         "+34"),
    ("🇮🇹 Italie",          "+39"),
    ("🇵🇹 Portugal",        "+351"),
    ("🇳🇱 Pays-Bas",        "+31"),
    ("🇨🇭 Suisse",          "+41"),
    ("🇸🇳 Sénégal",         "+221"),
    ("🇲🇷 Mauritanie",      "+222"),
    ("🇱🇾 Libye",           "+218"),
    ("🇪🇬 Égypte",          "+20"),
]
CC_DISPLAY  = [f"{name}  {code}" for name, code in COUNTRY_CODES]
CC_CODE_MAP = {f"{name}  {code}": code for name, code in COUNTRY_CODES}

# ── Phone number examples per country code (real format examples) ──
CC_EXAMPLES = {
    "+212": "Ex : 0661234567   — Mobile Maroc (IAM / Orange / Inwi)",
    "+33":  "Ex : 0623456789   — Mobile France (SFR / Orange / Free)",
    "+213": "Ex : 0551234567   — Mobile Algérie (Djezzy / Ooredoo / Mobilis)",
    "+216": "Ex : 20123456     — Mobile Tunisie (Tunisie Telecom / Ooredoo)",
    "+966": "Ex : 0501234567   — Mobile Arabie Saoudite (STC / Mobily)",
    "+971": "Ex : 0501234567   — Mobile UAE (Etisalat / du)",
    "+1":   "Ex : 2025551234   — Mobile USA (format 10 chiffres sans 0)",
    "+44":  "Ex : 07700900123  — Mobile UK (format 07XXX XXXXXX)",
    "+49":  "Ex : 01512345678  — Mobile Allemagne (Telekom / Vodafone)",
    "+32":  "Ex : 0470123456   — Mobile Belgique (Proximus / Base)",
    "+34":  "Ex : 612345678    — Mobile Espagne (Movistar / Vodafone)",
    "+39":  "Ex : 3201234567   — Mobile Italie (TIM / Vodafone / Wind)",
    "+351": "Ex : 912345678    — Mobile Portugal (MEO / NOS / Vodafone)",
    "+31":  "Ex : 0612345678   — Mobile Pays-Bas (KPN / T-Mobile)",
    "+41":  "Ex : 0791234567   — Mobile Suisse (Swisscom / Salt)",
    "+221": "Ex : 771234567    — Mobile Sénégal (Orange / Free)",
    "+222": "Ex : 36123456     — Mobile Mauritanie (Mauritel / Chinguitel)",
    "+218": "Ex : 0911234567   — Mobile Libye (Libyana / Madar)",
    "+20":  "Ex : 01001234567  — Mobile Égypte (Vodafone / Orange / Etisalat)",
}

# ═══════════════════════════════════════════════════════════
#  HELPERS — TIME
# ═══════════════════════════════════════════════════════════
def get_greeting():
    h = datetime.datetime.now().hour
    if h < 12:   return "Bonjour"
    elif h < 18: return "Bon après-midi"
    else:        return "Bonsoir"

# ═══════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════
def validate_name(name: str):
    name = name.strip()
    if len(name) < 2:
        return False, "Le nom doit contenir au moins 2 caractères."
    if re.search(r'\d', name):
        return False, "Le nom ne peut pas contenir de chiffres."
    if not re.match(r"^[a-zA-ZÀ-ÿ\s\-']+$", name):
        return False, "Uniquement lettres, espaces, tirets et apostrophes."
    return True, "Nom valide ✓"


def validate_phone(phone: str, country_code: str = "+212"):
    clean = re.sub(r'[\s\.\-\(\)]', '', phone)
    # Remove leading zeros for non-zero-start numbers
    patterns_by_code = {
        "+212": [r'^0[5-7]\d{8}$', r'^\d{9}$'],           # 06XXXXXXXX or 9 digits
        "+33":  [r'^0[1-9]\d{8}$', r'^\d{9}$'],
        "+213": [r'^0[5-7]\d{8}$', r'^\d{9}$'],
        "+216": [r'^\d{8}$'],
        "+966": [r'^0?5\d{8}$', r'^\d{9}$'],
        "+971": [r'^0?5\d{8}$', r'^\d{9}$'],
        "+1":   [r'^\d{10}$'],
        "+44":  [r'^0?\d{10}$', r'^\d{10}$'],
        "+49":  [r'^\d{10,12}$'],
        "+32":  [r'^\d{9}$'],
        "+34":  [r'^\d{9}$'],
        "+39":  [r'^\d{10}$'],
    }
    patterns = patterns_by_code.get(country_code, [r'^\d{7,15}$'])
    for p in patterns:
        if re.match(p, clean):
            return True, "Numéro valide ✓"
    return False, f"Format invalide pour {country_code}. Ex : 0612345678"


def validate_email(email: str):
    email = email.strip()
    if not email:
        return True, ""
    if re.match(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$', email):
        return True, "Email valide ✓"
    return False, "Format invalide. Ex : nom@hopital.ma"


def validate_cin(cin: str):
    """
    CIN marocaine : 1–2 lettres MAJUSCULES + 5 ou 6 chiffres
    Ex : A12345 · A123456 · BE12345 · BE123456
    """
    cin_up = cin.strip().upper()
    if not cin_up:
        return True, "", cin_up
    if re.match(r'^[A-Z]{1,2}\d{5,6}$', cin_up):
        return True, "CIN valide ✓", cin_up
    return False, "Format invalide. Ex : A12345 · BE123456 (1-2 lettres + 5-6 chiffres)", cin_up

# ═══════════════════════════════════════════════════════════
#  CSS — DARK PREMIUM + VISIBLE MEDICAL BACKGROUND
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── FORCE DARK THEME — override any Streamlit light-mode remnants ── */
:root {
    color-scheme: dark !important;
}
html {
    filter: none !important;
    background: #041018 !important;
}
/* Kill the white flash on initial load */
body { background-color: #041018 !important; }
/* Kill Streamlit's internal light dividers and backgrounds */
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] {
    background: rgba(4,16,24,0.95) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(0,212,180,0.12) !important;
}
[data-testid="stToolbar"] { background: transparent !important; }
/* Force all text inputs/selects dark */
input, select, textarea {
    background-color: rgba(14,26,48,0.95) !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.1) !important;
}
/* Force all white backgrounds to dark */
.stTextInput input, .stNumberInput input {
    background: rgba(14,26,48,0.95) !important;
    color: #e2e8f0 !important;
}
/* Plotly / chart containers */
.js-plotly-plot, .plotly { background: transparent !important; }
/* Expanders */
.streamlit-expanderHeader {
    background: rgba(10,22,40,0.9) !important;
    color: #e2e8f0 !important;
}
/* Tables */
.stDataFrame { color: #e2e8f0 !important; }
/* Remove any white overlay */
[class*="st-"] { background-color: transparent; }

/* ══════════════════════════════════════════════════════
   CLINICAL BACKGROUND — VISIBLE MEDICAL ATMOSPHERE
══════════════════════════════════════════════════════ */

/* Keyframes for animated ECG pulse */
@keyframes ecg-pulse {
    0%   { stroke-dashoffset: 1000; opacity: 0; }
    10%  { opacity: 0.35; }
    80%  { opacity: 0.25; }
    100% { stroke-dashoffset: 0; opacity: 0; }
}
@keyframes scan-line {
    0%   { top: -4px; }
    100% { top: 100%; }
}
@keyframes glow-pulse {
    0%, 100% { opacity: 0.55; transform: scale(1); }
    50%       { opacity: 0.80; transform: scale(1.08); }
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stAppViewBlockContainer"],
.stApp, .main {
    background-color: #041018 !important;
    color: #e2e8f0 !important;
    font-family: "DM Sans", sans-serif !important;
}

[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.block-container {
    background-color: transparent !important;
    color: #e2e8f0 !important;
    font-family: "DM Sans", sans-serif !important;
}

/* ── Rich layered background ── */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp {
    background-color: #030b14 !important;
    background-image:
        /* Fine dot grid — clearly visible */
        radial-gradient(circle, rgba(0,230,195,0.20) 1.8px, transparent 1.8px),
        /* Large teal bloom — top left */
        radial-gradient(ellipse 900px 700px at 0% 0%, rgba(0,210,180,0.26) 0%, transparent 60%),
        /* Emerald bloom — bottom right */
        radial-gradient(ellipse 800px 600px at 100% 100%, rgba(0,190,165,0.20) 0%, transparent 55%),
        /* Cyan accent — top right */
        radial-gradient(ellipse 500px 500px at 100% 0%, rgba(6,182,212,0.18) 0%, transparent 50%),
        /* Deep blue undertone — center */
        radial-gradient(ellipse 1100px 700px at 50% 55%, rgba(4,14,28,0.75) 0%, transparent 75%) !important;
    background-size: 28px 28px, auto, auto, auto, auto !important;
    background-attachment: fixed !important;
    position: relative;
}

/* Diagonal clinic grid overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        repeating-linear-gradient(
            -45deg,
            transparent,
            transparent 40px,
            rgba(0,220,185,0.025) 40px,
            rgba(0,220,185,0.025) 41px
        ),
        repeating-linear-gradient(
            45deg,
            transparent,
            transparent 40px,
            rgba(0,200,170,0.018) 40px,
            rgba(0,200,170,0.018) 41px
        );
    pointer-events: none;
    z-index: 0;
}

/* Bright top border line */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        transparent 0%,
        #00d4b4 15%,
        #06b6d4 35%,
        #00e5c7 55%,
        #06b6d4 75%,
        #00d4b4 85%,
        transparent 100%);
    opacity: 0.7;
    pointer-events: none;
    z-index: 9999;
}

/* ── SIDEBAR — Distinct clinical panel ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(2,14,26,0.99) 0%,
        rgba(1,10,18,1.00) 100%) !important;
    border-right: 1px solid rgba(0,212,180,0.20) !important;
    backdrop-filter: blur(20px);
    box-shadow: 4px 0 30px rgba(0,0,0,0.5);
    position: relative;
    overflow-y: auto !important;
    min-height: 100vh;
}
section[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    max-height: 100vh;
    padding-bottom: 2rem !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    overflow-y: auto !important;
    max-height: 100vh;
    padding-bottom: 2rem !important;
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 280px 350px at 50% 8%, rgba(0,212,180,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 200px 250px at 50% 95%, rgba(6,182,212,0.07) 0%, transparent 60%);
    pointer-events: none;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(0,212,180,0.10) !important; }

/* ── Radio buttons — clearly highlighted when selected ── */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 3px !important;
    display: flex;
    flex-direction: column;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: .84rem !important;
    font-weight: 500 !important;
    padding: .55rem .75rem !important;
    border-radius: 10px !important;
    margin: 2px 0 !important;
    transition: all .2s ease !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    cursor: pointer !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,212,180,0.08) !important;
    border-color: rgba(0,212,180,0.22) !important;
    color: #e2e8f0 !important;
}
/* Selected state — teal highlight for active item */
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(135deg,rgba(0,212,180,0.18),rgba(0,180,160,0.10)) !important;
    border-color: rgba(0,212,180,0.50) !important;
    color: #00d4b4 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 12px rgba(0,212,180,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) * {
    color: #00d4b4 !important;
}
/* Hide the default circle radio dot */
section[data-testid="stSidebar"] .stRadio label input[type="radio"] {
    display: none !important;
}
/* Active indicator dot — show before selected label */
section[data-testid="stSidebar"] .stRadio label:has(input:checked)::before {
    content: '';
    display: inline-block !important;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d4b4;
    box-shadow: 0 0 6px #00d4b4;
    margin-right: .55rem;
    flex-shrink: 0;
}
section[data-testid="stSidebar"] .stRadio label:not(:has(input:checked))::before {
    content: '';
    display: inline-block !important;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
    margin-right: .55rem;
    flex-shrink: 0;
}

.main .block-container {
    padding: 1.8rem 2.2rem 4rem;
    max-width: 1500px;
    position: relative; z-index: 1;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060d1b; }
::-webkit-scrollbar-thumb { background: rgba(0,212,180,0.2); border-radius: 3px; }

/* Inputs */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
    background: rgba(14,26,48,0.9) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
    backdrop-filter: blur(4px);
}
[data-baseweb="select"] option,
[data-baseweb="menu"] { background: #0a1628 !important; color: #e2e8f0 !important; }
[data-baseweb="popover"] { background: #0a1628 !important; }
[role="listbox"] { background: #0a1628 !important; }
[role="option"] { color: #e2e8f0 !important; }
[role="option"]:hover { background: rgba(0,212,180,0.07) !important; }
label, .stSelectbox label, .stSlider label,
.stNumberInput label, [data-testid="stWidgetLabel"] {
    color: #94a3b8 !important; font-size: .84rem !important;
}

.stSlider [data-baseweb="thumb"] { background: #00d4b4 !important; border-color: #00d4b4 !important; }
.stSlider [data-baseweb="track-fill"] { background: #00d4b4 !important; }
.stSlider [data-baseweb="track"] { background: rgba(255,255,255,0.07) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,22,40,0.8) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: .28rem !important;
    gap: .2rem; margin-bottom: 1rem;
    backdrop-filter: blur(8px);
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
    background: rgba(0,212,180,0.10) !important;
    color: #00d4b4 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

div[data-testid="metric-container"] {
    background: rgba(10,22,40,0.85) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
    backdrop-filter: blur(8px);
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: .74rem !important;
    text-transform: uppercase; letter-spacing: .06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.55rem !important;
}

.stButton > button {
    background: linear-gradient(135deg,#009e8a,#00d4b4) !important;
    color: #060d1b !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 2rem !important;
    font-family: 'DM Sans',sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    font-size: .88rem !important;
    box-shadow: 0 4px 20px rgba(0,212,180,0.22) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    opacity: .92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,212,180,0.35) !important;
}

.stDataFrame, [data-testid="stDataFrame"] {
    background: rgba(10,22,40,0.85) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
.stAlert { background: rgba(10,22,40,0.85) !important; color: #e2e8f0 !important; }
.stSpinner > div { border-top-color: #00d4b4 !important; }
.stNumberInput button {
    background: rgba(255,255,255,0.05) !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.08) !important;
}
hr { border-color: rgba(0,212,180,0.10) !important; }

/* ══════════════ CUSTOM COMPONENTS ══════════════ */

.page-banner {
    background: rgba(5,18,32,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative; overflow: hidden;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}
.page-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg,rgba(0,212,180,0.05) 0%,transparent 55%);
    pointer-events: none;
}
/* Medical cross watermark */
.page-banner::after {
    content: '✚';
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem; color: rgba(0,212,180,0.04);
    pointer-events: none; line-height: 1;
}
.banner-nurse::before  { background: linear-gradient(135deg,rgba(6,182,212,0.07) 0%,transparent 55%); }
.banner-doctor::before { background: linear-gradient(135deg,rgba(139,92,246,0.07) 0%,transparent 55%); }
.banner-eyebrow {
    font-size: .68rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; margin-bottom: .5rem;
}
.ey-nurse  { color: #06b6d4; }
.ey-doctor { color: #a78bfa; }
.banner-h1 {
    font-family: 'DM Serif Display',serif;
    font-size: 2rem; font-weight: 400; color: #f1f5f9;
    margin: 0 0 .4rem; line-height: 1.15;
}
.banner-sub { font-size: .88rem; color: #94a3b8; line-height: 1.6; margin: 0; }
.banner-tag {
    display: inline-block;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 5px; padding: .18rem .6rem;
    font-size: .7rem; font-weight: 600; color: #475569;
    margin: .6rem .3rem 0 0; font-family: 'JetBrains Mono',monospace;
}

.kpi-card {
    background: rgba(5,18,32,0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.3rem 1.2rem;
    position: relative; overflow: hidden;
    transition: border-color .22s,transform .22s;
    backdrop-filter: blur(8px);
}
.kpi-card:hover { border-color: rgba(0,212,180,0.25); transform: translateY(-2px); }
.kpi-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg,#00d4b4,#06b6d4); border-radius: 0 0 14px 14px;
}
.c-blue::after   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.c-green::after  { background: linear-gradient(90deg,#22c55e,#4ade80); }
.c-amber::after  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.c-red::after    { background: linear-gradient(90deg,#ef4444,#f87171); }
.c-violet::after { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-num { font-family:'DM Serif Display',serif; font-size:2rem; color:#00d4b4; line-height:1; margin-bottom:.3rem; }
.c-blue   .kpi-num { color:#3b82f6; }
.c-green  .kpi-num { color:#22c55e; }
.c-amber  .kpi-num { color:#f59e0b; }
.c-red    .kpi-num { color:#ef4444; }
.c-violet .kpi-num { color:#8b5cf6; }
.kpi-lbl { font-size:.69rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#64748b; }

.sec-head {
    font-family: 'DM Serif Display',serif;
    font-size: 1.1rem; color: #f1f5f9;
    margin: 2rem 0 1rem;
    display: flex; align-items: center; gap: .6rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid rgba(0,212,180,0.12);
}
.dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-teal   { background:#00d4b4; box-shadow:0 0 8px rgba(0,212,180,0.7); }
.dot-nurse  { background:#06b6d4; box-shadow:0 0 8px rgba(6,182,212,0.7); }
.dot-doctor { background:#8b5cf6; box-shadow:0 0 8px rgba(139,92,246,0.7); }
.dot-violet { background:#8b5cf6; box-shadow:0 0 8px rgba(139,92,246,0.7); }
.dot-green  { background:#22c55e; box-shadow:0 0 8px rgba(34,197,94,0.7); }
.dot-amber  { background:#f59e0b; box-shadow:0 0 8px rgba(245,158,11,0.7); }

.panel {
    background: rgba(5,18,32,0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 4px solid #00d4b4;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.p-teal   { border-left-color:#00d4b4; }
.p-blue   { border-left-color:#3b82f6; }
.p-green  { border-left-color:#22c55e; }
.p-amber  { border-left-color:#f59e0b; }
.p-red    { border-left-color:#ef4444; }
.p-violet { border-left-color:#8b5cf6; }
.p-nurse  { border-left-color:#06b6d4; }
.p-doctor { border-left-color:#8b5cf6; }
.panel-title { font-weight:700; font-size:.87rem; color:#00d4b4; margin-bottom:.3rem; letter-spacing:.03em; }
.p-green  .panel-title { color:#22c55e; }
.p-amber  .panel-title { color:#f59e0b; }
.p-red    .panel-title { color:#ef4444; }
.p-violet .panel-title { color:#a78bfa; }
.p-nurse  .panel-title { color:#06b6d4; }
.p-doctor .panel-title { color:#a78bfa; }
.panel-body { font-size:.84rem; color:#94a3b8; line-height:1.6; }

.model-card {
    background: rgba(5,18,32,0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.6rem 1.4rem;
    text-align: center; transition: all .22s;
    position: relative; overflow: hidden; height: 100%;
    backdrop-filter: blur(8px);
}
.model-card:hover { border-color: rgba(0,212,180,0.25); transform: translateY(-3px); }
.model-card.best {
    border-color: rgba(0,212,180,0.35);
    background: linear-gradient(160deg,rgba(0,212,180,0.06),rgba(10,22,40,0.85));
}
.model-card.best::before {
    content: '⭐ Modèle Sélectionné';
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,212,180,0.12); color: #00d4b4;
    border: 1px solid rgba(0,212,180,0.28); border-radius: 20px;
    padding: .18rem .7rem; font-size: .65rem; font-weight: 700; letter-spacing: .05em;
}
.model-icon  { font-size: 2.2rem; margin-bottom: .6rem; }
.model-name  { font-weight: 700; font-size: .95rem; color: #e2e8f0; margin-bottom: .35rem; }
.model-score { font-family:'DM Serif Display',serif; font-size:1.8rem; margin:.5rem 0; }

.result-box {
    border-radius: 16px; padding: 2.2rem; text-align: center; margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(5,18,32,0.82);
    position: relative; overflow: hidden;
    backdrop-filter: blur(12px);
}
.rb-green { border-color:rgba(34,197,94,.28); background:linear-gradient(160deg,rgba(34,197,94,.06),rgba(10,22,40,.85)); }
.rb-amber { border-color:rgba(245,158,11,.28); background:linear-gradient(160deg,rgba(245,158,11,.06),rgba(10,22,40,.85)); }
.rb-red   { border-color:rgba(239,68,68,.28); background:linear-gradient(160deg,rgba(239,68,68,.06),rgba(10,22,40,.85)); }
.result-emoji { font-size:3rem; display:block; margin-bottom:.8rem; }
.result-title { font-family:'DM Serif Display',serif; font-size:1.9rem; font-weight:400; }
.rb-green .result-title { color:#86efac; }
.rb-amber .result-title { color:#fcd34d; }
.rb-red   .result-title { color:#fca5a5; }
.result-imc  { font-family:'JetBrains Mono',monospace; font-size:.87rem; color:#64748b; margin:.5rem 0; }
.result-desc { font-size:.87rem; color:#94a3b8; margin-top:.5rem; }

.rec-card {
    background: rgba(5,18,32,0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 4px solid #00d4b4;
    border-radius: 10px; padding: 1rem 1.3rem; margin: .5rem 0;
    display: flex; align-items: flex-start; gap: 1rem;
    transition: transform .18s, border-color .18s;
    backdrop-filter: blur(8px);
}
.rec-card:hover { transform: translateX(4px); border-color: rgba(0,212,180,.3); }
.rc-green { border-left-color:#22c55e; }
.rc-amber { border-left-color:#f59e0b; }
.rc-red   { border-left-color:#ef4444; }
.rec-icon  { font-size:1.35rem; flex-shrink:0; margin-top:2px; }
.rec-title { font-weight:700; font-size:.87rem; color:#e2e8f0; margin-bottom:.2rem; }
.rec-text  { font-size:.79rem; color:#64748b; line-height:1.55; }

.imc-live {
    border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: .8rem;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(20,35,60,0.7);
    backdrop-filter: blur(8px);
}
.imc-label { font-size:.66rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#64748b; }
.imc-value { font-family:'DM Serif Display',serif; font-size:2.6rem; line-height:1.1; margin:.2rem 0; }
.imc-cat   { font-size:.82rem; font-weight:700; margin-top:2px; }

.chip {
    display: inline-flex; align-items: center; gap: .3rem;
    background: rgba(255,255,255,0.04); color: #94a3b8;
    border: 1px solid rgba(255,255,255,0.07); border-radius: 6px;
    padding: .22rem .7rem; font-size: .74rem; font-weight: 600;
    margin: .2rem; font-family: 'JetBrains Mono',monospace;
}
.chip-teal   { background:rgba(0,212,180,.08);  color:#00d4b4; border-color:rgba(0,212,180,.22); }
.chip-violet { background:rgba(139,92,246,.08); color:#a78bfa; border-color:rgba(139,92,246,.22); }
.chip-green  { background:rgba(34,197,94,.08);  color:#22c55e; border-color:rgba(34,197,94,.22); }
.chip-amber  { background:rgba(245,158,11,.08); color:#f59e0b; border-color:rgba(245,158,11,.22); }
.chip-red    { background:rgba(239,68,68,.08);  color:#ef4444; border-color:rgba(239,68,68,.22); }

.form-section {
    background: rgba(5,18,32,0.78);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}
.form-title {
    font-size: .76rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: 1rem;
    padding-bottom: .5rem; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.ft-nurse  { color: #06b6d4; }
.ft-doctor { color: #8b5cf6; }

.stat-row {
    display: flex; justify-content: space-between;
    padding: .38rem 0; border-bottom: 1px solid rgba(255,255,255,.04);
    font-size: .83rem;
}
.stat-row:last-child { border-bottom: none; }
.sk { color: #64748b; font-weight: 500; }
.sv { color: #e2e8f0; font-weight: 700; font-family:'JetBrains Mono',monospace; font-size:.8rem; }

/* Greeting */
.greeting-badge {
    background: linear-gradient(135deg,rgba(139,92,246,.10),rgba(0,212,180,.07));
    border: 1px solid rgba(139,92,246,.18);
    border-radius: 12px; padding: .85rem 1rem; margin-bottom: .8rem; text-align: center;
}
.greeting-time { font-size:.63rem; color:#475569; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.2rem; }
.greeting-text { font-family:'DM Serif Display',serif; font-size:1rem; color:#c4b5fd; }

/* Validation messages */
.val-ok  { font-size:.74rem; color:#22c55e; margin-top:-.4rem; margin-bottom:.5rem; display:block; }
.val-err { font-size:.74rem; color:#ef4444; margin-top:-.4rem; margin-bottom:.5rem; display:block; }

/* SHAP */
.shap-insight-card {
    background: rgba(13,23,38,0.9);
    border: 1px solid rgba(139,92,246,.16);
    border-radius: 12px; padding: 1.2rem 1.4rem; margin: .4rem 0;
    display: flex; align-items: flex-start; gap: .9rem;
    backdrop-filter: blur(8px);
}
.shap-insight-icon  { font-size:1.4rem; flex-shrink:0; }
.shap-insight-title { font-weight:700; font-size:.85rem; color:#c4b5fd; margin-bottom:.15rem; }
.shap-insight-text  { font-size:.79rem; color:#64748b; line-height:1.55; }

/* CMP table */
.cmp-table { width:100%; border-collapse:collapse; }
.cmp-table th {
    background: rgba(20,35,60,0.9); color:#64748b; font-size:.74rem;
    font-weight:700; letter-spacing:.08em; text-transform:uppercase;
    padding:.7rem 1rem; text-align:left; border-bottom:1px solid rgba(255,255,255,.06);
}
.cmp-table td {
    padding:.72rem 1rem; border-bottom:1px solid rgba(255,255,255,.04);
    color:#e2e8f0; font-size:.87rem; font-family:'JetBrains Mono',monospace;
}
.cmp-table tr:hover td { background: rgba(0,212,180,0.03); }
.cmp-best  { color:#00d4b4 !important; font-weight:700 !important; }
.cmp-worst { color:#ef4444 !important; }

/* Counter display */
.counter-display {
    background: linear-gradient(160deg,rgba(4,20,38,0.94),rgba(2,10,20,0.97));
    border: 2px solid rgba(0,212,180,0.22);
    border-radius: 28px; padding: 2.4rem 2rem; text-align: center;
    margin: 1rem 0 2rem 0;
    box-shadow: 0 20px 60px rgba(0,212,180,0.07), inset 0 1px 0 rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    position: relative; overflow: hidden;
}
.counter-display::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 300px 200px at 50% 0%, rgba(0,212,180,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.counter-lbl { font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#334155; margin-bottom:.8rem; }
.counter-num {
    font-family:'DM Serif Display',serif; font-size:6rem; font-weight:400;
    color:#e8f4ff; line-height:1; text-shadow:0 0 60px rgba(0,212,180,0.25);
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
CLASS_NAMES = {
    0:"Poids Insuffisant",1:"Poids Normal",2:"Obésité Type I",
    3:"Obésité Type II",4:"Obésité Type III",5:"Surpoids Niveau I",6:"Surpoids Niveau II",
}
CLASS_INFO = {
    0:("Poids Insuffisant",  "green","IMC < 18.5",      "Risque de carences nutritionnelles. Suivi médical recommandé."),
    1:("Poids Normal",       "green","18.5 ≤ IMC < 25", "Profil clinique sain. Maintenir les habitudes actuelles."),
    2:("Obésité Type I",     "red",  "30 ≤ IMC < 35",   "Risque cardiovasculaire modéré. Suivi médical requis."),
    3:("Obésité Type II",    "red",  "35 ≤ IMC < 40",   "Risque cardiovasculaire élevé. Consultation spécialiste."),
    4:("Obésité Type III",   "red",  "IMC ≥ 40",        "Obésité morbide. Prise en charge médicale urgente."),
    5:("Surpoids Niveau I",  "amber","25 ≤ IMC < 27.5", "Surveiller l'alimentation. Augmenter l'activité physique."),
    6:("Surpoids Niveau II", "amber","27.5 ≤ IMC < 30", "Bilan lipidique conseillé. Consultation diététicien."),
}
CLASS_HEX = {0:"#22c55e",1:"#3b82f6",2:"#f59e0b",3:"#ef4444",4:"#8b5cf6",5:"#fbbf24",6:"#f97316"}

ALGO_LIST   = ["LightGBM Classifier","Random Forest Classifier","XGBoost Classifier"]
ALGO_ICONS  = {"LightGBM Classifier":"⚡","Random Forest Classifier":"🌲","XGBoost Classifier":"🚀"}
ALGO_COLORS = {"LightGBM Classifier":"#00d4b4","Random Forest Classifier":"#3b82f6","XGBoost Classifier":"#f97316"}
BEST_ALGO   = "LightGBM Classifier"

GENDER_MAP = {"Féminin":0,"Masculin":1}
BINARY_MAP = {"Non":0,"Oui":1}
CAEC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
CALC_MAP   = {"Jamais":3,"Parfois":2,"Fréquemment":1,"Toujours":0}
MTRANS_MAP = {"Automobile":0,"Vélo":1,"Moto":2,"Transport en commun":3,"Marche":4}

ROLES       = ["👩‍⚕️  Infirmière — Saisie Patient","👨‍⚕️  Médecin — Analyse & Diagnostic"]
NURSE_PAGES = ["📋  Dossier Patient","📏  Questionnaire Clinique","🏥  Tableau de Bord"]
DOC_PAGES   = ["🩺  Diagnostic IA","📁  Historique Patients","📊  Statistiques Cliniques","💊  Protocoles de Soins","🔬  Analyse IA Globale"]

FEATURE_LABELS = {
    "Gender":"Genre","Age":"Âge","Height":"Taille (m)","Weight":"Poids (kg)",
    "family_history_with_overweight":"Ant. familiaux","FAVC":"Aliments caloriques",
    "FCVC":"Fréquence légumes","NCP":"Repas/jour","CAEC":"Grignotage",
    "SMOKE":"Tabagisme","CH2O":"Eau/jour (L)","SCC":"Surveillance cal.",
    "FAF":"Activité physique","TUE":"Temps écran","CALC":"Alcool","MTRANS":"Transport",
}

CAPACITE_MAX = 20


# ═══════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════
for key, default in [
    ("patient_history", []),
    ("patient_counter", 0),
    ("patient_log", []),
    ("patient", {}),
    ("dossier_submitted", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════
#  ML HELPERS
# ═══════════════════════════════════════════════════════════
def dark_fig(w=10, h=5, ncols=1, nrows=1):
    plt.rcParams.update({
        "figure.facecolor":"#0a1628","axes.facecolor":"#101f38",
        "axes.edgecolor":"#1e3050","axes.labelcolor":"#94a3b8",
        "xtick.color":"#64748b","ytick.color":"#64748b",
        "text.color":"#e2e8f0","grid.color":"#162036",
        "legend.facecolor":"#0a1628","legend.edgecolor":"#1e3050",
        "font.family":"DejaVu Sans","figure.dpi":110,
    })
    if ncols==1 and nrows==1:
        return plt.subplots(figsize=(w,h))
    return plt.subplots(nrows,ncols,figsize=(w,h))

def set_dark_mpl():
    plt.rcParams.update({
        "figure.facecolor":"#0a1628","axes.facecolor":"#101f38",
        "axes.edgecolor":"#1e3050","axes.labelcolor":"#94a3b8",
        "xtick.color":"#64748b","ytick.color":"#64748b",
        "text.color":"#e2e8f0","grid.color":"#162036",
        "legend.facecolor":"#0a1628","legend.edgecolor":"#1e3050",
        "font.family":"DejaVu Sans","figure.dpi":110,
    })

@st.cache_data
def load_data():
    import os
    # Resolve paths relative to this script file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    candidates = [
        # Same folder as app.py (development / flat layout)
        os.path.join(script_dir, "data_clean.csv"),
        # ../data/data_clean.csv  (new repo: app/app.py → data/data_clean.csv)
        os.path.join(script_dir, "..", "data", "data_clean.csv"),
        # Root-level data_clean.csv
        os.path.join(script_dir, "..", "data_clean.csv"),
        # CWD fallbacks
        "data_clean.csv",
        "data/data_clean.csv",
        "../data/data_clean.csv",
        # Upload path (Claude environment)
        "/mnt/user-data/uploads/data_clean__2_.csv",
        "/mnt/user-data/uploads/data_clean__3_.csv",
    ]
    for p in candidates:
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            pass
    st.error("❌ data_clean.csv introuvable. Vérifiez que le fichier est dans data/ ou à la racine du projet.")
    st.stop()

@st.cache_resource
def train_model(algo=BEST_ALGO):
    df   = load_data()
    X    = df.drop("NObeyesdad",axis=1)
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
    clf  = clfs[algo]
    clf.fit(Xtrs,ytr)
    yp   = clf.predict(Xtes)
    explainer    = shap.TreeExplainer(clf)
    Xtes_sample  = Xtes[:min(200,len(Xtes))]
    shap_values  = explainer.shap_values(Xtes_sample)
    return (clf, sc, X.columns.tolist(),
            accuracy_score(yte,yp), f1_score(yte,yp,average="weighted"),
            precision_score(yte,yp,average="weighted"), recall_score(yte,yp,average="weighted"),
            confusion_matrix(yte,yp), classification_report(yte,yp,output_dict=True),
            Xtes, yte, yp, explainer, shap_values, Xtes_sample)

def imc_color(imc):
    if imc<18.5:   return "#60a5fa","Poids Insuffisant"
    elif imc<25:   return "#22c55e","Poids Normal ✓"
    elif imc<30:   return "#f59e0b","Surpoids"
    else:          return "#ef4444","Obésité ⚠️"

def get_shap_class(sv, ci):
    return sv[ci] if isinstance(sv,list) else (sv[:,:,ci] if sv.ndim==3 else sv)

def get_ev(exp, ci):
    ev=exp.expected_value
    return float(ev[ci]) if hasattr(ev,'__len__') else float(ev)

def global_shap_imp(sv,nc):
    return np.mean([np.abs(get_shap_class(sv,c)).mean(axis=0) for c in range(nc)],axis=0)

def plot_waterfall(exp,psv,pc,pdata,fc,cname,ccol):
    set_dark_mpl()
    sv  = get_shap_class(psv,pc)[0]
    ord = np.argsort(np.abs(sv))[::-1][:10]
    fig,ax=plt.subplots(figsize=(9,5))
    fig.patch.set_facecolor("#0a1628"); ax.set_facecolor("#101f38")
    cols=[ccol if v>0 else "#3b82f6" for v in sv[ord]]
    bars=ax.barh(range(10),sv[ord][::-1],color=cols[::-1],edgecolor="none",height=.62)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{FEATURE_LABELS.get(fc[i],fc[i])}  = {pdata[i]:.2f}" for i in ord[::-1]],fontsize=8.5,color="#e2e8f0")
    ax.axvline(0,color="#475569",lw=1.2,linestyle="--")
    ax.set_xlabel("Contribution SHAP",fontsize=9,color="#94a3b8")
    ax.set_title(f"Explication — {cname}",fontsize=11,color="#e2e8f0",pad=12,fontweight="600")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",alpha=.15,linestyle="--")
    for bar,v in zip(bars[::-1],sv[ord][::-1]):
        if abs(v)>.005:
            ax.text(v+(0.003 if v>=0 else -0.003),bar.get_y()+bar.get_height()/2,
                    f"{v:+.3f}",va="center",ha="left" if v>=0 else "right",fontsize=7.5,color="#e2e8f0",fontweight="700")
    plt.tight_layout(); return fig

def plot_global_imp(sv, fc, nc, ccol):
    """
    Stacked bar chart — Feature Importance SHAP par classe.
    Reproduit le style du graphique officiel du projet (shap_bar.png).
    """
    set_dark_mpl()

    # Per-class mean |SHAP| for each feature
    per_class = []
    for c in range(nc):
        sv_c = get_shap_class(sv, c)
        per_class.append(np.abs(sv_c).mean(axis=0))  # shape (n_features,)

    # Sort by total importance descending
    total_imp = np.sum(per_class, axis=0)
    sidx = np.argsort(total_imp)  # ascending for barh (bottom = least)

    feat_labels = [FEATURE_LABELS.get(fc[i], fc[i]) for i in sidx]
    y_pos = np.arange(len(sidx))

    # Class colors matching the notebook palette
    CLASS_COLORS_BAR = [
        "#2196F3",  # Class 0 — blue
        "#9C27B0",  # Class 1 — violet
        "#E91E63",  # Class 2 — pink
        "#FF1744",  # Class 3 — red
        "#FF6D00",  # Class 4 — orange
        "#4CAF50",  # Class 5 — green
        "#00BCD4",  # Class 6 — cyan
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#101f38")

    lefts = np.zeros(len(sidx))
    for c in range(nc):
        vals = np.array([per_class[c][i] for i in sidx])
        ax.barh(y_pos, vals, left=lefts,
                color=CLASS_COLORS_BAR[c % len(CLASS_COLORS_BAR)],
                alpha=0.88, height=0.62, label=f"Class {c}",
                edgecolor="none")
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=9, color="#e2e8f0")
    ax.set_xlabel("Importance SHAP moyenne (impact sur la prédiction)", fontsize=9, color="#94a3b8")
    ax.set_title("Feature Importance — Valeurs SHAP Moyennes",
                 fontsize=11, color="#e2e8f0", pad=12, fontweight="600")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linestyle="--")
    leg = ax.legend(title="Classe", fontsize=7.5, title_fontsize=7.5,
                    loc="lower right",
                    facecolor="#0a1628", edgecolor="#2d3a52",
                    labelcolor="#e2e8f0")
    leg.get_title().set_color("#64748b")
    plt.tight_layout()
    return fig

def generate_insights(sv,fc,pc,cname):
    out=[]; order=np.argsort(np.abs(sv))[::-1]
    CLINICAL={"Weight":("⚖️","Le poids est le facteur n°{r}."),"Height":("📏","La taille est le facteur n°{r}."),"Age":("🎂","L'âge joue un rôle n°{r}."),"FAF":("🏃","L'activité physique est le facteur n°{r}."),"CH2O":("💧","L'hydratation est le facteur n°{r}."),"FCVC":("🥦","La consommation de légumes (rang {r})."),"family_history_with_overweight":("🧬","Les antécédents familiaux n°{r}."),"FAVC":("🍔","Les aliments caloriques (rang {r})."),"CAEC":("🍪","Le grignotage est le facteur n°{r}."),"SMOKE":("🚬","Le tabagisme est le facteur n°{r}."),"MTRANS":("🚗","Le mode de transport (rang {r}).")}
    for rank,(fi,sv_v) in enumerate(zip([fc[i] for i in order[:3]],[sv[i] for i in order[:3]]),1):
        icon,title = CLINICAL.get(fi,("📌",f"{FEATURE_LABELS.get(fi,fi)} est le facteur n°{{r}}."))
        title=title.format(r=rank)
        text=f"{'Augmente' if sv_v>0 else 'Réduit'} la probabilité de '{cname}' de {abs(sv_v):.3f} point SHAP."
        out.append((icon,title,text))
    return out


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.7rem 0 .4rem;text-align:center;position:relative'>
        <svg width="48" height="48" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"
             style="display:block;margin:0 auto .4rem">
            <defs>
                <radialGradient id="rg1" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#00d4b4" stop-opacity="0.25"/>
                    <stop offset="100%" stop-color="#00d4b4" stop-opacity="0.03"/>
                </radialGradient>
                <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#00d4b4"/>
                    <stop offset="50%" stop-color="#06b6d4"/>
                    <stop offset="100%" stop-color="#3b82f6"/>
                </linearGradient>
            </defs>
            <!-- Outer ring with glow -->
            <circle cx="32" cy="32" r="30" fill="url(#rg1)" stroke="url(#lg1)" stroke-width="1.4" stroke-opacity="0.55"/>
            <!-- Inner ring -->
            <circle cx="32" cy="32" r="22" fill="none" stroke="#00d4b4" stroke-width="0.6" stroke-opacity="0.18"/>
            <!-- ECG pulse line -->
            <polyline points="4,32 9,32 12,22 15,42 18,28 21,32 28,32"
                fill="none" stroke="#00d4b4" stroke-width="1.5" stroke-opacity="0.4" stroke-linejoin="round" stroke-linecap="round"/>
            <polyline points="36,32 43,32 46,22 49,42 52,28 55,32 60,32"
                fill="none" stroke="#00d4b4" stroke-width="1.5" stroke-opacity="0.4" stroke-linejoin="round" stroke-linecap="round"/>
            <!-- Medical cross -->
            <rect x="27" y="16" width="10" height="32" rx="2.5" fill="url(#lg1)" opacity="0.90"/>
            <rect x="16" y="27" width="32" height="10" rx="2.5" fill="url(#lg1)" opacity="0.90"/>
            <!-- Center circle -->
            <circle cx="32" cy="32" r="4" fill="#060d1b" opacity="0.8"/>
            <circle cx="32" cy="32" r="2" fill="#00d4b4" opacity="0.7"/>
        </svg>
        <div style='font-family:"DM Serif Display",serif;font-size:1.3rem;color:#f1f5f9;letter-spacing:.02em'>ObesoScan</div>
        <div style='font-size:.62rem;color:#334155;font-weight:700;letter-spacing:.14em;text-transform:uppercase;margin-top:3px'>
            Clinique IA · v2.1
        </div>
        <div style='display:inline-flex;align-items:center;gap:.35rem;margin-top:.5rem;
                    background:rgba(0,212,180,.08);border:1px solid rgba(0,212,180,.18);
                    border-radius:20px;padding:.18rem .7rem'>
            <span style='width:6px;height:6px;border-radius:50%;background:#00d4b4;
                         box-shadow:0 0 6px #00d4b4;flex-shrink:0'></span>
            <span style='font-size:.65rem;color:#00d4b4;font-weight:700;letter-spacing:.06em'>EN LIGNE</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid rgba(0,212,180,0.10);margin:.4rem 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.63rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:#334155;margin-bottom:.3rem'>Rôle</div>",unsafe_allow_html=True)
    role      = st.radio("", ROLES, label_visibility="collapsed")
    is_nurse  = role == ROLES[0]
    is_doctor = not is_nurse

    st.markdown("<hr style='border:none;border-top:1px solid rgba(0,212,180,0.10);margin:.4rem 0'>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:.63rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:#334155;margin-bottom:.3rem'>Navigation</div>",unsafe_allow_html=True)
    page = st.radio("", NURSE_PAGES if is_nurse else DOC_PAGES, label_visibility="collapsed")
    st.markdown("<hr style='border:none;border-top:1px solid rgba(0,212,180,0.10);margin:.4rem 0'>",unsafe_allow_html=True)

    if is_doctor:
        greeting = get_greeting()
        now_str  = datetime.datetime.now().strftime("%H:%M")
        n_hist   = len(st.session_state["patient_history"])
        # Greeting
        st.markdown(f"""
        <div class='greeting-badge'>
            <div class='greeting-time'>{now_str} &nbsp;·&nbsp; {datetime.datetime.now().strftime("%d/%m/%Y")}</div>
            <div class='greeting-text'>{greeting}, Docteur 👨‍⚕️</div>
        </div>""", unsafe_allow_html=True)
        # Model info card
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(0,212,180,0.10),rgba(0,180,160,0.06));
                    border:1px solid rgba(0,212,180,0.28);border-radius:12px;
                    padding:.9rem 1rem;margin-bottom:.5rem'>
            <div style='font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
                        color:#00d4b4;margin-bottom:.5rem'>🤖 Modèle Actif</div>
            <div style='font-size:.9rem;font-weight:700;color:#f1f5f9;margin-bottom:.25rem'>⚡ LightGBM</div>
            <div style='font-size:.72rem;color:#64748b;line-height:1.6'>
                Gradient Boosting · 7 classes<br>
                Meilleure accuracy sur ce dataset
            </div>
        </div>
        <div style='font-size:.72rem;color:#334155;line-height:2;padding:.3rem 0'>
            <span style='color:#22c55e'>📁 {n_hist} diagnostic(s) enregistré(s)</span>
        </div>""", unsafe_allow_html=True)
        algo = BEST_ALGO
    else:
        algo = BEST_ALGO
        pat_name = st.session_state["patient"].get("nom","")
        st.markdown(f"""
        <div style='font-size:.72rem;color:#334155;line-height:1.9;padding:.5rem 0'>
            <span style='color:#06b6d4;font-weight:700'>Mode Infirmière</span><br>
            {'<span style="color:#22c55e">✓ Dossier : '+pat_name+'</span>' if pat_name else '<span style="color:#334155">Aucun dossier saisi</span>'}
        </div>""", unsafe_allow_html=True)

df = load_data()


# ╔══════════════════════════════════════════════════════════╗
#  NURSE — Page 1 : Dossier Patient
# ╚══════════════════════════════════════════════════════════╝
if is_nurse and page == NURSE_PAGES[0]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Dossier Patient</div>
        <div class='banner-sub'>Saisie des données biométriques, administratives et comportementales</div>
        <span class='banner-tag'>saisie-initiale</span><span class='banner-tag'>validation-temps-réel</span>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        # ── Identity ──
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🪪 Identité & Coordonnées</div>", unsafe_allow_html=True)

        nom_val = st.text_input("Nom complet *", placeholder="Ex : Amina El Alaoui", key="n_nom")
        ok_nom, msg_nom = validate_name(nom_val) if nom_val else (None,"")
        if nom_val:
            cls="val-ok" if ok_nom else "val-err"; icon="✅" if ok_nom else "❌"
            st.markdown(f"<span class='{cls}'>{icon} {msg_nom}</span>", unsafe_allow_html=True)

        cin_val = st.text_input("CIN (Carte d'Identité Nationale)", placeholder="Ex : BE123456", key="n_cin")
        ok_cin, msg_cin, cin_display = validate_cin(cin_val) if cin_val else (True,"","")
        if cin_val:
            cls="val-ok" if ok_cin else "val-err"; icon="✅" if ok_cin else "❌"
            st.markdown(f"<span class='{cls}'>{icon} {msg_cin}</span>", unsafe_allow_html=True)

        # Country code + phone
        tel_cc = st.selectbox("Indicatif pays *", CC_DISPLAY, index=0, key="n_cc")
        sel_code = CC_CODE_MAP[tel_cc]
        phone_placeholder = CC_EXAMPLES.get(sel_code, f"Ex : numéro local ({sel_code})")
        tel_val = st.text_input(f"Numéro de téléphone * ({sel_code})", placeholder=phone_placeholder, key="n_tel")
        ok_tel, msg_tel = validate_phone(tel_val, sel_code) if tel_val else (None,"")
        if tel_val:
            cls="val-ok" if ok_tel else "val-err"; icon="✅" if ok_tel else "❌"
            st.markdown(f"<span class='{cls}'>{icon} {msg_tel}</span>", unsafe_allow_html=True)

        email_val = st.text_input("Email", placeholder="Ex : patient@gmail.com", key="n_email")
        ok_email, msg_email = validate_email(email_val)
        if email_val:
            cls="val-ok" if ok_email else "val-err"; icon="✅" if ok_email else "❌"
            st.markdown(f"<span class='{cls}'>{icon} {msg_email}</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Biometrics ──
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>📐 Biométrie</div>", unsafe_allow_html=True)
        gender = st.selectbox("Genre *", ["Féminin","Masculin"], key="n_gender")
        age    = st.number_input("Âge (années) *", min_value=10, max_value=90, value=28, step=1, key="n_age")
        c1, c2 = st.columns(2)
        height = c1.number_input("Taille (m) *", min_value=1.40, max_value=2.15, value=1.70, step=0.01, format="%.2f", key="n_height")
        weight = c2.number_input("Poids (kg) *", min_value=30.0, max_value=200.0, value=70.0, step=0.5, key="n_weight")
        st.markdown("</div>", unsafe_allow_html=True)

        imc = round(weight/(height**2), 1)
        imc_c, imc_t = imc_color(imc)
        st.markdown(f"""
        <div class='imc-live'>
            <div class='imc-label'>Indice de Masse Corporelle (IMC)</div>
            <div class='imc-value' style='color:{imc_c}'>{imc}</div>
            <div class='imc-cat' style='color:{imc_c}'>{imc_t}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='form-section' style='margin-top:1.2rem'><div class='form-title ft-nurse'>🩺 Antécédents & Statut</div>",unsafe_allow_html=True)
        family = st.selectbox("Antécédents familiaux d'obésité *", ["Non","Oui"], key="n_family")
        smoke  = st.selectbox("Tabagisme actif", ["Non","Oui"], key="n_smoke")
        scc    = st.selectbox("Surveillance calorique (SCC)", ["Non","Oui"], key="n_scc")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🍽️ Habitudes Alimentaires</div>",unsafe_allow_html=True)
        favc = st.selectbox("Aliments très caloriques (FAVC)", ["Non","Oui"], key="n_favc")
        fcvc = st.slider("Fréquence légumes (FCVC)", 1.0, 3.0, 2.0, 0.1, key="n_fcvc")
        ncp  = st.slider("Repas principaux / jour (NCP)", 1.0, 4.0, 3.0, 0.5, key="n_ncp")
        caec = st.selectbox("Alimentation entre repas (CAEC)", ["Jamais","Parfois","Fréquemment","Toujours"], key="n_caec")
        calc = st.selectbox("Consommation d'alcool (CALC)", ["Jamais","Parfois","Fréquemment","Toujours"], key="n_calc")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🏃 Activité & Mode de Vie</div>",unsafe_allow_html=True)
        ch2o   = st.slider("Eau / jour (litres)", 1.0, 3.0, 2.0, 0.1, key="n_ch2o")
        faf    = st.slider("Activité physique (jours/semaine)", 0.0, 3.0, 1.0, 0.1, key="n_faf")
        tue    = st.slider("Temps écran quotidien (heures)", 0.0, 2.0, 1.0, 0.1, key="n_tue")
        mtrans = st.selectbox("Transport principal", list(MTRANS_MAP.keys()), key="n_mtrans")
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick summary
        st.markdown(f"""
        <div class='panel p-nurse' style='margin-top:0'>
            <div class='panel-title'>📋 Récapitulatif rapide</div>
            <div style='font-size:.82rem;line-height:2;color:#94a3b8'>
                <b style='color:#e2e8f0'>Nom :</b> {nom_val or "—"}<br>
                <b style='color:#e2e8f0'>CIN :</b> {cin_display if cin_display else "—"}<br>
                <b style='color:#e2e8f0'>Tél :</b> {sel_code} {tel_val or "—"}<br>
                <b style='color:#e2e8f0'>IMC :</b> <span style='color:{imc_c};font-weight:700'>{imc} — {imc_t}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Save button ──
    st.markdown("<br>", unsafe_allow_html=True)
    save_btn = st.button("💾  Enregistrer & Admettre le Patient", use_container_width=False)

    if save_btn:
        errors = []
        if not nom_val:             errors.append("Nom du patient obligatoire")
        elif not ok_nom:            errors.append(f"Nom invalide : {msg_nom}")
        if cin_val and not ok_cin:  errors.append(f"CIN invalide : {msg_cin}")
        if not tel_val:             errors.append("Téléphone obligatoire")
        elif not ok_tel:            errors.append(f"Téléphone invalide : {msg_tel}")
        if email_val and not ok_email: errors.append(f"Email invalide : {msg_email}")

        if errors:
            for e in errors:
                st.markdown(f"""<div style='background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.22);
                    border-radius:8px;padding:.6rem 1rem;margin:.2rem 0;font-size:.83rem;color:#fca5a5'>❌ {e}</div>""",
                    unsafe_allow_html=True)
        else:
            full_phone = f"{sel_code} {tel_val.strip()}"
            patient_data = {
                "nom":nom_val.strip(), "cin":cin_display if cin_display else "—",
                "telephone":full_phone, "email":email_val.strip() if email_val else "—",
                "gender":gender,"age":age,"height":height,"weight":weight,
                "family":family,"smoke":smoke,"scc":scc,"favc":favc,
                "fcvc":fcvc,"ncp":ncp,"caec":caec,"calc":calc,
                "ch2o":ch2o,"faf":faf,"tue":tue,"mtrans":mtrans,
            }
            st.session_state["patient"] = patient_data
            st.session_state["dossier_submitted"] = True

            # Auto-increment counter
            st.session_state.patient_counter += 1
            st.session_state.patient_log.append({
                "heure":   datetime.datetime.now().strftime("%H:%M:%S"),
                "type":    "arrivée",
                "nom":     nom_val.strip(),
                "tel":     full_phone,
                "email":   email_val.strip() if email_val else "—",
                "cin":     cin_display if cin_display else "—",
                "message": f"Patient «{nom_val.strip()}» admis",
                "total":   st.session_state.patient_counter,
            })

            st.markdown(f"""
            <div class='panel p-green'>
                <div class='panel-title'>✅ Patient admis avec succès — Dossier enregistré</div>
                <div class='panel-body'>
                    <b>{nom_val.strip()}</b> est désormais compté dans le tableau de bord.
                    Le médecin peut accéder au <b>Diagnostic IA</b> pour la prédiction.
                </div>
            </div>""", unsafe_allow_html=True)


# ── NURSE Page 2 : Questionnaire Clinique ──
elif is_nurse and page == NURSE_PAGES[1]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Questionnaire Clinique</div>
        <div class='banner-sub'>Évaluation complémentaire des facteurs de risque comportementaux</div>
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
            st.markdown("<div class='form-section'><div class='form-title ft-nurse'>📋 Récapitulatif</div>",unsafe_allow_html=True)
            imc_q = round(pat["weight"]/(pat["height"]**2),1)
            rows=[("Patient",pat.get("nom","—")),("CIN",pat.get("cin","—")),
                  ("Téléphone",pat.get("telephone","—")),("Email",pat.get("email","—")),
                  ("Genre","Femme" if pat["gender"]=="Féminin" else "Homme"),
                  ("Âge",f'{pat["age"]} ans'),("Taille",f'{pat["height"]} m'),
                  ("Poids",f'{pat["weight"]} kg'),("IMC",f'{imc_q}'),
                  ("Ant. familiaux",pat["family"]),("Tabagisme",pat["smoke"])]
            html="".join([f"<div class='stat-row'><span class='sk'>{k}</span><span class='sv'>{v}</span></div>" for k,v in rows])
            st.markdown(f"{html}</div>",unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='form-section'><div class='form-title ft-nurse'>🔍 Évaluation des risques</div>",unsafe_allow_html=True)
            score,flags=0,[]
            if imc_q>=30:    score+=3; flags.append(("red","IMC ≥ 30 — Obésité clinique"))
            elif imc_q>=25:  score+=2; flags.append(("amber","IMC 25–30 — Zone Surpoids"))
            else:            flags.append(("green","IMC dans la norme"))
            if pat.get("family")=="Oui": score+=2; flags.append(("amber","Antécédents familiaux"))
            if pat.get("faf",1)<1.0:    score+=1; flags.append(("amber","Activité physique insuffisante"))
            if pat.get("smoke")=="Oui": score+=1; flags.append(("amber","Tabagisme actif"))
            if pat.get("caec") in ["Fréquemment","Toujours"]: score+=1; flags.append(("amber","Grignotage fréquent"))
            if pat.get("ch2o",2)<1.5:  score+=1; flags.append(("red","Hydratation insuffisante"))
            level="Risque Faible" if score<=2 else "Risque Modéré" if score<=4 else "Risque Élevé"
            lc="#22c55e" if score<=2 else "#f59e0b" if score<=4 else "#ef4444"
            st.markdown(f"""
            <div style='background:rgba(0,0,0,.15);border:1px solid {lc}28;border-radius:12px;
                        padding:1rem;text-align:center;margin-bottom:1rem'>
                <div style='font-size:.66rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#475569;margin-bottom:.3rem'>Score de risque</div>
                <div style='font-family:"DM Serif Display",serif;font-size:2rem;color:{lc}'>{score}/10</div>
                <div style='font-size:.81rem;font-weight:700;color:{lc};margin-top:.2rem'>{level}</div>
            </div>""", unsafe_allow_html=True)
            for col,msg in flags:
                icon="✅" if col=="green" else "⚠️" if col=="amber" else "🚨"
                c_f="#22c55e" if col=="green" else "#f59e0b" if col=="amber" else "#ef4444"
                st.markdown(f"""<div style='background:rgba(0,0,0,.1);border-left:3px solid {c_f};
                    border-radius:0 8px 8px 0;padding:.42rem .9rem;margin:.3rem 0;font-size:.8rem;color:#94a3b8'>
                    {icon} {msg}</div>""", unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)

        st.markdown("""<div class='panel p-nurse' style='margin-top:1.5rem'>
            <div class='panel-title'>📨 Dossier transmis au médecin</div>
            <div class='panel-body'>Le médecin peut accéder au <strong>Diagnostic IA</strong> pour la prédiction personnalisée.</div>
        </div>""", unsafe_allow_html=True)


# ── NURSE Page 3 : Tableau de Bord ──
elif is_nurse and page == NURSE_PAGES[2]:
    st.markdown("""
    <div class='page-banner banner-nurse'>
        <div class='banner-eyebrow ey-nurse'>👩‍⚕️ Interface Infirmière</div>
        <div class='banner-h1'>Tableau de Bord — Flux Patients</div>
        <div class='banner-sub'>Le dossier saisi est automatiquement reflété ici · Gestion de la capacité clinique</div>
        <span class='banner-tag'>auto-sync</span><span class='banner-tag'>capacité</span>
    </div>""", unsafe_allow_html=True)

    # ── Auto-populated patient card from dossier ──
    pat = st.session_state.get("patient",{})
    if pat:
        imc_tb = round(pat["weight"]/(pat["height"]**2),1)
        imc_tc, imc_tt = imc_color(imc_tb)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(6,182,212,0.07),rgba(10,22,40,0.9));
                    border:1px solid rgba(6,182,212,0.22);border-radius:16px;
                    padding:1.4rem 1.8rem;margin-bottom:1.5rem;backdrop-filter:blur(12px)'>
            <div style='font-size:.67rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
                        color:#06b6d4;margin-bottom:.8rem'>🔗 Dernier dossier enregistré — synchronisation automatique</div>
            <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem'>
                <div>
                    <div style='font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em'>Patient</div>
                    <div style='font-size:1rem;color:#e2e8f0;font-weight:700;margin-top:.2rem'>{pat.get("nom","—")}</div>
                    <div style='font-size:.75rem;color:#64748b'>CIN : {pat.get("cin","—")}</div>
                </div>
                <div>
                    <div style='font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em'>Contact</div>
                    <div style='font-size:.82rem;color:#94a3b8;margin-top:.2rem;font-family:"JetBrains Mono",monospace'>{pat.get("telephone","—")}</div>
                    <div style='font-size:.76rem;color:#475569'>{pat.get("email","—")}</div>
                </div>
                <div>
                    <div style='font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em'>Biométrie</div>
                    <div style='font-size:.82rem;color:#94a3b8;margin-top:.2rem'>{pat.get("height","—")} m · {pat.get("weight","—")} kg</div>
                    <div style='font-size:.82rem;color:{imc_tc};font-weight:700'>IMC : {imc_tb} — {imc_tt}</div>
                </div>
                <div>
                    <div style='font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em'>Activité</div>
                    <div style='font-size:.82rem;color:#94a3b8;margin-top:.2rem'>{pat.get("faf","—")} j/sem</div>
                    <div style='font-size:.76rem;color:#475569'>{pat.get("ch2o","—")} L/j · {pat.get("smoke","—")} tabac</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='panel p-amber'>
            <div class='panel-title'>⚠️ Aucun dossier actif</div>
            <div class='panel-body'>Enregistrez un patient dans <strong>Dossier Patient</strong> — les informations apparaîtront ici automatiquement.</div>
        </div>""", unsafe_allow_html=True)

    # ── Counter ──
    nb     = st.session_state.patient_counter
    libres = max(0, CAPACITE_MAX - nb)
    pct    = min(int(nb / CAPACITE_MAX * 100), 100)
    if pct==0:       stxt,scol,sico = "Clinique vide","#3b82f6","🔵"
    elif pct<50:     stxt,scol,sico = "Disponible","#22c55e","🟢"
    elif pct<80:     stxt,scol,sico = "Affluence modérée","#f59e0b","🟡"
    elif pct<100:    stxt,scol,sico = "Quasi complet","#f97316","🟠"
    else:            stxt,scol,sico = "COMPLET","#ef4444","🔴"

    col_cnt = st.columns([1,2,1])[1]
    with col_cnt:
        st.markdown(f"""
        <div class='counter-display'>
            <div class='counter-lbl'>Patients présents en consultation</div>
            <div class='counter-num'>{nb}</div>
            <div style='margin-top:.8rem'>
                <span style='background:rgba(0,212,180,.07);border:1px solid rgba(0,212,180,.18);
                             border-radius:20px;padding:.22rem .85rem;font-size:.72rem;color:#5eead4;font-weight:600'>
                    {sico} {stxt} &nbsp;·&nbsp; {libres} place(s) libre(s)
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    # KPIs
    kpc = "c-green" if libres>10 else "c-amber" if libres>4 else "c-red"
    kpp = "c-green" if pct<50 else "c-amber" if pct<80 else "c-red"
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(f"<div class='kpi-card c-blue'><div class='kpi-num'>{nb}</div><div class='kpi-lbl'>Présents</div></div>",unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi-card {kpc}'><div class='kpi-num'>{libres}</div><div class='kpi-lbl'>Disponibles / {CAPACITE_MAX}</div></div>",unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi-card {kpp}'><div class='kpi-num'>{pct}%</div><div class='kpi-lbl'>Occupation</div></div>",unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='kpi-card' style='border-bottom:3px solid {scol}'><div style='font-size:1.5rem;margin-bottom:.3rem'>{sico}</div><div style='font-family:\"DM Serif Display\",serif;font-size:.95rem;color:{scol};font-weight:700'>{stxt}</div><div class='kpi-lbl' style='margin-top:.3rem'>Statut</div></div>",unsafe_allow_html=True)

    # Segment bar
    segs=[]
    for i in range(CAPACITE_MAX):
        sc2 = "#34d399" if nb/CAPACITE_MAX<0.5 else "#fbbf24" if nb/CAPACITE_MAX<0.8 else "#f43f5e"
        segs.append(f"<div style='flex:1;height:22px;background:{''+sc2+'' if i<nb else 'rgba(255,255,255,0.04)'};border-radius:3px;margin:0 1px;transition:background .3s'></div>")
    st.markdown(f"""
    <div style='background:rgba(10,22,40,0.85);border:1px solid rgba(255,255,255,.06);
                border-radius:14px;padding:1.3rem 1.6rem;margin-top:.8rem;backdrop-filter:blur(8px)'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem'>
            <span style='color:#475569;font-size:.78rem;font-weight:600'>📊 {nb} / {CAPACITE_MAX}</span>
            <span style='background:rgba(0,212,180,.07);border:1px solid rgba(0,212,180,.16);border-radius:20px;
                         padding:.16rem .72rem;font-size:.71rem;color:#5eead4;font-weight:700'>{libres} libre(s)</span>
        </div>
        <div style='display:flex;gap:2px;'>{''.join(segs)}</div>
    </div>""", unsafe_allow_html=True)

    # Manual controls
    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Contrôles manuels</div>",unsafe_allow_html=True)
    bc1,bc2,bc3,bc4 = st.columns(4,gap="medium")
    with bc1:
        if st.button("➕  Patient arrivé (+1)", use_container_width=True, key="btn_add"):
            st.session_state.patient_counter += 1
            st.session_state.patient_log.append({"heure":datetime.datetime.now().strftime("%H:%M:%S"),"type":"arrivée","nom":"(manuel)","tel":"—","email":"—","cin":"—","message":"1 patient admis manuellement","total":st.session_state.patient_counter})
            st.rerun()
    with bc2:
        if st.button("➖  Patient sorti (-1)", use_container_width=True, key="btn_sub"):
            if st.session_state.patient_counter > 0:
                st.session_state.patient_counter -= 1
                st.session_state.patient_log.append({"heure":datetime.datetime.now().strftime("%H:%M:%S"),"type":"sortie","nom":"—","tel":"—","email":"—","cin":"—","message":"1 patient sorti","total":st.session_state.patient_counter})
            st.rerun()
    with bc3:
        quick = st.number_input("Nb",min_value=2,max_value=20,value=2,step=1,key="quick",label_visibility="collapsed")
        if st.button(f"⚡ Ajouter {quick}", use_container_width=True, key="btn_quick"):
            st.session_state.patient_counter += quick
            st.session_state.patient_log.append({"heure":datetime.datetime.now().strftime("%H:%M:%S"),"type":"arrivée","nom":"—","tel":"—","email":"—","cin":"—","message":f"{quick} patients admis","total":st.session_state.patient_counter})
            st.rerun()
    with bc4:
        if st.button("🔄  Remettre à zéro", use_container_width=True, key="btn_reset"):
            st.session_state.patient_log.append({"heure":datetime.datetime.now().strftime("%H:%M:%S"),"type":"reset","nom":"—","tel":"—","email":"—","cin":"—","message":f"Remise à zéro ({st.session_state.patient_counter} patients)","total":0})
            st.session_state.patient_counter = 0
            st.rerun()

    # Journal
    st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Journal des mouvements</div>",unsafe_allow_html=True)
    logs = st.session_state.patient_log
    if not logs:
        st.markdown("""<div style='background:rgba(10,22,40,0.85);border:1px solid rgba(255,255,255,.05);
            border-radius:12px;padding:1.5rem;text-align:center;color:#334155;font-size:.83rem'>
            📋 &nbsp;Aucun mouvement enregistré.</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:rgba(10,22,40,0.9);border:1px solid rgba(255,255,255,.06);
            border-radius:14px;overflow:hidden;backdrop-filter:blur(8px)'>
            <div style='display:grid;grid-template-columns:75px 1.2fr 1fr 1fr 1.8fr 75px;gap:0;
                        padding:.5rem 1rem;background:rgba(20,35,60,0.9);border-bottom:1px solid rgba(255,255,255,.05)'>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155'>Heure</span>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155'>Nom</span>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155'>Téléphone</span>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155'>CIN</span>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155'>Mouvement</span>
                <span style='font-size:.62rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#334155;text-align:right'>Total</span>
            </div>""", unsafe_allow_html=True)
        for idx,entry in enumerate(reversed(logs[-12:])):
            if entry["type"]=="arrivée": ec,ei,eb="#34d399","➕","rgba(52,211,153,.04)"
            elif entry["type"]=="sortie": ec,ei,eb="#f43f5e","➖","rgba(244,63,94,.04)"
            else: ec,ei,eb="#94a3b8","🔄","rgba(0,0,0,.02)"
            border="border-bottom:1px solid rgba(255,255,255,.03);" if idx<11 else ""
            st.markdown(
                f"<div style='background:{eb};{border}display:grid;"
                f"grid-template-columns:75px 1.2fr 1fr 1fr 1.8fr 75px;"
                f"align-items:center;gap:0;padding:.55rem 1rem'>"
                f"<span style='color:#475569;font-family:\"JetBrains Mono\",monospace;font-size:.7rem'>{entry['heure']}</span>"
                f"<span style='color:#cbd5e1;font-size:.8rem;font-weight:600'>{entry.get('nom','—')}</span>"
                f"<span style='color:#64748b;font-family:\"JetBrains Mono\",monospace;font-size:.74rem'>{entry.get('tel','—')}</span>"
                f"<span style='color:#475569;font-family:\"JetBrains Mono\",monospace;font-size:.74rem'>{entry.get('cin','—')}</span>"
                f"<span style='color:{ec};font-size:.8rem'>{ei} {entry['message']}</span>"
                f"<span style='background:{ec}1a;color:{ec};border-radius:20px;padding:.1rem .5rem;"
                f"font-size:.68rem;font-weight:700;text-align:center'>{entry['total']}</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<div style='margin-top:.6rem'></div>",unsafe_allow_html=True)
        if st.button("🗑️  Vider le journal", key="btn_clear_log"):
            st.session_state.patient_log=[]
            st.rerun()


# ╔══════════════════════════════════════════════════════════╗
#  DOCTOR — Page 2 : Diagnostic IA
# ╚══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[0]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Diagnostic Individuel IA</div>
        <div class='banner-sub'>Prédiction personnalisée · ⚡ LightGBM · ⭐ Meilleur modèle · 🔍 SHAP activé</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Chargement du modèle…"):
        (clf,sc_m,fc,acc,f1,prec,rec,cm,cr,Xtes,yte,yp,explainer,shap_values,Xtes_sample) = train_model(BEST_ALGO)

    n_classes = len(CLASS_NAMES)
    st.markdown(
        f"<span class='chip chip-teal'>⚡ LightGBM</span>"
        f"<span class='chip'>✅ Acc {acc*100:.1f}%</span>"
        f"<span class='chip'>F1 {f1*100:.1f}%</span>"
        f"<span class='chip chip-violet'>🔍 SHAP actif</span>",
        unsafe_allow_html=True)

    pat = st.session_state.get("patient",{})
    if pat:
        st.markdown(f"""<div class='panel p-nurse'>
            <div class='panel-title'>🔗 Dossier importé — {pat.get("nom","Patient")}</div>
            <div class='panel-body'>Données pré-chargées depuis le dossier infirmière. CIN : {pat.get("cin","—")} · {pat.get("telephone","—")}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Paramètres Patient</div>",unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3,gap="large")

    def pidx(lst,key,default):
        v=pat.get(key,default)
        try: return lst.index(v)
        except: return lst.index(default)

    with col1:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🪪 Biométrie</div>",unsafe_allow_html=True)
        gender = st.selectbox("Genre",["Féminin","Masculin"],index=pidx(["Féminin","Masculin"],"gender","Féminin"),key="d_gender")
        age    = st.slider("Âge",10,80,pat.get("age",26),key="d_age")
        height = st.slider("Taille (m)",1.40,2.10,float(pat.get("height",1.70)),0.01,key="d_height")
        weight = st.slider("Poids (kg)",30.0,170.0,float(pat.get("weight",70.0)),0.5,key="d_weight")
        family = st.selectbox("Antécédents familiaux",["Non","Oui"],index=pidx(["Non","Oui"],"family","Non"),key="d_family")
        st.markdown("</div>",unsafe_allow_html=True)
        imc_d=round(weight/(height**2),1); imc_dc,imc_dt=imc_color(imc_d)
        st.markdown(f"""<div class='imc-live'>
            <div class='imc-label'>IMC Calculé</div>
            <div class='imc-value' style='color:{imc_dc}'>{imc_d}</div>
            <div class='imc-cat' style='color:{imc_dc}'>{imc_dt}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🍽️ Alimentation</div>",unsafe_allow_html=True)
        favc = st.selectbox("Aliments caloriques (FAVC)",["Non","Oui"],index=pidx(["Non","Oui"],"favc","Non"),key="d_favc")
        fcvc = st.slider("Légumes (FCVC)",1.0,3.0,float(pat.get("fcvc",2.0)),0.1,key="d_fcvc")
        ncp  = st.slider("Repas / jour",1.0,4.0,float(pat.get("ncp",3.0)),0.5,key="d_ncp")
        caec = st.selectbox("Grignotage (CAEC)",["Jamais","Parfois","Fréquemment","Toujours"],index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"caec","Parfois"),key="d_caec")
        calc = st.selectbox("Alcool (CALC)",["Jamais","Parfois","Fréquemment","Toujours"],index=pidx(["Jamais","Parfois","Fréquemment","Toujours"],"calc","Jamais"),key="d_calc")
        st.markdown("</div>",unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='form-section'><div class='form-title ft-doctor'>🏃 Mode de Vie</div>",unsafe_allow_html=True)
        smoke  = st.selectbox("Tabagisme",["Non","Oui"],index=pidx(["Non","Oui"],"smoke","Non"),key="d_smoke")
        ch2o   = st.slider("Eau / jour (L)",1.0,3.0,float(pat.get("ch2o",2.0)),0.1,key="d_ch2o")
        scc    = st.selectbox("Surveillance cal.",["Non","Oui"],index=pidx(["Non","Oui"],"scc","Non"),key="d_scc")
        faf    = st.slider("Activité (j/sem)",0.0,3.0,float(pat.get("faf",1.0)),0.1,key="d_faf")
        tue    = st.slider("Temps écran (h/j)",0.0,2.0,float(pat.get("tue",1.0)),0.1,key="d_tue")
        mtrans = st.selectbox("Transport",list(MTRANS_MAP.keys()),index=pidx(list(MTRANS_MAP.keys()),"mtrans","Automobile"),key="d_mtrans")
        st.markdown("</div>",unsafe_allow_html=True)

    bcol,_ = st.columns([1,3])
    with bcol:
        diag_btn = st.button("🩺  Lancer le Diagnostic", use_container_width=True)

    if diag_btn:
        row={"Gender":GENDER_MAP[gender],"Age":float(age),"Height":height,"Weight":weight,
             "family_history_with_overweight":BINARY_MAP[family],"FAVC":BINARY_MAP[favc],
             "FCVC":fcvc,"NCP":ncp,"CAEC":CAEC_MAP[caec],"SMOKE":BINARY_MAP[smoke],
             "CH2O":ch2o,"SCC":BINARY_MAP[scc],"FAF":faf,"TUE":tue,
             "CALC":CALC_MAP[calc],"MTRANS":MTRANS_MAP[mtrans]}
        Xn=pd.DataFrame([row])[fc]; Xns=sc_m.transform(Xn)
        pred=int(clf.predict(Xns)[0])
        proba=clf.predict_proba(Xns)[0] if hasattr(clf,"predict_proba") else None
        info=CLASS_INFO[pred]
        rb_class={"green":"rb-green","amber":"rb-amber","red":"rb-red"}[info[1]]
        emoji="✅" if info[1]=="green" else "⚠️" if info[1]=="amber" else "🚨"
        pred_color=CLASS_HEX[pred]

        # Save to history
        entry={"timestamp":datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
               "nom":pat.get("nom","—"),"cin":pat.get("cin","—"),
               "telephone":pat.get("telephone","—"),"email":pat.get("email","—"),
               "age":age,"genre":"H" if gender=="Masculin" else "F",
               "imc":imc_d,"diagnostic":info[0],"classe":pred,
               "confiance":f"{max(proba)*100:.1f}%" if proba is not None else "—",
               "color":info[1]}
        hist=st.session_state["patient_history"]
        if not hist or hist[-1].get("nom")!=entry["nom"] or hist[-1].get("timestamp")!=entry["timestamp"]:
            hist.append(entry)

        st.markdown("---")
        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Résultat du Diagnostic</div>",unsafe_allow_html=True)
        rc1,rc2=st.columns([1.2,1],gap="large")
        with rc1:
            st.markdown(f"""<div class='result-box {rb_class}'>
                <span class='result-emoji'>{emoji}</span>
                <div class='result-title'>{info[0]}</div>
                <div class='result-imc'>IMC : {imc_d} &nbsp;·&nbsp; {info[2]}</div>
                <div class='result-desc'>{info[3]}</div>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""<div class='panel p-doctor' style='height:100%'>
                <div class='panel-title'>📋 Résumé Patient</div>
                <div style='font-size:.84rem;line-height:2.1;color:#94a3b8'>
                    <b style='color:#e2e8f0'>Patient :</b> {pat.get("nom","—")}<br>
                    <b style='color:#e2e8f0'>CIN :</b> {pat.get("cin","—")}<br>
                    <b style='color:#e2e8f0'>Genre / Âge :</b> {"Homme" if gender=="Masculin" else "Femme"} · {age} ans<br>
                    <b style='color:#e2e8f0'>IMC :</b> <span style='color:{imc_dc};font-weight:800;font-size:1rem;font-family:"JetBrains Mono",monospace'>{imc_d}</span><br>
                    <b style='color:#e2e8f0'>Activité :</b> {faf} j/sem · {ch2o} L/j<br>
                    <b style='color:#e2e8f0'>Confiance IA :</b> <span style='color:{pred_color}'>{f"{max(proba)*100:.1f}%" if proba is not None else "—"}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Probabilities
        if proba is not None:
            st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Probabilités diagnostiques</div>",unsafe_allow_html=True)
            fig_p,ax_p=dark_fig(11,4.5)
            bars_p=ax_p.bar([CLASS_NAMES[i] for i in range(7)],proba,
                            color=[CLASS_HEX[i] for i in range(7)],edgecolor="none",width=.58)
            for bar,av in zip(bars_p,[1.0 if i==pred else .32 for i in range(7)]): bar.set_alpha(av)
            ax_p.set_ylim(0,1.15); ax_p.set_ylabel("Probabilité",fontsize=9,color="#64748b")
            ax_p.spines[["top","right"]].set_visible(False); ax_p.grid(axis="y",alpha=.15,linestyle="--")
            plt.xticks(rotation=22,ha="right",fontsize=8.5,color="#94a3b8")
            for bar,p_v in zip(bars_p,proba):
                if p_v>.015: ax_p.text(bar.get_x()+bar.get_width()/2,p_v+.015,f"{p_v*100:.1f}%",ha="center",va="bottom",fontsize=8.5,color="#e2e8f0",fontweight="700")
            plt.tight_layout(); st.pyplot(fig_p,use_container_width=True)

        # Recommendations
        st.markdown("<div class='sec-head'><div class='dot dot-doctor'></div>Recommandations Médicales</div>",unsafe_allow_html=True)
        recs=[]
        if faf<1.0: recs.append(("red","🏃","Activité physique insuffisante","≥ 150 min/sem (OMS). Débuter 20 min/j de marche rapide."))
        elif faf>=2.5: recs.append(("green","🏃","Activité physique optimale",f"{faf} j/sem — Risque cardiovasculaire -30%."))
        else: recs.append(("amber","🏃","Activité à renforcer","Progresser vers 3–4 séances/sem."))
        if ch2o<1.5: recs.append(("red","💧","Hydratation insuffisante",f"{ch2o} L/j. Objectif : ≥ 2 L/j."))
        elif ch2o>=2.0: recs.append(("green","💧","Hydratation satisfaisante",f"{ch2o} L/j — conforme EFSA."))
        if caec in ["Fréquemment","Toujours"]: recs.append(("red","🍪","Grignotage excessif","+20-30% calories. Orienter vers diététicien."))
        if smoke=="Oui": recs.append(("red","🚬","Tabagisme actif","Perturbe le métabolisme. Consultation sevrage."))
        if family=="Oui": recs.append(("amber","🧬","Prédisposition génétique","Risque ×2–3. Suivi annuel."))
        if imc_d>=30: recs.append(("red","⚕️","Consultation spécialiste urgente","Bilan lipidique, glycémie, TA. Orientation endocrinologue."))
        elif 25<=imc_d<30: recs.append(("amber","⚕️","Suivi préventif recommandé","Bilan cardiovasculaire préventif."))
        else: recs.append(("green","⚕️","Profil clinique satisfaisant","IMC normal. Maintenir les habitudes."))
        rcm={"green":"rc-green","amber":"rc-amber","red":"rc-red"}
        for color,icon,title,text in recs:
            st.markdown(f"""<div class='rec-card {rcm[color]}'>
                <div class='rec-icon'>{icon}</div>
                <div><div class='rec-title'>{title}</div><div class='rec-text'>{text}</div></div>
            </div>""", unsafe_allow_html=True)

        # SHAP
        st.markdown("---")
        st.markdown("<div class='sec-head'><div class='dot dot-violet'></div>🔍 Explicabilité SHAP</div>",unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(139,92,246,.06),transparent);
                    border:1px solid rgba(139,92,246,.18);border-radius:14px;
                    padding:1.2rem 1.7rem;margin-bottom:1.4rem;backdrop-filter:blur(8px)'>
            <div style='font-size:.66rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#7c3aed;margin-bottom:.4rem'>SHAP · SHapley Additive exPlanations</div>
            <div style='font-size:.88rem;color:#e2e8f0;font-weight:600;margin-bottom:.3rem'>Décomposition variable par variable : <span style='color:{pred_color}'>{info[0]}</span></div>
            <div style='font-size:.81rem;color:#64748b;line-height:1.65'>Chaque barre indique dans quelle mesure et dans quel sens une variable a orienté le diagnostic — rendant l'IA <strong style='color:#a78bfa'>totalement transparente</strong>.</div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("🔬 Calcul SHAP…"):
            psv=explainer.shap_values(Xns)
        sv_pred=get_shap_class(psv,pred)[0]
        pdata=Xn.values[0]

        sc1,sc2=st.columns([1.35,1],gap="large")
        with sc1:
            st.markdown(f"<div style='font-size:.69rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#8b5cf6;margin-bottom:.6rem'>📊 Explication individuelle — {info[0]}</div>",unsafe_allow_html=True)
            fig_wf=plot_waterfall(explainer,psv,pred,pdata,fc,info[0],pred_color)
            st.pyplot(fig_wf,use_container_width=True); plt.close(fig_wf)
        with sc2:
            st.markdown("<div style='font-size:.69rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#8b5cf6;margin-bottom:.6rem'>🏆 Importance Globale (SHAP)</div>",unsafe_allow_html=True)
            fig_imp=plot_global_imp(shap_values,fc,n_classes,pred_color)
            st.pyplot(fig_imp,use_container_width=True); plt.close(fig_imp)

        st.markdown("<div style='font-size:.69rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#8b5cf6;margin:.9rem 0 .6rem'>🧠 Insights Médicaux</div>",unsafe_allow_html=True)
        insights=generate_insights(sv_pred,fc,pred,info[0])
        ic=st.columns(len(insights),gap="medium")
        for col_i,(icon,title,text) in zip(ic,insights):
            with col_i:
                st.markdown(f"""<div class='shap-insight-card'>
                    <div class='shap-insight-icon'>{icon}</div>
                    <div><div class='shap-insight-title'>{title}</div><div class='shap-insight-text'>{text}</div></div>
                </div>""", unsafe_allow_html=True)

        # SHAP table
        shap_df=pd.DataFrame({
            "Variable":[FEATURE_LABELS.get(f,f) for f in fc],
            "Valeur Patient":[f"{v:.3f}" for v in pdata],
            "SHAP":sv_pred,"│SHAP│":np.abs(sv_pred),
            "Sens":["↑ Augmente" if v>0 else "↓ Diminue" for v in sv_pred],
        }).sort_values("│SHAP│",ascending=False).reset_index(drop=True)
        shap_df["SHAP"]=shap_df["SHAP"].round(4); shap_df["│SHAP│"]=shap_df["│SHAP│"].round(4)
        def cs(val):
            if "↑" in str(val): return "color:#ef4444;font-weight:700"
            if "↓" in str(val): return "color:#3b82f6;font-weight:700"
            return ""
        def cs2(val):
            try:
                v=float(val)
                if v>0.01: return "color:#fca5a5;font-weight:700"
                if v<-0.01: return "color:#93c5fd;font-weight:700"
            except: pass
            return "color:#64748b"
        styled=(shap_df.style.applymap(cs,subset=["Sens"]).applymap(cs2,subset=["SHAP"]).background_gradient(subset=["│SHAP│"],cmap="Purples"))
        st.dataframe(styled,use_container_width=True,height=400)

        st.markdown(f"""
        <div style='background:rgba(13,23,38,0.9);border:1px solid rgba(139,92,246,.14);
                    border-radius:12px;padding:1.2rem 1.6rem;margin-top:1rem;
                    display:flex;align-items:flex-start;gap:1rem;backdrop-filter:blur(8px)'>
            <div style='font-size:1.5rem;flex-shrink:0'>🏥</div>
            <div>
                <div style='font-size:.85rem;font-weight:700;color:#c4b5fd;margin-bottom:.3rem'>
                    Diagnostic sauvegardé dans l'historique
                </div>
                <div style='font-size:.78rem;color:#475569;line-height:1.6'>
                    ⚡ LightGBM — Accuracy <strong style='color:#00d4b4'>{acc*100:.1f}%</strong> · F1 {f1*100:.1f}%.
                    Facteurs décisifs : <strong style='color:#e2e8f0'>
                    {", ".join([FEATURE_LABELS.get(fc[i],fc[i]) for i in np.argsort(np.abs(sv_pred))[::-1][:3]])}</strong>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#  DOCTOR — Page 3 : Historique Patients
# ╚══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[1]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Historique des Diagnostics</div>
        <div class='banner-sub'>Registre complet de tous les diagnostics IA — persistants par patient</div>
        <span class='banner-tag'>historique</span><span class='banner-tag'>traçabilité-xai</span>
    </div>""", unsafe_allow_html=True)

    hist=st.session_state["patient_history"]

    if not hist:
        st.markdown("""<div style='background:rgba(10,22,40,0.85);border:1px solid rgba(255,255,255,.06);
            border-radius:16px;padding:3rem;text-align:center;backdrop-filter:blur(8px)'>
            <div style='font-size:2.5rem;margin-bottom:1rem'>📭</div>
            <div style='font-family:"DM Serif Display",serif;font-size:1.2rem;color:#334155;margin-bottom:.5rem'>Aucun diagnostic enregistré</div>
            <div style='font-size:.83rem;color:#1e3050'>Lancez un diagnostic depuis <strong style='color:#8b5cf6'>Diagnostic IA</strong> pour voir l'historique ici.</div>
        </div>""", unsafe_allow_html=True)
    else:
        n_total=len(hist)
        n_obese=sum(1 for h in hist if h["color"]=="red")
        n_normal=sum(1 for h in hist if h["color"]=="green")
        n_sur=sum(1 for h in hist if h["color"]=="amber")
        avg_imc=round(np.mean([h.get("imc",0) for h in hist]),1)

        k1,k2,k3,k4,k5=st.columns(5)
        with k1: st.markdown(f"<div class='kpi-card c-violet'><div class='kpi-num'>{n_total}</div><div class='kpi-lbl'>Total</div></div>",unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-card c-green'><div class='kpi-num'>{n_normal}</div><div class='kpi-lbl'>Normal/Insuffisant</div></div>",unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-card c-amber'><div class='kpi-num'>{n_sur}</div><div class='kpi-lbl'>Surpoids</div></div>",unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-card c-red'><div class='kpi-num'>{n_obese}</div><div class='kpi-lbl'>Obésité</div></div>",unsafe_allow_html=True)
        with k5: st.markdown(f"<div class='kpi-card'><div class='kpi-num'>{avg_imc}</div><div class='kpi-lbl'>IMC moyen</div></div>",unsafe_allow_html=True)

        st.markdown("<div class='sec-head'><div class='dot dot-violet'></div>Registre des patients</div>",unsafe_allow_html=True)

        st.markdown("""<div style='background:rgba(10,22,40,0.9);border:1px solid rgba(255,255,255,.06);
            border-radius:14px;overflow:hidden;backdrop-filter:blur(8px)'>
            <div style='display:grid;grid-template-columns:1.6fr 0.9fr 0.9fr 0.5fr 0.5fr 1.5fr 0.8fr;
                        gap:0;padding:.55rem 1.1rem;background:rgba(20,35,60,0.9);border-bottom:1px solid rgba(255,255,255,.05)'>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>Patient</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>Téléphone</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>CIN</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>Âge</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>IMC</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>Diagnostic</span>
                <span style='font-size:.61rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#334155'>Date</span>
            </div>""", unsafe_allow_html=True)
        DIAG_EMOJI={"green":"✅","amber":"⚠️","red":"🚨"}
        for idx,h in enumerate(reversed(hist)):
            dcol={"green":"#22c55e","amber":"#f59e0b","red":"#ef4444"}.get(h["color"],"#94a3b8")
            de=DIAG_EMOJI.get(h["color"],"•")
            row_bg="rgba(0,0,0,.0)" if idx%2==0 else "rgba(255,255,255,.012)"
            border="border-bottom:1px solid rgba(255,255,255,.03);" if idx<len(hist)-1 else ""
            st.markdown(
                f"<div style='background:{row_bg};{border}display:grid;"
                f"grid-template-columns:1.6fr 0.9fr 0.9fr 0.5fr 0.5fr 1.5fr 0.8fr;"
                f"align-items:center;gap:0;padding:.65rem 1.1rem'>"
                f"<span style='color:#e2e8f0;font-size:.82rem;font-weight:600'>{h.get('nom','—')}"
                f" <span style='color:#475569;font-size:.71rem'>({h.get('genre','—')})</span></span>"
                f"<span style='color:#64748b;font-family:\"JetBrains Mono\",monospace;font-size:.74rem'>{h.get('telephone','—')}</span>"
                f"<span style='color:#475569;font-family:\"JetBrains Mono\",monospace;font-size:.74rem'>{h.get('cin','—')}</span>"
                f"<span style='color:#94a3b8;font-size:.8rem'>{h.get('age','—')}</span>"
                f"<span style='color:#94a3b8;font-family:\"JetBrains Mono\",monospace;font-size:.8rem'>{h.get('imc','—')}</span>"
                f"<span style='color:{dcol};font-size:.79rem;font-weight:700'>{de} {h.get('diagnostic','—')}"
                f" <span style='color:#334155;font-weight:400'>({h.get('confiance','—')})</span></span>"
                f"<span style='color:#334155;font-size:.71rem;line-height:1.5'>{h.get('timestamp','—')}</span>"
                f"</div>", unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

        # Distribution chart
        if len(hist)>1:
            st.markdown("<div class='sec-head'><div class='dot dot-green'></div>Distribution des diagnostics</div>",unsafe_allow_html=True)
            diag_counts={}
            for h in hist:
                d=h["diagnostic"]; diag_counts[d]=diag_counts.get(d,0)+1
            fig_h,ax_h=dark_fig(10,4)
            lbs=list(diag_counts.keys()); vs=list(diag_counts.values())
            cols_h=[]
            for lbl in lbs:
                cl=next((v[1] for k,v in CLASS_INFO.items() if v[0]==lbl),"green")
                cols_h.append({"green":"#22c55e","amber":"#f59e0b","red":"#ef4444"}.get(cl,"#94a3b8"))
            bars_h=ax_h.barh(lbs,vs,color=cols_h,edgecolor="none",height=.55)
            for b,v in zip(bars_h,vs):
                ax_h.text(v+.04,b.get_y()+b.get_height()/2,str(v),va="center",fontsize=9,color="#e2e8f0",fontweight="700")
            ax_h.set_xlabel("Nombre de patients",fontsize=9,color="#64748b")
            ax_h.spines[["top","right","left"]].set_visible(False)
            ax_h.grid(axis="x",alpha=.15,linestyle="--")
            plt.tight_layout(); st.pyplot(fig_h,use_container_width=True)

        st.markdown("<div style='margin-top:1.2rem'></div>",unsafe_allow_html=True)
        if st.button("🗑️  Effacer l'historique",key="btn_clear_hist"):
            st.session_state["patient_history"]=[]
            st.rerun()


# ╔══════════════════════════════════════════════════════════╗
#  DOCTOR — Page 3 : Statistiques Cliniques
# ╚══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[2]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Statistiques Cliniques</div>
        <div class='banner-sub'>Synthèse agrégée des patients diagnostiqués · Session en cours</div>
        <span class='banner-tag'>session-stats</span><span class='banner-tag'>épidémiologie</span>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state["patient_history"]

    if not hist:
        st.markdown("""
        <div class='panel p-amber'>
            <div class='panel-title'>⚠️ Aucun diagnostic disponible</div>
            <div class='panel-body'>Effectuez des diagnostics depuis <strong>Diagnostic IA</strong> pour voir les statistiques de session.</div>
        </div>""", unsafe_allow_html=True)
    else:
        n = len(hist)
        imcs   = [h.get("imc", 0) for h in hist]
        ages   = [h.get("age", 0) for h in hist]
        genres = [h.get("genre","—") for h in hist]
        diags  = [h.get("diagnostic","—") for h in hist]
        colors = [h.get("color","green") for h in hist]

        n_h    = sum(1 for c in colors if c == "red")
        n_s    = sum(1 for c in colors if c == "amber")
        n_n    = sum(1 for c in colors if c == "green")
        n_m    = sum(1 for g in genres if g == "H")
        n_f    = sum(1 for g in genres if g == "F")
        imc_m  = round(np.mean(imcs), 1)
        imc_max= round(max(imcs), 1)
        age_m  = round(np.mean(ages), 1)
        risk_pct = round(n_h / n * 100, 1) if n else 0

        # KPI row
        st.markdown("<div class='sec-head'><div class='dot dot-teal'></div>Indicateurs de session</div>", unsafe_allow_html=True)
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        kpis = [
            (k1,"c-violet", str(n), "Patients diagnostiqués"),
            (k2,"c-red",    str(n_h), "Cas obésité"),
            (k3,"c-amber",  str(n_s), "Cas surpoids"),
            (k4,"c-green",  str(n_n), "Profil normal"),
            (k5,"c-blue",   str(imc_m), "IMC moyen"),
            (k6,"c-amber",  f"{risk_pct}%", "Taux obésité"),
        ]
        for col, cls, val, lbl in kpis:
            col.markdown(f"<div class='kpi-card {cls}'><div class='kpi-num'>{val}</div><div class='kpi-lbl'>{lbl}</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown("<div class='sec-head'><div class='dot dot-violet'></div>Répartition des diagnostics</div>", unsafe_allow_html=True)
            diag_counts = {}
            for d, col in zip(diags, colors):
                diag_counts[d] = diag_counts.get(d, (0, col))
                diag_counts[d] = (diag_counts[d][0] + 1, col)
            fig_d, ax_d = dark_fig(6, 4)
            labels_d = list(diag_counts.keys())
            vals_d   = [v[0] for v in diag_counts.values()]
            cols_d   = [{"green":"#22c55e","amber":"#f59e0b","red":"#ef4444"}.get(v[1],"#94a3b8") for v in diag_counts.values()]
            bars_d = ax_d.barh(labels_d, vals_d, color=cols_d, edgecolor="none", height=.6)
            for b, v in zip(bars_d, vals_d):
                ax_d.text(v + .03, b.get_y() + b.get_height()/2, str(v), va="center", fontsize=9, color="#e2e8f0", fontweight="700")
            ax_d.spines[["top","right","left"]].set_visible(False)
            ax_d.grid(axis="x", alpha=.15, linestyle="--")
            ax_d.set_xlabel("Nombre de patients", fontsize=9, color="#64748b")
            plt.tight_layout(); st.pyplot(fig_d, use_container_width=True)

        with c2:
            st.markdown("<div class='sec-head'><div class='dot dot-nurse'></div>Distribution des IMC</div>", unsafe_allow_html=True)
            fig_i, ax_i = dark_fig(6, 4)
            ax_i.hist(imcs, bins=min(15, n), color="#00d4b4", edgecolor="none", alpha=.85)
            ax_i.axvline(imc_m, color="#f59e0b", lw=2, linestyle="--", label=f"Moy {imc_m}")
            ax_i.axvline(25, color="#ef4444", lw=1.5, linestyle=":", alpha=.7, label="Seuil surpoids (25)")
            ax_i.axvline(30, color="#8b5cf6", lw=1.5, linestyle=":", alpha=.7, label="Seuil obésité (30)")
            ax_i.set_xlabel("IMC", fontsize=9, color="#64748b")
            ax_i.set_ylabel("Patients", fontsize=9, color="#64748b")
            ax_i.spines[["top","right"]].set_visible(False)
            ax_i.grid(alpha=.15, linestyle="--")
            ax_i.legend(fontsize=7.5)
            plt.tight_layout(); st.pyplot(fig_i, use_container_width=True)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown("<div class='sec-head'><div class='dot dot-green'></div>Genre & Tranche d'âge</div>", unsafe_allow_html=True)
            fig_g, axes_g = dark_fig(6, 4, ncols=2)
            # Gender
            axes_g[0].bar(["Hommes","Femmes"], [n_m, n_f], color=["#3b82f6","#ec4899"], edgecolor="none", width=.5)
            axes_g[0].set_title("Genre", fontsize=9, color="#94a3b8")
            axes_g[0].spines[["top","right"]].set_visible(False)
            # Age groups
            age_groups = {"< 25": 0, "25-40": 0, "40-55": 0, "55+": 0}
            for a in ages:
                if a < 25:    age_groups["< 25"] += 1
                elif a < 40:  age_groups["25-40"] += 1
                elif a < 55:  age_groups["40-55"] += 1
                else:         age_groups["55+"]   += 1
            axes_g[1].bar(age_groups.keys(), age_groups.values(), color=["#00d4b4","#3b82f6","#f59e0b","#ef4444"], edgecolor="none", width=.55)
            axes_g[1].set_title("Tranche d'âge", fontsize=9, color="#94a3b8")
            axes_g[1].spines[["top","right"]].set_visible(False)
            for ax in axes_g: ax.grid(axis="y", alpha=.15, linestyle="--")
            plt.tight_layout(); st.pyplot(fig_g, use_container_width=True)

        with c4:
            st.markdown("<div class='sec-head'><div class='dot dot-amber'></div>Résumé épidémiologique</div>", unsafe_allow_html=True)
            epi_rows = [
                ("Patients analysés", str(n)),
                ("IMC moyen", f"{imc_m}"),
                ("IMC maximum", f"{imc_max}"),
                ("Âge moyen", f"{age_m} ans"),
                ("Hommes / Femmes", f"{n_m} / {n_f}"),
                ("Taux obésité", f"{risk_pct}%"),
                ("Taux surpoids", f"{round(n_s/n*100,1)}%"),
                ("Profil sain", f"{round(n_n/n*100,1)}%"),
            ]
            html_rows = "".join([
                f"<div class='stat-row'><span class='sk'>{k}</span><span class='sv'>{v}</span></div>"
                for k, v in epi_rows
            ])
            st.markdown(f"<div class='form-section'>{html_rows}</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#  DOCTOR — Page 4 : Protocoles de Soins
# ╚══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[3]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Protocoles de Soins</div>
        <div class='banner-sub'>Recommandations cliniques officielles OMS par classe d'obésité · Référence médicale</div>
        <span class='banner-tag'>OMS-2024</span><span class='banner-tag'>guidelines</span>
    </div>""", unsafe_allow_html=True)

    PROTOCOLS = {
        0: {
            "label": "Poids Insuffisant", "imc": "IMC < 18.5", "color": "#3b82f6",
            "icon": "⚖️",
            "bilan": ["Bilan nutritionnel complet", "NFS + ferritine + albumine", "TSH — bilan thyroïdien", "ECG si bradycardie associée"],
            "alimentation": ["Augmenter les apports caloriques : +500 kcal/j", "3 repas + 2 collations riches en protéines", "Supplémentation : vitamines B12, D, Fer, Oméga-3", "Éviter le jeûne prolongé"],
            "activite": ["Musculation légère 2×/sem pour gain de masse", "Yoga ou étirements — éviter cardio intense", "Suivi régulier de la composition corporelle"],
            "suivi": "Mensuel — nutritionniste + médecin généraliste",
            "urgence": False,
        },
        1: {
            "label": "Poids Normal", "imc": "18.5 ≤ IMC < 25", "color": "#22c55e",
            "icon": "✅",
            "bilan": ["Bilan lipidique annuel", "Glycémie à jeun tous les 3 ans", "Tension artérielle à chaque consultation"],
            "alimentation": ["Maintenir alimentation équilibrée — régime méditerranéen", "5 fruits et légumes/jour (OMS)", "Limiter ultra-transformés et sucres ajoutés"],
            "activite": ["150 min/sem d'activité modérée (OMS)", "Renforcement musculaire 2×/sem", "Marche quotidienne recommandée"],
            "suivi": "Annuel — médecin généraliste",
            "urgence": False,
        },
        5: {
            "label": "Surpoids Niveau I", "imc": "25 ≤ IMC < 27.5", "color": "#fbbf24",
            "icon": "⚠️",
            "bilan": ["Bilan lipidique (CT, HDL, LDL, TG)", "Glycémie à jeun + HbA1c", "Tension artérielle + fréquence cardiaque", "Tour de taille"],
            "alimentation": ["Déficit calorique modéré : -300 à -500 kcal/j", "Réduire sucres raffinés et graisses saturées", "Augmenter fibres alimentaires (légumineuses, légumes)", "Journalisation alimentaire recommandée"],
            "activite": ["200 min/sem d'activité modérée", "Marche rapide 30 min/j minimum", "Limiter sédentarité — pause active toutes les heures"],
            "suivi": "Trimestriel — médecin généraliste + diététicien",
            "urgence": False,
        },
        6: {
            "label": "Surpoids Niveau II", "imc": "27.5 ≤ IMC < 30", "color": "#f97316",
            "icon": "⚠️",
            "bilan": ["Bilan lipidique complet + apolipoprotéines", "Test de tolérance au glucose (HGPO)", "Bilan hépatique (stéatose hépatique)", "Échographie abdominale"],
            "alimentation": ["Déficit calorique : -500 à -750 kcal/j", "Régime pauvre en graisses saturées et en sucres", "Consultation diététicien pour plan personnalisé", "Arrêt des boissons sucrées"],
            "activite": ["250 min/sem d'activité physique", "Natation, vélo, marche nordique recommandés", "Kinésithérapie si douleurs articulaires"],
            "suivi": "Bimestriel — diététicien + médecin + cardiologue si facteur de risque",
            "urgence": False,
        },
        2: {
            "label": "Obésité Type I", "imc": "30 ≤ IMC < 35", "color": "#ef4444",
            "icon": "🚨",
            "bilan": ["Bilan lipidique + cardiovasculaire complet", "HbA1c + HOMA-IR (résistance à l'insuline)", "Bilan hépatique + échographie", "Polysomnographie si suspicion SAOS", "ECG + échocardiographie"],
            "alimentation": ["Déficit calorique : -750 kcal/j sous supervision", "Régimes structurés : VLCD si IMC > 32 avec comorbidités", "Supplémentation protéique et micronutriments", "Suivi nutritionnel hebdomadaire"],
            "activite": ["300 min/sem minimum d'activité adaptée", "Programme supervisé par kinésithérapeute", "Éviter sports à fort impact (risque articulaire)"],
            "suivi": "Mensuel — équipe pluridisciplinaire (endocrinologue, diéticien, cardiologue)",
            "urgence": True,
        },
        3: {
            "label": "Obésité Type II", "imc": "35 ≤ IMC < 40", "color": "#dc2626",
            "icon": "🆘",
            "bilan": ["Bilan cardiovasculaire complet + coronarographie si indiqué", "Diabétologie : HbA1c, C-peptide, insulinémie", "Bilan respiratoire : spirométrie + gazométrie", "Bilan orthopédique", "Évaluation psychologique"],
            "alimentation": ["Régime très basse calorie (VLCD) : 800–1000 kcal/j sous surveillance", "Supplémentation complète obligatoire", "Évaluation chirurgie bariatrique (sleeve, bypass)"],
            "activite": ["Programme de réadaptation physique supervisé", "Hydrothérapie si mobilité réduite", "Objectif : -5 à -10% du poids corporel/6 mois"],
            "suivi": "Hebdomadaire — unité de prise en charge de l'obésité (CHU)",
            "urgence": True,
        },
        4: {
            "label": "Obésité Type III (morbide)", "imc": "IMC ≥ 40", "color": "#991b1b",
            "icon": "🏥",
            "bilan": ["Hospitalisation pour bilan complet multidisciplinaire", "Évaluation pré-chirurgicale bariatrique obligatoire", "Bilan psychiatrique + psychologique", "Bilan anesthésique pré-opératoire"],
            "alimentation": ["Préparation nutritionnelle pré-opératoire obligatoire", "VLCD ou nutrition entérale si nécessaire", "Suivi post-opératoire nutritionnel à vie"],
            "activite": ["Programme pré-chirurgical de perte de poids (10%)", "Rééducation physique post-opératoire", "Activité très progressive sous supervision médicale stricte"],
            "suivi": "Suivi à vie — centre spécialisé CHU + chirurgien bariatrique",
            "urgence": True,
        },
    }

    # Selector
    class_order = [1, 5, 6, 0, 2, 3, 4]
    sel_labels = [f"{PROTOCOLS[c]['icon']} {PROTOCOLS[c]['label']} ({PROTOCOLS[c]['imc']})" for c in class_order]
    sel = st.selectbox("Sélectionner une classe clinique", sel_labels, key="proto_sel")
    sel_idx = class_order[sel_labels.index(sel)]
    proto = PROTOCOLS[sel_idx]
    pcol = proto["color"]

    # Pre-build header HTML to avoid nested f-string rendering issues
    urgence_badge = (
        "<div style='margin-left:auto;background:rgba(239,68,68,0.12);"
        "border:1px solid rgba(239,68,68,0.3);border-radius:8px;"
        "padding:.3rem .8rem;font-size:.72rem;font-weight:700;color:#fca5a5'>"
        "&#9888;&#65039; PRISE EN CHARGE URGENTE</div>"
    ) if proto["urgence"] else ""

    proto_header_html = (
        f"<div style='background:linear-gradient(135deg,{pcol}18,rgba(5,18,32,0.92));"
        f"border:1px solid {pcol}55;border-radius:18px;"
        f"padding:1.6rem 2rem;margin:1rem 0;backdrop-filter:blur(12px)'>"
        f"<div style='display:flex;align-items:center;gap:1rem;margin-bottom:.5rem'>"
        f"<span style='font-size:2.5rem'>{proto['icon']}</span>"
        f"<div style='flex:1'>"
        f"<div style='font-size:1.45rem;font-weight:700;color:#f1f5f9;margin-bottom:.2rem'>{proto['label']}</div>"
        f"<div style='font-size:.78rem;color:{pcol};font-weight:700;font-family:monospace'>{proto['imc']}</div>"
        f"</div>"
        f"{urgence_badge}"
        f"</div>"
        f"</div>"
    )
    st.markdown(proto_header_html, unsafe_allow_html=True)

    pc1, pc2 = st.columns(2, gap="large")

    with pc1:
        # Bilan
        bilan_html = "".join([
            f"<div style='display:flex;gap:.6rem;align-items:flex-start;padding:.3rem 0;"
            f"border-bottom:1px solid rgba(255,255,255,.04)'>"
            f"<span style='color:#3b82f6;flex-shrink:0;font-weight:700'>›</span>"
            f"<span style='font-size:.84rem;color:#94a3b8'>{item}</span></div>"
            for item in proto["bilan"]
        ])
        st.markdown(f"""
        <div class='panel p-blue'>
            <div class='panel-title'>🔬 Bilan Biologique Recommandé</div>
            <div class='panel-body'>{bilan_html}</div>
        </div>""", unsafe_allow_html=True)

        # Suivi
        st.markdown(f"""
        <div class='panel p-violet'>
            <div class='panel-title'>📅 Fréquence de Suivi</div>
            <div class='panel-body'>{proto["suivi"]}</div>
        </div>""", unsafe_allow_html=True)

    with pc2:
        # Alimentation
        alim_html = "".join([
            f"<div style='display:flex;gap:.6rem;align-items:flex-start;padding:.3rem 0;"
            f"border-bottom:1px solid rgba(255,255,255,.04)'>"
            f"<span style='color:#22c55e;flex-shrink:0;font-weight:700'>›</span>"
            f"<span style='font-size:.84rem;color:#94a3b8'>{item}</span></div>"
            for item in proto["alimentation"]
        ])
        st.markdown(f"""
        <div class='panel p-green'>
            <div class='panel-title'>🥗 Plan Alimentaire</div>
            <div class='panel-body'>{alim_html}</div>
        </div>""", unsafe_allow_html=True)

        # Activité
        activ_html = "".join([
            f"<div style='display:flex;gap:.6rem;align-items:flex-start;padding:.3rem 0;"
            f"border-bottom:1px solid rgba(255,255,255,.04)'>"
            f"<span style='color:#f59e0b;flex-shrink:0;font-weight:700'>›</span>"
            f"<span style='font-size:.84rem;color:#94a3b8'>{item}</span></div>"
            for item in proto["activite"]
        ])
        st.markdown(f"""
        <div class='panel p-amber'>
            <div class='panel-title'>🏃 Programme d'Activité Physique</div>
            <div class='panel-body'>{activ_html}</div>
        </div>""", unsafe_allow_html=True)

    # IMC reference chart
    st.markdown("<div class='sec-head'><div class='dot dot-teal'></div>Référence IMC — Classification OMS</div>", unsafe_allow_html=True)
    ref_data = [
        ("< 18.5",  "Poids Insuffisant",  "#3b82f6", 18.5),
        ("18.5–25", "Poids Normal",        "#22c55e", 6.5),
        ("25–27.5", "Surpoids Niveau I",   "#fbbf24", 2.5),
        ("27.5–30", "Surpoids Niveau II",  "#f97316", 2.5),
        ("30–35",   "Obésité Type I",      "#ef4444", 5.0),
        ("35–40",   "Obésité Type II",     "#dc2626", 5.0),
        ("≥ 40",    "Obésité Type III",    "#991b1b", 5.0),
    ]
    fig_ref, ax_ref = dark_fig(12, 3.5)
    left = 10.0
    for imc_range, label, col, width in ref_data:
        is_sel = label == proto["label"]
        alpha = 0.95 if is_sel else 0.5
        ax_ref.barh([0], [width], left=left, color=col, alpha=alpha, edgecolor="#060d1b", height=.65)
        ax_ref.text(left + width/2, 0, f"{imc_range}\n{label[:12]}", ha="center", va="center",
                    fontsize=7.2, color="white" if is_sel else "#94a3b8",
                    fontweight="bold" if is_sel else "normal")
        if is_sel:
            ax_ref.text(left + width/2, .45, "▲", ha="center", fontsize=10, color=col)
        left += width
    ax_ref.set_xlim(9, 50); ax_ref.set_ylim(-.5, .8)
    ax_ref.set_yticks([]); ax_ref.set_xticks([18.5, 25, 27.5, 30, 35, 40])
    ax_ref.tick_params(labelsize=8, colors="#64748b")
    ax_ref.spines[["top","right","left","bottom"]].set_visible(False)
    ax_ref.grid(axis="x", alpha=.2, linestyle="--")
    plt.tight_layout(); st.pyplot(fig_ref, use_container_width=True)


# ╔══════════════════════════════════════════════════════════╗
#  DOCTOR — Page 5 : Analyse IA Globale
# ╚══════════════════════════════════════════════════════════╝
elif is_doctor and page == DOC_PAGES[4]:
    st.markdown("""
    <div class='page-banner banner-doctor'>
        <div class='banner-eyebrow ey-doctor'>👨‍⚕️ Interface Médecin</div>
        <div class='banner-h1'>Analyse IA Globale</div>
        <div class='banner-sub'>Facteurs décisifs du modèle LightGBM · Explicabilité SHAP sur population entière</div>
        <span class='banner-tag'>SHAP-global</span><span class='banner-tag'>XAI</span>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Chargement de l'analyse IA…"):
        (clf,sc_m,fc,acc,f1,prec,rec,cm,cr,Xtes,yte,yp,explainer,shap_values,Xtes_sample) = train_model(BEST_ALGO)
    n_classes = len(CLASS_NAMES)

    st.markdown(
        f"<span class='chip chip-teal'>⚡ LightGBM</span>"
        f"<span class='chip'>Acc {acc*100:.1f}%</span>"
        f"<span class='chip'>F1 {f1*100:.1f}%</span>"
        f"<span class='chip chip-violet'>🔍 SHAP global</span>",
        unsafe_allow_html=True)

    # Global SHAP importance
    st.markdown("<div class='sec-head'><div class='dot dot-violet'></div>Importance Globale des Variables (SHAP)</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='panel p-violet'>
        <div class='panel-title'>📖 Lecture du graphique</div>
        <div class='panel-body'>
            L'importance SHAP mesure la <strong>contribution moyenne de chaque variable</strong> sur l'ensemble
            des patients du jeu de test. Plus la barre est longue, plus la variable influence fortement la
            prédiction d'obésité, toutes classes confondues.
        </div>
    </div>""", unsafe_allow_html=True)

    fig_imp = plot_global_imp(shap_values, fc, n_classes, "#00d4b4")
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

    # SHAP Beeswarm population
    st.markdown("<div class='sec-head'><div class='dot dot-violet'></div>Distribution SHAP — Vue Population</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='panel p-violet'>
        <div class='panel-title'>📖 Lecture du beeswarm</div>
        <div class='panel-body'>
            Chaque point = un patient du jeu de test.
            <span style='color:#ef4444;font-weight:700'>Rouge = valeur haute</span> de la variable,
            <span style='color:#3b82f6;font-weight:700'>Bleu = valeur basse</span>.
            Position à droite → augmente le risque · Position à gauche → le réduit.
        </div>
    </div>""", unsafe_allow_html=True)

    # Build beeswarm
    set_dark_mpl()
    imp_g = global_shap_imp(shap_values, n_classes)
    order_g = np.argsort(imp_g)[::-1][:12]
    sv_mean = np.mean([get_shap_class(shap_values, c) for c in range(n_classes)], axis=0)
    fig_b, ax_b = plt.subplots(figsize=(11, 7))
    fig_b.patch.set_facecolor("#0a1628"); ax_b.set_facecolor("#101f38")
    for rank, fi in enumerate(order_g[::-1]):
        sv_f  = sv_mean[:, fi]
        raw_f = Xtes_sample[:, fi]
        vmin, vmax = raw_f.min(), raw_f.max()
        norm = (raw_f - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(raw_f)
        colors_b = plt.cm.RdBu_r(norm)
        jitter = np.random.normal(0, 0.09, size=len(sv_f))
        ax_b.scatter(sv_f, rank + jitter, c=colors_b, s=16, alpha=.65, linewidths=0)
    feat_labels_b = [FEATURE_LABELS.get(fc[i], fc[i]) for i in order_g[::-1]]
    ax_b.set_yticks(range(12)); ax_b.set_yticklabels(feat_labels_b, fontsize=9.5, color="#e2e8f0")
    ax_b.axvline(0, color="#475569", lw=1.3, linestyle="--")
    ax_b.set_xlabel("Valeur SHAP (impact sur la prédiction)", fontsize=9, color="#94a3b8")
    ax_b.set_title("Distribution SHAP — Top 12 variables · Population complète",
                   fontsize=11, color="#e2e8f0", pad=12, fontweight="600")
    ax_b.spines[["top","right","left"]].set_visible(False)
    ax_b.grid(axis="x", alpha=.12, linestyle="--")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(0,1))
    sm.set_array([])
    cbar = fig_b.colorbar(sm, ax=ax_b, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_label("Valeur\n(bleu=bas · rouge=haut)", fontsize=7.5, color="#64748b")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#64748b")
    cbar.outline.set_edgecolor("#2d3a52")
    plt.tight_layout(); st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)

    # Top insights
    st.markdown("<div class='sec-head'><div class='dot dot-teal'></div>Insights Clés du Modèle</div>", unsafe_allow_html=True)
    imp_sorted = np.argsort(imp_g)[::-1]
    top5 = imp_sorted[:5]
    INSIGHTS_MAP = {
        "Weight":  ("⚖️","#ef4444","Le poids est le facteur le plus déterminant. Un IMC élevé est fortement corrélé aux classes obésité III et II."),
        "Height":  ("📏","#3b82f6","La taille intervient via le calcul de l'IMC — elle module directement la classification diagnostique."),
        "Age":     ("🎂","#f59e0b","Le vieillissement favorise la prise de poids métabolique et réduit le métabolisme basal."),
        "FAF":     ("🏃","#22c55e","L'activité physique est un facteur protecteur majeur. 3+ j/sem réduit significativement le risque d'obésité."),
        "FCVC":    ("🥦","#22c55e","La fréquence de consommation de légumes est inversement corrélée au risque d'obésité."),
        "CH2O":    ("💧","#06b6d4","Une hydratation insuffisante est associée à un métabolisme ralenti et à une prise de poids."),
        "CAEC":    ("🍪","#f97316","Le grignotage entre les repas augmente significativement l'apport calorique total quotidien."),
        "FAVC":    ("🍔","#ef4444","La consommation régulière d'aliments caloriques est un prédicteur fort d'obésité."),
        "family_history_with_overweight":("🧬","#8b5cf6","La prédisposition génétique multiplie par 2-3 le risque d'obésité — facteur non modifiable."),
        "NCP":     ("🍽️","#fbbf24","Le nombre de repas par jour influence le cycle métabolique et la régulation de la faim."),
        "MTRANS":  ("🚗","#64748b","Le mode de transport reflète le niveau d'activité physique quotidienne intégrée."),
        "SMOKE":   ("🚬","#94a3b8","Le tabagisme perturbe le métabolisme lipidique et peut masquer des problèmes de poids."),
    }
    ins_cols = st.columns(min(3, len(top5)))
    for col_i, fi in zip(ins_cols, top5[:3]):
        fname = fc[fi]
        icon, col_c, text = INSIGHTS_MAP.get(fname, ("📌","#94a3b8", f"{FEATURE_LABELS.get(fname,fname)} : variable influente."))
        with col_i:
            st.markdown(f"""
            <div class='shap-insight-card' style='border-color:{col_c}30'>
                <div class='shap-insight-icon'>{icon}</div>
                <div>
                    <div class='shap-insight-title' style='color:{col_c}'>{FEATURE_LABELS.get(fname,fname)}</div>
                    <div class='shap-insight-text'>{text}</div>
                    <div style='margin-top:.5rem;font-size:.7rem;font-family:"JetBrains Mono",monospace;color:#334155'>
                        SHAP moyen : {imp_g[fi]:.4f}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    ins_cols2 = st.columns(2)
    for col_i, fi in zip(ins_cols2, top5[3:5]):
        fname = fc[fi]
        icon, col_c, text = INSIGHTS_MAP.get(fname, ("📌","#94a3b8", f"{FEATURE_LABELS.get(fname,fname)} : variable influente."))
        with col_i:
            st.markdown(f"""
            <div class='shap-insight-card' style='border-color:{col_c}30'>
                <div class='shap-insight-icon'>{icon}</div>
                <div>
                    <div class='shap-insight-title' style='color:{col_c}'>{FEATURE_LABELS.get(fname,fname)}</div>
                    <div class='shap-insight-text'>{text}</div>
                    <div style='margin-top:.5rem;font-size:.7rem;font-family:"JetBrains Mono",monospace;color:#334155'>
                        SHAP moyen : {imp_g[fi]:.4f}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
