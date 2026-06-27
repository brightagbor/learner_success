"""
Learner Success Prediction Dashboard
Federal Polytechnic Nekede, Owerri — Department of Computer Engineering
Production-ready with theme switch, CSV/PDF export, robust session state.
"""

import io
import csv
import datetime
import warnings
import pickle
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Learner Success Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# THEME TOGGLE
# ──────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True   # default: dark (matches original)

with st.sidebar:
    st.markdown("### ⚙️ Display")
    _lbl = "🌞 Switch to Light" if st.session_state["dark_mode"] else "🌙 Switch to Dark"
    if st.button(_lbl, use_container_width=True):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]
        st.rerun()
    dark = st.session_state["dark_mode"]

# ──────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ──────────────────────────────────────────────────────────────────────────────
if dark:
    T = dict(
        bg        = "#07090f",
        surface   = "#0e1118",
        surface2  = "#141824",
        border    = "#1e2535",
        accent    = "#6c8eff",
        accent2   = "#00d4aa",
        danger    = "#ff5f6d",
        warning   = "#ffb347",
        text      = "#dde4f0",
        muted     = "#6b7794",
        sidebar   = "#0e1118",
        card_hi   = "#0e1530",
        badge_teal= "rgba(0,212,170,0.10)",
        badge_blue= "rgba(108,142,255,0.12)",
        badge_amb = "rgba(255,179,71,0.10)",
    )
else:
    T = dict(
        bg        = "#f4f6fb",
        surface   = "#ffffff",
        surface2  = "#eef1f8",
        border    = "#dde3ef",
        accent    = "#3b5bdb",
        accent2   = "#0ca678",
        danger    = "#e03131",
        warning   = "#e67700",
        text      = "#0f172a",
        muted     = "#64748b",
        sidebar   = "#f8faff",
        card_hi   = "#e8f4ff",
        badge_teal= "rgba(12,166,120,0.10)",
        badge_blue= "rgba(59,91,219,0.10)",
        badge_amb = "rgba(230,119,0,0.10)",
    )

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {{
    background: {T['bg']} !important;
    color: {T['text']} !important;
    font-family: 'Inter', sans-serif !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

[data-testid="block-container"] {{
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1280px !important;
    margin: auto !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {T['sidebar']} !important;
    border-right: 1px solid {T['border']} !important;
}}
[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: {T['muted']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}
.sidebar-section {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: {T['accent']};
    border-top: 1px solid {T['border']};
    padding-top: 0.9rem;
    margin: 1.2rem 0 0.6rem;
}}

/* ── Masthead ── */
.masthead {{
    background: {T['surface']};
    border: 1px solid {T['border']};
    border-left: 5px solid {T['accent']};
    border-radius: 12px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
}}
.masthead::after {{
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, {T['accent']}18 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}}
.masthead-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {T['accent']};
    margin-bottom: 0.5rem;
}}
.masthead-title {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.5rem);
    font-weight: 500;
    color: {T['text']};
    line-height: 1.15;
    margin-bottom: 0.4rem;
}}
.masthead-title em {{ color: {T['accent2']}; font-style: normal; }}
.masthead-sub {{
    font-size: 0.88rem;
    color: {T['muted']};
    line-height: 1.6;
    max-width: 640px;
}}
.badge-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1.2rem; }}
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
}}
.badge.blue  {{ background: {T['badge_blue']}; border: 1px solid {T['accent']}44; color: {T['accent']}; }}
.badge.teal  {{ background: {T['badge_teal']}; border: 1px solid {T['accent2']}44; color: {T['accent2']}; }}
.badge.amber {{ background: {T['badge_amb']};  border: 1px solid {T['warning']}44; color: {T['warning']}; }}

/* ── Metric cards ── */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}}
@media (max-width: 900px) {{
    .metric-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}
.mcard {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 10px;
    padding: 0.9rem 1rem;
}}
.mcard .mlabel {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {T['muted']};
    margin-bottom: 0.3rem;
}}
.mcard .mval {{
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
}}

/* ── Gauge ── */
.gauge-wrap {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 1.75rem;
    text-align: center;
}}
.gauge-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {T['muted']};
    margin-bottom: 0.9rem;
}}
.gauge-score {{
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.35rem;
}}
.gauge-track {{
    height: 12px;
    background: {T['border']};
    border-radius: 99px;
    overflow: hidden;
    margin: 0.6rem 0 0.3rem;
}}
.gauge-ticks {{
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: {T['muted']};
    margin-top: 4px;
}}
.band-row {{ display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.9rem; }}
.band-pill {{
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    border: 1px solid;
    opacity: 0.28;
    transition: opacity 0.2s;
}}
.band-pill.active {{ opacity: 1; }}

/* ── Panel ── */
.panel {{
    background: {T['surface2']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}}
.panel-title {{
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: {T['muted']};
    border-bottom: 1px solid {T['border']};
    padding-bottom: 0.55rem;
    margin-bottom: 0.9rem;
}}

/* ── Chips ── */
.chips {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }}
.chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    padding: 0.3rem 0.7rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 500;
    border: 1px solid;
}}
.chip.risk  {{ background: {T['danger']}18;  border-color: {T['danger']}44;  color: {T['danger']}; }}
.chip.warn  {{ background: {T['warning']}18; border-color: {T['warning']}44; color: {T['warning']}; }}
.chip.rec   {{ background: {T['accent']}18;  border-color: {T['accent']}44;  color: {T['accent']}; }}
.chip.ok    {{ background: {T['accent2']}18; border-color: {T['accent2']}44; color: {T['accent2']}; }}

/* ── Feature bars ── */
.fbar-row {{ margin: 0.45rem 0; }}
.fbar-label {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    margin-bottom: 3px;
    color: {T['text']};
}}
.fbar-pct {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: {T['muted']};
}}
.fbar-track {{
    height: 6px;
    background: {T['border']};
    border-radius: 99px;
    overflow: hidden;
}}
.fbar-fill {{
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, {T['accent']}, {T['accent2']});
}}

/* ── Breakdown rows ── */
.brow {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid {T['border']};
}}
.brow-lbl {{ color: {T['muted']}; }}
.brow-val {{ font-weight: 500; }}

/* ── Section heading ── */
.sec-head {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {T['border']};
}}
.sec-icon {{
    width: 26px; height: 26px;
    background: {T['accent']};
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem;
    flex-shrink: 0;
}}
.sec-text {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {T['muted']};
}}

/* ── Buttons ── */
.stButton > button {{
    width: 100% !important;
    background: linear-gradient(135deg, {T['accent']}, #4a6ef5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

[data-testid="stDownloadButton"] > button {{
    background: {T['surface2']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 8px !important;
    color: {T['text']} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.25rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    border-color: {T['accent']} !important;
    color: {T['accent']} !important;
}}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {{
    background: {T['accent']} !important;
    border: 2px solid {T['surface']} !important;
}}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {{
    background: {T['surface']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 6px !important;
    color: {T['text']} !important;
}}

/* DataFrame */
div[data-testid="stDataFrame"] {{
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
}}

/* Expander */
.stExpander {{
    border: 1px solid {T['border']} !important;
    border-radius: 10px !important;
    background: {T['surface2']} !important;
}}

/* Results animation */
.results-wrap {{ animation: fadeUp 0.35s ease both; }}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* Footer */
.footer {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    color: {T['muted']};
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid {T['border']};
}}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    m = joblib.load("results/XGBoost_model.pkl")
    with open("results/scaler.pkl", "rb") as f:
        s = pickle.load(f)
    return m, s

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)
    model, scaler = None, None

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
NICE = {
    "G1":"1st Period Grade","G2":"2nd Period Grade",
    "absences":"Absences","failures":"Past Failures",
    "studytime":"Study Time","higher":"Higher Ed Goal",
    "Medu":"Mother Education","Fedu":"Father Education",
    "goout":"Social Outings","famsup":"Family Support",
    "famrel":"Family Relations","health":"Health Status",
    "schoolsup":"School Support","internet":"Internet Access",
    "activities":"Activities","traveltime":"Travel Time",
}

def grade_band(s):
    if s >= 90: return ("A+", T['accent2'], T['badge_teal'])
    if s >= 80: return ("A",  T['accent'],  T['badge_blue'])
    if s >= 70: return ("B",  "#a78bfa",    "rgba(167,139,250,0.12)")
    if s >= 60: return ("C",  T['warning'],  T['badge_amb'])
    if s >= 50: return ("D",  "#ff9a5c",    "rgba(255,154,92,0.12)")
    return              ("F",  T['danger'],  "rgba(255,95,109,0.12)")

def score_color(s):
    if s >= 70: return T['accent2']
    if s >= 50: return T['warning']
    return T['danger']

def trend_arrow(g1, g2):
    d = g2 - g1
    if d > 5:  return "↑", T['accent2'], f"+{d:.0f}"
    if d < -5: return "↓", T['danger'],  f"{d:.0f}"
    return "→", T['muted'], f"{d:+.0f}"

bmap = {"yes": 1, "no": 0}

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUTS  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(
        f'<p style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;'
        f'margin-bottom:0.15rem;color:{T["text"]};">Student Profile</p>'
        f'<p style="font-size:0.75rem;color:{T["muted"]};margin-bottom:1rem;">'
        f'Fill in the student\'s details</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">📊 Prior Grades</div>', unsafe_allow_html=True)
    G1 = st.select_slider("G1 — Test",
        options=[15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95], value=60)
    G2 = st.select_slider("G2 — CBT Score",
        options=[0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95], value=60)
    st.markdown('<div class="sidebar-section">📚 Academic History</div>', unsafe_allow_html=True)
    failures = st.selectbox("Previous Failures", [0,1,2,3], index=0)
    absences = st.slider("Total Absences", 0, 75, 4)
    st.markdown('<div class="sidebar-section">⏱ Study Behaviour</div>', unsafe_allow_html=True)
    studytime  = st.selectbox("Weekly Study Time", [1,2,3,4],
                               format_func=lambda x:{1:"< 2 hrs",2:"2–5 hrs",3:"5–10 hrs",4:"> 10 hrs"}[x], index=1)
    traveltime = st.selectbox("Travel Time to School", [1,2,3,4],
                               format_func=lambda x:{1:"< 15 min",2:"15–30 min",3:"30–60 min",4:"> 60 min"}[x])
    st.markdown('<div class="sidebar-section">👨‍👩‍👧 Family & Home</div>', unsafe_allow_html=True)
    Medu   = st.selectbox("Mother's Education",[0,1,2,3,4],
                           format_func=lambda x:{0:"None",1:"Primary",2:"Middle",3:"Secondary",4:"Higher"}[x], index=3)
    Fedu   = st.selectbox("Father's Education",[0,1,2,3,4],
                           format_func=lambda x:{0:"None",1:"Primary",2:"Middle",3:"Secondary",4:"Higher"}[x], index=2)
    famsup = st.selectbox("Family Academic Support", ["yes","no"])
    famrel = st.slider("Family Relationship Quality", 1, 5, 4)
    st.markdown('<div class="sidebar-section">🌱 Goals & Lifestyle</div>', unsafe_allow_html=True)
    higher     = st.selectbox("Aims for Higher Education", ["yes","no"])
    internet   = st.selectbox("Has Internet at Home", ["yes","no"])
    schoolsup  = st.selectbox("Extra School Support", ["yes","no"])
    activities = st.selectbox("Extra-Curricular Activities", ["yes","no"])
    goout  = st.slider("Social Outings Frequency", 1, 5, 2, help="1=rarely  5=very often")
    health = st.slider("Health Status", 1, 5, 3, help="1=very poor  5=excellent")

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE DATAFRAME  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
features = pd.DataFrame({
    "studytime":[studytime],"failures":[failures],"absences":[absences],
    "Medu":[Medu],"Fedu":[Fedu],
    "schoolsup":[bmap[schoolsup]],"famsup":[bmap[famsup]],
    "higher":[bmap[higher]],"internet":[bmap[internet]],
    "activities":[bmap[activities]],"traveltime":[traveltime],
    "famrel":[famrel],"goout":[goout],"health":[health],
    "G1":[G1],"G2":[G2],
})

# ──────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="masthead">
  <div class="masthead-eyebrow">🎓 AI-Powered · For Teachers, Counsellors &amp; Parents</div>
  <div class="masthead-title">
    ML-Driven Learner Success Prediction<br>
    <em>Dept. of Computer Engineering, Federal Polytechnic Nekede, Owerri</em>
  </div>
  <div class="masthead-sub">
    Predicts each student's expected final score (0–100) from their academic profile
    and surfaces the specific factors — and interventions — that matter most.
  </div>
  <div class="badge-row">
    <span class="badge blue">XGBoost Regression</span>
    <span class="badge teal">Continuous Score 0–100</span>
    <span class="badge amber">Early Intervention Tool</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"**Could not load model files from `results/`.** `{load_error}`")
    st.info("Ensure `results/XGBoost_model.pkl` and `results/scaler.pkl` exist alongside this app.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PREDICT  (live — no button, 100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
scaled     = scaler.transform(features)
pred_score = float(np.clip(model.predict(scaled)[0], 0, 100))
grade, gc, gbg = grade_band(pred_score)
sc_col = score_color(pred_score)
arrow, acol, adiff = trend_arrow(G1, G2)
avg_prior = (G1 + G2) / 2

# Persist latest prediction
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.session_state["pred_score"]  = pred_score
st.session_state["grade"]       = grade
st.session_state["features"]    = features.copy()
st.session_state["scaled"]      = scaled.copy()
st.session_state["timestamp"]   = ts

# ──────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ──────────────────────────────────────────────────────────────────────────────
g1c  = T['accent2'] if G1>=60 else T['warning'] if G1>=40 else T['danger']
g2c  = T['accent2'] if G2>=60 else T['warning'] if G2>=40 else T['danger']
abc  = T['danger']  if absences>20 else T['warning'] if absences>10 else T['accent2']
flc  = T['danger']  if failures>1  else T['warning'] if failures==1  else T['accent2']

st.markdown(f"""
<div class="metric-grid">
  <div class="mcard"><div class="mlabel">G1 Score</div>
    <div class="mval" style="color:{g1c};">{G1}</div></div>
  <div class="mcard"><div class="mlabel">G2 Score</div>
    <div class="mval" style="color:{g2c};">{G2}</div></div>
  <div class="mcard"><div class="mlabel">G1→G2 Trend</div>
    <div class="mval" style="color:{acol};">{arrow} {adiff}</div></div>
  <div class="mcard"><div class="mlabel">Avg Prior Grade</div>
    <div class="mval" style="color:{T['accent']};">{avg_prior:.0f}</div></div>
  <div class="mcard"><div class="mlabel">Absences</div>
    <div class="mval" style="color:{abc};">{absences}</div></div>
  <div class="mcard"><div class="mlabel">Past Failures</div>
    <div class="mval" style="color:{flc};">{failures}</div></div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# THREE-COLUMN RESULTS  (100% logic preserved)
# ──────────────────────────────────────────────────────────────────────────────
col_gauge, col_analysis, col_factors = st.columns([1, 1.1, 1.2])

with col_gauge:
    bands = [
        ("F",  T['danger'],  "rgba(255,95,109,0.12)",  0,  49),
        ("D",  "#ff9a5c",   "rgba(255,154,92,0.12)",  50,  59),
        ("C",  T['warning'], T['badge_amb'],            60,  69),
        ("B",  "#a78bfa",   "rgba(167,139,250,0.12)", 70,  79),
        ("A",  T['accent'],  T['badge_blue'],           80,  89),
        ("A+", T['accent2'], T['badge_teal'],           90, 100),
    ]
    pills = "".join(
        f'<span class="band-pill {"active" if lo<=pred_score<=hi else ""}" '
        f'style="color:{bc};background:{bbg};border-color:{bc};">'
        f'{bl} ({lo}–{hi})</span>'
        for bl, bc, bbg, lo, hi in bands
    )
    outcome       = "🎉 Likely to Succeed" if pred_score >= 50 else "⚠️ At Risk of Failing"
    outcome_color = T['accent2'] if pred_score >= 50 else T['danger']

    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-label">Predicted Final Score</div>
      <div class="gauge-score" style="color:{sc_col};">{pred_score:.1f}</div>
      <div style="font-size:1rem;font-weight:700;color:{gc};background:{gbg};
           display:inline-block;padding:0.2rem 1rem;border-radius:6px;
           border:1px solid {gc}55;margin-bottom:0.7rem;">Grade {grade}</div><br>
      <div style="font-size:0.88rem;font-weight:600;color:{outcome_color};
           margin-bottom:0.8rem;">{outcome}</div>
      <div class="gauge-track">
        <div style="height:100%;width:{int(pred_score)}%;border-radius:99px;
          background:linear-gradient(90deg,{gc}99,{gc});"></div>
      </div>
      <div class="gauge-ticks">
        <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
      </div>
      <div class="band-row">{pills}</div>
      <div style="margin-top:1.1rem;font-size:0.75rem;color:{T['muted']};
           background:{T['surface']};border:1px solid {T['border']};
           border-radius:8px;padding:0.6rem 0.9rem;text-align:left;">
        ℹ️ Pass threshold is <strong style="color:{T['text']};">50 / 100</strong>.
        Use alongside teacher judgment.
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_analysis:
    # ── Risk flags ──
    risks = []
    if failures > 1:          risks.append(("risk","🔴","Multiple past failures"))
    elif failures == 1:       risks.append(("warn","🟡","1 previous failure"))
    if absences > 20:         risks.append(("risk","🔴","Very high absenteeism"))
    elif absences > 10:       risks.append(("warn","🟡","Moderate absenteeism"))
    if studytime < 2:         risks.append(("risk","🔴","Very low study time"))
    if goout > 4:             risks.append(("warn","🟡","High social frequency"))
    if health < 2:            risks.append(("warn","🟡","Poor health status"))
    if G1 < 40 or G2 < 40:   risks.append(("risk","🔴","Very low prior grade(s)"))
    elif G1 < 50 or G2 < 50: risks.append(("warn","🟡","Below-average prior grades"))
    if G2 < G1 - 10:         risks.append(("risk","🔴","Sharp grade decline G1→G2"))
    if higher == "no":        risks.append(("warn","🟡","No higher-ed ambition"))
    if famsup == "no":        risks.append(("warn","🟡","No family support"))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚠️ Risk Flags</div>', unsafe_allow_html=True)
    if not risks:
        st.markdown('<div class="chips"><span class="chip ok">✓ No major risks detected</span></div>', unsafe_allow_html=True)
    else:
        chips = "".join(f'<span class="chip {c}">{i} {t}</span>' for c,i,t in risks)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    # ── Score breakdown ──
    dist_pass = max(0.0, 50.0 - pred_score)
    st.markdown(f'<div class="panel-title" style="margin-top:1.3rem;">📐 Score Breakdown</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("Prior grade average",    f"{avg_prior:.0f} / 100"),
        ("Predicted final score",  f"{pred_score:.1f} / 100"),
        ("Grade band",             f"Grade {grade}"),
        ("Pass threshold",         "50 / 100"),
        ("To reach pass",
         f"{dist_pass:.1f} pts still needed" if dist_pass > 0 else "✓ Above pass threshold"),
    ]:
        vc = T['accent2'] if "✓" in val else T['text']
        st.markdown(
            f'<div class="brow"><span class="brow-lbl">{lbl}</span>'
            f'<span class="brow-val" style="color:{vc};">{val}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ──
    recs = []
    if studytime < 2:                          recs.append("Increase weekly study hours")
    if absences > 15:                          recs.append("Improve class attendance")
    if bmap[famsup] == 0:                      recs.append("Engage family in academics")
    if goout > 4:                              recs.append("Balance social & study time")
    if health < 2:                             recs.append("Address health & wellbeing")
    if bmap[schoolsup] == 0 and pred_score<55: recs.append("Enrol in school support program")
    if G2 < G1 - 5:                            recs.append("Investigate cause of grade drop")
    if failures > 0:                           recs.append("Reinforce weak subjects")
    if pred_score >= 70 and studytime >= 3:    recs.append("Encourage advanced coursework")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💡 Recommendations for Teachers &amp; Parents</div>', unsafe_allow_html=True)
    if not recs:
        st.markdown('<div class="chips"><span class="chip ok">✓ Student on a healthy trajectory</span></div>', unsafe_allow_html=True)
    else:
        chips = "".join(f'<span class="chip rec">→ {r}</span>' for r in recs)
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_factors:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📈 What Drives the Prediction</div>', unsafe_allow_html=True)

    imp_dict = None
    try:
        imp_dict = dict(zip(features.columns, model.feature_importances_))
    except AttributeError:
        try:
            imps = [e.feature_importances_ for _, e in model.estimators_
                    if hasattr(e, "feature_importances_")]
            if imps:
                imp_dict = dict(zip(features.columns, np.mean(imps, axis=0)))
        except Exception:
            pass

    if imp_dict:
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        max_imp = sorted_imp[0][1]
        bars = "".join(
            f'<div class="fbar-row"><div class="fbar-label">'
            f'<span>{NICE.get(f, f)}</span>'
            f'<span class="fbar-pct">{v*100:.1f}%</span>'
            f'</div><div class="fbar-track">'
            f'<div class="fbar-fill" style="width:{int(v/max_imp*100)}%;"></div>'
            f'</div></div>'
            for f, v in sorted_imp
        )
        st.markdown(bars, unsafe_allow_html=True)
    else:
        st.info("Feature importance not available for this model type.")

    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# EXPORT SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="sec-head" style="margin-top:1.75rem;">
  <div class="sec-icon">💾</div>
  <span class="sec-text">Export Results</span>
</div>
""", unsafe_allow_html=True)

ex1, ex2 = st.columns(2, gap="medium")

# ── CSV ──
with ex1:
    csv_buf = io.StringIO()
    writer  = csv.writer(csv_buf)
    writer.writerow(["Learner Success Prediction Report"])
    writer.writerow([f"Generated: {ts}"])
    writer.writerow(["Model: XGBoost Regression  |  Dataset: UCI Student Performance"])
    writer.writerow([])
    writer.writerow(["=== PREDICTION SUMMARY ==="])
    writer.writerow(["Predicted Final Score", f"{pred_score:.1f} / 100"])
    writer.writerow(["Grade Band",            f"Grade {grade}"])
    writer.writerow(["Outcome",               "Likely to Succeed" if pred_score >= 50 else "At Risk of Failing"])
    writer.writerow(["Pass Threshold",        "50 / 100"])
    writer.writerow(["Points to Pass",        f"{max(0.0, 50-pred_score):.1f}"])
    writer.writerow([])
    writer.writerow(["=== STUDENT INPUTS ==="])
    writer.writerow(["Feature", "Value"])
    for col in features.columns:
        writer.writerow([NICE.get(col, col), features[col].values[0]])
    writer.writerow([])
    if imp_dict:
        writer.writerow(["=== FEATURE IMPORTANCE ==="])
        writer.writerow(["Feature", "Importance (%)"])
        for f, v in sorted(imp_dict.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([NICE.get(f, f), f"{v*100:.2f}%"])
    writer.writerow([])
    if risks:
        writer.writerow(["=== RISK FLAGS ==="])
        for _, _, t in risks:
            writer.writerow([t])
    if recs:
        writer.writerow([])
        writer.writerow(["=== RECOMMENDATIONS ==="])
        for r in recs:
            writer.writerow([r])

    st.download_button(
        label     = "⬇️  Download CSV Report",
        data      = csv_buf.getvalue().encode(),
        file_name = f"learner_report_{ts.replace(' ','_').replace(':','-')}.csv",
        mime      = "text/csv",
        use_container_width=True,
    )

# ── PDF ──
with ex2:
    def build_pdf() -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, HRFlowable,
            )

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story  = []

            ac_rl  = colors.HexColor("#3b5bdb" if not dark else "#6c8eff")
            ac2_rl = colors.HexColor("#0ca678" if not dark else "#00d4aa")
            ink_rl = colors.HexColor("#0f172a")
            mid_rl = colors.HexColor("#64748b")

            h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                                 fontSize=18, textColor=ink_rl, spaceAfter=4)
            h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                                 fontSize=10, textColor=ac_rl, fontName="Helvetica-Bold",
                                 spaceBefore=14, spaceAfter=4)
            body = ParagraphStyle("Body", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.HexColor("#334155"), leading=14)

            story.append(Paragraph("Learner Success Prediction Report", h1))
            story.append(Paragraph(
                "Department of Computer Engineering · Federal Polytechnic Nekede, Owerri", body))
            story.append(HRFlowable(width="100%", thickness=2, color=ac_rl, spaceAfter=10))
            story.append(Paragraph(f"Generated: {ts}", body))
            story.append(Spacer(1, 10))

            # Summary
            story.append(Paragraph("Prediction Summary", h2))
            outcome_str = "Likely to Succeed" if pred_score >= 50 else "At Risk of Failing"
            summ = [
                ["Metric", "Value"],
                ["Predicted Final Score", f"{pred_score:.1f} / 100"],
                ["Grade Band",            f"Grade {grade}"],
                ["Outcome",               outcome_str],
                ["Pass Threshold",        "50 / 100"],
                ["Points Still Needed",   f"{max(0.0, 50-pred_score):.1f}"],
            ]
            t = Table(summ, colWidths=[8*cm, 8.5*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), ac_rl),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 9),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 8))

            # Student inputs
            story.append(Paragraph("Student Input Features", h2))
            feat_rows = [["Feature", "Value"]]
            for col in features.columns:
                feat_rows.append([NICE.get(col, col), str(features[col].values[0])])
            ft = Table(feat_rows, colWidths=[8*cm, 8.5*cm])
            ft.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), ac_rl),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 9),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f8fafc"), colors.white]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(ft)
            story.append(Spacer(1, 8))

            # Feature importance
            if imp_dict:
                story.append(Paragraph("Feature Importance", h2))
                imp_rows = [["Feature", "Importance (%)"]]
                for f, v in sorted(imp_dict.items(), key=lambda x: x[1], reverse=True):
                    imp_rows.append([NICE.get(f, f), f"{v*100:.2f}%"])
                it = Table(imp_rows, colWidths=[10*cm, 6.5*cm])
                it.setStyle(TableStyle([
                    ("BACKGROUND",  (0, 0), (-1, 0), ac2_rl),
                    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0, 0), (-1, 0), 9),
                    ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE",    (0, 1), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.HexColor("#f0fdf4"), colors.white]),
                    ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                    ("TOPPADDING",  (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(it)
                story.append(Spacer(1, 8))

            # Risk & Recs
            if risks:
                story.append(Paragraph("Risk Flags", h2))
                for _, _, txt in risks:
                    story.append(Paragraph(f"• {txt}", body))
            if recs:
                story.append(Paragraph("Recommendations", h2))
                for r in recs:
                    story.append(Paragraph(f"→ {r}", body))

            story.append(Spacer(1, 14))
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#cbd5e1"), spaceBefore=4))
            story.append(Paragraph(
                "Model: XGBoost Regression · Dataset: UCI Student Performance · "
                "Federal Polytechnic Nekede, Owerri", body))

            doc.build(story)
            return buf.getvalue()

        except ImportError:
            fallback = (
                f"Learner Success Prediction Report\n"
                f"Generated: {ts}\n"
                f"Predicted Score: {pred_score:.1f} / 100\n"
                f"Grade: {grade}\n"
                f"(Install reportlab for full PDF output)"
            )
            return fallback.encode()

    st.download_button(
        label     = "⬇️  Download PDF Report",
        data      = build_pdf(),
        file_name = f"learner_report_{ts.replace(' ','_').replace(':','-')}.pdf",
        mime      = "application/pdf",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# EXPANDERS  (100% preserved)
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Full Student Data Summary"):
    st.markdown("**Raw input features:**")
    st.dataframe(features.rename(columns=NICE), use_container_width=True)
    st.markdown("**Scaled values fed to the model:**")
    st.dataframe(
        pd.DataFrame(scaled, columns=features.columns).rename(columns=NICE).round(4),
        use_container_width=True,
    )

with st.expander("📚 Feature & Grade Glossary"):
    st.markdown("""
**Features Used**

| Feature | Description |
|---|---|
| **G1** | First period exam score (0–100) — strongest predictor |
| **G2** | Second period exam score (0–100) — strongest predictor |
| **Failures** | Number of previously failed classes (0–3) |
| **Absences** | Total school absences during the year |
| **Study Time** | 1=<2hrs · 2=2–5hrs · 3=5–10hrs · 4=>10hrs per week |
| **Medu / Fedu** | Parental education: 0=none · 1=primary · 2=middle · 3=secondary · 4=higher |
| **Family Support** | Whether family actively encourages academic success |
| **School Support** | Whether the student receives extra school support |
| **Internet** | Internet access at home |
| **Activities** | Extra-curricular activity participation |
| **Higher** | Intends to pursue higher education |
| **Go Out** | Social outing frequency (1=rarely – 5=very often) |
| **Health** | Self-rated health (1=very poor – 5=excellent) |
| **Family Rel.** | Family relationship quality (1=very bad – 5=excellent) |
| **Travel Time** | Travel time to school (1=<15 min – 4=>60 min) |

**Grade Bands**

| Grade | Score | Interpretation |
|---|---|---|
| A+ | 90–100 | Outstanding |
| A | 80–89 | Excellent |
| B | 70–79 | Good |
| C | 60–69 | Satisfactory |
| D | 50–59 | Passing — may need support |
| F | 0–49 | At risk — intervention recommended |
""")

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  Model: XGBoost Regression &nbsp;·&nbsp;
  Dataset: UCI Student Performance &nbsp;·&nbsp;
  Department of Computer Engineering · Federal Polytechnic Nekede, Owerri
</div>
""", unsafe_allow_html=True)