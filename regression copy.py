import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Learner Success Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --bg:#07090f; --surface:#0e1118; --surface2:#141824; --border:#1e2535;
    --accent:#6c8eff; --accent2:#00d4aa; --danger:#ff5f6d; --warning:#ffb347;
    --text:#dde4f0; --muted:#6b7794; --radius:14px;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2.5rem 4rem;max-width:1280px;margin:auto;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stSlider label{font-size:0.72rem!important;font-weight:600!important;color:var(--muted)!important;text-transform:uppercase;letter-spacing:0.07em;}
.sidebar-section{font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:var(--accent);border-top:1px solid var(--border);padding-top:1rem;margin:1.2rem 0 0.6rem;}
.hero{background:linear-gradient(135deg,#0e1a2e 0%,#0b1220 60%,var(--bg) 100%);border:1px solid var(--border);border-radius:var(--radius);padding:2.8rem 3rem;margin-bottom:2rem;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;bottom:-80px;right:-80px;width:260px;height:260px;background:radial-gradient(circle,rgba(108,142,255,0.10) 0%,transparent 65%);border-radius:50%;pointer-events:none;}
.hero-eyebrow{font-size:0.68rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:var(--accent);margin-bottom:0.6rem;}
.hero-title{font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:800;color:var(--text);line-height:1.1;margin-bottom:0.8rem;}
.hero-title em{color:var(--accent2);font-style:normal;}
.hero-sub{color:var(--muted);font-size:0.92rem;max-width:600px;line-height:1.6;}
.badge-row{display:flex;gap:0.6rem;margin-top:1.4rem;flex-wrap:wrap;}
.badge{display:inline-flex;align-items:center;gap:0.35rem;padding:0.28rem 0.75rem;border-radius:99px;font-size:0.72rem;font-weight:600;}
.badge.blue{background:rgba(108,142,255,0.12);border:1px solid rgba(108,142,255,0.3);color:var(--accent);}
.badge.teal{background:rgba(0,212,170,0.10);border:1px solid rgba(0,212,170,0.3);color:var(--accent2);}
.badge.amber{background:rgba(255,179,71,0.10);border:1px solid rgba(255,179,71,0.3);color:var(--warning);}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;margin-bottom:1.5rem;}
.mcard{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;}
.mcard .mlabel{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--muted);margin-bottom:0.35rem;}
.mcard .mval{font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:700;line-height:1;}
.gauge-wrap{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:2rem;text-align:center;}
.gauge-label{font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:var(--muted);margin-bottom:1rem;}
.gauge-score{font-family:'Syne',sans-serif;font-size:5rem;font-weight:800;line-height:1;margin-bottom:0.4rem;}
.gauge-track{height:12px;background:var(--border);border-radius:99px;overflow:hidden;margin:0.5rem 0;}
.gauge-fill{height:100%;border-radius:99px;}
.gauge-ticks{display:flex;justify-content:space-between;font-size:0.65rem;color:var(--muted);margin-top:4px;}
.band-row{display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.8rem;}
.band-pill{padding:0.35rem 0.8rem;border-radius:8px;font-size:0.7rem;font-weight:600;border:1px solid;opacity:0.3;}
.band-pill.active{opacity:1;}
.panel{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:1.4rem 1.6rem;}
.panel-title{font-family:'Syne',sans-serif;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;color:var(--muted);border-bottom:1px solid var(--border);padding-bottom:0.6rem;margin-bottom:1rem;}
.fbar-row{margin:0.5rem 0;}
.fbar-label{display:flex;justify-content:space-between;font-size:0.78rem;margin-bottom:3px;}
.fbar-track{height:6px;background:var(--border);border-radius:99px;overflow:hidden;}
.fbar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2));}
.chips{display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.6rem;}
.chip{display:inline-flex;align-items:center;gap:0.3rem;padding:0.3rem 0.7rem;border-radius:99px;font-size:0.75rem;font-weight:500;}
.chip.risk{background:rgba(255,95,109,0.10);border:1px solid rgba(255,95,109,0.3);color:var(--danger);}
.chip.rec{background:rgba(108,142,255,0.10);border:1px solid rgba(108,142,255,0.3);color:var(--accent);}
.chip.ok{background:rgba(0,212,170,0.10);border:1px solid rgba(0,212,170,0.3);color:var(--accent2);}
.chip.warn{background:rgba(255,179,71,0.10);border:1px solid rgba(255,179,71,0.3);color:var(--warning);}
.stButton>button{width:100%!important;background:linear-gradient(135deg,var(--accent),#4a6ef5)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:0.75rem 1.5rem!important;font-weight:600!important;font-size:0.9rem!important;font-family:'Inter',sans-serif!important;}
.stButton>button:hover{opacity:0.88!important;}
div[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:var(--radius)!important;}
.stExpander{border:1px solid var(--border)!important;border-radius:var(--radius)!important;background:var(--surface2)!important;}
</style>
""", unsafe_allow_html=True)


# ── Load saved model & scaler ─────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    if s >= 90: return ("A+","#00d4aa","#0a2e28")
    if s >= 80: return ("A","#6c8eff","#0e1530")
    if s >= 70: return ("B","#a78bfa","#1a1030")
    if s >= 60: return ("C","#ffb347","#2a1e08")
    if s >= 50: return ("D","#ff9a5c","#2a1508")
    return ("F","#ff5f6d","#2a0c0e")

def score_color(s):
    if s >= 70: return "var(--accent2)"
    if s >= 50: return "var(--warning)"
    return "var(--danger)"

def trend_arrow(g1, g2):
    d = g2 - g1
    if d > 5:  return "↑","var(--accent2)",f"+{d:.0f}"
    if d < -5: return "↓","var(--danger)",f"{d:.0f}"
    return "→","var(--muted)",f"{d:+.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p style="font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;margin-bottom:0.2rem;">Student Profile</p>'
        '<p style="font-size:0.75rem;color:var(--muted);margin-bottom:1rem;">Fill in the student\'s details</p>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-section">📊 Prior Grades</div>', unsafe_allow_html=True)
    G1 = st.select_slider("G1 — Test",
                           options=[15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95], value=60)
    G2 = st.select_slider("G2 — CBT Score",
                           options=[0,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95], value=60)
    st.markdown('<div class="sidebar-section">📚 Academic History</div>', unsafe_allow_html=True)
    failures   = st.selectbox("Previous Failures", [0,1,2,3], index=0)
    absences   = st.slider("Total Absences", 0, 75, 4)
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
    famsup = st.selectbox("Family Academic Support",["yes","no"])
    famrel = st.slider("Family Relationship Quality",1,5,4)
    st.markdown('<div class="sidebar-section">🌱 Goals & Lifestyle</div>', unsafe_allow_html=True)
    higher     = st.selectbox("Aims for Higher Education",["yes","no"])
    internet   = st.selectbox("Has Internet at Home",["yes","no"])
    schoolsup  = st.selectbox("Extra School Support",["yes","no"])
    activities = st.selectbox("Extra-Curricular Activities",["yes","no"])
    goout  = st.slider("Social Outings Frequency",1,5,2,help="1=rarely  5=very often")
    health = st.slider("Health Status",1,5,3,help="1=very poor  5=excellent")


# ── Feature DataFrame ─────────────────────────────────────────────────────────

bmap = {"yes":1,"no":0}
features = pd.DataFrame({
    "studytime":[studytime],"failures":[failures],"absences":[absences],
    "Medu":[Medu],"Fedu":[Fedu],
    "schoolsup":[bmap[schoolsup]],"famsup":[bmap[famsup]],
    "higher":[bmap[higher]],"internet":[bmap[internet]],
    "activities":[bmap[activities]],"traveltime":[traveltime],
    "famrel":[famrel],"goout":[goout],"health":[health],
    "G1":[G1],"G2":[G2],
})


# ── Hero ──────────────────────────────────────────────────────────────────────

# st.markdown("""
# <div class="hero">
#   <div class="hero-eyebrow">🎓 AI-Powered · For Teachers, Counsellors &amp; Parents</div>
#   <div class="hero-title">An Intelligent Educational Analytics System for Learner Success Prediction<br><em>Dept. of Computer Engineering, Federal Polytechnic Nekede, Owerri</em></div>
#   <div class="hero-sub">
#     Helps teachers, counsellors, and parents understand whether a student is likely to
#     succeed in a course — and <strong>what to do if they are at risk</strong>.
#     Predicts the student's expected final score (0–100) from their academic profile.
#   </div>
#   <div class="badge-row">
#     <span class="badge blue">Ensemble Regression Model</span>
#     <span class="badge teal">Continuous Score (0–100)</span>
#     <span class="badge amber">Early Intervention Tool</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

st.markdown("""
<h1 style="
    text-align:center;
    font-family:'Poppins', sans-serif;
    font-size:36px;
    font-weight:700;
    margin-bottom:10px;
    color:#111111;">
    ML-Driven Learner Success Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    font-family:'Poppins', sans-serif;
    font-size:24px;
    color:#555555;
    margin-top:-20px;
    margin-bottom:40px;">
    Dept. of Comp. Engineering, Federal Polytechnic Nekede, Owerri
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Could not load model files from `results/`. Error: `{load_error}`")
    st.info("Ensure `results/ensemble_model_model.pkl` and `results/scaler.pkl` exist alongside this app.")
    st.stop()


# ── Predict (live — no button needed) ────────────────────────────────────────

scaled     = scaler.transform(features)
pred_score = float(np.clip(model.predict(scaled)[0], 0, 100))
grade, gc, gbg = grade_band(pred_score)
sc_col = score_color(pred_score)
arrow, acol, adiff = trend_arrow(G1, G2)
avg_prior = (G1 + G2) / 2


# ── Metric cards ─────────────────────────────────────────────────────────────

g1c = "var(--accent2)" if G1>=60 else "var(--warning)" if G1>=40 else "var(--danger)"
g2c = "var(--accent2)" if G2>=60 else "var(--warning)" if G2>=40 else "var(--danger)"
abc = "var(--danger)"  if absences>20 else "var(--warning)" if absences>10 else "var(--accent2)"
flc = "var(--danger)"  if failures>1  else "var(--warning)" if failures==1  else "var(--accent2)"

st.markdown(f"""
<div class="metric-grid">
  <div class="mcard"><div class="mlabel">G1 Score</div><div class="mval" style="color:{g1c};">{G1}</div></div>
  <div class="mcard"><div class="mlabel">G2 Score</div><div class="mval" style="color:{g2c};">{G2}</div></div>
  <div class="mcard"><div class="mlabel">G1 → G2 Trend</div><div class="mval" style="color:{acol};">{arrow} {adiff}</div></div>
  <div class="mcard"><div class="mlabel">Avg Prior Grade</div><div class="mval" style="color:var(--accent);">{avg_prior:.0f}</div></div>
  <div class="mcard"><div class="mlabel">Absences</div><div class="mval" style="color:{abc};">{absences}</div></div>
  <div class="mcard"><div class="mlabel">Past Failures</div><div class="mval" style="color:{flc};">{failures}</div></div>
</div>
""", unsafe_allow_html=True)


# ── Three columns ─────────────────────────────────────────────────────────────

col_gauge, col_analysis, col_factors = st.columns([1, 1.1, 1.2])

with col_gauge:
    bands = [
        ("F","#ff5f6d","#2a0c0e",0,49),("D","#ff9a5c","#2a1508",50,59),
        ("C","#ffb347","#2a1e08",60,69),("B","#a78bfa","#1a1030",70,79),
        ("A","#6c8eff","#0e1530",80,89),("A+","#00d4aa","#0a2e28",90,100),
    ]
    pills = "".join(
        f'<span class="band-pill {"active" if lo<=pred_score<=hi else ""}" '
        f'style="color:{bc};background:{bbg};border-color:{bc};">{bl} ({lo}–{hi})</span>'
        for bl,bc,bbg,lo,hi in bands
    )
    outcome       = "🎉 Likely to Succeed" if pred_score >= 50 else "⚠️ At Risk of Failing"
    outcome_color = "var(--accent2)"       if pred_score >= 50 else "var(--danger)"

    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-label">Predicted Final Score</div>
      <div class="gauge-score" style="color:{sc_col};">{pred_score:.1f}</div>
      <div style="font-size:1rem;font-weight:600;color:{gc};background:{gbg};
           display:inline-block;padding:0.2rem 1rem;border-radius:6px;
           border:1px solid {gc}44;margin-bottom:0.8rem;">Grade {grade}</div><br>
      <div style="font-size:0.9rem;font-weight:600;color:{outcome_color};margin-bottom:0.8rem;">{outcome}</div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{int(pred_score)}%;background:linear-gradient(90deg,{gc}99,{gc});"></div>
      </div>
      <div class="gauge-ticks"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
      <div class="band-row">{pills}</div>
      <div style="margin-top:1.2rem;font-size:0.78rem;color:var(--muted);background:var(--surface);
           border:1px solid var(--border);border-radius:8px;padding:0.6rem 0.9rem;text-align:left;">
        ℹ️ Pass threshold is <strong style="color:var(--text);">50 / 100</strong>.
        Use alongside teacher judgment.
      </div>
    </div>
    """, unsafe_allow_html=True)


with col_analysis:
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
        st.markdown('<div class="chips">'+"".join(f'<span class="chip {c}">{i} {t}</span>' for c,i,t in risks)+'</div>', unsafe_allow_html=True)

    dist_pass = max(0.0, 50.0 - pred_score)
    st.markdown('<div class="panel-title" style="margin-top:1.4rem;">📐 Score Breakdown</div>', unsafe_allow_html=True)
    for lbl, val in [
        ("Prior grade average", f"{avg_prior:.0f} / 100"),
        ("Predicted final score", f"{pred_score:.1f} / 100"),
        ("Grade band", f"Grade {grade}"),
        ("Pass threshold", "50 / 100"),
        ("To reach pass", f"{dist_pass:.1f} pts still needed" if dist_pass > 0 else "✓ Above pass threshold"),
    ]:
        cv = "var(--accent2)" if "✓" in val else "var(--text)"
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;'
            f'padding:0.35rem 0;border-bottom:1px solid var(--border);">'
            f'<span style="color:var(--muted);">{lbl}</span>'
            f'<span style="color:{cv};font-weight:500;">{val}</span></div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

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

    st.markdown('<div class="panel" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">💡 Recommendations for Teachers &amp; Parents</div>', unsafe_allow_html=True)
    if not recs:
        st.markdown('<div class="chips"><span class="chip ok">✓ Student on a healthy trajectory</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chips">'+"".join(f'<span class="chip rec">→ {r}</span>' for r in recs)+'</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


with col_factors:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📈 What Drives the Prediction</div>', unsafe_allow_html=True)

    imp_dict = None
    try:
        imp_dict = dict(zip(features.columns, model.feature_importances_))
    except AttributeError:
        try:
            imps = [e.feature_importances_ for _, e in model.estimators_ if hasattr(e, "feature_importances_")]
            if imps:
                imp_dict = dict(zip(features.columns, np.mean(imps, axis=0)))
        except Exception:
            pass

    if imp_dict:
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        max_imp = sorted_imp[0][1]
        bars = "".join(
            f'<div class="fbar-row"><div class="fbar-label">'
            f'<span>{NICE.get(f,f)}</span>'
            f'<span style="color:var(--muted);font-size:0.7rem;">{v*100:.1f}%</span>'
            f'</div><div class="fbar-track">'
            f'<div class="fbar-fill" style="width:{int(v/max_imp*100)}%;"></div>'
            f'</div></div>'
            for f, v in sorted_imp
        )
        st.markdown(bars, unsafe_allow_html=True)
    else:
        st.info("Feature importance not available for this model type.")

    st.markdown('</div>', unsafe_allow_html=True)


# ── Expanders ─────────────────────────────────────────────────────────────────

with st.expander("📋 Full Student Data Summary"):
    st.markdown("**Raw input features:**")
    st.dataframe(features.rename(columns=NICE), use_container_width=True)
    st.markdown("**Scaled values fed to the model:**")
    st.dataframe(pd.DataFrame(scaled, columns=features.columns).rename(columns=NICE).round(4), use_container_width=True)

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