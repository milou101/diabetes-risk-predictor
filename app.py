import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* global */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b2d 0%, #162032 100%);
}
section[data-testid="stSidebar"] * { color: #d0dce8 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #2563eb; }
section[data-testid="stSidebar"] label { font-size: 0.82rem; letter-spacing: 0.04em; text-transform: uppercase; }

/* metric cards */
div[data-testid="metric-container"] {
    background: #0f1b2d;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
div[data-testid="metric-container"] label { color: #7ca0c5 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2eaf5 !important; font-size: 1.6rem; font-weight: 700; }

/* section header */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.2rem 0 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e3a5f;
}

/* result banner */
.banner-high   { background:#3d0a0a; border:1px solid #dc2626; border-radius:10px; padding:1rem 1.4rem; color:#fca5a5; font-size:1.15rem; font-weight:700; }
.banner-mod    { background:#3d2600; border:1px solid #d97706; border-radius:10px; padding:1rem 1.4rem; color:#fcd34d; font-size:1.15rem; font-weight:700; }
.banner-low    { background:#052e16; border:1px solid #16a34a; border-radius:10px; padding:1rem 1.4rem; color:#86efac; font-size:1.15rem; font-weight:700; }

/* factor pills */
.pill-risk     { display:inline-block; background:#3d0a0a; border:1px solid #dc2626; color:#fca5a5; border-radius:20px; padding:0.25rem 0.75rem; margin:0.2rem; font-size:0.82rem; }
.pill-good     { display:inline-block; background:#052e16; border:1px solid #16a34a; color:#86efac; border-radius:20px; padding:0.25rem 0.75rem; margin:0.2rem; font-size:0.82rem; }
.pill-warn     { display:inline-block; background:#3d2600; border:1px solid #d97706; color:#fcd34d; border-radius:20px; padding:0.25rem 0.75rem; margin:0.2rem; font-size:0.82rem; }

/* disclaimer */
.disclaimer {
    background: #0f1b2d;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    color: #7ca0c5;
    font-size: 0.8rem;
    margin-top: 1rem;
}

/* hide default Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    try:
        bundle = joblib.load('diabetes_optimized_model.pkl')
        return bundle
    except FileNotFoundError:
        return None


bundle = load_model_bundle()

if bundle is None:
    st.error("❌ **Model file not found: `diabetes_optimized_model.pkl`**")
    st.info("Run the Jupyter notebook (`diabetes.ipynb`) to train and save the model first.")
    st.stop()


# ── Feature engineering (mirrors notebook Cell 37) ───────────────────────────
def engineer_features(raw: dict, feature_names: list) -> np.ndarray:
    g   = raw['Glucose']
    b   = raw['BMI']
    a   = raw['Age']
    ins = raw['Insulin']

    engineered = {
        **raw,
        'Glucose_BMI':           g * b / 1000,
        'Age_Glucose':           a * g / 1000,
        'Glucose_squared':       (g / 100) ** 2,
        'Insulin_Glucose_ratio': g / (ins + 1),
        'BMI_category':          3 if b >= 30 else (2 if b >= 25 else (1 if b >= 18.5 else 0)),
        'Age_group':             3 if a >= 60 else (2 if a >= 45 else (1 if a >= 30 else 0)),
    }
    return np.array([[engineered[f] for f in feature_names]])


def predict_diabetes(raw: dict, mdl_bundle: dict) -> dict:
    model     = mdl_bundle['model']
    threshold = mdl_bundle['threshold']
    X         = engineer_features(raw, mdl_bundle['feature_names'])
    proba     = model.predict_proba(X)[0, 1]
    pred      = int(proba >= threshold)
    confidence = (
        'High'   if abs(proba - 0.5) > 0.25 else
        'Medium' if abs(proba - 0.5) > 0.10 else
        'Low'
    )
    return {
        'prediction':  'Diabetic' if pred == 1 else 'Non-diabetic',
        'probability': round(float(proba), 4),
        'confidence':  confidence,
        'threshold':   threshold,
        'pred_int':    pred,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Input")

    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    age         = st.slider('Age (years)',         21, 100, 30)
    pregnancies = st.number_input('Pregnancies',    0,  20,  0)

    st.markdown('<p class="section-header">Clinical Measurements</p>', unsafe_allow_html=True)
    glucose  = st.slider('Glucose (mg/dL)',          0, 200, 120)
    bp       = st.slider('Blood Pressure (mm Hg)',   0, 130,  70)
    skin     = st.slider('Skin Thickness (mm)',       0, 100,  20)
    insulin  = st.slider('Insulin (mu U/ml)',         0, 900,  80)
    bmi      = st.number_input('BMI', 10.0, 70.0, 25.0, 0.1, format="%.1f")
    dpf      = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.47, 0.01)

    st.markdown("---")

    model_info = bundle.get('model_name', 'Unknown')
    acc_info   = f"{bundle.get('test_accuracy', 0):.1%}"
    f1_info    = f"{bundle.get('test_f1', 0):.1%}"
    auc_info   = f"{bundle.get('test_auc', 0):.1%}"
    thresh_info = f"{bundle.get('threshold', 0.5):.2f}"

    st.markdown(f"""
    <div style="font-size:0.75rem; color:#4b7099; line-height:1.8;">
    <b style="color:#7ca0c5">Model</b> · {model_info}<br>
    <b style="color:#7ca0c5">Accuracy</b> · {acc_info}<br>
    <b style="color:#7ca0c5">F1 Score</b> · {f1_info}<br>
    <b style="color:#7ca0c5">ROC-AUC</b> · {auc_info}<br>
    <b style="color:#7ca0c5">Threshold</b> · {thresh_info}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Prediction", type="primary", use_container_width=True)


# ── Main page header ──────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2rem; font-weight:800; color:#e2eaf5; margin-bottom:0;">
  🩺 Diabetes Risk Predictor
</h1>
<p style="color:#4b7099; margin-top:0.2rem; margin-bottom:1.5rem;">
  Optimized ML pipeline · Stratified imputation · Feature engineering · Threshold-tuned
</p>
""", unsafe_allow_html=True)

# ── Idle state ────────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", bundle.get('model_name', '—'))
    c2.metric("Accuracy", f"{bundle.get('test_accuracy', 0):.1%}")
    c3.metric("ROC-AUC",  f"{bundle.get('test_auc', 0):.1%}")
    c4.metric("Features", str(len(bundle.get('feature_names', []))))

    st.markdown("""
    <div style="margin-top:2rem; padding:1.5rem; background:#0f1b2d; border-radius:12px; border:1px solid #1e3a5f; color:#7ca0c5; font-size:0.9rem; line-height:1.9;">
    <b style="color:#e2eaf5">How to use:</b><br>
    1. Fill in the patient values in the left sidebar.<br>
    2. Click <b style="color:#3b82f6">⚡ Run Prediction</b>.<br>
    3. Review the risk gauge, factor analysis, and recommendations.<br><br>
    <b style="color:#e2eaf5">What makes this model different from v1:</b><br>
    • Biologically-impossible zeros replaced via stratified median imputation<br>
    • 6 engineered features (Glucose×BMI, Age×Glucose, insulin resistance proxy, etc.)<br>
    • <code>RobustScaler</code> + <code>class_weight='balanced'</code><br>
    • GridSearchCV over SVM and Random Forest with 5-fold stratified CV<br>
    • Decision threshold optimized for F1 / Recall
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Prediction ────────────────────────────────────────────────────────────────
raw_input = {
    'Pregnancies':             pregnancies,
    'Glucose':                 glucose,
    'BloodPressure':           bp,
    'SkinThickness':           skin,
    'Insulin':                 insulin,
    'BMI':                     float(bmi),
    'DiabetesPedigreeFunction': dpf,
    'Age':                     age,
}

result = predict_diabetes(raw_input, bundle)
proba       = result['probability']
pred_int    = result['pred_int']
prob_pct    = proba * 100
prob_neg    = (1 - proba) * 100
confidence  = result['confidence']


# ── Result banner ─────────────────────────────────────────────────────────────
if pred_int == 1:
    if prob_pct >= 70:
        banner_cls = "banner-high"
        banner_txt = "🔴 HIGH RISK — Diabetic"
    else:
        banner_cls = "banner-mod"
        banner_txt = "🟡 MODERATE RISK — Diabetic"
else:
    if prob_pct < 30:
        banner_cls = "banner-low"
        banner_txt = "🟢 LOW RISK — Non-Diabetic"
    else:
        banner_cls = "banner-mod"
        banner_txt = "🟡 BORDERLINE — Non-Diabetic"

st.markdown(f'<div class="{banner_cls}">{banner_txt} &nbsp;·&nbsp; Confidence: {confidence}</div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Diabetic Probability",    f"{prob_pct:.1f}%")
m2.metric("Non-Diabetic Probability", f"{prob_neg:.1f}%")
m3.metric("Decision Threshold",      f"{result['threshold']:.2f}")
m4.metric("Confidence Level",        confidence)

st.markdown("---")

# ── Gauge + Radar ─────────────────────────────────────────────────────────────
col_gauge, col_radar = st.columns([1, 1])

with col_gauge:
    st.markdown("#### 📊 Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        delta={'reference': 50, 'valueformat': '.1f', 'suffix': '%'},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#e2eaf5'}},
        title={'text': "Diabetes Probability", 'font': {'color': '#7ca0c5', 'size': 14}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#4b7099', 'tickfont': {'color': '#4b7099'}},
            'bar':  {'color': '#3b82f6', 'thickness': 0.22},
            'bgcolor': '#0f1b2d',
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0,  30], 'color': '#052e16'},
                {'range': [30, 70], 'color': '#3d2600'},
                {'range': [70, 100], 'color': '#3d0a0a'},
            ],
            'threshold': {
                'line': {'color': '#60a5fa', 'width': 3},
                'thickness': 0.75,
                'value': result['threshold'] * 100,
            },
        }
    ))
    fig_gauge.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=10),
        font_color='#7ca0c5',
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_radar:
    st.markdown("#### 🕸️ Patient Profile")
    # Normalise each input to 0-1 against realistic max values for radar
    radar_labels = ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin', 'DPF']
    radar_max    = [200,        70,    100,   130,              900,       2.5]
    radar_vals   = [glucose, float(bmi), age, bp, insulin, dpf]
    radar_norm   = [v / m for v, m in zip(radar_vals, radar_max)]
    radar_norm  += radar_norm[:1]
    radar_labels_closed = radar_labels + [radar_labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_norm,
        theta=radar_labels_closed,
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
        name='Patient',
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(15,27,45,0.6)',
            radialaxis=dict(visible=True, range=[0, 1], tickfont_color='#4b7099', gridcolor='#1e3a5f', linecolor='#1e3a5f'),
            angularaxis=dict(tickfont=dict(color='#7ca0c5', size=11), gridcolor='#1e3a5f', linecolor='#1e3a5f'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=40, r=40, t=40, b=10),
        showlegend=False,
        font_color='#7ca0c5',
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ── Risk factor analysis ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### ⚠️ Risk Factor Analysis")

risk_pills  = []
good_pills  = []
warn_pills  = []

# Glucose
if glucose >= 126:
    risk_pills.append("Glucose ≥ 126 mg/dL (diabetic range)")
elif glucose >= 100:
    warn_pills.append("Glucose 100–125 mg/dL (pre-diabetic range)")
else:
    good_pills.append("Glucose < 100 mg/dL (normal)")

# BMI
if float(bmi) >= 30:
    risk_pills.append("BMI ≥ 30 (obese)")
elif float(bmi) >= 25:
    warn_pills.append("BMI 25–30 (overweight)")
else:
    good_pills.append("BMI < 25 (healthy weight)")

# Blood Pressure
if bp >= 90:
    risk_pills.append("Blood Pressure ≥ 90 mm Hg (stage 2 hypertension)")
elif bp >= 80:
    warn_pills.append("Blood Pressure 80–90 mm Hg (elevated)")
elif 60 <= bp < 80:
    good_pills.append("Blood Pressure 60–79 mm Hg (normal)")

# Age
if age >= 60:
    risk_pills.append("Age ≥ 60")
elif age >= 45:
    warn_pills.append("Age 45–59")
else:
    good_pills.append("Age < 45")

# DPF
if dpf >= 1.0:
    risk_pills.append("DPF ≥ 1.0 (strong family history)")
elif dpf >= 0.5:
    warn_pills.append("DPF 0.5–1.0 (moderate genetic risk)")
else:
    good_pills.append("DPF < 0.5 (low genetic risk)")

# Insulin
if insulin == 0:
    warn_pills.append("Insulin = 0 (may indicate missing data)")
elif insulin > 200:
    warn_pills.append("Insulin > 200 mu U/ml (elevated)")

pills_html = ""
if risk_pills:
    pills_html += "<b style='color:#fca5a5'>Risk factors:</b><br>"
    pills_html += "".join(f'<span class="pill-risk">🔴 {p}</span>' for p in risk_pills) + "<br><br>"
if warn_pills:
    pills_html += "<b style='color:#fcd34d'>Watch points:</b><br>"
    pills_html += "".join(f'<span class="pill-warn">🟡 {p}</span>' for p in warn_pills) + "<br><br>"
if good_pills:
    pills_html += "<b style='color:#86efac'>Positive indicators:</b><br>"
    pills_html += "".join(f'<span class="pill-good">🟢 {p}</span>' for p in good_pills)

st.markdown(pills_html, unsafe_allow_html=True)


# ── Engineered features bar chart ────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔬 Engineered Feature Values (sent to model)")

feat_names = bundle['feature_names']
X_display  = engineer_features(raw_input, feat_names)[0]
feat_df    = pd.DataFrame({'Feature': feat_names, 'Value': X_display})

# Separate base vs engineered
base_feats = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
              'Insulin','BMI','DiabetesPedigreeFunction','Age']
feat_df['Type'] = feat_df['Feature'].apply(lambda f: 'Base' if f in base_feats else 'Engineered')
colors = feat_df['Type'].map({'Base': '#3b82f6', 'Engineered': '#10b981'})

fig_bar = go.Figure(go.Bar(
    x=feat_df['Feature'],
    y=feat_df['Value'],
    marker_color=colors.tolist(),
    text=[f"{v:.3f}" for v in feat_df['Value']],
    textposition='outside',
    textfont=dict(color='#7ca0c5', size=9),
))
fig_bar.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(15,27,45,0.6)',
    height=300,
    margin=dict(l=10, r=10, t=10, b=100),
    xaxis=dict(tickangle=-35, tickfont=dict(color='#7ca0c5', size=10), gridcolor='#1e3a5f'),
    yaxis=dict(tickfont=dict(color='#4b7099'), gridcolor='#1e3a5f'),
    font_color='#7ca0c5',
    showlegend=False,
)
st.plotly_chart(fig_bar, use_container_width=True)

# Legend
st.markdown("""
<span style="display:inline-block;width:12px;height:12px;background:#3b82f6;border-radius:2px;margin-right:5px;"></span>
<span style="color:#7ca0c5;font-size:0.8rem;">Base feature</span>
&nbsp;&nbsp;
<span style="display:inline-block;width:12px;height:12px;background:#10b981;border-radius:2px;margin-right:5px;"></span>
<span style="color:#7ca0c5;font-size:0.8rem;">Engineered feature</span>
""", unsafe_allow_html=True)


# ── Recommendations ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Recommendations")

if pred_int == 1:
    st.markdown("""
    <div style="background:#3d0a0a;border:1px solid #dc2626;border-radius:10px;padding:1rem 1.4rem;color:#fca5a5;">
    <b>Immediate actions recommended:</b><br>
    • Consult a healthcare professional as soon as possible<br>
    • Request comprehensive diabetes screening (HbA1c, fasting glucose)<br>
    • Monitor blood glucose regularly at home<br>
    • Review diet: reduce refined carbohydrates and sugars<br>
    • Aim for at least 150 min/week of moderate physical activity
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background:#052e16;border:1px solid #16a34a;border-radius:10px;padding:1rem 1.4rem;color:#86efac;">
    <b>Maintain healthy practices:</b><br>
    • Annual health check-ups and routine blood work<br>
    • Balanced diet rich in whole grains, vegetables, and lean protein<br>
    • At least 30 min of moderate exercise daily<br>
    • Monitor weight and BMI regularly<br>
    • Manage stress — chronic stress raises blood glucose
    </div>
    """, unsafe_allow_html=True)


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <b>Medical disclaimer:</b> This tool is for educational and research purposes only.
It does <b>not</b> constitute medical advice and must not replace consultation with a
qualified healthcare professional. Always seek professional medical guidance for health concerns.
</div>
""", unsafe_allow_html=True)