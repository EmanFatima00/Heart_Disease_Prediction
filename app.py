"""
Heart Disease Prediction App
Author: Eman Fatima | BS-AI @ PAF-IAST
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #DC2626, #991B1B);
        padding: 2rem; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #DC2626;
    }
    .metric-card h3 { font-size: 1.8rem; color: #DC2626; margin: 0; }
    .metric-card p  { color: #64748B; margin: 0; font-size: 0.9rem; }
    .result-high {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border: 2px solid #DC2626; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
        border: 2px solid #059669; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }
    .info-box {
        background: #EFF6FF; border-left: 4px solid #3B82F6;
        padding: 1rem; border-radius: 6px; margin: 1rem 0;
        font-size: 0.9rem; color: #1E40AF;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load(os.path.join(OUTPUT_DIR, "heart_disease_model.pkl"))
    scaler        = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(OUTPUT_DIR, "feature_names.pkl"))
    metrics_path  = os.path.join(OUTPUT_DIR, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {
        "accuracy": 83.15, "roc_auc": 0.9179, "samples": 920, "models": 5
    }
    return model, scaler, feature_names, metrics

model, scaler, feature_names, metrics = load_model()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>AI-powered cardiovascular risk assessment — UCI Heart Disease Dataset</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><h3>{metrics["accuracy"]}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><h3>{metrics["roc_auc"]}</h3><p>ROC-AUC</p></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><h3>{metrics["samples"]}</h3><p>Training Samples</p></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><h3>{metrics["models"]}</h3><p>Models Compared</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 Patient Information")
    st.markdown("---")
    st.markdown("### 👤 Demographics")
    age = st.slider("Age", 20, 90, 55)
    sex = st.selectbox("Sex", ["Male", "Female"])

    st.markdown("### 💓 Cardiac Symptoms")
    cp    = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal","asymptomatic"])
    exang = st.selectbox("Exercise Induced Angina", ["No","Yes"])

    st.markdown("### 🔬 Clinical Measurements")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No","Yes"])
    thalch   = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)

    st.markdown("### 📋 ECG & Scan Results")
    restecg = st.selectbox("Resting ECG", ["normal","lv hypertrophy","st-t abnormality"])
    slope   = st.selectbox("ST Slope", ["upsloping","flat","downsloping"])
    ca      = st.slider("Major Vessels Colored (0-3)", 0, 3, 0)
    thal    = st.selectbox("Thalassemia", ["normal","fixed defect","reversable defect"])

    predict_btn = st.button("🔍 Predict Risk", type="primary", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Insights", "ℹ️ About"])

with tab1:
    if predict_btn:
        sex_enc     = 1 if sex == "Male" else 0
        exang_enc   = 1 if exang == "Yes" else 0
        fbs_enc     = 1 if fbs == "Yes" else 0
        cp_map      = {"typical angina":0,"atypical angina":1,"non-anginal":2,"asymptomatic":3}
        restecg_map = {"normal":0,"lv hypertrophy":1,"st-t abnormality":2}
        slope_map   = {"upsloping":0,"flat":1,"downsloping":2}
        thal_map    = {"normal":0,"fixed defect":1,"reversable defect":2}

        age_group  = 0 if age<=40 else (1 if age<=55 else (2 if age<=65 else 3))
        bp_cat     = 0 if trestbps<=120 else (1 if trestbps<=130 else (2 if trestbps<=140 else 3))
        chol_risk  = 1 if chol > 240 else 0
        hr_reserve = thalch - age

        input_data = pd.DataFrame([[
            age, sex_enc, cp_map[cp], trestbps, chol, fbs_enc,
            restecg_map[restecg], thalch, exang_enc, oldpeak,
            slope_map[slope], ca, thal_map[thal],
            age_group, bp_cat, chol_risk, hr_reserve
        ]], columns=feature_names)

        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]
        risk_pct     = probability[1] * 100

        col_res, col_detail = st.columns([1, 1])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h2>⚠️ HIGH RISK</h2>
                    <h1 style="color:#DC2626;font-size:3rem;">{risk_pct:.1f}%</h1>
                    <p style="color:#7F1D1D;">Probability of Heart Disease</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h2>✅ LOW RISK</h2>
                    <h1 style="color:#059669;font-size:3rem;">{100-risk_pct:.1f}%</h1>
                    <p style="color:#064E3B;">Probability of No Disease</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_g, ax_g = plt.subplots(figsize=(5, 2.5))
            ax_g.barh(["Risk"], [risk_pct], color='#DC2626' if risk_pct>50 else '#059669', height=0.4)
            ax_g.barh(["Risk"], [100-risk_pct], left=[risk_pct], color='#F1F5F9', height=0.4)
            ax_g.set_xlim(0, 100); ax_g.axvline(50, color='#94A3B8', linestyle='--', linewidth=1)
            ax_g.set_xlabel("Risk Probability (%)"); ax_g.set_title("Risk Gauge", fontweight='bold')
            ax_g.text(risk_pct/2, 0, f"{risk_pct:.1f}%", ha='center', va='center',
                      color='white', fontweight='bold', fontsize=11)
            plt.tight_layout(); st.pyplot(fig_g); plt.close()

        with col_detail:
            st.markdown("#### 📋 Patient Summary")
            summary = {
                "Age": f"{age} years", "Sex": sex, "Chest Pain": cp,
                "Blood Pressure": f"{trestbps} mmHg", "Cholesterol": f"{chol} mg/dL",
                "Max Heart Rate": f"{thalch} bpm", "HR Reserve": f"{hr_reserve} bpm",
                "Chol Risk": "High" if chol_risk else "Normal",
            }
            for k, v in summary.items():
                a, b = st.columns([1.2, 1]); a.write(f"**{k}**"); b.write(v)

            st.markdown("""
            <div class="info-box">
            ⚕️ <b>Disclaimer:</b> For educational purposes only.
            Not a substitute for professional medical advice.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748B;">
            <h2>👈 Fill in patient details on the left</h2>
            <p>Enter clinical parameters and click <b>Predict Risk</b></p>
        </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Model Performance & Analysis")
    c1, c2 = st.columns(2)
    imgs_left  = ["model_comparison.png", "confusion_matrix.png"]
    imgs_right = ["roc_curves.png", "feature_importance.png"]
    captions_l = ["Model Comparison", "Confusion Matrix"]
    captions_r = ["ROC Curves", "Feature Importance"]
    with c1:
        for img, cap in zip(imgs_left, captions_l):
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path):
                st.image(path, caption=cap, use_container_width=True)
    with c2:
        for img, cap in zip(imgs_right, captions_r):
            path = os.path.join(OUTPUT_DIR, img)
            if os.path.exists(path):
                st.image(path, caption=cap, use_container_width=True)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        path = os.path.join(OUTPUT_DIR, "age_chol_distribution.png")
        if os.path.exists(path):
            st.image(path, caption="Age & Cholesterol Distribution", use_container_width=True)
    with c4:
        path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
        if os.path.exists(path):
            st.image(path, caption="Feature Correlation Heatmap", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Results Summary")
    perf = {
        "Model":    ["Logistic Regression","Random Forest ✅","XGBoost","Gradient Boosting","SVM","Ensemble"],
        "Accuracy": ["84.78%","84.24%","84.78%","84.78%","83.70%","84.24%"],
        "ROC-AUC":  ["0.9058","0.9177","0.9044","0.9015","0.9048","0.9138"],
        "F1 Score": ["0.8480","0.8425","0.8478","0.8478","0.8371","0.8424"],
    }
    st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🔬 About This Project
        AI-powered heart disease risk assessment using advanced ML pipeline
        trained on the UCI Heart Disease Dataset (920 samples, 4 hospitals).

        **Pipeline:**
        - ✅ 5 ML models trained and compared
        - ✅ SMOTE for class imbalance
        - ✅ 4 engineered features
        - ✅ 5-fold stratified cross-validation
        - ✅ Ensemble (RF + XGBoost + Gradient Boosting)
        - ✅ Final: Random Forest (ROC-AUC: 0.9177)
        """)
    with c2:
        st.markdown("""
        ### 👩‍💻 Developer
        **Eman Fatima**
        BS Artificial Intelligence — Semester 6 | PAF-IAST

        - 🏢 ML Intern @ ProSensia
        - 🎓 Dean's List — SGPA 3.72
        - 🤖 HR Manager @ CtrlAltCrew

        **Stack:** Python · Scikit-learn · XGBoost · SMOTE · Streamlit

        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/eman-fatima-99962230b)
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/EmanFatima00)
        """)
