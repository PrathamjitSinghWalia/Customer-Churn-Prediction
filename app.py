import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model artifacts
model = joblib.load("churn_model.pkl")
threshold = joblib.load("threshold.pkl")
scaler = joblib.load("scaler.pkl")
scale_cols = joblib.load("scale_cols.pkl")

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# Custom CSS - Clean Professional Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Hide default streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header section */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .header-box p {
        color: #a0aec0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Section card */
    .section-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Result cards */
    .result-high {
        background: #fff5f5;
        border: 1px solid #fc8181;
        border-left: 5px solid #e53e3e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-medium {
        background: #fffbeb;
        border: 1px solid #f6ad55;
        border-left: 5px solid #dd6b20;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-low {
        background: #f0fff4;
        border: 1px solid #68d391;
        border-left: 5px solid #38a169;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
    }
    .result-sub {
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
        opacity: 0.8;
    }

    /* Probability display */
    .prob-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prob-number {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .prob-label {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }

    /* Segment box */
    .segment-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .segment-label {
        font-size: 0.75rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 0.3rem 0;
    }
    .segment-value {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
        border: none;
    }

    /* Input labels */
    label {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #4a5568 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-box">
    <h1>Customer Churn Predictor</h1>
    <p>Enter customer details below to assess churn risk and receive a retention strategy recommendation.</p>
</div>
""", unsafe_allow_html=True)

# Input Section
st.markdown('<div class="section-card"><div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    cltv = st.slider("Customer Lifetime Value ($)", 2000, 6500, 4000)
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

# Predict Button
predict = st.button("Run Churn Analysis", use_container_width=True)

if predict:
    # Build input
    feature_cols = model.get_booster().feature_names
    input_dict = {col: 0 for col in feature_cols}

    input_dict['Tenure Months'] = tenure
    input_dict['Monthly Charges'] = monthly_charges
    input_dict['Total Charges'] = monthly_charges * tenure
    input_dict['CLTV'] = cltv
    input_dict['Service Count'] = (
        (1 if tech_support == "Yes" else 0) +
        (1 if online_security == "Yes" else 0)
    )
    input_dict['Charges Per Month'] = monthly_charges / (tenure + 1)
    input_dict['Senior Citizen'] = 1 if senior_citizen == "Yes" else 0
    input_dict['Partner'] = 1 if partner == "Yes" else 0

    if contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1

    if internet_service == "Fiber optic":
        input_dict['Internet Service_Fiber optic'] = 1
    elif internet_service == "No":
        input_dict['Internet Service_No'] = 1

    if payment_method == "Credit card (automatic)":
        input_dict['Payment Method_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        input_dict['Payment Method_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_dict['Payment Method_Mailed check'] = 1

    if tech_support == "No internet service":
        input_dict['Tech Support_No internet service'] = 1
    elif tech_support == "Yes":
        input_dict['Tech Support_Yes'] = 1

    if online_security == "No internet service":
        input_dict['Online Security_No internet service'] = 1
    elif online_security == "Yes":
        input_dict['Online Security_Yes'] = 1

    if tenure <= 12:
        pass
    elif tenure <= 36:
        input_dict['Tenure Group_Mid'] = 1
    else:
        input_dict['Tenure Group_Loyal'] = 1

    if cltv <= 3000:
        pass
    elif cltv <= 4500:
        input_dict['CLTV Tier_Medium'] = 1
    else:
        input_dict['CLTV Tier_High'] = 1

    input_df = pd.DataFrame([input_dict])
    cols_to_scale = [c for c in scale_cols if c in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)

    # Risk level
    if prob >= 0.6:
        risk_label = "HIGH RISK"
        risk_color = "#e53e3e"
        result_class = "result-high"
        verdict = "This customer is likely to churn"
    elif prob >= 0.35:
        risk_label = "MEDIUM RISK"
        risk_color = "#dd6b20"
        result_class = "result-medium"
        verdict = "This customer shows early churn signals"
    else:
        risk_label = "LOW RISK"
        risk_color = "#38a169"
        result_class = "result-low"
        verdict = "This customer is likely to stay"

    # Results section
    st.markdown('<div class="section-card"><div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="prob-box">
            <p class="prob-label">Churn Probability</p>
            <p class="prob-number" style="color:{risk_color}">{prob:.1%}</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="prob-box">
            <p class="prob-label">Risk Level</p>
            <p class="prob-number" style="color:{risk_color};font-size:1.4rem;padding-top:0.6rem">{risk_label}</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        pred_label = "CHURN" if prediction == 1 else "STAY"
        pred_color = "#e53e3e" if prediction == 1 else "#38a169"
        st.markdown(f"""
        <div class="prob-box">
            <p class="prob-label">Prediction</p>
            <p class="prob-number" style="color:{pred_color};font-size:1.4rem;padding-top:0.6rem">{pred_label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{result_class}" style="margin-top:1rem">
        <p class="result-title">{verdict}</p>
        <p class="result-sub">Model confidence: {prob:.1%} churn probability — threshold set at {threshold}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Segment & Action
    if prob >= 0.55 and cltv > 4500:
        segment = "Priority — Retain Now"
        action = "Immediate personal outreach + premium discount offer"
        reason = "High value customer with high churn risk. Biggest revenue loss if they leave."
        seg_color = "#e53e3e"
    elif prob >= 0.55 and cltv > 3000:
        segment = "High Risk — Monitor"
        action = "Automated retention email + service upgrade offer"
        reason = "Medium value customer at risk. Worth saving with targeted intervention."
        seg_color = "#dd6b20"
    elif prob >= 0.55:
        segment = "High Risk — Low Value"
        action = "Low-cost retention only (automated chatbot offer)"
        reason = "High risk but low CLTV. Not worth expensive retention spend."
        seg_color = "#d69e2e"
    elif prob >= 0.35:
        segment = "Medium Risk — Engage"
        action = "Enroll in loyalty program + send satisfaction survey"
        reason = "Showing early risk signals. Engage proactively before risk increases."
        seg_color = "#3182ce"
    else:
        segment = "Stable — Maintain"
        action = "Regular engagement + explore upsell opportunities"
        reason = "Happy, stable customer. Focus on expanding their service portfolio."
        seg_color = "#38a169"

    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Retention Strategy</div>
        <div class="segment-box" style="border-left: 4px solid {seg_color}">
            <p class="segment-label">Customer Segment</p>
            <p class="segment-value" style="color:{seg_color}">{segment}</p>
        </div>
        <div class="segment-box">
            <p class="segment-label">Recommended Action</p>
            <p class="segment-value">{action}</p>
        </div>
        <div class="segment-box">
            <p class="segment-label">Rationale</p>
            <p class="segment-value" style="font-weight:400;color:#718096">{reason}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)