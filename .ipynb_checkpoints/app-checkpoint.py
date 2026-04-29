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

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details to predict churn risk and get retention recommendations.")
st.divider()

# Input Form
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

with col2:
    cltv = st.slider("CLTV", 2000, 6500, 4000)
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])

st.divider()

if st.button("Predict Churn Risk", use_container_width=True):

    # Build input dict with EXACT column names model was trained on
    feature_cols = model.get_booster().feature_names
    input_dict = {col: 0 for col in feature_cols}

    # Numeric features
    input_dict['Tenure Months'] = tenure
    input_dict['Monthly Charges'] = monthly_charges
    input_dict['Total Charges'] = monthly_charges * tenure
    input_dict['CLTV'] = cltv
    input_dict['Service Count'] = (
        (1 if tech_support == "Yes" else 0) +
        (1 if online_security == "Yes" else 0)
    )
    input_dict['Charges Per Month'] = monthly_charges / (tenure + 1)

    # Demographics
    input_dict['Senior Citizen'] = 1 if senior_citizen == "Yes" else 0
    input_dict['Partner'] = 1 if partner == "Yes" else 0

    # Contract
    if contract == "One year":
        input_dict['Contract_One year'] = 1
    elif contract == "Two year":
        input_dict['Contract_Two year'] = 1

    # Internet Service
    if internet_service == "Fiber optic":
        input_dict['Internet Service_Fiber optic'] = 1
    elif internet_service == "No":
        input_dict['Internet Service_No'] = 1

    # Payment Method
    if payment_method == "Credit card (automatic)":
        input_dict['Payment Method_Credit card (automatic)'] = 1
    elif payment_method == "Electronic check":
        input_dict['Payment Method_Electronic check'] = 1
    elif payment_method == "Mailed check":
        input_dict['Payment Method_Mailed check'] = 1

    # Tech Support
    if tech_support == "No internet service":
        input_dict['Tech Support_No internet service'] = 1
    elif tech_support == "Yes":
        input_dict['Tech Support_Yes'] = 1

    # Online Security
    if online_security == "No internet service":
        input_dict['Online Security_No internet service'] = 1
    elif online_security == "Yes":
        input_dict['Online Security_Yes'] = 1

    # Tenure Group
    if tenure <= 12:
        pass
    elif tenure <= 36:
        input_dict['Tenure Group_Mid'] = 1
    else:
        input_dict['Tenure Group_Loyal'] = 1

    # CLTV Tier
    if cltv <= 3000:
        pass
    elif cltv <= 4500:
        input_dict['CLTV Tier_Medium'] = 1
    else:
        input_dict['CLTV Tier_High'] = 1

    # Create dataframe and scale
    input_df = pd.DataFrame([input_dict])
    cols_to_scale = [c for c in scale_cols if c in input_df.columns]
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= threshold)

    # Results
    st.subheader("Prediction Result")

    if prob >= 0.6:
        color = "RED"
        risk_label = "HIGH RISK"
    elif prob >= 0.35:
        color = "YELLOW"
        risk_label = "MEDIUM RISK"
    else:
        color = "GREEN"
        risk_label = "LOW RISK"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob:.1%}")
    with col2:
        st.metric("Risk Level", risk_label)

    st.progress(float(prob))

    if prediction == 1:
        st.error("This customer is predicted to CHURN")
    else:
        st.success("This customer is predicted to STAY")

    st.divider()
    st.subheader("Business Segment and Recommended Action")

    if prob >= 0.55 and cltv > 4500:
        segment = "Priority - Retain Now"
        action = "Immediate personal outreach + premium discount offer"
        reason = "High value customer with high churn risk - biggest revenue loss if they leave"
    elif prob >= 0.55 and cltv > 3000:
        segment = "High Risk - Monitor"
        action = "Send automated retention email + service upgrade offer"
        reason = "Medium value customer at risk - worth saving with low-cost intervention"
    elif prob >= 0.55:
        segment = "High Risk - Low Value"
        action = "Low-cost retention only (automated chatbot offer)"
        reason = "High risk but low CLTV - not worth expensive retention spend"
    elif prob >= 0.35:
        segment = "Medium Risk - Engage"
        action = "Enroll in loyalty program + send satisfaction survey"
        reason = "Showing early risk signals - engage before risk increases"
    else:
        segment = "Stable - Maintain"
        action = "Regular engagement + explore upsell opportunities"
        reason = "Happy customer - focus on expanding their services"

    st.info("Segment: " + segment)
    st.warning("Recommended Action: " + action)
    st.caption("Why: " + reason)