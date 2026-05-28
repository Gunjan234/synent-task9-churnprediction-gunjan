import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    return model, scaler, feature_names, le_dict

model, scaler, feature_names, le_dict = load_models()

st.title("📞 Customer Churn Prediction System")
st.markdown("---")

st.write("""
### Predict whether a customer will churn (leave the company)
This model analyzes customer data and predicts the likelihood of churn.
""")

st.sidebar.header("📋 Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 24)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

if st.sidebar.button("🔮 Predict Churn", key="predict_btn"):
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support
    }
    
    input_df = pd.DataFrame([input_data])
    
    for col in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']:
        if col in le_dict:
            input_df[col] = le_dict[col].transform([input_df[col].iloc[0]])
    
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    if prediction == 1:
        with col1:
            st.error("⚠️ HIGH RISK")
        with col2:
            st.metric("Churn Probability", f"{probability[1]:.1%}")
        with col3:
            st.info("Retention action recommended!")
        
        st.warning("""
        ### Recommendation:
        This customer is at high risk of churning. Consider:
        - Discount on services
        - Loyalty rewards
        - Upgrade options
        """)
    else:
        with col1:
            st.success("✅ LOW RISK")
        with col2:
            st.metric("Churn Probability", f"{probability[1]:.1%}")
        with col3:
            st.info("Customer likely to stay")

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
**Model:** Random Forest Classifier
**Accuracy:** 82%
**Features:** 20 customer attributes
**Training Data:** 7,043 customers
""")
