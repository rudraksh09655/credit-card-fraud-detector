import streamlit as st
import numpy as np
import joblib

model = joblib.load('model/fraud_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🔒")
st.title("🔒 Credit Card Fraud Detection")
st.markdown("Enter transaction details to check if it's **fraudulent or legitimate**.")

st.sidebar.header("About This Model")
st.sidebar.info("""
- **Model:** Random Forest Classifier
- **Dataset:** 284,807 transactions
- **Technique:** SMOTE for class imbalance
- **ROC-AUC:** ~0.999
""")

st.subheader("Transaction Details")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
with col2:
    time = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=50000.0)

st.markdown("**PCA Feature Values (V1 – V28)**")
st.caption("In a real system these come from the bank. For demo, adjust the sliders.")

v_values = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        v = st.slider(f"V{i}", -5.0, 5.0, 0.0, 0.1, key=f"v{i}")
        v_values.append(v)

if st.button("🔍 Analyze Transaction", type="primary"):
    amount_scaled = scaler.transform([[amount]])[0][0]
    time_scaled = (time - 94813.0) / 47488.0
    features = np.array(v_values + [amount_scaled, time_scaled]).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **FRAUD DETECTED** — Confidence: {probability[1]*100:.1f}%")
        st.warning("This transaction has been flagged for review.")
    else:
        st.success(f"✅ **LEGITIMATE TRANSACTION** — Confidence: {probability[0]*100:.1f}%")
        st.info("Transaction appears normal.")

    col1, col2 = st.columns(2)
    col1.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
    col2.metric("Legit Probability", f"{probability[0]*100:.2f}%")

st.markdown("---")
st.caption("Built with Scikit-learn & Streamlit | Dataset: Kaggle Credit Card Fraud Detection")