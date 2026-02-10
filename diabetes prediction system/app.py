import streamlit as st
import numpy as np
import pickle
from utils import generate_ai_explanation

st.set_page_config(page_title="AI Disease Predictor", layout="wide")

with open("models.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
features = data["features"]

st.title("🧠 AI-Powered Disease Prediction System")
st.caption("Machine Learning–based health risk analysis")

st.sidebar.header("🧾 Patient Details")

inputs = []
for feature in features:
    value = st.sidebar.number_input(feature, 0.0, 300.0, 50.0)
    inputs.append(value)

input_array = np.array([inputs])
scaled_input = scaler.transform(input_array)

if st.sidebar.button("🔍 Analyze Health Risk"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Risk Score")
        st.progress(int(probability * 100))
        st.metric("Diabetes Risk", f"{probability*100:.2f}%")

        if prediction == 1:
            st.error("⚠️ High Risk Detected")
        else:
            st.success("✅ Low Risk Detected")

    with col2:
        st.subheader("🤖 AI Explanation")
        explanation = generate_ai_explanation(
            scaled_input[0], features, model
        )
        st.markdown(explanation)

    st.info(
        "⚠️ This system is for educational purposes only and not a medical diagnosis."
    )
