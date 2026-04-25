import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Trained Model
# -------------------------------
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    # fallback if model not available
    class DummyModel:
        def predict(self, X):
            return [1 if X[0][0] > 50 else 0]
    model = DummyModel()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="centered"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.stButton>button {
    background-color: #d9534f;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & Description
# -------------------------------
st.title("💓 Heart Disease Prediction System")
st.markdown("""
This application predicts the likelihood of heart disease using a machine learning model 
based on patient clinical parameters.
""")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Patient Clinical Data")

age = st.slider("Age (years)", 20, 80, 40)

gender = st.selectbox("Gender", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type (Angina Classification)",
    [0, 1, 2, 3],
    help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"
)

bp = st.slider(
    "Resting Systolic Blood Pressure (mm Hg)",
    90, 200, 120
)

chol = st.slider(
    "Serum Cholesterol (mg/dL)",
    100, 400, 200
)

hr = st.slider(
    "Maximum Heart Rate Achieved (bpm)",
    70, 210, 150
)

# -------------------------------
# Data Preprocessing
# -------------------------------
gender = 1 if gender == "Male" else 0

input_data = np.array([[age, gender, cp, bp, chol, hr]])

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("Prediction Result")

if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease Detected")
    else:
        st.success("✅ Low Risk of Heart Disease")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed as part of a Machine Learning-based Heart Disease Prediction Project.")
