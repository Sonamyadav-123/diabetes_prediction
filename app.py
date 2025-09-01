import streamlit as st
import pandas as pd
import joblib

# ----------------- Load model, encoders and scaler -----------------
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')  # Only for gender and smoking_history
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("Diabetes Prediction App 🩺")
st.write("Enter your details below to predict diabetes risk:")

# ----------------- Input Fields -----------------
age = st.number_input("Age", 1, 120, 25)
gender = st.selectbox("Gender", ["Male", "Female"])
hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current", "No Info"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 20.0, 5.5)
blood_glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

# ----------------- Create DataFrame -----------------
input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking_history],
    "bmi": [bmi],
    "HbA1c_level": [hba1c],
    "blood_glucose_level": [blood_glucose]
})

# ----------------- Encode categorical columns -----------------
for col in ["gender", "smoking_history"]:
    input_data[col] = encoders[col].transform(input_data[col])

# ----------------- Scale numerical features -----------------
numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# ----------------- Ensure columns order matches training -----------------
final_cols = ["gender", "age", "hypertension", "heart_disease",
              "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
input_data = input_data[final_cols]

# ----------------- Prediction -----------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Prediction: Positive for Diabetes\nRisk Probability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Prediction: Negative for Diabetes\nRisk Probability: {prob*100:.2f}%")
