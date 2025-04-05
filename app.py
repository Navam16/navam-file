
import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Retention Predictor")

# Input fields
marital_status = st.selectbox("Marital Status", ["single", "married", "widowed", "divorced", "fact union"])
application_mode = st.selectbox("Application Mode", ["online", "in-person", "other"])
application_order = st.number_input("Application Order", min_value=1)
course = st.selectbox("Course", ["Engineering", "Management", "Others"])
attendance = st.radio("Attendance", ["daytime", "evening"])
prev_qualification = st.selectbox("Previous Qualification", ["high school", "bachelor", "other"])
nationality = st.selectbox("Nationality", ["local", "foreign", "other"])
gender = st.radio("Gender", ["Male", "Female"])

if st.button("Predict"):
    input_dict = {
        "marital_status": marital_status,
        "application_mode": application_mode,
        "application_order": application_order,
        "course": course,
        "attendance": attendance,
        "prev_qualification": prev_qualification,
        "nationality": nationality,
        "gender": gender,
    }

    # Convert to DataFrame and process
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    
    for col in model.feature_names_in_:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model.feature_names_in_]
    input_scaled = scaler.transform(input_encoded)
    
    prediction = model.predict(input_scaled)[0]
    st.success(f"The student is predicted to: **{prediction}**")
