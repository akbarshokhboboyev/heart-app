import streamlit as st
import pickle
import pandas as pd

model_path = 'xgboost_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.title("Heart Disease Predictor")


age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
thalachh = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=100)
exang = st.selectbox("Exercise-induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x - 1])


if st.button("Predict Heart Disease"):
    user_input = [
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalachh, exang, oldpeak, slope, ca, thal
    ]

   
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalachh", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    input_df = pd.DataFrame([user_input], columns=features)

 
    result = model.predict(input_df)

 
    if result[0] == 1:
        st.warning("Prognoz: Yurak kasalligi xavfi yuqori!")
    else:
        st.success("Prognoz: Yurak kasalligi xavfi past!")
