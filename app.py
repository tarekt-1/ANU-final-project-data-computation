import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import os

st.title("Heart Disease Prediction App ❤️")

# Load model and metadata
model_exists = os.path.exists("svm_model.pkl")

if not model_exists:
    st.error("❌ Model not found!")
    st.info("""
    **اتبع هذه الخطوات:**
    
    1. افتح ملف `project.ipynb` الـ notebook
    2. شغّل جميع الخلايا من الأعلى للأسفل (Cell → Run All)
    3. تأكد من طباعة رسالة "✓ Model saved with 21 features" في الخلية الأخيرة
    4. بعدها ارجع إلى هنا
    
    """)
    st.stop()

try:
    model = joblib.load("svm_model.pkl")
    
    # Try to load metadata if available
    if os.path.exists("model_metadata.pkl"):
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        expected_features = metadata.get('feature_names', [])
        n_features = metadata.get('n_features', 0)
    else:
        expected_features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                           'Diabetes', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                           'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 
                           'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
        n_features = 21
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.write("Enter patient data:")

HighBP = st.selectbox("High Blood Pressure", [0,1])
HighChol = st.selectbox("High Cholesterol", [0,1])
CholCheck = st.selectbox("Cholesterol Check (last 5 yrs)", [0,1])
BMI = st.number_input("BMI", value=25.0)
Smoker = st.selectbox("Smoker", [0,1])
Stroke = st.selectbox("Stroke", [0,1])
Diabetes = st.selectbox("Diabetes", [0,1])
PhysActivity = st.selectbox("Physical Activity", [0,1])
Fruits = st.selectbox("Fruits", [0,1])
Veggies = st.selectbox("Veggies", [0,1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol", [0,1])
AnyHealthcare = st.selectbox("Any Healthcare", [0,1])
NoDocbcCost = st.selectbox("No Doctor due to Cost", [0,1])
GenHlth = st.slider("General Health (1-5)", 1, 5)
MentHlth = st.number_input("Mental Health Days", min_value=0, value=0)
PhysHlth = st.number_input("Physical Health Days", min_value=0, value=0)
DiffWalk = st.selectbox("Difficulty Walking", [0,1])
Sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
Age = st.slider("Age", 1, 13)
Education = st.slider("Education", 1, 6)
Income = st.slider("Income", 1, 8)

if st.button("Predict"):
    try:
        # Create DataFrame with exact column names and order from training data
        input_data = pd.DataFrame({
            'HighBP': [float(HighBP)],
            'HighChol': [float(HighChol)],
            'CholCheck': [float(CholCheck)],
            'BMI': [float(BMI)],
            'Smoker': [float(Smoker)],
            'Stroke': [float(Stroke)],
            'Diabetes': [float(Diabetes)],
            'PhysActivity': [float(PhysActivity)],
            'Fruits': [float(Fruits)],
            'Veggies': [float(Veggies)],
            'HvyAlcoholConsump': [float(HvyAlcoholConsump)],
            'AnyHealthcare': [float(AnyHealthcare)],
            'NoDocbcCost': [float(NoDocbcCost)],
            'GenHlth': [float(GenHlth)],
            'MentHlth': [float(MentHlth)],
            'PhysHlth': [float(PhysHlth)],
            'DiffWalk': [float(DiffWalk)],
            'Sex': [float(Sex)],
            'Age': [float(Age)],
            'Education': [float(Education)],
            'Income': [float(Income)]
        })
        
        # Ensure columns are in correct order if metadata is available
        if expected_features:
            input_data = input_data[expected_features]

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease ❤️‍🩹")
        else:
            st.success("✓ Low Risk of Heart Disease 💚")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.warning(f"Expected {n_features} features, got {input_data.shape[1] if 'input_data' in locals() else 'unknown'}")
