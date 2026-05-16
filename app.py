import streamlit as st
import numpy as np
import joblib

# لو حفظت الموديل
model = joblib.load("model.pkl")

st.title("Heart Disease Prediction App ❤️")

st.write("Enter patient data:")

BMI = st.number_input("BMI")
Smoker = st.selectbox("Smoker", [0,1])
Stroke = st.selectbox("Stroke", [0,1])
Diabetes = st.selectbox("Diabetes", [0,1])
PhysActivity = st.selectbox("Physical Activity", [0,1])
Fruits = st.selectbox("Fruits", [0,1])
Veggies = st.selectbox("Veggies", [0,1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol", [0,1])
GenHlth = st.slider("General Health (1-5)", 1, 5)
MentHlth = st.number_input("Mental Health Days")
PhysHlth = st.number_input("Physical Health Days")
DiffWalk = st.selectbox("Difficulty Walking", [0,1])
Sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
Age = st.slider("Age", 1, 13)
Education = st.slider("Education", 1, 6)
Income = st.slider("Income", 1, 8)

if st.button("Predict"):
    input_data = np.array([[BMI, Smoker, Stroke, Diabetes, PhysActivity,
                            Fruits, Veggies, HvyAlcoholConsump,
                            GenHlth, MentHlth, PhysHlth, DiffWalk,
                            Sex, Age, Education, Income]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk of Heart Disease ❤️‍🩹")
    else:
        st.success("Low Risk of Heart Disease 💚")