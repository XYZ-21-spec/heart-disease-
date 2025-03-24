import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained LSTM model
model = load_model('lstm_heart_model.h5')  # Make sure this file exists in the same folder

st.title("❤️ Heart Disease Prediction (LSTM Model)")

# User input (13 features)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", [0, 1])  # 0 = female, 1 = male
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])  # Adjust based on encoding

if st.button("Predict"):
    # Create input array with shape (1, 1, 13) for LSTM
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]], dtype=np.float32)
    input_data = input_data.reshape((1, 1, 13))  # (samples, time_steps, features)

    # Make prediction
    prediction = model.predict(input_data)
    result = int(prediction[0][0] > 0.5)  # Convert probability to 0 or 1

    if result == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
