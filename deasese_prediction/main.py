import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ⬇️ Load model and label encoder
model = joblib.load("./random_forest_disease_model.pkl")
label_encoder = joblib.load("./label_encoder.pkl")

# 🧠 Load symptom list from training dataset (replace with your CSV if needed)
# You can also hardcode it if you prefer
symptoms = [col for col in pd.read_csv("../Dataset/data.csv", nrows=1).columns if col != 'diseases']

# 🖥️ UI Title
st.title("🩺 Disease Prediction System")
st.markdown("Select the symptoms the patient is experiencing, and we’ll predict the likely disease.")

# 🤒 Symptom Selection
selected_symptoms = st.multiselect("Choose symptoms:", symptoms)

# 🔍 Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Convert selected symptoms to binary feature vector
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

        # Predict disease
        prediction = model.predict([input_vector])
        predicted_disease = label_encoder.inverse_transform(prediction)[0]

        # 🎯 Output
        st.success(f"🔬 **Predicted Disease:** {predicted_disease}")
