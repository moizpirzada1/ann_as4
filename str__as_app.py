import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load trained models
with open("ai_health_advisor_models.pkl", "rb") as f:
    models = pickle.load(f)

scaler = models['scaler']
mlp_model = models['mlp']

st.title("🧠 AI Health Advisor - Liver Disease Predictor")

st.markdown("""
This application uses an Artificial Neural Network to predict whether a patient has **liver disease** based on clinical and demographic data.
""")

# Input form
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 45)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.sidebar.number_input("Total Bilirubin", 0.0, 100.0, 1.2)
direct_bilirubin = st.sidebar.number_input("Direct Bilirubin", 0.0, 10.0, 0.3)
alk_phos = st.sidebar.number_input("Alkaline Phosphotase", 0, 2000, 300)
alamine_aminotransferase = st.sidebar.number_input("Alamine Aminotransferase", 0, 2000, 50)
aspartate_aminotransferase = st.sidebar.number_input("Aspartate Aminotransferase", 0, 2000, 50)
total_protiens = st.sidebar.number_input("Total Proteins", 0.0, 10.0, 6.5)
albumin = st.sidebar.number_input("Albumin", 0.0, 10.0, 3.5)
agr = st.sidebar.number_input("Albumin and Globulin Ratio", 0.0, 5.0, 1.0)

# Gender encoding
gender_encoded = 1 if gender == "Male" else 0

# Prepare input
input_data = pd.DataFrame([[
    age, gender_encoded, total_bilirubin, direct_bilirubin,
    alk_phos, alamine_aminotransferase, aspartate_aminotransferase,
    total_protiens, albumin, agr
]], columns=[
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
])

# Predict button
if st.button("🔍 Predict"):
    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = mlp_model.predict(input_scaled)[0]
    probability = mlp_model.predict_proba(input_scaled)[0][prediction]

    # Label and color
    result_text = "Liver Disease" if prediction == 1 else "No Liver Disease"
    result_color = "error" if prediction == 1 else "success"

    # Show result
    st.write(f"### 🧾 Prediction Result")
    getattr(st, result_color)(f"**{result_text}** (Confidence: {probability:.2f})")
    st.dataframe(input_data)

    # Save to history
    history_entry = input_data.copy()
    history_entry["Prediction"] = result_text
    history_entry["Confidence"] = round(probability, 2)

    if os.path.exists("history.csv"):
        history_df = pd.read_csv("history.csv")
        history_df = pd.concat([history_df, history_entry], ignore_index=True)
    else:
        history_df = history_entry

    history_df.to_csv("history.csv", index=False)

# Show history
if st.checkbox("📜 Show Prediction History"):
    if os.path.exists("history.csv"):
        st.subheader("🧾 Prediction History")
        hist_df = pd.read_csv("history.csv")
        st.dataframe(hist_df)
    else:
        st.info("No predictions have been made yet.")
