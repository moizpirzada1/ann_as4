import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

# Load models
with open("models/perceptron_model.pkl", "rb") as f:
    perceptron = pickle.load(f)

with open("models/mlp_model.pkl", "rb") as f:
    mlp = pickle.load(f)

with open("models/rbf_model.pkl", "rb") as f:
    rbf = pickle.load(f)

with open("models/som_model.pkl", "rb") as f:
    som = pickle.load(f)

with open("models/bam_model.pkl", "rb") as f:
    bam = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define input fields
st.title("🧠 AI Health Advisor - Liver Disease Prediction")

st.write("Enter patient details to predict the risk of liver disease:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 45)
total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, 1.0)
direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.3)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 50, 2000, 200)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase", 0, 2000, 30)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 2000, 40)
total_proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
albumin = st.number_input("Albumin", 0.0, 6.0, 3.0)
agr = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)

if st.button("Predict"):
    gender_encoded = 1 if gender == "Male" else 0
    input_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                            alkaline_phosphotase, alamine_aminotransferase,
                            aspartate_aminotransferase, total_proteins,
                            albumin, agr]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Get predictions
    p_pred = perceptron.predict(input_scaled)[0]
    mlp_pred = mlp.predict(input_scaled)[0]
    rbf_pred = rbf.predict(input_scaled)[0]

    # SOM: find BMU (Best Matching Unit)
    som_position = som.winner(input_scaled[0])
    som_group = f"Cluster {som_position[0]}-{som_position[1]}"

    # BAM: use bipolar version
    bam_input = np.where(input_data > 0, 1, -1)
    bam_output = bam.recall(bam_input)
    bam_pred = 1 if bam_output[0][0] > 0 else 0

    result = {
        "Perceptron": "Liver Disease" if p_pred == 1 else "No Disease",
        "MLP": "Liver Disease" if mlp_pred == 1 else "No Disease",
        "RBF": "Liver Disease" if rbf_pred == 1 else "No Disease",
        "BAM": "Liver Disease" if bam_pred == 1 else "No Disease",
        "SOM Group": som_group
    }

    st.success("✅ Predictions Complete")
    st.write(result)

    # Save to history
    history = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Gender": gender,
        "Age": age,
        "Perceptron": result["Perceptron"],
        "MLP": result["MLP"],
        "RBF": result["RBF"],
        "BAM": result["BAM"],
        "SOM Cluster": result["SOM Group"]
    }

    if os.path.exists("prediction_history.csv"):
        history_df = pd.read_csv("prediction_history.csv")
        history_df = pd.concat([history_df, pd.DataFrame([history])], ignore_index=True)
    else:
        history_df = pd.DataFrame([history])

    history_df.to_csv("prediction_history.csv", index=False)

# Show history
st.markdown("---")
if st.checkbox("📜 Show Prediction History"):
    if os.path.exists("prediction_history.csv"):
        df_history = pd.read_csv("prediction_history.csv")
        st.dataframe(df_history)
    else:
        st.info("No history available yet.")
