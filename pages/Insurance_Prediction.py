import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import io

@st.cache_resource
def load_model_and_encoders(csv_file_content):
    # ✅ Load CSV with proper encoding
    try:
        data = pd.read_csv(io.StringIO(csv_file_content), encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(io.StringIO(csv_file_content), encoding="ISO-8859-1")

    # ✅ Clean Column Names (Remove spaces and weird characters)
    data.columns = data.columns.str.strip().str.replace("Â", "")

    # ✅ Rename Temperature column safely
    for col in data.columns:
        if "Temperature" in col:
            data.rename(columns={col: "Temperature_C"}, inplace=True)
            break
    if "Temperature_C" not in data.columns:
        st.error("🚨 Critical Error: 'Temperature (°C)' column is missing! Check CSV format.")
        st.stop()

    # ✅ Encode categorical features
    label_encoders = {}
    for column in ['Crop Type', 'Location', 'Damage Type']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # ✅ Scale numerical features
    numerical_features = ['Rainfall (mm)', 'Temperature_C', 'Claim Amount ($)']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # ✅ Define Features (X) and Target (y)
    X = data.drop('Claim Approval', axis=1)
    y = data['Claim Approval'].map({'Yes': 1, 'No': 0})  # Convert labels to binary

    # ✅ Train Model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)

    return model, label_encoders, scaler, X.columns

# ✅ Load Data (Ensure File Exists)
try:
    with open("DATA/Insurance_prediction.csv", "r") as f:
        csv_file_content = f.read()
except FileNotFoundError:
    st.error("🚨 Error: 'Insurance_prediction.csv' not found! Upload the correct file.")
    st.stop()

# ✅ Load Model & Preprocessing Data
model, label_encoders, scaler, feature_names = load_model_and_encoders(csv_file_content)

# ✅ Streamlit UI

st.markdown("<h1 style='text-align: center; color: #008000;'> 🌾 Crop Insurance Claim Prediction </h1>", unsafe_allow_html=True)
st.divider()

# ✅ Input Form
crop_type = st.selectbox("Crop Type", list(label_encoders['Crop Type'].classes_))  
location = st.selectbox("Location", list(label_encoders['Location'].classes_))  
rainfall = st.number_input("Rainfall (mm)", value=0.0)
temperature = st.number_input("Temperature (°C)", value=0.0)
damage_type = st.selectbox("Damage Type", list(label_encoders['Damage Type'].classes_))  
claim_amount = st.number_input("Claim Amount ($)", value=0.0)

# ✅ Prediction Button
if st.button("Predict Claim Approval"):
    # ✅ Preprocess Input Data
    input_data = pd.DataFrame({
        'Crop Type': [crop_type],
        'Location': [location],
        'Rainfall (mm)': [rainfall],
        'Temperature_C': [temperature],  # Correct column name
        'Damage Type': [damage_type],
        'Claim Amount ($)': [claim_amount]
    })

    # ✅ Encode Categorical Features
    input_data['Crop Type'] = label_encoders['Crop Type'].transform(input_data['Crop Type'])
    input_data['Location'] = label_encoders['Location'].transform(input_data['Location'])
    input_data['Damage Type'] = label_encoders['Damage Type'].transform(input_data['Damage Type'])

    # ✅ Scale Numerical Features
    numerical_features = ['Rainfall (mm)', 'Temperature_C', 'Claim Amount ($)']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # ✅ Ensure Input Data Matches Training Columns
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # ✅ Make Prediction
    prediction = model.predict(input_data)[0]
    claim_approval = "✅ Yes" if prediction == 1 else "❌ No"

    # ✅ Display Final Result
    st.subheader(f"📢 Predicted Claim Approval: {claim_approval}")  