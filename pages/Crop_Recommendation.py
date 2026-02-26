import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import google.generativeai as genai
import os
import base64

# ==============================
# 🎨 Background Image Function
# ==============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    page_bg = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.1), rgba(0,0,0,0.1)),
            url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1, h2, h3, label {{
        color: white !important;
    }}

    .stButton>button {{
        border-radius: 10px;
        height: 50px;
        font-size: 16px;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# 🔥 Set background image
set_background("/workspaces/AgroTechHub/pages/CropRback.avif")

# ==============================
# 🔐 Initialize Gemini AI (SAFE)
# ==============================
from credentials import GEMINI_API_KEY

# make sure you set GEMINI_API_KEY in credentials.py
genai.configure(api_key=GEMINI_API_KEY)

def ask_gemini(question):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception:
        return "⚠️ AI response unavailable."

# ==============================
# 🚀 Load & Train Model
# ==============================
@st.cache_resource
def load_model():
    df = pd.read_csv("DATA/crop_recommendation.csv")

    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# ==============================
# 🌾 Page Content
# ==============================

st.markdown("<h1 style='color: #008000;'>🌾 AI Crop Recommendation System</h1>", unsafe_allow_html=True)
st.divider()

model, scaler = load_model()

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 150.0)
    P = st.number_input("Phosphorus (P)", 0.0, 150.0)
    K = st.number_input("Potassium (K)", 0.0, 150.0)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    ph = st.number_input("Soil pH", 0.0, 14.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0)

if st.button("🔍 Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    crop = model.predict(input_scaled)[0]
    st.success(f"✅ Recommended Crop: **{crop}**")

    years = list(range(2010, 2025))
    yields = np.random.uniform(2.5, 5.5, len(years))

    fig, ax = plt.subplots()
    ax.plot(years, yields)
    ax.set_title(f"{crop} Yield Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (T/Ha)")
    st.pyplot(fig)

    with st.expander("💡 AI Farming Tips"):
        st.write(
            ask_gemini(
                f"Give best farming practices, fertilizer advice, and climate conditions for growing {crop} in India."
            )
        )
