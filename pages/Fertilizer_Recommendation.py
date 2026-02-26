import streamlit as st
import pickle
import google.generativeai as genai
import os
from google.api_core import exceptions
import time

# 🎯 Configure Gemini API Key
# Try to get API key from environment, then credentials.py, then streamlit secrets
api_key = os.getenv("AIzaSyAumH1szAgvZbc7M6oqxIRqELE-gT0JACo")
if not api_key:
    try:
        from credentials import GEMINI_API_KEY
        api_key = GEMINI_API_KEY
    except ImportError:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    st.error("❌ Gemini API key not configured. Please set GEMINI_API_KEY environment variable or in credentials.py")
    st.stop()

genai.configure(api_key=api_key)

# 🌾 Available Crop Types (Dropdown options)
crop_types = [
    'Rice', 'Jowar(Sorghum)', 'Barley(JAV)', 'Maize', 'Ragi( naachnnii)',
    'Chickpeas(Channa)', 'French Beans(Farasbi)', 'Fava beans (Papdi - Val)',
    'Lima beans(Pavta)', 'Cluster Beans(Gavar)'
]

# ================================
# 🌱 Streamlit UI
# ================================

st.set_page_config(page_title="Fertilizer Recommendation", )

# 🎨 Title & Description
st.markdown("<h1 style='text-align: center; color: #008000;'>🌿 Fertilizer Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter soil parameters to get the best fertilizer recommendation for your crop.</p>", unsafe_allow_html=True)
st.divider()

# 📌 Crop Selection Dropdown
crop = st.selectbox("🔽 Select a Crop Type", crop_types)

# 🧪 Soil Parameter Inputs
col1, col2 = st.columns(2)
with col1:
    nitrogen = st.number_input("🌱 Enter Nitrogen (N) value", min_value=0.0, format="%.2f")
    potassium = st.number_input("🍀 Enter Potassium (K) value", min_value=0.0, format="%.2f")

with col2:
    phosphorus = st.number_input("🌾 Enter Phosphorus (P) value", min_value=0.0, format="%.2f")
    ph = st.number_input("🧪 Enter Soil pH Level", min_value=0.0, format="%.2f")
    # 🌍 Language Selection
languages = ["English", "Hindi", "Marathi", "Gujarati", "Tamil"]

language = st.selectbox("🌍 Select Insight Language", languages)


# ================================
# 🚀 Gemini AI Fertilizer Recommendation
# ================================
def get_fertilizer_recommendation(crop, nitrogen, phosphorus, potassium, ph, language):
    """Queries Gemini AI for fertilizer recommendation with error handling and retry logic."""
    prompt = (
        f"Crop: {crop}\n"
        f"Soil Nitrogen (N): {nitrogen}\n"
        f"Soil Phosphorus (P): {phosphorus}\n"
        f"Soil Potassium (K): {potassium}\n"
        f"Soil pH: {ph}\n"
        "\nBased on these soil conditions, recommend the best fertilizers to improve crop yield."
        f"Provide the recommendation in {language}."
    )

    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")  # Using Gemini 1.5 Flash model
            response = model.generate_content(prompt)
            return response.text if response else "No recommendation received."
        except exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                st.warning(f"⏳ API quota exceeded. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                return "❌ API quota exceeded. Please try again later."
        except exceptions.Unauthenticated:
            return "❌ Authentication failed. Please check your API key."
        except exceptions.PermissionDenied:
            return "❌ Permission denied. Your API key may not have access to this model."
        except exceptions.DeadlineExceeded:
            if attempt < max_retries - 1:
                st.warning(f"⏳ Request timeout. Retrying... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                return "❌ Request timeout. Please try again."
        except exceptions.GoogleAPIError as e:
            return f"❌ API Error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
    
    return "❌ Failed to get recommendation after multiple attempts."

# 📌 Generate Recommendation Button
if st.button("📊 Get Fertilizer Recommendation"):
    if nitrogen and phosphorus and potassium and ph:
        recommendation = get_fertilizer_recommendation(crop, nitrogen, phosphorus, potassium, ph, language)
        
        # 🎯 Display Recommendation in a Styled Card
        st.markdown(
            f"""
            <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 10px; 
            border-left: 5px solid #28a745; font-weight: bold;">
            ✅ <strong>Fertilizer Recommendation for {crop}</strong> <br>{recommendation}
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Please enter valid values for all soil parameters!")

# ================================
# 💾 Save Model Option (Optional)
# ===============================
