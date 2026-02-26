import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# ================================
# 🚀 Configure Google Gemini API
# ================================
GEMINI_API_KEY = "AIzaSyDMFdeC5sxov55qvRGce9Rc5RiHr2jLqOM"  # Add your API Key
genai.configure(api_key=GEMINI_API_KEY)

# ================================
# 🚀 Load and Prepare Dataset
# ================================
df = pd.read_csv("DATA\Expanded_Crop_price.csv")

# Ensure 'Price per kg' exists
if 'Price per kg' not in df.columns:
    st.error("Error: 'Price per kg' column is missing.")
    st.stop()

# Convert 'Month' column to numerical format
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    'july': 7, 'sept': 9
}
df['Month'] = df['Month'].map(month_mapping)

# Define features and target variable
X = df.drop('Price per kg', axis=1)
y = df['Price per kg']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['Vegetable', 'Season', 'Vegetable condition', 'Deasaster Happen in last 3month']
numerical_features = ['Month', 'Farm size'] if 'Farm size' in X_train.columns else ['Month']

# Ensure numerical features exist
missing_numerical_features = [feature for feature in numerical_features if feature not in X_train.columns]
if missing_numerical_features:
    st.error(f"Error: Missing numerical features: {missing_numerical_features}")
    st.stop()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Train model
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# 🚀 AI Insights Function
# ================================
def get_gemini_insights(prompt):
    """Fetch AI insights from Gemini API."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No insights available."
    except Exception as e:
        return f"Error fetching AI insights: {str(e)}"

# ================================
# 🚀 Streamlit UI
# ================================
st.set_page_config(page_title="AgroTechHub", layout="wide")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🌾 Crop Prediction", "📊 Crop Trends"])


st.markdown("<h1 style='text-align: center; color: #008000;'> 🌾 Crop Price Prediction & AI Insights </h1>", unsafe_allow_html=True)
st.divider()

if page == "🌾 Crop Prediction":
    st.subheader("💡 Get Smart Predictions for Your Crops")

    # Store user input in session state
    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = ""

    col1, col2 = st.columns(2)

    with col1:
        vegetable = st.selectbox("🌱 Select Vegetable", df['Vegetable'].unique())
        season = st.selectbox("🗓️ Select Season", df['Season'].unique())

    with col2:
        condition = st.selectbox("🥦 Select Vegetable Condition", df['Vegetable condition'].unique())
        disaster = st.selectbox("🌍 Any Disaster in Last 3 Months", df['Deasaster Happen in last 3month'].unique())

    month_name = st.selectbox("📅 Select Month", list(month_mapping.keys()))
    month = month_mapping[month_name]

    city = st.text_input("📍 Enter your City")
    state = st.text_input("🏛️ Enter your State")

    # Predict Crop Price
    if st.button("💰 Predict Crop Price"):
        input_data = pd.DataFrame({
            'Vegetable': [vegetable],
            'Season': [season],
            'Vegetable condition': [condition],
            'Deasaster Happen in last 3month': [disaster],
            'Month': [month]
        })

        input_data = preprocessor.transform(input_data)
        predicted_price = model.predict(input_data)[0]

        st.success(f"💰 **Predicted Price for {vegetable}:** ₹{predicted_price:.2f} per kg")
        st.session_state.predicted_price = predicted_price

    # AI Insights Button
    if st.button("🔍 Get AI-Powered Insights"):
        if city.strip() and state.strip():
            with st.spinner("Fetching AI insights..."):
                prompt = f"Provide market trends and future insights for {vegetable} in {city}, {state}."
                st.session_state.ai_insights = get_gemini_insights(prompt)
                st.success("✅ AI Insights Fetched Successfully!")
        else:
            st.warning("⚠️ Please enter your City and State before fetching insights.")

    # Display AI Insights
    if st.session_state.ai_insights:
        st.markdown(
            f"""
            <div style="background-color:#000000;padding:15px;border-radius:10px;border:1px solid #ddd;">
                <h4>📢 AI-Powered Market Insights</h4>
                <p>{st.session_state.ai_insights}</p>
            </div>
            """, unsafe_allow_html=True
        )

elif page == "📊 Crop Trends":
    st.subheader("📊 Crop Price Trends Over Time")

    selected_crop = st.selectbox("🌿 Select Crop for Trend Analysis", df['Vegetable'].unique())

    # Filter data for the selected crop
    crop_data = df[df['Vegetable'] == selected_crop].groupby("Month")["Price per kg"].mean()

    # Plot the trend
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(crop_data.index, crop_data.values, marker='o', linestyle='-', color='green')
    ax.set_title(f"📈 Price Trend for {selected_crop}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price per kg (₹)")
    ax.grid(True)

    st.pyplot(fig)