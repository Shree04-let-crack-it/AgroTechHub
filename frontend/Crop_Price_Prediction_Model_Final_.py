import pandas as pd
import numpy as np
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
# 🚀 Load and Prepare the Dataset
# ================================
df = pd.read_csv("DATA/Expanded_Crop_price.csv")

# Ensure 'Price per kg' exists
if 'Price per kg' not in df.columns:
    raise ValueError("Error: 'Price per kg' column is missing from the dataset.")

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
    raise ValueError(f"Error: Missing numerical features: {missing_numerical_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Fit and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model training completed successfully!")

# ================================
# 🤖 Gemini AI for Deep Insights
# ================================
def get_gemini_insights(prompt):
    """Fetch deep AI insights with interactive follow-up questions."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No insights available."
    except Exception as e:
        return "Error fetching AI insights."

# ================================
# 🚀 Predict & Provide Actionable Plan
# ================================
def predict_crop_price():
    print("\n💬 Enter details to predict crop price and receive expert advice\n")

    # Dropdown options
    vegetable_options = df['Vegetable'].unique().tolist()
    season_options = df['Season'].unique().tolist()
    condition_options = df['Vegetable condition'].unique().tolist()
    disaster_options = df['Deasaster Happen in last 3month'].unique().tolist()
    years = list(range(2015, 2035))

    # Collect details from farmer
    vegetable = input(f"🌱 Select Vegetable {vegetable_options}: ").strip()
    season = input(f"🗓️ Select Season {season_options}: ").strip()
    condition = input(f"🥦 Select Vegetable Condition {condition_options}: ").strip()
    disaster = input(f"🌍 Any Disaster in Last 3 Months {disaster_options}: ").strip()
    year = int(input(f"📅 Select Year {years}: ").strip())

    month_name = input("📅 Enter Month (e.g., jan, feb, mar, apr): ").strip().lower()
    month = month_mapping.get(month_name, None)
    if month is None:
        print(f"⚠️ Invalid month '{month_name}', defaulting to January.")
        month = 1

    city = input("📍 Enter your city: ").strip()
    state = input("🏛️ Enter your state: ").strip()
    previous_price = float(input("💰 What was the previous selling price per kg? (Enter 0 if unknown): "))

    # Additional Market & Storage Details
    farming_method = input("🌿 Type of Farming (Organic/Conventional): ").strip()
    irrigation = input("💧 Irrigation Method (Drip/Sprinkler/Canal/Borewell): ").strip()
    farm_size = input("🚜 Farm Size (Small/Medium/Large): ").strip()
    storage_option = input("📦 Do you have cold storage available? (Yes/No): ").strip()

    # Prepare input data
    input_data = pd.DataFrame({
        'Vegetable': [vegetable],
        'Season': [season],
        'Vegetable condition': [condition],
        'Deasaster Happen in last 3month': [disaster],
        'Month': [month]
    })

    input_data = preprocessor.transform(input_data)
    predicted_price = model.predict(input_data)[0]

    print(f"\n💰 **Predicted Crop Price for {vegetable} in {year}:** ₹{predicted_price:.2f} per kg\n")

    # ================================
    # 📌 Personalized Action Plan
    # ================================
    action_plan = f"""
    📢 **Farmer Action Plan for {vegetable} in {city}, {state}:**

    ✅ **Market Trend Analysis:** Previous price ₹{previous_price}, Predicted price ₹{predicted_price:.2f}
    ✅ **Storage Advice:** {'Use cold storage' if storage_option.lower() == 'yes' else 'Sell immediately to avoid losses'}
    ✅ **Profit Maximization:** Explore direct selling via eNAM or Farmer Producer Organizations (FPOs)
    ✅ **Irrigation Suggestion:** Recommended: {irrigation} system for better yield
    """

    print(action_plan)

    # ================================
    # 🤖 Get Detailed Insights from Gemini
    # ================================
    more_details = input("\n🔍 Want detailed AI insights? (yes/no): ").strip().lower()
    if more_details == "yes":
        ai_insights = get_gemini_insights(f"Give deep market insights for {vegetable} in {city}, {state}.")
        print("\n📢 **Deep Market Insights from AI:**")
        print(ai_insights)

# Run prediction
predict_crop_price()