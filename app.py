import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠")

# --- Title and Description ---
st.title("🏠 Real Estate Price Predictor")
st.write("""
This application predicts real estate prices based on various property features. 
The model was trained on the California Housing Dataset using a Random Forest Regressor.
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'real_estate_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please run src/model_training.py first.")
        return None

model = load_model()

# --- Sidebar: User Inputs ---
st.sidebar.header("Property Features")

def user_input_features():
    MedInc = st.sidebar.slider('Median Income (in $10,000s)', 0.5, 15.0, 3.5)
    HouseAge = st.sidebar.slider('House Age (years)', 0, 52, 25)
    AveRooms = st.sidebar.slider('Average Rooms', 1.0, 15.0, 5.0)
    AveBedrms = st.sidebar.slider('Average Bedrooms', 0.5, 5.0, 1.0)
    Population = st.sidebar.slider('Population', 3, 35000, 1000)
    AveOccup = st.sidebar.slider('Average Occupancy', 1.0, 10.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 37.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -122.0)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Display User Inputs ---
st.subheader("User Input Features")
st.write(input_df)

# --- Prediction ---
if model is not None:
    if st.button("Predict Price"):
        prediction = model.predict(input_df)
        predicted_price = prediction[0] * 100000  # Convert back to dollars
        
        st.success(f"### Estimated Price: ${predicted_price:,.2f}")
        
        st.info("""
        **Note:** This is a machine learning estimate based on the California Housing Dataset. 
        Actual market prices may vary based on location-specific factors, condition, and market trends.
        """)
else:
    st.warning("Please ensure the model is trained and saved in the 'models' folder.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit & Scikit-Learn | [View Code on GitHub](https://github.com/JohnNessime/real-estate-price-predictor)")