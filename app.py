import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠", layout="wide")

# --- Title and Description ---
st.title("🏠 Real Estate Price Predictor")
st.write("""
This application predicts real estate prices based on property features using a 
**Linear Regression** model trained on the California Housing Dataset.
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'real_estate_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found. Please run `python src/model_training.py` first.")
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

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("User Input Features")
    st.write(input_df)

    # --- Prediction ---
    if model is not None:
        if st.button("Predict Price"):
            prediction = model.predict(input_df)
            predicted_price = prediction[0] * 100000  # Convert back to dollars
            
            st.success(f"### Estimated Price: ${predicted_price:,.2f}")
            
            # Simple Feature Importance (Coefficients for Linear Regression)
            if hasattr(model, 'coef_'):
                feature_names = input_df.columns
                coef = model.coef_
                importance = pd.DataFrame({'Feature': feature_names, 'Importance': coef})
                importance = importance.sort_values(by='Importance', ascending=False)
                
                st.subheader("Key Price Drivers")
                fig, ax = plt.subplots()
                ax.barh(importance['Feature'], importance['Importance'])
                ax.set_xlabel('Coefficient Impact')
                ax.set_title('How Features Influence Price')
                st.pyplot(fig)

with col2:
    st.info("""
    **About the Model:**
    - **Algorithm:** Linear Regression
    - **Dataset:** California Housing (1990 Census)
    - **R² Score:** ~0.60 (Simplified for portability)
    
    *Note: Prices are in USD. This is a demonstration model.*
    """)

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit & Scikit-Learn | [View Code on GitHub](https://github.com/JohnNessime/real-estate-price-predictor)")
