import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data():
    """Load the cleaned dataset."""
    # Navigate from src/ to data/
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'housing_clean.csv')
    df = pd.read_csv(data_path)
    return df

def prepare_data(df):
    """Separate features and target variable."""
    # Drop the target column from features
    X = df.drop('Price', axis=1)
    y = df['Price']
    return X, y

def train_model(X, y):
    """Train a Random Forest Regressor."""
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Complete!")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model

def save_model(model):
    """Save the trained model to a file."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'real_estate_model.pkl')
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    df = load_data()
    
    # 2. Prepare Data
    print("Preparing data...")
    X, y = prepare_data(df)
    
    # 3. Train Model
    print("Training model...")
    model = train_model(X, y)
    
    # 4. Save Model
    print("Saving model...")
    save_model(model)