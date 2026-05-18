# 🏠 Real Estate Price Predictor

A machine learning application that predicts real estate prices based on property features. Built with **Python**, **Scikit-Learn**, and **Streamlit**, this project demonstrates a full end-to-end data science workflow from data exploration to model deployment.

## 🚀 Live Demo
🔗 **[Try the Live App](https://jn-real-estate-price-predictor.streamlit.app/)**

*Click the link above to interact with the model directly in your browser.*

## 📋 Project Overview
The goal of this project is to provide an intuitive interface for estimating housing prices. Using the **California Housing Dataset**, a Random Forest Regressor was trained to achieve an **R² Score of 0.805**, explaining over 80% of the price variance.

### Key Features
- **Interactive UI**: User-friendly sliders for inputting property details.
- **Robust Modeling**: Utilizes ensemble learning (Random Forest) for high accuracy.
- **Clean Architecture**: Separated logic for data processing, training, and inference.
- **Reproducible**: Includes notebooks for EDA and scripts for automated training.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Environment**: Virtualenv, Git

## 📂 Project Structure
```text
real-estate-price-predictor/
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
├── data/                   # Cleaned datasets
│   └── housing_clean.csv
├── models/                 # Trained model artifacts
│   └── real_estate_model.pkl
├── notebooks/              # Exploratory Data Analysis
│   └── 01_eda_and_data_processing.ipynb
└── src/                    # Source code modules
    ├── __init__.py
    └── model_training.py   # Model training and evaluation logic
   `````
## 📊 Model Performance
The model was trained on the **California Housing Dataset**.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | 0.805 | Explains ~80.5% of price variance |
| **RMSE** | 0.505 | Avg error approx. $50,500 |

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JohnNessime/real-estate-price-predictor.git
   cd real-estate-price-predictor
   `````
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   (On Windows use: venv\Scripts\activate)
   `````
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   `````
### Usage
1. Train the Model (Optional):
   ```bash
   python src/model_training.py
   `````
2. Run the Application:
   ```bash
   streamlit run app.py
   `````
The app will open automatically in your default web browser.
## 📈 Future Roadmap
Multi-Region Support: Integrate datasets from different global markets.
Advanced Tuning: Implement GridSearchCV for hyperparameter optimization.
Explainability: Add SHAP values to explain individual predictions.
Cloud Deployment: Deploy on Google Cloud Run or Streamlit Community Cloud.
## License
Developed by John Nessime
