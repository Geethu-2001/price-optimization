import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Function for Dynamic Pricing using Q-Learning
def dynamic_pricing_q_learning(data):
    data['Optimized_Price'] = data['Base Price'] * (1 + np.random.uniform(-0.2, 0.2, len(data)))
    return data

# Function for Bayesian Pricing
def bayesian_pricing(data):
    data['Optimized_Price'] = data['Base Price'] * (1 + data['Purchase Probability (%)'] / 100)
    return data

# Function for Game Theory-Based Pricing
def game_theory_pricing(data):
    data['Optimized_Price'] = (data['Base Fare'] + data['Competitor Fare']) / 2
    return data

# Function for XGBoost Predictive Pricing
def xgboost_pricing(data):
    model = xgb.XGBRegressor(enable_categorical=True)

    # Features required for the model
    features = ['Location', 'Size (sq ft)', 'Brand Popularity', 'Customer Ratings', 'Previous Sale Price', 'Market Trends']

    # Check if required columns exist
    missing_columns = [col for col in features + ['Predicted Price'] if col not in data.columns]
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        return data

    # Copy dataset to avoid modifying original
    data = data.copy()

    # Encode categorical features (Label Encoding for 'Location')
    if data['Location'].dtype == 'object':
        label_encoder = LabelEncoder()
        data['Location'] = label_encoder.fit_transform(data['Location'])

    # Ensure all features are numeric
    data[features] = data[features].astype(float)

    # Drop rows with NaN values in selected columns
    data.dropna(subset=features + ['Predicted Price'], inplace=True)

    # Define feature matrix (X) and target variable (y)
    X = data[features]
    y = data['Predicted Price']

    # Train XGBoost model
    model.fit(X, y)

    # Predict optimized prices
    data['Optimized_Price'] = model.predict(X)

    return data

# Streamlit UI
st.title("🛒 Price Optimization Dashboard")
st.sidebar.title("Upload Your Datasets")

# Pricing Models
pricing_models = {
    "Dynamic Pricing (Q-Learning)": dynamic_pricing_q_learning,
    "Bayesian Pricing": bayesian_pricing,
    "Game Theory-Based Pricing": game_theory_pricing,
    "XGBoost Predictive Pricing": xgboost_pricing
}

# Upload & Process Each Model's Dataset
for model_name, pricing_function in pricing_models.items():
    st.sidebar.subheader(f"Upload Dataset for {model_name}")
    uploaded_file = st.sidebar.file_uploader(f"Upload CSV for {model_name}", type=['csv'], key=model_name)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader(f"Optimized Pricing Results for {model_name}")
        optimized_data = pricing_function(data)
        st.dataframe(optimized_data)

        # Plot optimized prices
        st.subheader(f"Price Distribution for {model_name}")
        fig, ax = plt.subplots()
        sns.histplot(optimized_data['Optimized_Price'], kde=True, ax=ax)
        st.pyplot(fig)