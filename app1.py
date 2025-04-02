import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Function to calculate and display error metrics
def calculate_error_metrics(optimized_data, actual_column='Predicted Price'):
    if actual_column in optimized_data.columns:
        y_true = optimized_data[actual_column]
        y_pred = optimized_data['Optimized_Price']
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_true, y_pred)
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_true, y_pred)
        
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Mean Absolute Error (MAE): {mae}")
    else:
        st.warning("Actual price column ('Predicted Price') not found for error calculation.")

# Function for Dynamic Pricing using Q-Learning
def dynamic_pricing_q_learning(data):
    # Split into training and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Apply Q-learning pricing on the training set
    train_data['Optimized_Price'] = train_data['Base Price'] * (1 + np.random.uniform(-0.2, 0.2, len(train_data)))
    
    # Predict optimized prices on the test set
    test_data['Optimized_Price'] = test_data['Base Price'] * (1 + np.random.uniform(-0.2, 0.2, len(test_data)))
    
    # Combine the train and test data back
    optimized_data = pd.concat([train_data, test_data])
    
    # Calculate error metrics
    calculate_error_metrics(optimized_data)
    return optimized_data

# Function for Bayesian Pricing
def bayesian_pricing(data):
    # Split into training and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Apply Bayesian pricing on the training set
    train_data['Optimized_Price'] = train_data['Base Price'] * (1 + train_data['Purchase Probability (%)'] / 100)
    
    # Predict optimized prices on the test set
    test_data['Optimized_Price'] = test_data['Base Price'] * (1 + test_data['Purchase Probability (%)'] / 100)
    
    # Combine the train and test data back
    optimized_data = pd.concat([train_data, test_data])
    
    # Calculate error metrics
    calculate_error_metrics(optimized_data)
    return optimized_data

# Function for Game Theory-Based Pricing
def game_theory_pricing(data):
    # Split into training and test sets (80% train, 20% test)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Apply Game Theory pricing on the training set
    train_data['Optimized_Price'] = (train_data['Base Fare'] + train_data['Competitor Fare']) / 2
    
    # Predict optimized prices on the test set
    test_data['Optimized_Price'] = (test_data['Base Fare'] + test_data['Competitor Fare']) / 2
    
    # Combine the train and test data back
    optimized_data = pd.concat([train_data, test_data])
    
    # Calculate error metrics
    calculate_error_metrics(optimized_data)
    return optimized_data

# Function for XGBoost Predictive Pricing with Train-Test Split
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

    # Split the data into training and testing sets (80% train, 20% test)
    X = data[features]
    y = data['Predicted Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model.fit(X_train, y_train)

    # Predict optimized prices
    data['Optimized_Price'] = model.predict(X)

    # Evaluate model on the test set (optional, for feedback)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE) for XGBoost Pricing: {mse}")

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
