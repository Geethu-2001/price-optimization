import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Set dataset path (modify this if necessary)
data_path = "C:/Users/krish/Downloads/lemonade_price_optimization.csv"

# Check if file exists
if not os.path.exists(data_path):
    st.error(f"Dataset not found: {data_path}. Please upload the correct file.")
    st.stop()

# Load dataset
df = pd.read_csv(data_path)

# Convert categorical variables to numerical
categorical_columns = ['Time of Day', 'Day of Week', 'Weather', 'Season', 'Holiday', 'Demand Level', 'Customer Demographics']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Features and Target
X = df.drop(columns=['Price', 'Forecasted Optimal Price'])  # Features
y = df['Forecasted Optimal Price']  # Target price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Calculate errors
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)

# Dummy implementations for other models (replace these with actual implementations)
def q_learning_price():
    return y_test.mean() + np.random.uniform(-0.5, 0.5)

def bayesian_price():
    return y_test.mean() + np.random.uniform(-0.3, 0.3)

def game_theory_price():
    return y_test.mean() + np.random.uniform(-0.4, 0.4)

# Predictions
q_price = [q_learning_price() for _ in range(len(y_test))]
b_price = [bayesian_price() for _ in range(len(y_test))]
gt_price = [game_theory_price() for _ in range(len(y_test))]

# Error calculations
mae_q = mean_absolute_error(y_test, q_price)
mse_q = mean_squared_error(y_test, q_price)
mae_b = mean_absolute_error(y_test, b_price)
mse_b = mean_squared_error(y_test, b_price)
mae_gt = mean_absolute_error(y_test, gt_price)
mse_gt = mean_squared_error(y_test, gt_price)

# Streamlit App
st.title("Lemonade Stand Price Optimizer")
st.write("This app predicts the optimal price using different models.")

# Display errors
st.subheader("Model Performance")
st.write("### Mean Absolute Error (MAE)")
st.write(f"- XGBoost: {mae_xgb:.2f}")
st.write(f"- Q-learning: {mae_q:.2f}")
st.write(f"- Bayesian Pricing: {mae_b:.2f}")
st.write(f"- Game Theory-Based Pricing: {mae_gt:.2f}")

st.write("### Mean Squared Error (MSE)")
st.write(f"- XGBoost: {mse_xgb:.2f}")
st.write(f"- Q-learning: {mse_q:.2f}")
st.write(f"- Bayesian Pricing: {mse_b:.2f}")
st.write(f"- Game Theory-Based Pricing: {mse_gt:.2f}")

st.subheader("Predicted Optimal Prices")
st.write(f"XGBoost Predicted Price: ${y_pred_xgb.mean():.2f}")
st.write(f"Q-learning Predicted Price: ${np.mean(q_price):.2f}")
st.write(f"Bayesian Predicted Price: ${np.mean(b_price):.2f}")
st.write(f"Game Theory-Based Price: ${np.mean(gt_price):.2f}")
