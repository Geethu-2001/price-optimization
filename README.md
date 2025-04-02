# Lemonade Stand Price Optimizer

## Overview
The **Lemonade Stand Price Optimizer** is a machine learning-based application built with **Streamlit** that predicts the optimal price of lemonade based on various factors. The app implements multiple pricing models, including:

- **XGBoost Regression** (Supervised Learning)
- **Q-Learning** (Reinforcement Learning)
- **Bayesian Pricing** (Probabilistic Approach)
- **Game Theory-Based Pricing** (Competitive Pricing Model)

## Features
- **Dynamic Feature Selection**: Users can choose different sets of features for each pricing model.
- **Multiple Pricing Models**: Compares different pricing strategies and evaluates their performance.
- **Model Performance Metrics**: Displays Mean Absolute Error (MAE) and Mean Squared Error (MSE) for each model.
- **User-Friendly Interface**: Built with Streamlit for an interactive experience.

## Dataset
The application uses a dataset (`lemonade_price_optimization.csv`) containing the following columns:
- **Price**: Historical price data
- **Forecasted Optimal Price**: Target price for prediction
- **Time of Day, Day of Week, Weather, Season, Holiday**: Contextual data
- **Demand Level, Customer Demographics**: Business-related factors

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed. You also need the following dependencies:
```sh
pip install streamlit pandas numpy xgboost scikit-learn
```

## Usage
1. **Clone the repository**
```sh
git clone https://github.com/your-repo/lemonade-price-optimizer.git
cd lemonade-price-optimizer
```
2. **Run the Streamlit app**
```sh
streamlit run app.py
```
3. **Select Features & Compare Models**
   - Choose features for each pricing model.
   - View predicted prices and model performance.

## Pricing Models Explained
### **XGBoost Regression**
Uses **supervised learning** to predict the optimal price based on historical data. It minimizes errors using decision trees and gradient boosting.

### **Q-Learning Pricing**
A **reinforcement learning** approach where the model learns pricing strategies over time by maximizing rewards based on demand.

### **Bayesian Pricing**
Estimates the optimal price using a **Bayesian probability distribution**, assuming price variations follow a normal distribution.

### **Game Theory-Based Pricing (Bertrand Model)**
Models pricing decisions based on **competitive market behavior**, assuming sellers compete by adjusting their prices strategically.

## Performance Evaluation
The app calculates the following for each model:
- **Mean Absolute Error (MAE)**: Measures average pricing deviation.
- **Mean Squared Error (MSE)**: Penalizes large pricing errors.

## Future Enhancements
- Implement **Deep Q-Learning** for better reinforcement learning performance.
- Integrate **real-time pricing updates** based on market demand.
- Add **dynamic competitor analysis** using web scraping.

## Contributing
Pull requests are welcome! Please ensure any new features are well-documented.

## License
This project is licensed under the **MIT License**.

## Contact
For any inquiries, reach out to [Your Email or GitHub].

