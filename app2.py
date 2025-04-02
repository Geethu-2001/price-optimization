import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm

# Configure page
st.set_page_config(page_title="Price Optimization Suite", layout="wide")
st.title("💰 Advanced Price Optimization Dashboard")

# Custom styling
st.markdown("""
<style>
.section { padding: 20px; border-radius: 10px; margin: 10px 0; background-color: #f0f2f6 }
.plot-container { margin: 20px 0 }
</style>
""", unsafe_allow_html=True)

# Dynamic Pricing with Q-Learning
def dynamic_pricing_q_learning(data):
    if 'Base Price' not in data.columns or 'Demand Level' not in data.columns:
        st.error("Missing required columns: Base Price and Demand Level")
        return data
    
    class QLearningOptimizer:
        def __init__(self, n_bins=20):
            self.q_table = {}
            self.price_bins = None
            self.n_bins = n_bins
            
        def create_price_bins(self, base_prices):
            min_price = base_prices.min() * 0.8
            max_price = base_prices.max() * 1.2
            self.price_bins = np.linspace(min_price, max_price, self.n_bins)
            
        def get_price_state(self, price):
            return np.digitize(price, self.price_bins)
        
        def optimize(self, data, episodes=100):
            self.create_price_bins(data['Base Price'])
            states = data['Base Price'].apply(self.get_price_state).unique()
            
            # Initialize Q-table
            for state in states:
                self.q_table[state] = {action: 0 for action in self.price_bins}
                
            # Q-learning process
            for _ in range(episodes):
                for _, row in data.iterrows():
                    current_state = self.get_price_state(row['Base Price'])
                    action = np.random.choice(self.price_bins)
                    reward = (action - row['Base Price']) * row['Demand Level']
                    
                    # Update Q-value
                    max_next_q = max(self.q_table[current_state].values())
                    self.q_table[current_state][action] += 0.1 * (reward + 0.9 * max_next_q - self.q_table[current_state][action])
            
            # Generate optimized prices
            data['Optimized_Price'] = data['Base Price'].apply(
                lambda x: self.price_bins[np.argmax(list(self.q_table[self.get_price_state(x)].values()))]
            )
            return data
    
    optimizer = QLearningOptimizer()
    return optimizer.optimize(data)

# Bayesian Pricing
def bayesian_pricing(data):
    required_cols = ['Base Price', 'Purchase Probability (%)', 'Competitor Price']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        return data
    
    def calculate_optimal_price(row):
        prior = norm(loc=row['Base Price'], scale=row['Base Price']*0.1)
        likelihood = norm(loc=row['Competitor Price'], scale=row['Competitor Price']*0.15)
        posterior_mean = (prior.mean() + likelihood.mean() * row['Purchase Probability (%)']/100) / 2
        return posterior_mean * 1.1  # Adding margin
    
    data['Optimized_Price'] = data.apply(calculate_optimal_price, axis=1)
    return data

# Game Theory-Based Pricing
def game_theory_pricing(data):
    required_cols = ['Base Fare', 'Competitor Fare', 'Market_Demand']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        return data
    
    def nash_equilibrium(row):
        price_range = np.linspace(row['Base Fare']*0.8, row['Base Fare']*1.2, 50)
        profits = [(p - row['Base Fare']) * (1000 - 20*p + 10*row['Competitor Fare']) for p in price_range]
        return price_range[np.argmax(profits)]
    
    data['Optimized_Price'] = data.apply(nash_equilibrium, axis=1)
    return data

# XGBoost Predictive Pricing
def xgboost_pricing(data):
    required_cols = ['Location', 'Size (sq ft)', 'Brand Popularity', 
                    'Customer Ratings', 'Previous Sale Price', 'Predicted Price']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns: {required_cols}")
        return data
    
    # Preprocessing
    data = data.copy()
    le = LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])
    
    # Feature engineering
    data['Price_Ratio'] = data['Previous Sale Price'] / data.groupby('Location')['Previous Sale Price'].transform('mean')
    data['Brand_Rating'] = data['Brand Popularity'] * data['Customer Ratings']
    
    # Model training
    features = ['Location', 'Size (sq ft)', 'Brand Popularity', 
               'Customer Ratings', 'Previous Sale Price', 'Price_Ratio', 'Brand_Rating']
    X = data[features]
    y = data['Predicted Price']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    
    # Generate predictions
    data['Optimized_Price'] = model.predict(X)
    
    # Feature importance visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig)
    
    return data

# Streamlit UI Components
def main():
    st.sidebar.title("Configuration")
    pricing_models = {
        "Dynamic Pricing (Q-Learning)": dynamic_pricing_q_learning,
        "Bayesian Pricing": bayesian_pricing,
        "Game Theory Pricing": game_theory_pricing,
        "XGBoost Pricing": xgboost_pricing
    }
    
    selected_model = st.sidebar.selectbox("Select Pricing Model", list(pricing_models.keys()))
    
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        optimizer = pricing_models[selected_model]
        optimized_data = optimizer(data)
        
        with st.expander("🔍 View Raw Data", expanded=False):
            st.dataframe(data.head())
        
        with st.container():
            st.subheader("📊 Optimization Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Optimized Prices Preview:")
                st.dataframe(optimized_data[['Base Price', 'Optimized_Price']].head())
                
            with col2:
                st.write("Price Distribution Comparison:")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.kdeplot(data['Base Price'], label='Original Prices', ax=ax)
                sns.kdeplot(optimized_data['Optimized_Price'], label='Optimized Prices', ax=ax)
                ax.set_title("Price Distribution Comparison")
                st.pyplot(fig)
        
        with st.container():
            st.subheader("📈 Detailed Analysis")
            if selected_model == "XGBoost Pricing":
                st.write("Feature Importance Already Displayed Above")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                if selected_model == "Dynamic Pricing (Q-Learning)":
                    price_changes = optimized_data['Optimized_Price'] - optimized_data['Base Price']
                    sns.histplot(price_changes, kde=True, ax=ax)
                    ax.set_title("Price Change Distribution")
                elif selected_model == "Bayesian Pricing":
                    sns.scatterplot(x='Competitor Price', y='Optimized_Price', 
                                  hue='Purchase Probability (%)', data=optimized_data, ax=ax)
                elif selected_model == "Game Theory Pricing":
                    sns.heatmap(pd.crosstab(optimized_data['Market_Demand'], 
                              optimized_data['Optimized_Price'], 
                              annot=True, fmt=".0f", cmap="YlGnBu", ax=ax))
                st.pyplot(fig)

if __name__ == "__main__":
    main()