import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load your dataset
def load_data():
    # Assuming you have a CSV file named 'data.csv'
    df = pd.read_csv('C:/Users/Puneet Makkar/Desktop/Customer Churn prediction using streamlit/global_development_data.csv')
    return df

# Train the model
def train_model():
    df = load_data()
    
    # Define features and target
    X = df[['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']]
    y = df['Life Expectancy (IHME)']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'model.joblib')

# Load the model
def load_model():
    return joblib.load('model.joblib')

# Make predictions
def predict_life_expectancy(model, gdp, poverty_line, year):
    input_data = np.array([[gdp, poverty_line, year]])
    return model.predict(input_data)[0]

# Feature importance
def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)