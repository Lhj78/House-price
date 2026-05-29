#!/usr/bin/env python3
"""
Housing Price Prediction Model
This script trains a gradient boosting model to predict housing prices
and generates predictions for train and test datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load training and test data"""
    train_data = pd.read_csv('train (5).csv')
    test_data = pd.read_csv('test (4).csv')
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """Preprocess and encode features"""
    # Extract target and features
    y_train = train_data['SalePrice']
    X_train = train_data.drop(['SalePrice', 'Id'], axis=1)
    X_test = test_data.drop('Id', axis=1)
    
    # Combine datasets for consistent preprocessing
    combined = pd.concat([X_train, X_test], keys=['train', 'test'])
    
    # Handle missing values
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    categorical_cols = combined.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        combined[col].fillna(combined[col].median(), inplace=True)
    
    # Fill categorical missing values
    for col in categorical_cols:
        combined[col].fillna(combined[col].mode()[0] if len(combined[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
    
    # Debug: report missing values after imputation and encoding
    na_counts = combined.isnull().sum()
    print("Missing values after imputation (per column):")
    print(na_counts[na_counts > 0])
    
    # Fill any remaining NaNs with zeros
    combined.fillna(0, inplace=True)

    # Separate back
    X_train_processed = combined.loc['train']
    X_test_processed = combined.loc['test']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_scaled, X_test_scaled, y_train, train_data, test_data

def train_and_predict(X_train, X_test, y_train, train_data, test_data):
    """Train model and generate predictions"""
    # Train gradient boosting model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Create prediction DataFrames
    train_pred_df = pd.DataFrame({
        'Id': train_data['Id'],
        'Actual_SalePrice': train_data['SalePrice'],
        'Predicted_SalePrice': y_train_pred
    })
    
    test_pred_df = pd.DataFrame({
        'Id': test_data['Id'],
        'Predicted_SalePrice': y_test_pred
    })
    
    return train_pred_df, test_pred_df

def main():
    """Main execution function"""
    print("Loading data...")
    train_data, test_data = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, train_data, test_data = preprocess_data(train_data, test_data)
    
    print("Training model and generating predictions...")
    train_pred, test_pred = train_and_predict(X_train, X_test, y_train, train_data, test_data)
    
    # Save predictions
    train_pred.to_csv('train_predictions.csv', index=False)
    test_pred.to_csv('test_predictions.csv', index=False)
    
    print("\nPredictions saved successfully!")
    print(f"✓ train_predictions.csv - {len(train_pred)} records")
    print(f"✓ test_predictions.csv - {len(test_pred)} records")
    
    return train_pred, test_pred

if __name__ == "__main__":
    train_pred, test_pred = main()
