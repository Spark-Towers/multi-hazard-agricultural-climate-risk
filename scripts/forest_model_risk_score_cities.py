import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import argparse
import os

def load_and_preprocess_data(file_path):
    """
    Loads dataset, filters relevant rows, and prepares features.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    df_filtered = df[df['Cluster_irrig'] < 100].copy()
    df_filtered['cluster'] = df_filtered['cluster'].astype(str)
    return df_filtered

def test_residual_normality(df_filtered):
    """
    Tests normality of residuals using Shapiro-Wilk test.
    """
    formula = 'Q("Adjusted yield") ~ cluster * Q("Risk Score") + Q("Irrigation(%)")'
    y, X = dmatrices(formula, data=df_filtered, return_type='dataframe')
    linear_model = sm.OLS(y, X).fit()
    
    stat, shapiro_p_value = shapiro(linear_model.resid)
    print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
    return shapiro_p_value

def train_random_forest(df_filtered, n_estimators=100):
    """
    Trains a Random Forest model on the dataset.
    """
    X_rf = df_filtered[['cluster', 'Risk Score', 'Irrigation(%)']]
    y_rf = df_filtered['Adjusted yield']
    
    X_rf = pd.get_dummies(X_rf, columns=['cluster'])
    
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_rf, y_rf)
    return rf_model, X_rf, y_rf

def perform_cross_validation(rf_model, X_rf, y_rf, n_splits=5):
    """
    Performs k-fold cross-validation for Random Forest model.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = cross_val_score(rf_model, X_rf, y_rf, cv=kf, scoring=make_scorer(mean_squared_error))
    r2_scores = cross_val_score(rf_model, X_rf, y_rf, cv=kf, scoring=make_scorer(r2_score))
    
    print(f"Mean MSE: {np.mean(mse_scores)}")
    print(f"Mean R^2: {np.mean(r2_scores)}")
    return mse_scores, r2_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model with cross-validation.")
    parser.add_argument("file", type=str, help="Path to the input Excel file")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of cross-validation splits")
    
    args = parser.parse_args()
    df = load_and_preprocess_data(args.file)
    
    if df is not None:
        p_value = test_residual_normality(df)
        if p_value < 0.05:
            print("Residuals are not normally distributed. Using Random Forest.")
            rf_model, X_rf, y_rf = train_random_forest(df, args.n_estimators)
            mse_scores, r2_scores = perform_cross_validation(rf_model, X_rf, y_rf, args.cv_splits)
        else:
            print("Residuals are normally distributed. Consider using OLS instead.")
