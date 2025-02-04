import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.model_selection import KFold
from scipy.stats import shapiro
import argparse
import os

def load_and_preprocess_data(file_path):
    """
    Loads dataset, filters relevant rows, and preprocesses clusters.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    
    df_filtered = df[df['Irrigation(%)'] < 100].copy()
    df_filtered['cluster'] = df_filtered['cluster'].astype(str).astype('category')
    return df_filtered

def run_model_with_reference(df_filtered, cluster_ref):
    """
    Runs an OLS model using a specified cluster as the reference.
    """
    categories = df_filtered['cluster'].cat.categories
    new_order = [str(cluster_ref)] + [str(c) for c in categories if str(c) != str(cluster_ref)]
    df_filtered['cluster'] = df_filtered['cluster'].cat.reorder_categories(new_order, ordered=True)
    
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index") + Q("Irrigation(%)")'
    y, X = dmatrices(formula, data=df_filtered, return_type='dataframe')
    model = sm.OLS(y, X).fit()
    
    shapiro_p_value = shapiro(model.resid)[1]
    print(f"Shapiro-Wilk test p-value: {shapiro_p_value}")
    return model, shapiro_p_value

def cross_validate_model(df_filtered, cluster_ref, n_splits=5):
    """
    Performs k-fold cross-validation for OLS models.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index") + Q("Irrigation(%)")'
    
    coefficients_list = []
    for train_index, _ in kf.split(df_filtered):
        train_data = df_filtered.iloc[train_index]
        y_train, X_train = dmatrices(formula, data=train_data, return_type='dataframe')
        model = sm.OLS(y_train, X_train).fit()
        coefficients_list.append(model.params)
    
    return pd.DataFrame(coefficients_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OLS models with cross-validation.")
    parser.add_argument("file", type=str, help="Path to the input Excel file")
    parser.add_argument("--cluster_ref", type=str, default=None, help="Reference cluster for OLS model")
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of cross-validation splits")
    
    args = parser.parse_args()
    df = load_and_preprocess_data(args.file)
    
    if df is not None and args.cluster_ref:
        model, p_value = run_model_with_reference(df, args.cluster_ref)
        cv_results = cross_validate_model(df, args.cluster_ref, args.cv_splits)
        print("Cross-validation results:")
        print(cv_results)
    else:
        print("Please provide a valid cluster reference.")
