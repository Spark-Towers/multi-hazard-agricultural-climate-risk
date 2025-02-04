import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

def train_random_forest(df_filtered, n_estimators=100):
    """
    Trains a Random Forest model on the dataset.
    """
    X_rf = df_filtered[['cluster', 'Risk Score', 'Irrigation(%)']]
    y_rf = df_filtered['Adjusted yield']
    
    # Convert categorical variables to dummies
    X_rf = pd.get_dummies(X_rf, columns=['cluster'])
    
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_rf, y_rf)
    
    return rf_model, X_rf

def predict_yield(rf_model, X_rf, df_filtered):
    """
    Generates yield predictions across a range of Risk Scores.
    """
    gii_values = np.linspace(df_filtered['Risk Score'].min(), df_filtered['Risk Score'].max(), 100)
    clusters = df_filtered['cluster'].unique()
    
    predictions = {}
    for cluster in clusters:
        X_cluster = pd.DataFrame({
            'Risk Score': gii_values,
            'Irrigation(%)': np.mean(df_filtered['Irrigation(%)'])
        })
        
        for cl in clusters:
            X_cluster[f'cluster_{cl}'] = 1 if cl == cluster else 0
        
        X_cluster = X_cluster.reindex(columns=X_rf.columns, fill_value=0)
        predictions[cluster] = rf_model.predict(X_cluster)
    
    return gii_values, predictions

def plot_results(gii_values, predictions):
    """
    Plots the predicted adjusted yield values against Risk Scores.
    """
    plt.figure(figsize=(11, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'P', '*']
    
    for (cluster, pred), color, marker in zip(predictions.items(), colors, markers):
        plt.plot(gii_values, pred, label=f'Cluster {cluster}', color=color, marker=marker)
    
    plt.xlabel('Risk Score')
    plt.ylabel('Adjusted Yield')
    plt.legend(title='Cluster')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model and predict adjusted yield.")
    parser.add_argument("file", type=str, help="Path to the input Excel file")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in Random Forest")
    
    args = parser.parse_args()
    df = load_and_preprocess_data(args.file)
    
    if df is not None:
        rf_model, X_rf = train_random_forest(df, args.n_estimators)
        gii_values, predictions = predict_yield(rf_model, X_rf, df)
        plot_results(gii_values, predictions)