import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.multivariate.manova import MANOVA

def load_data(file_path, variables):
    """Loads and selects relevant columns from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df[variables]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit()

def scale_data(df):
    """Standardizes the dataset using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def apply_pca(X_scaled, n_components=35):
    """Performs PCA and returns transformed data and variance ratio."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Variance explained by {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_pca, pca.explained_variance_ratio_

def determine_optimal_k(X_pca, k_range=(2, 10)):
    """Finds the optimal number of clusters using the elbow and silhouette methods."""
    inertia = []
    silhouette_scores = []
    
    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_pca)
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_pca, labels))
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(k_range[0], k_range[1]), inertia, marker='o', label='Elbow')
    plt.plot(range(k_range[0], k_range[1]), silhouette_scores, marker='s', label='Silhouette')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def cluster_data(X_pca, n_clusters):
    """Clusters the data using KMeans."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    return model.fit_predict(X_pca)

def save_results(df, labels, output_path):
    """Saves clustering results to an Excel file."""
    df['Cluster'] = labels
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the Excel file")
    parser.add_argument("--output", type=str, default="clustering_results.xlsx", help="Output file path")
    args = parser.parse_args()

    variables = ['SPI_9', 'SPI_10', 'SPI_11','SPI_12', 'SPI_1', 'SPI_2','SPI_3', 'SPI_4', 'SPI_5','SPI_6', 'SPI_7', 'SPI_8',
                       'SPI_91011', 'SPI_1212', 'SPI_345','SPI_678', 'SPI_16', 'SPI_612','SPI_YEAR', 'SPEI_9', 'SPEI_10', 'SPEI_11',
                       'SPEI_12', 'SPEI_1', 'SPEI_2','SPEI_3', 'SPEI_4', 'SPEI_5','SPEI_6', 'SPEI_7', 'SPEI_8',
                       'SPEI_91011', 'SPEI_1212', 'SPEI_345','SPEI_678', 'SPEI_16', 'SPEI_612','SPEI_YEAR','Frost_sepnov','Frost_marmay','Frost_junaug',
                       'TX10_sepnov','TX10_decfeb','TX10_marmay','TX10_junaug', 'TN10_sepnov','TN10_decfeb'	, 'TN10_marmay',
                       'TN10_junaug','Disease_sepnov','Disease_decfeb','Disease_marmay','Disease_junaug','pmax','Ptot_sepnov','Ptot_decfeb','Ptot_marmay','Ptot_junaug',
                       'R20_sepnov','R20_decfeb','R20_marmay','R20_junaug','Heat_days_sepnov','Intensity_sepnov','Severity_sepnov','Heat_days_decfeb','Intensity_decfeb',
                       'Severity_decfeb','Heat_days_marmay','Intensity_marmay','Severity_marmay','TX90_sepnov',	'TX90_decfeb',
                       'TX90_marmay','TX90_junaug','TN90_sepnov','TN90_decfeb','TN90_marmay','TN90_junaug','SMDI_drought_totaldays_sepnov','SMDI_drought_totaldays_decfeb','SMDI_drought_totaldays_marmay',
                        'SMDI_drought_totaldays_junaug','SMDI_drought_intensity_sepnov','SMDI_drought_intensity_decfeb',
                       'SMDI_drought_intensity_marmay','SMDI_drought_intensity_junaug',	'SMDI_drought_severity_sepnov','SMDI_drought_severity_decfeb',
                       'SMDI_drought_severity_marmay','SMDI_drought_severity_junaug','SMDI_water_totaldays_sepnov','SMDI_water_totaldays_decfeb',
                       'SMDI_water_totaldays_marmay','SMDI_water_totaldays_junaug','SMDI_water_intensity_sepnov','SMDI_water_intensity_decfeb',
                       'SMDI_water_intensity_marmay','SMDI_water_intensity_junaug',	'SMDI_water_severity_sepnov','SMDI_water_severity_decfeb',
                       'SMDI_water_severity_marmay','SMDI_water_severity_junaug']
    
    df = load_data(args.file, variables)
    X_scaled = scale_data(df)
    X_pca, _ = apply_pca(X_scaled)
    
    determine_optimal_k(X_pca)
    
    labels = cluster_data(X_pca, n_clusters=5)
    save_results(df, labels, args.output)

if __name__ == "__main__":
    main()
