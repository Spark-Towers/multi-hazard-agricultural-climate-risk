import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_and_label_clusters(file_path, label):
    """Loads cluster data from an Excel file and labels clusters."""
    try:
        df = pd.read_excel(file_path)
        df['cluster'] = df['cluster'].apply(lambda x: f"{label}-{x}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()

def standardize_variables(df, variables):
    """Standardizes selected climatic variables."""
    scaler = StandardScaler()
    df[variables] = scaler.fit_transform(df[variables])
    return df

def compute_standardized_means(df_list, variables):
    """Computes standardized means and standard deviations for clusters."""
    combined_df = pd.concat(df_list, axis=0)
    means = combined_df.groupby('cluster')[variables].mean()
    stds = combined_df.groupby('cluster')[variables].std()
    
    scaler = StandardScaler()
    means_standardized = pd.DataFrame(scaler.fit_transform(means), columns=means.columns, index=means.index)
    return means_standardized

def plot_heatmap(df, title):
    """Generates a heatmap for the given DataFrame."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
    plt.title(title)
    plt.xlabel('Hazard groups')
    plt.ylabel('Climatic profiles')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing cluster files")
    parser.add_argument("--output", type=str, default="group_stats.xlsx", help="Output file path")
    args = parser.parse_args()

    # Load datasets
    df1 = load_and_label_clusters(os.path.join(args.folder, 'clusteringA.xlsx'), 'A')
    df2 = load_and_label_clusters(os.path.join(args.folder, 'clusteringB.xlsx'), 'B')
    df3 = load_and_label_clusters(os.path.join(args.folder, 'clusteringC.xlsx'), 'C')

    # Combine datasets
    df_combined = pd.concat([df1, df2, df3])

    # Define climatic variables
    variables_climatiques = ['SPI_9', 'SPI_10', 'SPEI_9', 'TX90_sepnov', 'R20_sepnov']

    # Standardize variables
    df_combined = standardize_variables(df_combined, variables_climatiques)

    # Compute standardized means
    means_standardized = compute_standardized_means([df1, df2, df3], variables_climatiques)

    # Save results
    means_standardized.to_excel(args.output)
    print(f"Results saved to {args.output}")

    # Plot heatmap
    plot_heatmap(means_standardized, "Heatmap of standardized means by hazard groups")

if __name__ == "__main__":
    main()
