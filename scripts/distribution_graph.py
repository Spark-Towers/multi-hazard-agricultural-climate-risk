import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_distribution_graph(file_path, x_column, y_column, cluster_column, output_file=None):
    """
    Plots a distribution graph using scatter plots for different clusters.
    
    Parameters:
    - file_path (str): Path to the Excel file.
    - x_column (str): Column name for the x-axis.
    - y_column (str): Column name for the y-axis.
    - cluster_column (str): Column name indicating cluster groups.
    - output_file (str, optional): If provided, saves the plot to this file.
    """
    # Load data
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Define unique markers for clusters
    symbols = ['o', 's', '^', 'D', 'P', '*', 'X', '<', '>']
    clusters_unique = data[cluster_column].unique()
    cluster_symbols = {cluster: symbols[i % len(symbols)] for i, cluster in enumerate(clusters_unique)}
    
    # Create the figure
    plt.figure(figsize=(8, 7))
    
    # Scatter plot for each cluster
    for cluster_value in clusters_unique:
        subset = data[data[cluster_column] == cluster_value]
        plt.scatter(
            subset[x_column], subset[y_column],
            label=f'{cluster_value}',
            s=50, alpha=0.7,
            marker=cluster_symbols[cluster_value]
        )
    
    # Customize plot
    plt.xlabel(x_column, fontsize=20)
    plt.ylabel("", fontsize=0)  # Remove default y-axis label
    plt.gca().annotate(
        "Yield (t/ha)", xy=(0, 1.01), xycoords='axes fraction',
        fontsize=20, ha='center', va='bottom'
    )
    plt.xlim(0, 13)
    plt.ylim(0, 14)
    plt.legend(title='Cluster')
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a distribution graph from an Excel dataset.")
    parser.add_argument("file", type=str, help="Path to the input Excel file")
    parser.add_argument("--x_column", type=str, default="Risk Score", help="Column name for X-axis")
    parser.add_argument("--y_column", type=str, default="Adjusted yield", help="Column name for Y-axis")
    parser.add_argument("--cluster_column", type=str, default="cluster", help="Column indicating clusters")
    parser.add_argument("--output", type=str, default=None, help="Optional: Save output graph to file")
    
    args = parser.parse_args()
    plot_distribution_graph(args.file, args.x_column, args.y_column, args.cluster_column, args.output)
