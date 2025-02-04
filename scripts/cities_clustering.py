import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths (Modify as needed)
DATA_DIR = "path/to/directory"
INPUT_FILE = os.path.join(DATA_DIR, "Cities.xlsx")
OUTPUT_CLUSTER_FILE = os.path.join(DATA_DIR, "Irrigation/resultats_clusters.xlsx")
OUTPUT_SUMMARY_FILE = os.path.join(DATA_DIR, "Irrigation/resume_clusters.xlsx")

# Load dataset
try:
    df = pd.read_excel(INPUT_FILE)
    logging.info(f"Loaded data from {INPUT_FILE}")
except FileNotFoundError:
    logging.error(f"File {INPUT_FILE} not found. Please check the path.")
    exit()

# Select relevant features
variables = ['Irrigation(%)']
X = df[variables]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Hierarchical Clustering Dendrogram ---
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# --- Clustering Performance Metrics ---
silhouette_scores = []
calinski_harabasz_scores = []
K_range = range(2, 15)

for k in K_range:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = clustering.fit_predict(X_scaled)

    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))

# Plot silhouette scores
plt.figure(figsize=(10, 7))
plt.plot(K_range, silhouette_scores, 'bx-')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method for Determining Clusters")
plt.show()

# Plot Calinski-Harabasz Index
plt.figure(figsize=(10, 7))
plt.plot(K_range, calinski_harabasz_scores, 'bx-')
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.title("Calinski-Harabasz Index for Determining Clusters")
plt.show()

# --- Final Clustering with Chosen Number of Clusters ---
n_clusters = 3  # Adjust based on results
agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
df['cluster'] = agglomerative.fit_predict(X_scaled)

# Display cluster sizes
cluster_sizes = df['cluster'].value_counts()
logging.info(f"Cluster sizes:\n{cluster_sizes}")

# --- Variance Calculation ---
global_centroid = np.mean(X_scaled, axis=0)
total_variance = np.sum((X_scaled - global_centroid) ** 2)

intra_cluster_variance = sum(
    np.sum((X_scaled[df['cluster'] == cluster] - np.mean(X_scaled[df['cluster'] == cluster], axis=0)) ** 2)
    for cluster in np.unique(df['cluster']) if cluster != -1
)

explained_variance = 1 - (intra_cluster_variance / total_variance)
logging.info(f"Explained variance by clusters: {explained_variance * 100:.2f}%")

# --- Save Results ---
df.to_excel(OUTPUT_CLUSTER_FILE, index=False)
df.groupby('cluster')[variables].mean().to_excel(OUTPUT_SUMMARY_FILE)

logging.info(f"Cluster results saved to {OUTPUT_CLUSTER_FILE}")
logging.info(f"Cluster summary saved to {OUTPUT_SUMMARY_FILE}")
