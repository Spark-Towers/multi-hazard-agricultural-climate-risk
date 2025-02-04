import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import  AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# Charger les données depuis un fichier Excel
df = pd.read_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Cities.xlsx')

# Sélectionner les colonnes pertinentes pour le clustering
variables = ['Irrigation(%)']

X = df[variables]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Dendrogramme ----
linked = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogramme')
plt.xlabel('Points de données')
plt.ylabel('Distance')
plt.show()

# ---- Méthode du Coude et Score de Silhouette ----
sum_of_squared_distances = []
silhouette_scores = []
calinski_harabasz_scores = []
K = range(2, 15)

for k in K:
    agglomerative = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = agglomerative.fit_predict(X_scaled)

    # Vous devrez ajuster pour utiliser un score similaire à l'inertie
    sum_of_squared_distances.append(agglomerative.fit(X_scaled))
    silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))

# Tracer la méthode du coude (en utilisant ici le score de silhouette à la place de SSE)
plt.figure(figsize=(10, 7))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Score de Silhouette')
plt.title('Méthode du Coude et Score de Silhouette pour déterminer le nombre de clusters')
plt.show()

# Tracer l'indice de Calinski-Harabasz
plt.figure(figsize=(10, 7))
plt.plot(K, calinski_harabasz_scores, 'bx-')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Indice de Calinski-Harabasz')
plt.title('Indice de Calinski-Harabasz pour déterminer le nombre de clusters')
plt.show()

# Appliquer l'Agglomération Hiérarchique dans l'espace UMAP
agglomerative = AgglomerativeClustering(n_clusters=3)
df['cluster'] = agglomerative.fit_predict(X_scaled)

# Afficher les nouvelles tailles des clusters après fusion
new_cluster_sizes = df['cluster'].value_counts()
print("\nTailles des clusters :\n", new_cluster_sizes)

# Calcul du centroïde global
global_centroid = np.mean(X_scaled, axis=0)

# Calcul de la variance totale (par rapport au centroïde global)
total_variance = np.sum((X_scaled - global_centroid) ** 2)

# Calcul de la variance intra-cluster
intra_cluster_variance = 0
for cluster in np.unique(df['cluster']):
    if cluster != -1:  # Ignorer le bruit (cluster -1)
        cluster_points = X_scaled[df['cluster'] == cluster]
        cluster_centroid = np.mean(cluster_points, axis=0)
        intra_cluster_variance += np.sum((cluster_points - cluster_centroid) ** 2)

# Calcul du pourcentage de variance expliquée
explained_variance = 1 - (intra_cluster_variance / total_variance)

print(f"Pourcentage de variance totale expliquée par les clusters HDBSCAN : {explained_variance * 100:.2f}%")


# Sauvegarder le DataFrame avec les clusters dans un fichier Excel
df.to_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/resultats_clusters0.xlsx', index=False)

# Créer un résumé des clusters pour chaque variable
cluster_summary = df.groupby('cluster')[variables].mean()

# Sauvegarder le résumé des clusters dans un fichier Excel
cluster_summary.to_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/resume_cluster0.xlsx')

print("Les clusters ont été créés et sauvegardés dans 'resultats_clusters_0.xlsx'.")
print("Le résumé des clusters a été sauvegardé dans 'resume_clusters_0.xlsx'.")
