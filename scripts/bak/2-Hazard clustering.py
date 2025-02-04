import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA
from sklearn.cluster import  AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Charger les données
file_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterC/ClusterC.xlsx'
df = pd.read_excel(file_path)

# Variables climatiques
variables_climatiques = ['SPI_9', 'SPI_10', 'SPI_11','SPI_12', 'SPI_1', 'SPI_2','SPI_3', 'SPI_4', 'SPI_5','SPI_6', 'SPI_7', 'SPI_8',
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

# Standardiser les données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[variables_climatiques])

# Appliquer PCA
n_components = 35  # Vous pouvez ajuster ce nombre
pca = PCA(n_components=n_components)
data_pca = pca.fit_transform(data_scaled)

# Afficher la variance expliquée par les composantes principales
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Variance expliquée par les {n_components} premières composantes : {np.sum(explained_variance_ratio):.4f}")
print(f"Variance expliquée par chaque composante : {explained_variance_ratio}")

# Elbow method (Méthode du coude) pour trouver le nombre optimal de clusters
range_n_clusters = range(2, 10)  # Tester entre 2 et 10 clusters
inertia = []  # Inertie (somme des distances des points à leur centre de cluster)
silhouette_avg_scores = []  # Score de silhouette pour chaque nombre de clusters

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    #agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    kmeans.fit(data_pca)
    inertia.append(kmeans.inertia_)
    #cluster_labels = agglomerative.fit_predict(data_pca)

    # Calcul du score de silhouette
    cluster_labels = kmeans.predict(data_pca)
    silhouette_avg = silhouette_score(data_pca, cluster_labels)
    silhouette_avg_scores.append(silhouette_avg)

# Tracer la courbe du coude (Elbow) et le score de silhouette
plt.figure(figsize=(14, 6))

# Elbow plot
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertia, marker='o')
plt.title("Elbow")
plt.xlabel("Number of cluster")
plt.ylabel("Inertia (Within-cluster sum of squares)")

# Silhouette plot
plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_avg_scores, marker='o')
plt.title("Silhouette index")
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette score")

plt.tight_layout()
plt.show()

# Appliquer KMeans avec le nombre de clusters spécifié (par exemple, 3)
n_clusters_specified =5# Vous pouvez ajuster ce nombre
kmeans = KMeans(n_clusters=n_clusters_specified, n_init=10, random_state=42)
#kmeans = AgglomerativeClustering(n_clusters=n_clusters_specified, linkage='ward')
cluster_labels = kmeans.fit_predict(data_pca)

# Calculer et afficher le score de silhouette pour ce nombre de clusters
silhouette_avg = silhouette_score(data_pca, cluster_labels)
print(f"Score de silhouette pour {n_clusters_specified} clusters et {n_components} composantes : {silhouette_avg}")

# Ajouter les clusters au DataFrame original pour analyse
df['cluster'] = cluster_labels

# Afficher le nombre d'individus par cluster
cluster_sizes = df['cluster'].value_counts().sort_index()
print("\nNombre d'individus par cluster :")
print(cluster_sizes)

# Sauvegarder le DataFrame avec les clusters
df.to_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/clustering42.xlsx', index=False)

# Calculer la moyenne des variables climatiques par cluster
cluster_means = df.groupby('cluster')[variables_climatiques].mean()

# Standardiser les moyennes par variable (Z-score normalisation)
cluster_means_standardized = (cluster_means - cluster_means.mean()) / cluster_means.std()
# Enregistrer les moyennes standardisées par cluster
cluster_means_standardized.to_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/cluster_means_standardized42.xlsx')

# Calculer les statistiques (moyenne, écart-type, min, max) des variables climatiques par cluster
cluster_stats = df.groupby('cluster')[variables_climatiques].agg(['mean', 'std', 'min', 'max'])

# Sauvegarder les résultats dans un fichier Excel
cluster_stats.to_excel('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/cluster_stats42.xlsx')

# Créer une heatmap des moyennes standardisées par cluster pour les variables significatives
plt.figure(figsize=(40, 37))
sns.heatmap(cluster_means_standardized, annot=True, cmap='coolwarm', linewidths=0.5,annot_kws={"size": 7})
plt.title('Heatmap of standardized means of significant climate variables by cluster')
plt.xlabel('Climatic variables')
plt.ylabel('Cluster')
plt.savefig('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/cluster_heatmap42.png')
plt.show()

# Calculer les écarts-types des variables climatiques par cluster
cluster_std = df.groupby('cluster')[variables_climatiques].std()

# Standardiser les écarts-types par variable (Z-score normalisation pour une visualisation plus claire)
cluster_std_standardized = (cluster_std - cluster_std.mean()) / cluster_std.std()

# Créer une heatmap des écarts-types standardisés pour visualiser l'homogénéité des clusters
plt.figure(figsize=(40, 37))
sns.heatmap(cluster_std_standardized, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
plt.title('Heatmap of standardized standard deviations of climate variables by cluster')
plt.xlabel('Climatic variables')
plt.ylabel('Cluster')
plt.savefig('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/cluster_std_heatmap42.png')
plt.show()

# Définir des groupes de variables climatiques
groupes_variables = {
    'SPI_SPEI_sepnov': ['SPI_9', 'SPI_10', 'SPI_11','SPI_91011', 'SPEI_9', 'SPEI_10', 'SPEI_11',  'SPEI_91011'],
'SPI_SPEI_decfeb': ['SPI_12', 'SPI_1', 'SPI_2', 'SPI_1212', 'SPEI_12', 'SPEI_1', 'SPEI_2', 'SPEI_1212'],
'SPI_SPEI_marmay': ['SPI_3', 'SPI_4', 'SPI_5', 'SPI_345', 'SPEI_3', 'SPEI_4', 'SPEI_5', 'SPEI_345'],
'SPI_SPEI_junaug': ['SPI_6', 'SPI_7', 'SPI_8','SPI_678', 'SPEI_6', 'SPEI_7', 'SPEI_8','SPEI_678'],
    'Cold_sepnov': ['TX10_sepnov','TN10_sepnov'],
    'Cold_decfeb': ['TX10_decfeb','TN10_decfeb'],
    'Cold_marmay': ['TX10_marmay', 'TN10_marmay'],
    'Cold_junaug': ['TX10_junaug','TN10_junaug'],
'Frost_sepnov': ['Frost_sepnov'],
'Frost_marmay': ['Frost_marmay'],
'Frost_junaug': ['Frost_junaug'],
'Disease_sepnov': ['Disease_sepnov'],
'Disease_decfeb': ['Disease_decfeb'],
'Disease_marmay': ['Disease_marmay'],
'Disease_junaug': ['Disease_junaug'],
    'Precipitations_sepnov': ['Ptot_sepnov','R20_sepnov'],
    'Precipitations_decfeb': ['Ptot_decfeb','R20_decfeb'],
    'Precipitations_marmay': ['Ptot_marmay', 'R20_marmay'],
    'Precipitations_junaug': ['Ptot_junaug','R20_junaug'],
    'Precipitations': ['pmax'],
    'Heat_sepnov': ['Heat_days_sepnov', 'Intensity_sepnov', 'Severity_sepnov','TX90_sepnov','TN90_sepnov'],
    'Heat_decfeb': ['Heat_days_decfeb', 'Intensity_decfeb', 'Severity_decfeb','TX90_decfeb','TN90_decfeb'],
    'Heat_marmay': ['Heat_days_marmay', 'Intensity_marmay', 'Severity_marmay', 'TX90_marmay', 'TN90_marmay'],
    'Heat_junaug': ['TX90_junaug','TN90_junaug'],
    'SMDI_drought_sepnov': ['SMDI_drought_totaldays_sepnov', 'SMDI_drought_intensity_sepnov', 'SMDI_drought_severity_sepnov'],
'SMDI_drought_decfeb': ['SMDI_drought_totaldays_decfeb', 'SMDI_drought_intensity_decfeb', 'SMDI_drought_severity_decfeb'],
'SMDI_drought_marmay': ['SMDI_drought_totaldays_marmay', 'SMDI_drought_intensity_marmay', 'SMDI_drought_severity_marmay'],
'SMDI_drought_junaug': ['SMDI_drought_totaldays_junaug','SMDI_drought_intensity_junaug','SMDI_drought_severity_junaug'],
'SMDI_water_sepnov': ['SMDI_water_totaldays_sepnov','SMDI_water_intensity_sepnov', 'SMDI_water_severity_sepnov'],
'SMDI_water_decfeb': ['SMDI_water_totaldays_decfeb','SMDI_water_intensity_decfeb', 'SMDI_water_severity_decfeb'],
'SMDI_water_marmay': ['SMDI_water_totaldays_marmay', 'SMDI_water_intensity_marmay','SMDI_water_severity_marmay'],
'SMDI_water_junaug': ['SMDI_water_totaldays_junaug','SMDI_water_intensity_junaug','SMDI_water_severity_junaug']
}

# Créer un DataFrame pour stocker les moyennes standardisées par groupe
group_means_standardized = pd.DataFrame()

# Calculer la moyenne des variables standardisées pour chaque groupe
for group_name, variables in groupes_variables.items():
    # Standardiser les variables de chaque groupe individuellement
    scaler = StandardScaler()
    group_data_scaled = scaler.fit_transform(df[variables])

    # Calculer la moyenne des variables standardisées pour chaque individu dans le groupe
    group_means_standardized[group_name] = pd.DataFrame(group_data_scaled, columns=variables).mean(axis=1)

# Ajouter les labels de clusters
group_means_standardized['cluster'] = df['cluster'].values

# Re-standardiser les moyennes par groupe après calcul
scaler_means = StandardScaler()
group_means_standardized.iloc[:, :-1] = scaler_means.fit_transform(group_means_standardized.iloc[:, :-1])

# Calculer les moyennes par groupe pour chaque cluster
cluster_means = group_means_standardized.groupby('cluster').mean()

# Visualiser les moyennes standardisées par cluster pour chaque groupe de variables
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1,annot_kws={"size": 7})
plt.title('Heatmap of standardized means by variable groups and clusters (Standardized)')
plt.xlabel('Variable groups')
plt.ylabel('Clusters')
plt.tight_layout()
plt.show()

# Créer un DataFrame pour stocker les écarts-types par variable et par cluster
group_std_devs = pd.DataFrame()

# Calculer l'écart type de chaque variable par cluster
for group_name, variables in groupes_variables.items():
    for variable in variables:
        # Calculer l'écart type pour chaque variable au sein de chaque cluster
        std_dev = df.groupby('cluster')[variable].std()
        group_std_devs[group_name + '_' + variable] = std_dev

# Vérifier les colonnes d'écarts types
print("Colonnes d'écarts types :")
print(group_std_devs.columns)

# Standardiser les écarts types par groupe
scaler = StandardScaler()
group_std_devs_standardized = scaler.fit_transform(group_std_devs)

# Créer un DataFrame pour les écarts types standardisés
group_std_devs_df = pd.DataFrame(group_std_devs_standardized, index=group_std_devs.index, columns=group_std_devs.columns)

# Calculer la moyenne des écarts types standardisés pour chaque groupe
group_means_std = group_std_devs_df.groupby(level=0).mean()  # Regroupe par cluster et calcule la moyenne

# Re-standardiser les moyennes des écarts types
scaler_means = StandardScaler()
group_means_std_standardized = scaler_means.fit_transform(group_means_std)

# Créer un DataFrame pour les moyennes d'écarts types standardisées
group_means_std_df = pd.DataFrame(group_means_std_standardized, index=group_means_std.index, columns=group_means_std.columns)

# Création de la heatmap des écarts-types
plt.figure(figsize=(12, 8))
sns.heatmap(group_means_std_df, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7})
plt.title('Heatmap of standardized standard deviations by variable groups and clusters')
plt.xlabel('Variable groups')
plt.ylabel('Clusters')
plt.tight_layout()
plt.show()

# Créer un DataFrame pour stocker les écarts-types par variable et par cluster
group_std_devs = pd.DataFrame()

# Calculer l'écart type de chaque variable par cluster
for group_name, variables in groupes_variables.items():
    # Calculer l'écart type pour chaque variable au sein de chaque cluster
    for variable in variables:
        # Calculer l'écart type pour chaque variable dans chaque cluster
        std_dev = df.groupby('cluster')[variable].std()
        group_std_devs[group_name] = std_dev if group_name not in group_std_devs else group_std_devs[group_name].combine_first(std_dev)

# Vérifier les colonnes d'écarts types
print("Colonnes d'écarts types :")
print(group_std_devs.columns)

# Standardiser les écarts types par groupe
scaler = StandardScaler()
group_std_devs_standardized = scaler.fit_transform(group_std_devs)

# Créer un DataFrame pour les écarts types standardisés
group_std_devs_df = pd.DataFrame(group_std_devs_standardized, index=group_std_devs.index, columns=group_std_devs.columns)

# Calculer la moyenne des écarts types standardisés pour chaque groupe
group_means_std = group_std_devs_df.groupby(group_std_devs_df.index).mean()  # Regroupe par cluster et calcule la moyenne

# Re-standardiser les moyennes d'écarts types entre tous les groupes et clusters
scaler_means = StandardScaler()
group_means_std_all_standardized = scaler_means.fit_transform(group_means_std)

# Créer un DataFrame pour les moyennes d'écarts types standardisées
group_means_std_all_df = pd.DataFrame(group_means_std_all_standardized, index=group_means_std.index, columns=group_means_std.columns)

# Création de la heatmap des écarts-types
plt.figure(figsize=(12, 8))
sns.heatmap(group_means_std_all_df, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 7})
plt.title('Heatmap of re-standardized standard deviations by variable groups and clusters')
plt.xlabel('Variable groups')
plt.ylabel('Clusters')
plt.tight_layout()
plt.show()

# Save the results to an Excel file
excel_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/test/cluster_means_standardized_heatmap42.xlsx'
cluster_means.to_excel(excel_path)


# Calculer le score de silhouette en utilisant les moyennes standardisées par groupe
# Exclure la colonne 'cluster' lors du calcul du score de silhouette
silhouette_avg = silhouette_score(group_means_standardized.drop(columns='cluster'), group_means_standardized['cluster'])

# Afficher le score de silhouette global basé sur les moyennes des groupes de variables standardisées
print(f"Score de silhouette global basé sur les regroupements de variables standardisées : {silhouette_avg:.4f}")

# ---- MANOVA pour tester les différences multivariées entre les clusters ----
manova_formula = ' + '.join(variables_climatiques) + ' ~ cluster'
manova_test = MANOVA.from_formula(manova_formula, data=df)
manova_results = manova_test.mv_test()

print("\nRésultats du test MANOVA :")
print(manova_results)

# Calcul de la variance cumulée
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Tracer le graphique de variance ajoutée cumulée
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='o', linestyle='-', label='Cumulative Variance')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Threshold')  # Lignes guides pour 90%
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Création d'une carte de couleurs discrète
unique_clusters = np.unique(cluster_labels)  # Clusters uniques (0 à 4)
colors = plt.cm.tab10(range(len(unique_clusters)))  # Une couleur unique par cluster
cmap = mcolors.ListedColormap(colors)

# Scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    data_pca[:, 0],
    data_pca[:, 1],
    c=cluster_labels,
    cmap=cmap,  # Utiliser une carte de couleurs discrète
    s=50,
    alpha=0.7
)

# Colorbar alignée avec les clusters
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(unique_clusters) + 0.5), ncolors=len(unique_clusters))
cbar = plt.colorbar(scatter, ticks=unique_clusters, boundaries=np.arange(-0.5, len(unique_clusters) + 0.5))
cbar.set_label('Cluster')

plt.title('2D PCA Projection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot 3D
sc = ax.scatter(
    data_pca[:, 0],
    data_pca[:, 1],
    data_pca[:, 2],
    c=cluster_labels,
    cmap=cmap,  # Utiliser une carte de couleurs discrète
    s=50,
    alpha=0.7
)

# Colorbar alignée avec les clusters
cbar = plt.colorbar(sc, ticks=unique_clusters, boundaries=np.arange(-0.5, len(unique_clusters) + 0.5))
cbar.set_label('Cluster')

ax.set_title('3D PCA Projection')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.tight_layout()
plt.show()


# Obtenir les coefficients de chaque variable pour les trois premières composantes
pca_loadings = pd.DataFrame(
    pca.components_[:3].T,  # Les trois premières composantes principales
    index=variables_climatiques,
    columns=['PCA 1', 'PCA 2', 'PCA 3']
)

# Fonction pour tracer un diagramme en cercle
def plot_pca_pie(component_loadings, component_name):
    component_loadings_abs = component_loadings.abs()
    component_loadings_sorted = component_loadings_abs.sort_values(ascending=False)[:10]  # Les 10 plus importantes
    plt.figure(figsize=(8, 8))
    plt.pie(
        component_loadings_sorted,
        labels=component_loadings_sorted.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title(f'{component_name}: Top 10 Variable Contributions')
    plt.tight_layout()
    plt.show()

# Tracer les diagrammes en cercle pour PCA 1, PCA 2 et PCA 3
for i, component_name in enumerate(['PCA 1', 'PCA 2', 'PCA 3']):
    plot_pca_pie(pca_loadings[component_name], component_name)
