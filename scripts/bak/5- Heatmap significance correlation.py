import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import pearsonr

# Fonction pour charger un fichier et renommer les clusters
def load_and_label_clusters(file_path, label):
    df = pd.read_excel(file_path)
    # Renommer les clusters en ajoutant la lettre du fichier
    df['cluster'] = df['cluster'].apply(lambda x: f"{label}-{x}")
    return df

# Charger les trois fichiers en ajoutant le label
df1 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterA/clusteringA2.xlsx', 'R')
df2 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterB/clusteringB2.xlsx', 'MI')
df3 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterC/clusteringC.xlsx', 'HI')

# Combiner les trois DataFrames pour un traitement uniforme
df_combined = pd.concat([df1, df2, df3])

# Variables climatiques
variables_climatiques = [
    'SPI_9', 'SPI_10', 'SPI_11', 'SPI_12', 'SPI_1', 'SPI_2', 'SPI_3', 'SPI_4', 'SPI_5', 'SPI_6', 'SPI_7', 'SPI_8',
    'SPI_91011', 'SPI_1212', 'SPI_345', 'SPI_678', 'SPI_16', 'SPI_612', 'SPI_YEAR', 'SPEI_9', 'SPEI_10', 'SPEI_11',
    'SPEI_12', 'SPEI_1', 'SPEI_2', 'SPEI_3', 'SPEI_4', 'SPEI_5', 'SPEI_6', 'SPEI_7', 'SPEI_8', 'SPEI_91011',
    'SPEI_1212', 'SPEI_345', 'SPEI_678', 'SPEI_16', 'SPEI_612', 'SPEI_YEAR', 'Frost_sepnov', 'Frost_marmay',
    'Frost_junaug', 'TX10_sepnov', 'TX10_decfeb', 'TX10_marmay', 'TX10_junaug', 'TN10_sepnov', 'TN10_decfeb',
    'TN10_marmay', 'TN10_junaug', 'Disease_sepnov', 'Disease_decfeb', 'Disease_marmay', 'Disease_junaug', 'pmax',
    'Ptot_sepnov', 'Ptot_decfeb', 'Ptot_marmay', 'Ptot_junaug', 'R20_sepnov', 'R20_decfeb', 'R20_marmay',
    'R20_junaug', 'Heat_days_sepnov', 'Intensity_sepnov', 'Severity_sepnov', 'Heat_days_decfeb', 'Intensity_decfeb',
    'Severity_decfeb', 'Heat_days_marmay', 'Intensity_marmay', 'Severity_marmay', 'TX90_sepnov', 'TX90_decfeb',
    'TX90_marmay', 'TX90_junaug', 'TN90_sepnov', 'TN90_decfeb', 'TN90_marmay', 'TN90_junaug',
    'SMDI_drought_totaldays_sepnov', 'SMDI_drought_totaldays_decfeb', 'SMDI_drought_totaldays_marmay',
    'SMDI_drought_totaldays_junaug', 'SMDI_drought_intensity_sepnov', 'SMDI_drought_intensity_decfeb',
    'SMDI_drought_intensity_marmay', 'SMDI_drought_intensity_junaug', 'SMDI_drought_severity_sepnov',
    'SMDI_drought_severity_decfeb', 'SMDI_drought_severity_marmay', 'SMDI_drought_severity_junaug',
    'SMDI_water_totaldays_sepnov', 'SMDI_water_totaldays_decfeb', 'SMDI_water_totaldays_marmay',
    'SMDI_water_totaldays_junaug', 'SMDI_water_intensity_sepnov', 'SMDI_water_intensity_decfeb',
    'SMDI_water_intensity_marmay', 'SMDI_water_intensity_junaug', 'SMDI_water_severity_sepnov',
    'SMDI_water_severity_decfeb', 'SMDI_water_severity_marmay', 'SMDI_water_severity_junaug'
]


# Standardiser les variables climatiques
scaler = StandardScaler()
df_combined[variables_climatiques] = scaler.fit_transform(df_combined[variables_climatiques].fillna(0))


# Modifier la fonction pour inclure les p-valeurs et gérer les constantes
def compute_correlations_for_clusters_with_significance(df, rendement_var, groupes_variables):
    cluster_correlations = {}
    cluster_significance = {}
    for cluster in df['cluster'].unique():
        df_cluster = df[df['cluster'] == cluster]
        aggregated_means = pd.DataFrame()
        for group_name, variables in groupes_variables.items():
            aggregated_means[group_name] = df_cluster[variables].mean(axis=1)
        if rendement_var in df_cluster.columns:
            aggregated_means[rendement_var] = df_cluster[rendement_var]
            corr_matrix = aggregated_means.corr()
            p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)

            # Calcul des p-valeurs, en évitant les colonnes constantes
            for col in aggregated_means.columns:
                for row in aggregated_means.columns:
                    if aggregated_means[col].nunique() > 1 and aggregated_means[row].nunique() > 1:
                        _, p_val = pearsonr(aggregated_means[col], aggregated_means[row])
                    else:
                        p_val = 1  # Pas significatif si constant
                    p_matrix.loc[row, col] = p_val

            cluster_correlations[cluster] = corr_matrix[rendement_var].drop(rendement_var, errors='ignore').fillna(0)
            cluster_significance[cluster] = p_matrix[rendement_var].drop(rendement_var, errors='ignore').fillna(1)
        else:
            cluster_correlations[cluster] = pd.Series(0, index=groupes_variables.keys())  # Corrélation à 0 si absente
            cluster_significance[cluster] = pd.Series(1, index=groupes_variables.keys())  # Non significatif si absent
    return cluster_correlations, cluster_significance


# Ajouter les étoiles directement dans la matrice des corrélations
def annotate_correlation_matrix_with_significance(correlations, significance):
    annotated_matrix = pd.DataFrame(index=correlations.index, columns=correlations.columns)
    for i in correlations.index:
        for j in correlations.columns:
            corr = correlations.loc[i, j]
            p_val = significance.loc[i, j]
            if p_val <= 0.01:
                annotated_matrix.loc[i, j] = f"{corr:.2f}**"
            elif p_val <= 0.05:
                annotated_matrix.loc[i, j] = f"{corr:.2f}*"
            else:
                annotated_matrix.loc[i, j] = f"{corr:.2f}"
    return annotated_matrix


# Définir les groupes de variables climatiques
groupes_variables = {
'SPI_SPEI_sepnov': ['SPI_9', 'SPI_10', 'SPI_11', 'SPI_91011', 'SPEI_9', 'SPEI_10', 'SPEI_11', 'SPEI_91011'],
    'SPI_SPEI_decfeb': ['SPI_12', 'SPI_1', 'SPI_2', 'SPI_1212', 'SPEI_12', 'SPEI_1', 'SPEI_2', 'SPEI_1212'],
    'SPI_SPEI_marmay': ['SPI_3', 'SPI_4', 'SPI_5', 'SPI_345', 'SPEI_3', 'SPEI_4', 'SPEI_5', 'SPEI_345'],
    'SPI_SPEI_junaug': ['SPI_6', 'SPI_7', 'SPI_8', 'SPI_678', 'SPEI_6', 'SPEI_7', 'SPEI_8', 'SPEI_678'],
    'Cold_sepnov': ['TX10_sepnov', 'TN10_sepnov'],
    'Cold_decfeb': ['TX10_decfeb', 'TN10_decfeb'],
    'Cold_marmay': ['TX10_marmay', 'TN10_marmay'],
    'Cold_junaug': ['TX10_junaug', 'TN10_junaug'],
    'Frost_sepnov': ['Frost_sepnov'],
    'Frost_marmay': ['Frost_marmay'],
    'Frost_junaug': ['Frost_junaug'],
    'Disease_sepnov': ['Disease_sepnov'],
    'Disease_decfeb': ['Disease_decfeb'],
    'Disease_marmay': ['Disease_marmay'],
    'Disease_junaug': ['Disease_junaug'],
    'Precipitations_sepnov': ['Ptot_sepnov', 'R20_sepnov'],
    'Precipitations_decfeb': ['Ptot_decfeb', 'R20_decfeb'],
    'Precipitations_marmay': ['Ptot_marmay', 'R20_marmay'],
    'Precipitations_junaug': ['Ptot_junaug', 'R20_junaug'],
    'Precipitations': ['pmax'],
    'Heat_sepnov': ['Heat_days_sepnov', 'Intensity_sepnov', 'Severity_sepnov', 'TX90_sepnov', 'TN90_sepnov'],
    'Heat_decfeb': ['Heat_days_decfeb', 'Intensity_decfeb', 'Severity_decfeb', 'TX90_decfeb', 'TN90_decfeb'],
    'Heat_marmay': ['Heat_days_marmay', 'Intensity_marmay', 'Severity_marmay', 'TX90_marmay', 'TN90_marmay'],
    'Heat_junaug': ['TX90_junaug', 'TN90_junaug'],
    'SMDI_drought_sepnov': ['SMDI_drought_totaldays_sepnov', 'SMDI_drought_intensity_sepnov', 'SMDI_drought_severity_sepnov'],
    'SMDI_drought_decfeb': ['SMDI_drought_totaldays_decfeb', 'SMDI_drought_intensity_decfeb', 'SMDI_drought_severity_decfeb'],
    'SMDI_drought_marmay': ['SMDI_drought_totaldays_marmay', 'SMDI_drought_intensity_marmay', 'SMDI_drought_severity_marmay'],
    'SMDI_drought_junaug': ['SMDI_drought_totaldays_junaug', 'SMDI_drought_intensity_junaug', 'SMDI_drought_severity_junaug'],
    'SMDI_water_sepnov': ['SMDI_water_totaldays_sepnov', 'SMDI_water_intensity_sepnov', 'SMDI_water_severity_sepnov'],
    'SMDI_water_decfeb': ['SMDI_water_totaldays_decfeb', 'SMDI_water_intensity_decfeb', 'SMDI_water_severity_decfeb'],
    'SMDI_water_marmay': ['SMDI_water_totaldays_marmay', 'SMDI_water_intensity_marmay', 'SMDI_water_severity_marmay'],
    'SMDI_water_junaug': ['SMDI_water_totaldays_junaug', 'SMDI_water_intensity_junaug', 'SMDI_water_severity_junaug']
}

# Calculer les corrélations et significativités pour chaque cluster
correlations_by_cluster, significance_by_cluster = compute_correlations_for_clusters_with_significance(
    df_combined, 'Adjusted yield', groupes_variables
)

# Créer les DataFrames nécessaires
all_correlations_df = pd.DataFrame(correlations_by_cluster).T
all_significance_df = pd.DataFrame(significance_by_cluster).T

# Annoter directement la matrice des corrélations
annotated_matrix = annotate_correlation_matrix_with_significance(all_correlations_df, all_significance_df)

# Générer une matrice contenant uniquement les étoiles de significativité
def generate_stars_only_matrix(significance):
    stars_matrix = significance.copy()
    for i in significance.index:
        for j in significance.columns:
            p_val = significance.loc[i, j]
            if p_val <= 0.01:
                stars_matrix.loc[i, j] = "**"
            elif p_val <= 0.05:
                stars_matrix.loc[i, j] = "*"
            else:
                stars_matrix.loc[i, j] = ""
    return stars_matrix

# Créer la matrice d'étoiles
stars_only_matrix = generate_stars_only_matrix(pd.DataFrame(significance_by_cluster).T)

# Générer la heatmap avec uniquement les étoiles
plt.figure(figsize=(30, 25))
sns.heatmap(
    all_correlations_df.astype(float),  # Utiliser les données numériques pour les couleurs
    annot=stars_only_matrix,  # Afficher uniquement les étoiles
    fmt="",
    cmap="RdYlGn",
    cbar=True,
    annot_kws={"size": 32},  # Taille des étoiles
    linewidths=0.5,
    linecolor="black",
    cbar_kws={"shrink": 1, "aspect": 20, "label": "Correlation scale"}
)

plt.title("", fontsize=45)
plt.tick_params(axis='x', labelsize=32)
plt.tick_params(axis='y', labelsize=32)
plt.xlabel("Hazard group", fontsize=40)
# Déplacer la légende de l'axe des y en haut de l'axe et l'écrire horizontalement
plt.ylabel("", fontsize=0)  # Supprimer la légende standard de l'axe des y
plt.gca().annotate(
    "Hazard profile",
    xy=(0, 1.0),  # Position en haut de l'axe des y
    xycoords='axes fraction',
    fontsize=40,
    ha='center',  # Centrer horizontalement
    va='bottom',  # Aligner en bas du texte
    rotation=0  # Horizontal
)


# Ajuster la taille des labels de la colorbar
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=31)
cbar.ax.set_ylabel("Correlation Scale", fontsize=33, labelpad=10)

# Ajuster les marges
plt.subplots_adjust(
    top=0.93,
    bottom=0.25,
    left=0.10,
    right=0.99
)

# Sauvegarder la figure
plt.savefig('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/heatmap_with_significance_stars_only2.png')
plt.show()
