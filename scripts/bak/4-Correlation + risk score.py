import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Fonction pour charger un fichier et renommer les clusters
def load_and_label_clusters(file_path, label):
    df = pd.read_excel(file_path)
    # Renommer les clusters en ajoutant la lettre du fichier
    df['cluster'] = df['cluster'].apply(lambda x: f"{label}-{x}")
    return df

# Charger les trois fichiers en ajoutant le label
df1 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterA/clusteringA.xlsx', 'A')
df2 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterB/clusteringB.xlsx', 'B')
df3 = load_and_label_clusters('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/ClusterC/clusteringC.xlsx', 'C')

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

# Fonction pour calculer la matrice de corrélation pour chaque combinaison de cluster
def compute_correlations_for_clusters(df, rendement_var, groupes_variables):
    cluster_correlations = {}
    for cluster in df['cluster'].unique():
        df_cluster = df[df['cluster'] == cluster]
        aggregated_means = pd.DataFrame()
        for group_name, variables in groupes_variables.items():
            aggregated_means[group_name] = df_cluster[variables].mean(axis=1)
        if rendement_var in df_cluster.columns:
            aggregated_means[rendement_var] = df_cluster[rendement_var]
            corr_matrix = aggregated_means.corr()
            cluster_correlations[cluster] = corr_matrix[rendement_var].drop(rendement_var, errors='ignore').fillna(0)
        else:
            cluster_correlations[cluster] = pd.Series(0, index=groupes_variables.keys())  # Corrélation à 0 si absente
    return cluster_correlations


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

# Calculer les corrélations pour chaque cluster
rendement_var = 'Adjusted yield'
correlations_by_cluster = compute_correlations_for_clusters(df_combined, rendement_var, groupes_variables)

# Fonction pour calculer le score de risque pour chaque individu
def calculate_risk_score(df, correlations_by_cluster, groupes_variables):
    risk_scores = []
    for index, row in df.iterrows():
        cluster = row['cluster']
        correlations = correlations_by_cluster.get(cluster, pd.Series(0, index=groupes_variables.keys()))
        risk_score = 0
        for group_name, variables in groupes_variables.items():
            standardized_value = row[variables].mean()
            correlation_value = correlations.get(group_name, 0)  # Corrélation à 0 si absente
            if correlation_value > 0:
                standardized_value = -standardized_value
            risk_score += standardized_value * abs(correlation_value)
        risk_scores.append(risk_score)
    df['Risk Score'] = risk_scores
    return df

# Calculer le score de risque pour chaque individu
df_with_risk_score = calculate_risk_score(df_combined, correlations_by_cluster, groupes_variables)

# Sauvegarder le DataFrame avec les scores de risque
output_file = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_combined_2.xlsx'
df_with_risk_score.to_excel(output_file, index=False)

print(f"Le fichier avec les scores de risque individuels a été sauvegardé sous le nom : {output_file}")

# Combiner les corrélations de tous les clusters dans un seul DataFrame pour une grande heatmap
all_correlations_df = pd.DataFrame(correlations_by_cluster)

# Transposer le DataFrame pour mettre les groupes de variables en abscisses et les clusters en ordonnées
all_correlations_df = all_correlations_df.T

# Générer la heatmap avec les groupes de variables en abscisses
plt.figure(figsize=(45, 40))  # Ajuster la taille pour une meilleure lisibilité si nécessaire
heatmap =sns.heatmap(all_correlations_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,  annot_kws={"size": 31},linewidths=0.5, linecolor="black",cbar_kws={"shrink": 1, "aspect": 20, "label": "Correlation scale"})
plt.title("Heatmap of correlations to yields for each climatic profile by group of hazard", fontsize=45)
plt.tick_params(axis='x', labelsize=32)  # Taille des ticks de l'axe X
plt.tick_params(axis='y', labelsize=32)  # Taille des ticks de l'axe Y
plt.xlabel("Hazard group", fontsize=40)
plt.ylabel("Climatic profile", fontsize=40)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=31)  # Ajuster la taille des labels
# Accéder à la colorbar et modifier la taille du titre
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel("Correlation Scale", fontsize=33, labelpad=10)  # Taille et espace du titre
# Ajuster manuellement les marges pour plus d'espace en haut
plt.subplots_adjust(
    top=0.97,    # Plus d'espace en haut
    bottom=0.15,  # Moins d'espace en bas
    left=0.05,    # Espace à gauche
    right=0.99   # Espace à droite
)
heatmap_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/heatmap_correlation_all_clusters_transposed_2.png'
plt.savefig(heatmap_path)
plt.show()
print(f"Heatmap combinée avec groupes de variables en abscisses sauvegardée sous : {heatmap_path}")

# Sauvegarder les corrélations combinées transposées dans un fichier Excel
correlation_combined_output_file = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/correlations_combined_transposed.xlsx'
all_correlations_df.to_excel(correlation_combined_output_file, index=True)

print(f"Le fichier Excel avec les corrélations combinées transposées a été sauvegardé sous le nom : {correlation_combined_output_file}")
