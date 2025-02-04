import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Standardiser chaque variable individuellement (comme dans le Code 2)
scaler = StandardScaler()
df_combined[variables_climatiques] = scaler.fit_transform(df_combined[variables_climatiques])

# Diviser les données combinées en trois DataFrames distincts après la standardisation
df1 = df_combined[df_combined['cluster'].str.startswith('A')]
df2 = df_combined[df_combined['cluster'].str.startswith('B')]
df3 = df_combined[df_combined['cluster'].str.startswith('C')]

# Calculer les moyennes et écarts-types pour chaque fichier
def compute_cluster_stats(df, variables):
    means = df.groupby('cluster')[variables].mean()
    stds = df.groupby('cluster')[variables].std()
    return means, stds

means1, stds1 = compute_cluster_stats(df1, variables_climatiques)
means2, stds2 = compute_cluster_stats(df2, variables_climatiques)
means3, stds3 = compute_cluster_stats(df3, variables_climatiques)

# Combiner les résultats de tous les fichiers
means_combined = pd.concat([means1, means2, means3], axis=0)
stds_combined = pd.concat([stds1, stds2, stds3], axis=0)

# Standardiser les moyennes et écarts-types de manière globale
scaler = StandardScaler()
means_standardized = pd.DataFrame(scaler.fit_transform(means_combined), columns=means_combined.columns, index=means_combined.index)
stds_standardized = pd.DataFrame(scaler.fit_transform(stds_combined), columns=stds_combined.columns, index=stds_combined.index)

# Heatmap des moyennes standardisées
plt.figure(figsize=(30, 25))
sns.heatmap(means_standardized, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
plt.title('Heatmap of standardized means of hazard groups')
plt.xlabel('Hazard group')
plt.ylabel('Climatic profiles')
plt.tight_layout()
plt.show()

# Heatmap des écarts-types standardisés
plt.figure(figsize=(30, 25))
sns.heatmap(stds_standardized, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
plt.title('Heatmap of standardized standard deviations of Hazard groups')
plt.xlabel('Hazard groups')
plt.ylabel('Climatic profiles')
plt.tight_layout()
plt.show()
# ---- Pour les Variables Regroupées ----
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


# Fonction pour calculer les moyennes et écarts-types des groupes de variables par cluster
def compute_group_stats(df, groupes_variables):
    group_means = {}
    group_stds = {}

    for group_name, variables in groupes_variables.items():
        # Calculer les moyennes et écarts-types pour chaque cluster
        group_means[group_name] = df.groupby('cluster')[variables].mean().mean(axis=1)
        group_stds[group_name] = df.groupby('cluster')[variables].std().mean(axis=1)

    # Créer des DataFrames à partir des dictionnaires
    group_means_df = pd.DataFrame(group_means)
    group_stds_df = pd.DataFrame(group_stds)

    return group_means_df, group_stds_df


# Calculer les statistiques pour chaque fichier
group_means1, group_stds1 = compute_group_stats(df1, groupes_variables)
group_means2, group_stds2 = compute_group_stats(df2, groupes_variables)
group_means3, group_stds3 = compute_group_stats(df3, groupes_variables)

# Combiner les résultats pour tous les fichiers
group_means_combined = pd.concat([group_means1, group_means2, group_means3], axis=0)
group_stds_combined = pd.concat([group_stds1, group_stds2, group_stds3], axis=0)

# Standardiser les moyennes et écarts-types des groupes
scaler = StandardScaler()
group_means_standardized = pd.DataFrame(scaler.fit_transform(group_means_combined),
                                        columns=group_means_combined.columns, index=group_means_combined.index)
group_stds_standardized = pd.DataFrame(scaler.fit_transform(group_stds_combined), columns=group_stds_combined.columns,
                                       index=group_stds_combined.index)

# Calculer les percentiles pour les moyennes standardisées
percentiles = group_means_standardized.apply(
    lambda col: pd.Series({
        "25th Percentile": np.percentile(col, 25),
        "50th Percentile (Median)": np.percentile(col, 50),
        "75th Percentile": np.percentile(col, 75)
    }), axis=0
).T

# Sauvegarder les résultats dans un fichier Excel
output_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/group_stats.xlsx'
with pd.ExcelWriter(output_path) as writer:
    group_means_standardized.to_excel(writer, sheet_name='Standardized Means')
    group_stds_standardized.to_excel(writer, sheet_name='Standardized Std Devs')
    percentiles.to_excel(writer, sheet_name='Percentiles')

print(f"Les résultats ont été sauvegardés dans le fichier : {output_path}")

# Heatmap des moyennes standardisées des groupes
plt.figure(figsize=(12, 8))
sns.heatmap(group_means_standardized, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
plt.title('Heatmap of standardized means by hazard groups')
plt.xlabel('Hazard groups')
plt.ylabel('Climatic profiles')
plt.tight_layout()
plt.show()

# Heatmap des écarts-types standardisés des groupes
plt.figure(figsize=(12, 8))
sns.heatmap(group_stds_standardized, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 7})
plt.title('Heatmap of standardized standard deviations by hazard groups')
plt.xlabel('Hazard groups')
plt.ylabel('Climatic profiles')
plt.tight_layout()
plt.show()



# Supposons que la variable de rendement ajusté soit 'rdt_ajuste'
rendement_var = 'Adjusted yield'

# Vérifier que la variable de rendement ajusté 'Adjusted yield' est présente dans le DataFrame combiné
if rendement_var in df_combined.columns:
    # Calculer les moyennes et écarts-types des rendements ajustés par cluster
    yield_means_by_cluster = df_combined.groupby('cluster')[rendement_var].mean()
    yield_stds_by_cluster = df_combined.groupby('cluster')[rendement_var].std()

    # Combiner les résultats dans un DataFrame
    yield_stats_df = pd.DataFrame({
        'Mean Adjusted Yield': yield_means_by_cluster,
        'Standard Deviation Adjusted Yield': yield_stds_by_cluster
    })

    # Afficher le DataFrame des statistiques de rendement ajusté
    print(yield_stats_df)

    # Sauvegarder le DataFrame des statistiques de rendement ajusté dans un fichier Excel
    yield_stats_file_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/adjusted_yield_stats_by_cluster.xlsx'
    yield_stats_df.to_excel(yield_stats_file_path, index=True)
    print(f"Les statistiques de rendement ajusté par cluster ont été sauvegardées dans : {yield_stats_file_path}")
else:
    print("La variable de rendement ajusté n'est pas présente dans le DataFrame combiné.")
