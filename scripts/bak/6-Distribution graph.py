import pandas as pd
import matplotlib.pyplot as plt

# Charger les données du fichier CSV
fichier = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_C.xlsx'
data = pd.read_excel(fichier)

# Définir les noms des colonnes à tracer
colonne_x = 'Risk Score'  # Colonne pour l'axe X
colonne_y = 'Adjusted yield'  # Colonne pour l'axe Y
colonne_cluster = 'cluster'  # Colonne contenant les informations de cluster

# Définir une liste de symboles (marqueurs) à utiliser
symboles = ['o', 's', '^', 'D', 'P', '*', 'X', '<', '>']  # Rond, carré, triangle, etc.

# Associer chaque cluster unique à un symbole
clusters_uniques = data[colonne_cluster].unique()
cluster_symboles = {cluster: symboles[i % len(symboles)] for i, cluster in enumerate(clusters_uniques)}

# Ajouter les légendes et titres
plt.figure(figsize=(8, 7))  # Augmenter la taille de l'image

# Boucle pour créer un sous-ensemble pour chaque cluster unique
for cluster_value in clusters_uniques:
    subset = data[data[colonne_cluster] == cluster_value]
    plt.scatter(
        subset[colonne_x],
        subset[colonne_y],
        label=f'{cluster_value}',
        s=50,
        alpha=0.7,
        marker=cluster_symboles[cluster_value]
    )

# Ajouter les légendes et titres
plt.xlabel(colonne_x, fontsize=20)
# Déplacer la légende de l'axe des y en haut de l'axe et l'écrire horizontalement
plt.ylabel("", fontsize=0)  # Supprimer la légende standard de l'axe des y
plt.gca().annotate(
    "Yield (t/ha)",
    xy=(0, 1.01),  # Position en haut de l'axe des y
    xycoords='axes fraction',
    fontsize=20,
    ha='center',  # Centrer horizontalement
    va='bottom',  # Aligner en bas du texte
    rotation=0  # Horizontal
)

plt.xlim(0, 13)  # Ajuster l'échelle des abscisses
plt.ylim(0, 14)  # Ajuster l'échelle des ordonnées
plt.title('Yield distribution as a function of Risk score\nby hazard profile', fontsize=20, loc='center', pad=20)
# Supprimer le quadrillage
plt.gca().set_facecolor('white')  # Fond blanc
plt.grid(False)

# Afficher le graphique
plt.show()
