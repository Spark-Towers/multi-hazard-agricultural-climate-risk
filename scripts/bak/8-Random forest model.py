import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Charger le fichier avec les données incluant les clusters et le GII
file_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_A.xlsx'
df = pd.read_excel(file_path)

# Filtrer les individus avec irrigation inférieure à 100%
df_filtered = df[df['Cluster_irrig'] < 100]
df_filtered['cluster'] = df_filtered['cluster'].astype(str)

# Préparer les variables pour la forêt aléatoire
X_rf = df_filtered[['cluster', 'Risk Score', 'Irrigation(%)']]
y_rf = df_filtered['Adjusted yield']

# Convertir les clusters en variables numériques pour le modèle
X_rf = pd.get_dummies(X_rf, columns=['cluster'])

# Définir le modèle de forêt aléatoire et entraîner sur toutes les données
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_rf, y_rf)

# Générer une gamme de valeurs pour le GII pour les prédictions
gii_values = np.linspace(df_filtered['Risk Score'].min(),
                         df_filtered['Risk Score'].max(), 100)

# Palette de couleurs et marqueurs pour les clusters
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'P', '*']
clusters = df_filtered['cluster'].unique()

# Créer un graphique pour visualiser les relations GII-rendement ajusté
plt.figure(figsize=(11, 8))

for cluster, color, marker in zip(clusters, colors, markers):
    # Créer les données pour le cluster donné et les différentes valeurs de GII
    X_cluster = pd.DataFrame({
        'Risk Score': gii_values,
        'Irrigation(%)': np.mean(df_filtered['Irrigation(%)']),  # Fixer l'irrigation à la moyenne
    })

    # Ajouter les colonnes de cluster (dummy variables)
    for cl in clusters:
        X_cluster[f'cluster_{cl}'] = 1 if cl == cluster else 0

    # S'assurer que les colonnes de X_cluster correspondent à celles de X_rf
    X_cluster = X_cluster.reindex(columns=X_rf.columns, fill_value=0)

    # Faire les prédictions avec le modèle de forêt aléatoire
    y_pred = rf_model.predict(X_cluster)

    # Tracer la courbe des valeurs prédites avec la couleur unique et des marqueurs
    plt.plot(gii_values, y_pred, label=f'Cluster {cluster}', color=color, lw=2)
    plt.scatter(gii_values[::10], y_pred[::10], color=color, marker=marker, s=50, label=None)  # Points espacés

# Ajouter des détails au graphique
plt.title('Yields predicted as a function of Risk Score for each hazard profile\n (Random Forest with Cross-Validation)', fontsize=20, loc='center', pad=20)
plt.xlabel('Risk Score', fontsize=20)
# Déplacer la légende de l'axe des y en haut de l'axe et l'écrire horizontalement
plt.ylabel("", fontsize=0)  # Supprimer la légende standard de l'axe des y
plt.gca().annotate(
    "Predicted yield (t/ha)",
    xy=(0, 1.01),  # Position en haut de l'axe des y
    xycoords='axes fraction',
    fontsize=20,
    ha='center',  # Centrer horizontalement
    va='bottom',  # Aligner en bas du texte
    rotation=0  # Horizontal
)
plt.xlim(0, 10)  # Ajuster l'échelle des abscisses
plt.ylim(0, 10)  # Ajuster l'échelle des ordonnées

# Supprimer le quadrillage
plt.gca().set_facecolor('white')  # Assurer que le fond est blanc
plt.grid(False)

# Ajuster les marges
plt.tight_layout()



# Afficher le graphique
plt.show()

