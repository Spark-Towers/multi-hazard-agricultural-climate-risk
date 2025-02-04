import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import shapiro

# Charger le fichier avec les données incluant les clusters et le GII
file_path = ('/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_A.xlsx')
df = pd.read_excel(file_path)

# Filtrer les individus avec irrigation inférieure à 90% (si nécessaire)
df_filtered = df[df['Cluster_irrig'] < 100]
df_filtered['cluster'] = df_filtered['cluster'].astype(str)

# Préparer les variables pour le test de Shapiro avec une régression linéaire incluant l'interaction
formula = 'Q("Adjusted yield") ~ cluster * Q("Risk Score") + Q("Irrigation(%)")'
y, X = dmatrices(formula, data=df_filtered, return_type='dataframe')
linear_model = sm.OLS(y, X).fit()

# Effectuer le test de Shapiro sur les résidus du modèle incluant l'interaction
stat, shapiro_p_value = shapiro(linear_model.resid)
print(f"Test de Shapiro-Wilk pour la normalité des résidus (p-value) : {shapiro_p_value}")

# Justification
if shapiro_p_value < 0.05:
    print("Les résidus ne suivent pas une distribution normale (p-value < 0.05). Utilisation de la Forêt Aléatoire pour capturer les relations non-linéaires.")
else:
    print("Les résidus suivent une distribution normale (p-value >= 0.05).")

# Préparer les variables pour la forêt aléatoire
X_rf = df_filtered[['cluster', 'Risk Score', 'Irrigation(%)']]
y_rf = df_filtered['Adjusted yield']

# Convertir les clusters en variables numériques pour le modèle
X_rf = pd.get_dummies(X_rf, columns=['cluster'])

# Définir la validation croisée en k plis (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Définir le modèle de forêt aléatoire
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Utiliser la validation croisée pour calculer le MSE
mse_scores = cross_val_score(rf_model, X_rf, y_rf, cv=kf, scoring=make_scorer(mean_squared_error))
r2_scores = cross_val_score(rf_model, X_rf, y_rf, cv=kf, scoring=make_scorer(r2_score))

# Calculer la moyenne et l'écart-type des MSE et R^2 sur les k plis
mse_mean = np.mean(mse_scores)
mse_std = np.std(mse_scores)
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)

print(f"Performance de la Forêt Aléatoire avec validation croisée (k=5):")
print(f" - MSE moyen: {mse_mean} (± {mse_std})")
print(f" - R^2 moyen: {r2_mean} (± {r2_std})")

# Entraîner le modèle sur la totalité des données pour les prédictions finales
rf_model.fit(X_rf, y_rf)

# Générer une gamme de valeurs pour le GII pour les prédictions
gii_values = np.linspace(df_filtered['Risk Score'].min(),
                         df_filtered['Risk Score'].max(), 100)

# Palette de couleurs pour les clusters
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22',
          '#ff9896', '#f1ce63', '#76b7b2', '#98df8a']

# Calculer les pentes et intercepts pour chaque cluster
clusters = df_filtered['cluster'].unique()
combined_results = {}

# Créer un graphique pour visualiser les relations GII-rendement ajusté
plt.figure(figsize=(10, 6))

# Assigner une couleur unique à chaque cluster
for cluster, color in zip(clusters, colors):
    # Créer les données pour le cluster donné et les différentes valeurs de GII
    X_cluster = pd.DataFrame({
        'Risk Score': gii_values,
        'Irrigation(%)': np.mean(df_filtered['Irrigation(%)']),  # Fixer l'irrigation à la moyenne pour simplifier
    })

    # Ajouter les colonnes de cluster (dummy variables) avec la valeur correspondante à chaque cluster
    for cl in clusters:
        X_cluster[f'cluster_{cl}'] = 1 if cl == cluster else 0

    # S'assurer que les colonnes de X_cluster correspondent à celles de X_rf
    X_cluster = X_cluster.reindex(columns=X_rf.columns, fill_value=0)

    # Faire les prédictions avec le modèle de forêt aléatoire
    y_pred = rf_model.predict(X_cluster)
    # Stocker les valeurs prédites pour chaque cluster
    combined_results[cluster] = {'gii_values': gii_values, 'predicted_yields': y_pred}

    # Tracer la courbe des valeurs prédites avec la couleur unique
    plt.plot(gii_values, y_pred, label=f' {cluster}', color=color)


# Ajouter des détails au graphique
plt.title('Yields predicted as a function of Risk Score\n (Random Forest with Cross-Validation)', fontsize=20)
plt.xlabel('Risk Score', fontsize=20)
plt.ylabel('Predicted yield (t/ha)', fontsize=20)

# Définir l'échelle des axes
plt.xlim(0, 10)  # Ajuster l'échelle des abscisses (exemple : de 0 à 1)
plt.ylim(0, 10)  # Ajuster l'échelle des ordonnées (exemple : de 0 à 10)


plt.grid(True)
plt.tight_layout()

# Afficher le graphique
plt.show()


# Calculer les rendements max et min pour tous les GII confondus
def calculate_global_max_min_rf(rf_model, clusters, X_rf_columns, gii_range, irrigation_mean):
    predicted_yields = []

    for gii in gii_range:
        for cluster in clusters:
            # Construire les données d'entrée pour chaque cluster et GII
            X_input = pd.DataFrame({
                'Risk Score': [gii],
                'Irrigation(%)': [irrigation_mean],  # Utiliser la moyenne de l'irrigation
            })
            for cl in clusters:
                X_input[f'cluster_{cl}'] = 1 if cl == cluster else 0

            # Assurer que les colonnes correspondent à X_rf
            X_input = X_input.reindex(columns=X_rf_columns, fill_value=0)

            # Prédire le rendement avec le modèle
            predicted_value = rf_model.predict(X_input)[0]
            predicted_yields.append(predicted_value)

    # Retourner le max et le min globalement
    return max(predicted_yields), min(predicted_yields)

# Fonction pour calculer le score de risque normalisé
def calculate_normalized_risk_score_rf(row, rf_model, clusters, X_rf_columns, global_max, global_min):
    gii = row['Risk Score']
    irrigation = row['Irrigation(%)']
    target_cluster = row['cluster']

    # Calculer le rendement prédit pour l'observation dans son cluster
    X_target = pd.DataFrame({
        'Risk Score': [gii],
        'Irrigation(%)': [irrigation],
    })
    for cl in clusters:
        X_target[f'cluster_{cl}'] = 1 if cl == target_cluster else 0
    X_target = X_target.reindex(columns=X_rf_columns, fill_value=0)

    predicted_yield_target = rf_model.predict(X_target)[0]

    # Normaliser le score
    if global_max == global_min:  # Éviter la division par zéro
        return 0
    else:
        return (global_max - predicted_yield_target) / (global_max - global_min)

# Étape 1 : Calculer le max et le min globalement pour tous les clusters et GII
gii_range = np.linspace(df_filtered['Risk Score'].min(), df_filtered['Risk Score'].max(), 100)
clusters = df_filtered['cluster'].unique()  # Obtenir la liste des clusters
X_rf_columns = X_rf.columns  # Obtenir les colonnes utilisées pour le modèle
irrigation_mean = df_filtered['Irrigation(%)'].mean()  # Moyenne de l'irrigation

global_max, global_min = calculate_global_max_min_rf(
    rf_model, clusters, X_rf_columns, gii_range, irrigation_mean
)

# Étape 2 : Calculer les scores normalisés pour chaque observation
df_filtered['Normalized Risk Score'] = df_filtered.apply(
    calculate_normalized_risk_score_rf,
    axis=1,
    rf_model=rf_model,
    clusters=clusters,
    X_rf_columns=X_rf_columns,
    global_max=global_max,
    global_min=global_min
)

# Étape 3 : Calculer les statistiques agrégées par ville
aggregated_scores = df_filtered.groupby('Town')['Normalized Risk Score'].agg(['sum', 'mean']).reset_index()
aggregated_scores.columns = ['Town', 'Total Normalized Risk Score', 'Average Normalized Risk Score']

# Contribution des clusters dans chaque ville
cluster_contributions = df_filtered.groupby(['Town', 'cluster'])['Normalized Risk Score'].agg(
    ['sum', 'mean']).reset_index()
cluster_contributions.columns = ['Town', 'Cluster', 'Cluster Contribution to Normalized Risk Score',
                                  'Cluster Average Normalized Risk Score']

# Pourcentage d'occurrence des clusters
total_occurrences_by_town = df_filtered.groupby('Town').size().reset_index(name='Total Occurrences')
cluster_occurrences = df_filtered.groupby(['Town', 'cluster']).size().reset_index(name='Cluster Occurrences')
cluster_occurrences = pd.merge(cluster_occurrences, total_occurrences_by_town, on='Town')
cluster_occurrences['Cluster Occurrence Percentage'] = (cluster_occurrences['Cluster Occurrences'] /
                                                        cluster_occurrences['Total Occurrences']) * 100

# Fusionner les résultats pour obtenir un tableau final
final_output = pd.merge(aggregated_scores, cluster_contributions, on='Town', how='left')




# Étape 5 : Sauvegarder les résultats dans un fichier Excel
output_file = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/output_adjusted_risk_scores_A_4.xlsx'
with pd.ExcelWriter(output_file) as writer:
    aggregated_scores.to_excel(writer, sheet_name='Aggregated Scores by Town', index=False)
    cluster_contributions.to_excel(writer, sheet_name='Cluster Contributions', index=False)
    final_output.to_excel(writer, sheet_name='Final Output', index=False)
    cluster_occurrences.to_excel(writer, sheet_name='Cluster Occurrence Percentage', index=False)

print(f"Les scores ajustés et agrégés ont été sauvegardés sous le nom : {output_file}")



# Calculer les valeurs min, max et moyenne des rendements pour des Risk Scores spécifiques
def calculate_yield_stats_for_scores(rf_model, clusters, X_rf_columns, risk_scores, irrigation_mean):
    stats = []

    for score in risk_scores:
        predicted_yields = []
        for cluster in clusters:
            # Construire les données d'entrée pour chaque cluster
            X_input = pd.DataFrame({
                'Risk Score': [score],
                'Irrigation(%)': [irrigation_mean],
            })
            for cl in clusters:
                X_input[f'cluster_{cl}'] = 1 if cl == cluster else 0

            # Assurer que les colonnes correspondent à celles du modèle
            X_input = X_input.reindex(columns=X_rf_columns, fill_value=0)

            # Prédire le rendement pour ce cluster et ce Risk Score
            predicted_value = rf_model.predict(X_input)[0]
            predicted_yields.append(predicted_value)

        # Calculer les statistiques pour ce Risk Score
        stats.append({
            'Risk Score': score,
            'Min Yield': min(predicted_yields),
            'Max Yield': max(predicted_yields),
            'Mean Yield': np.mean(predicted_yields)
        })

    return pd.DataFrame(stats)

# Définir les Risk Scores pour lesquels les statistiques seront calculées
risk_scores = [0, 4, 8]

# Calculer les statistiques
yield_stats = calculate_yield_stats_for_scores(
    rf_model, clusters, X_rf_columns, risk_scores, irrigation_mean
)

# Afficher les résultats
print("Statistiques des rendements prédits pour les Risk Scores spécifiques :")
print(yield_stats)

