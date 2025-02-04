import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro

# Charger le fichier avec les données incluant les clusters et le GII
file_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_C.xlsx'
df = pd.read_excel(file_path)

# Filtrer les individus avec irrigation inférieure à 100%
df_filtered = df[df['Irrigation(%)'] < 100]
print("Clusters présents dans df_filtered :")
print(df_filtered['cluster'].unique())

# Convertir les clusters en chaînes de caractères et catégories
df_filtered['cluster'] = df_filtered['cluster'].astype(str)
df_filtered['cluster'] = df_filtered['cluster'].astype('category')

# Fonction pour ajuster le modèle en prenant un cluster donné comme référence
def run_model_with_reference(cluster_ref, df_filtered):
    df_filtered['cluster'] = df_filtered['cluster'].astype('category')
    categories = df_filtered['cluster'].cat.categories
    new_order = [str(cluster_ref)] + [str(c) for c in categories if str(c) != str(cluster_ref)]
    df_filtered['cluster'] = df_filtered['cluster'].cat.reorder_categories(new_order, ordered=True)
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index") + Q("Irrigation(%)")'
    y, X = dmatrices(formula, data=df_filtered, return_type='dataframe')
    model = sm.OLS(y, X).fit()

    # Diagnostic de la normalité des résidus avec le test de Shapiro-Wilk
    shapiro_p_value = shapiro(model.resid)[1]
    print(f"Test de Shapiro-Wilk pour la normalité des résidus (p-value) : {shapiro_p_value}")

    return model, shapiro_p_value

# Fonction pour effectuer la validation croisée
def cross_validate_model(df_filtered, cluster_ref, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index") + Q("Irrigation(%)")'

    coefficients_list = []

    for train_index, test_index in kf.split(df_filtered):
        train_data = df_filtered.iloc[train_index]
        y_train, X_train = dmatrices(formula, data=train_data, return_type='dataframe')
        model = sm.OLS(y_train, X_train).fit()
        coefficients_list.append(model.params)

    # Calculer la moyenne et l'écart-type des coefficients sur les plis
    coefficients_df = pd.DataFrame(coefficients_list)
    coefficients_mean = coefficients_df.mean()
    coefficients_std = coefficients_df.std()

    print(f"\nValidation croisée des coefficients pour le cluster {cluster_ref} (moyenne et écart-type) :")
    print(pd.DataFrame({'mean': coefficients_mean, 'std': coefficients_std}))

    return coefficients_mean, coefficients_std

# Initialiser une figure pour tracer les courbes
plt.figure(figsize=(10, 6))
clusters = df_filtered['cluster'].astype('category').cat.categories
combined_results = {}
recap_info = []

# Fonction pour stocker les relations significatives à tracer
def store_combined_results(model, ref_cluster, coefficients_mean, coefficients_std):
    coeff_intercept = coefficients_mean['Intercept']
    coeff_gii = coefficients_mean['Q("standardized_global_influence_index")']
    combined_results[ref_cluster] = {'intercept': coeff_intercept, 'gii': coeff_gii, 'significant': False}

    # Vérifier si les pentes et positions des autres clusters par rapport à la référence sont significatives
    for cluster in clusters:
        if cluster != ref_cluster:
            term_cluster = f'cluster[T.{cluster}]'
            term_gii_cluster = f'cluster[T.{cluster}]:Q("standardized_global_influence_index")'
            significant_position = term_cluster in model.pvalues and model.pvalues[term_cluster] < 0.05
            significant_gii = term_gii_cluster in model.pvalues and model.pvalues[term_gii_cluster] < 0.05

            # Stocker les informations significatives si présentes
            if significant_position or significant_gii:
                combined_results[cluster] = combined_results.get(cluster, {})
                if significant_position:
                    combined_results[cluster]['intercept'] = coeff_intercept + coefficients_mean[term_cluster]
                if significant_gii:
                    combined_results[cluster]['gii'] = coeff_gii + coefficients_mean[term_gii_cluster]
                combined_results[cluster]['significant'] = True

                # Ajouter les informations dans le récapitulatif
                recap_info.append({
                    'Cluster': cluster,
                    'Reference': ref_cluster,
                    'Significant Position': significant_position,
                    'Significant Slope': significant_gii,
                    'p-value Position': model.pvalues.get(term_cluster, np.nan),
                    'p-value Slope': model.pvalues.get(term_gii_cluster, np.nan)
                })

# Étudier chaque cluster comme référence
for cluster_ref in clusters:
    try:
        # Vérifier si le cluster est présent dans les données
        if cluster_ref not in df_filtered['cluster'].cat.categories:
            print(f"Cluster {cluster_ref} non présent dans les données, passage au suivant.")
            continue

        model, shapiro_p_value = run_model_with_reference(cluster_ref, df_filtered)
        print(f"Modèle avec cluster {cluster_ref} comme référence :")
        print(model.summary())

        # Effectuer la validation croisée pour le cluster de référence
        coefficients_mean, coefficients_std = cross_validate_model(df_filtered, cluster_ref)

        # Stocker les relations significatives par rapport à la référence actuelle
        store_combined_results(model, cluster_ref, coefficients_mean, coefficients_std)

    except Exception as e:
        print(f"Erreur pour le cluster {cluster_ref} comme référence : {e}")
        print("Passage au cluster suivant.")
        continue

# Générer une gamme de valeurs de GII pour les prédictions
gii_values = np.linspace(df_filtered['standardized_global_influence_index'].min(),
                         df_filtered['standardized_global_influence_index'].max(), 100)

# Définir des couleurs pour chaque cluster
cluster_colors = {
    'C-0': 'blue',
    'C-1': 'orange',
    'C-2': 'green',
    'C-3': 'red',
    'C-4': 'purple',
    'B-5': 'brown',
    'B-6': 'pink',
    'B-7': 'gray',
}

# Définir des symboles pour les courbes
markers = ['o', 's', '^', 'D', 'P', '*', 'X', '<', '>']

# Tracer les courbes combinées avec des marqueurs et suppression de la légende/quadrillage
plt.figure(figsize=(11, 6))
sorted_clusters = sorted(combined_results.keys())  # Tri alphabétique des clusters

for i, cluster in enumerate(sorted_clusters):
    params = combined_results[cluster]
    intercept = params['intercept']
    gii = params['gii']
    color = cluster_colors.get(cluster, 'black')  # Noir par défaut si cluster non spécifié

    # Tracer la courbe avec des marqueurs réguliers
    y_values = intercept + gii * gii_values
    plt.plot(gii_values, y_values, color=color, lw=2)  # Tracé de la courbe
    plt.scatter(gii_values[::10], y_values[::10], color=color, marker=markers[i % len(markers)], s=50)  # Marqueurs

# Ajouter des détails au graphique
plt.title('Yields predicted as a function of Risk Score for each hazard profile\n(OLS model)', fontsize=20, loc='center', pad=20)
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


# Supprimer le quadrillage et la légende
plt.gca().set_facecolor('white')  # Assurer un fond blanc
plt.grid(False)

# Ajuster les marges
plt.tight_layout()

# Afficher les équations des courbes
for cluster, params in combined_results.items():
    intercept = params['intercept']
    slope = params['gii']
    print(f"Équation pour le cluster {cluster} : y = {intercept:.4f} + {slope:.4f} * x")


# Afficher le graphique
plt.show()

# Calculer les statistiques des rendements prédits pour des Risk Scores spécifiques
def calculate_yield_stats_for_scores(combined_results, risk_scores):
    stats = []

    for score in risk_scores:
        predicted_yields = []
        for cluster, params in combined_results.items():
            intercept = params['intercept']
            slope = params['gii']
            yield_prediction = intercept + slope * score
            predicted_yields.append(yield_prediction)

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
yield_stats = calculate_yield_stats_for_scores(combined_results, risk_scores)

# Afficher les résultats
print("Statistiques des rendements prédits pour les Risk Scores spécifiques :")
print(yield_stats)

