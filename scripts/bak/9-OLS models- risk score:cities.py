import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from scipy.stats import shapiro
import scipy.stats as stats

# Charger le fichier avec les données incluant les clusters et le GII
file_path = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/risk_scores_individuals_C.xlsx'
df = pd.read_excel(file_path)

# Filtrer les individus avec irrigation inférieure à 35% (si nécessaire)
df_filtered = df[df['Irrigation(%)'] <100]
print("Clusters présents dans df_filtered :")
print(df_filtered['cluster'].unique())

# Convertir les clusters en chaînes de caractères si nécessaire
df_filtered['cluster'] = df_filtered['cluster'].astype(str)
df_filtered['cluster'] = df_filtered['cluster'].astype('category')

# Fonction pour ajuster le modèle en prenant un cluster donné comme référence
def run_model_with_reference(cluster_ref, df_filtered):
    df_filtered['cluster'] = df_filtered['cluster'].astype('category')
    categories = df_filtered['cluster'].cat.categories
    new_order = [str(cluster_ref)] + [str(c) for c in categories if str(c) != str(cluster_ref)]
    df_filtered['cluster'] = df_filtered['cluster'].cat.reorder_categories(new_order, ordered=True)
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index")+ Q("Irrigation(%)")'
    y, X = dmatrices(formula, data=df_filtered, return_type='dataframe')
    model = sm.OLS(y, X).fit()

    # Diagnostic de la normalité des résidus avec le test de Shapiro-Wilk
    shapiro_p_value = shapiro(model.resid)[1]
    print(f"Test de Shapiro-Wilk pour la normalité des résidus (p-value) : {shapiro_p_value}")

    return model, shapiro_p_value


# Fonction pour effectuer la validation croisée
def cross_validate_model(df_filtered, cluster_ref, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    formula = 'Q("Adjusted yield") ~ cluster * Q("standardized_global_influence_index")+ Q("Irrigation(%)")'

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

# Générer une gamme de valeurs de GII pour les prédictions
gii_values = np.linspace(df_filtered['standardized_global_influence_index'].min(),
                         df_filtered['standardized_global_influence_index'].max(), 100)

# Initialiser une figure pour tracer les courbes
plt.figure(figsize=(10, 6))
clusters = df_filtered['cluster'].astype('category').cat.categories
combined_results = {}
recap_info = []  # Liste pour stocker le récapitulatif
adjusted_p_values = []

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

# Initialiser `all_p_values` comme une liste vide
all_p_values = []

# Remplir `all_p_values` avec les p-values des positions et des pentes
all_p_values = [entry['p-value Position'] for entry in recap_info] + [entry['p-value Slope'] for entry in recap_info]

# Appliquer une correction de Bonferroni sur les p-values pour les positions et les pentes
if all_p_values:  # Vérifier si `all_p_values` contient des valeurs
    rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='bonferroni')

    # Mettre à jour les p-values corrigées dans le récapitulatif
    for i, entry in enumerate(recap_info):
        entry['Adjusted p-value Position'] = corrected_p_values[i]
        entry['Adjusted p-value Slope'] = corrected_p_values[len(recap_info) + i]
else:
    print("Aucune p-value n'a été générée pour la correction de Bonferroni.")

# Définir les couleurs pour chaque cluster (ajustez les couleurs selon vos préférences)
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

# Générer une gamme de valeurs de GII pour les prédictions
gii_values = np.linspace(df_filtered['standardized_global_influence_index'].min(),
                         df_filtered['standardized_global_influence_index'].max(), 100)

# Trier les clusters en tant que chaînes
sorted_clusters = sorted(combined_results.keys())  # Tri alphabétique standard


# Tracer les courbes combinées en utilisant les résultats et indiquant significativité
for cluster in sorted_clusters:
    params = combined_results[cluster]
    intercept = params['intercept']
    gii = params['gii']

    # Chercher la significativité pour la position et la pente
    has_significant_position = any(
        entry['Cluster'] == cluster and entry['Adjusted p-value Position'] < 0.06 for entry in recap_info
    )
    has_significant_slope = any(
        entry['Cluster'] == cluster and entry['Adjusted p-value Slope'] < 0.06 for entry in recap_info
    )

    # Construire le libellé avec des mentions spécifiques
    label_details = []
    if has_significant_position:
        label_details.append("Position sig")
    if has_significant_slope:
        label_details.append("Slope sig")
    if not label_details:
        label_details.append("No sig")

    # Ajouter un saut de ligne entre "Position sig" et "Slope sig" si nécessaire
    if len(label_details) > 1:
        joined_label = " & ".join(label_details)
        wrapped_label = joined_label.replace(" & ", "\n")  # Remplacer " & " par un retour à la ligne
    else:
        wrapped_label = label_details[0]

    # Libellé final
    label = f' {cluster}\n({wrapped_label})'

    # Récupérer la couleur pour le cluster actuel
    color = cluster_colors.get(cluster, 'black')  # Noir par défaut si cluster non spécifié

    # Tracer la courbe avec la couleur assignée
    plt.plot(gii_values, intercept + gii * gii_values, label=label, color=color)

# Ajouter des détails au graphique
plt.title(' Yields predicted as a function of Risk Score\n(OLS model)', fontsize=18)
plt.xlabel('Risk Score', fontsize=16)
plt.ylabel('Predicted yield (t/ha)', fontsize=16)
# Définir l'échelle des axes
plt.xlim(0, 10)  # Ajuster l'échelle des abscisses (exemple : de 0 à 1)
plt.ylim(0, 10)  # Ajuster l'échelle des ordonnées (exemple : de 0 à 10)
plt.legend(title="Hazard profile",title_fontsize=16,bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=16)
# Augmenter la taille des ticks des axes
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.tight_layout()

# Afficher le graphique
plt.show()


# Calculer les rendements max et min pour tous les GII confondus
def calculate_global_max_min_all(combined_results, gii_range):
    predicted_yields = []

    for gii in gii_range:
        for cluster, params in combined_results.items():
            intercept = params['intercept']
            slope = params['gii']
            predicted_yield = intercept + slope * gii
            predicted_yields.append(predicted_yield)

    # Retourner les rendements maximum et minimum globalement
    return max(predicted_yields), min(predicted_yields)


# Fonction pour calculer le score de risque normalisé
def calculate_normalized_risk_score_global(row, combined_results, global_max, global_min):
    gii = row['standardized_global_influence_index']
    target_cluster = row['cluster']

    # Calculer le rendement prédit pour l'observation dans son cluster
    params = combined_results[target_cluster]
    intercept = params['intercept']
    slope = params['gii']
    predicted_yield = intercept + slope * gii

    # Normaliser le score avec les max/min globaux
    if global_max == global_min:  # Éviter la division par zéro
        return 0
    else:
        return (global_max - predicted_yield) / (global_max - global_min)


# Étape 1 : Calculer le max et le min globalement pour tous les clusters et GII
gii_range = np.linspace(df_filtered['standardized_global_influence_index'].min(),
                        df_filtered['standardized_global_influence_index'].max(), 100)

global_max, global_min = calculate_global_max_min_all(combined_results, gii_range)

# Étape 2 : Appliquer la normalisation à chaque observation
df_filtered['Normalized Risk Score'] = df_filtered.apply(
    calculate_normalized_risk_score_global,
    axis=1,
    combined_results=combined_results,
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
output_file = '/Users/sophiegrosse/Documents/Greenresilience/Doctorat/Database/Irrigation/output_adjusted_risk_scores_B_4.xlsx'
with pd.ExcelWriter(output_file) as writer:
    aggregated_scores.to_excel(writer, sheet_name='Aggregated Scores by Town', index=False)
    cluster_contributions.to_excel(writer, sheet_name='Cluster Contributions', index=False)
    final_output.to_excel(writer, sheet_name='Final Output', index=False)
    cluster_occurrences.to_excel(writer, sheet_name='Cluster Occurrence Percentage', index=False)

print(f"Les scores ajustés et agrégés ont été sauvegardés sous le nom : {output_file}")

