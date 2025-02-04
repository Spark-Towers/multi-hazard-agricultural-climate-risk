import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import argparse

def load_and_label_clusters(file_path, label):
    """Loads cluster data from an Excel file and labels clusters."""
    df = pd.read_excel(file_path)
    df['cluster'] = df['cluster'].apply(lambda x: f"{label}-{x}")
    return df

def standardize_variables(df, variables):
    """Standardizes selected climatic variables."""
    scaler = StandardScaler()
    df[variables] = scaler.fit_transform(df[variables].fillna(0))
    return df

def compute_correlations_for_clusters(df, rendement_var, groupes_variables):
    """Computes correlation matrices for each cluster."""
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
            cluster_correlations[cluster] = pd.Series(0, index=groupes_variables.keys())
    return cluster_correlations

def calculate_risk_score(df, correlations_by_cluster, groupes_variables):
    """Calculates risk scores for each individual based on correlations."""
    risk_scores = []
    for index, row in df.iterrows():
        cluster = row['cluster']
        correlations = correlations_by_cluster.get(cluster, pd.Series(0, index=groupes_variables.keys()))
        risk_score = sum(row[variables].mean() * abs(correlations.get(group_name, 0)) for group_name, variables in groupes_variables.items())
        risk_scores.append(risk_score)
    df['Risk Score'] = risk_scores
    return df

def plot_heatmap(df, title, output_path):
    """Generates and saves a heatmap for the given DataFrame."""
    plt.figure(figsize=(45, 40))
    heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 31}, linewidths=0.5, linecolor="black")
    plt.title(title, fontsize=45)
    plt.tick_params(axis='x', labelsize=32)
    plt.tick_params(axis='y', labelsize=32)
    plt.xlabel("Hazard group", fontsize=40)
    plt.ylabel("Climatic profile", fontsize=40)
    plt.savefig(output_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing cluster files")
    parser.add_argument("--output", type=str, default="risk_scores.xlsx", help="Output file path")
    args = parser.parse_args()

    df1 = load_and_label_clusters(os.path.join(args.folder, 'clusteringA.xlsx'), 'A')
    df2 = load_and_label_clusters(os.path.join(args.folder, 'clusteringB.xlsx'), 'B')
    df3 = load_and_label_clusters(os.path.join(args.folder, 'clusteringC.xlsx'), 'C')
    df_combined = pd.concat([df1, df2, df3])

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
    df_combined = standardize_variables(df_combined, variables_climatiques)
    
    rendement_var = 'Adjusted yield'
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

    correlations_by_cluster = compute_correlations_for_clusters(df_combined, rendement_var, groupes_variables)
    df_with_risk_score = calculate_risk_score(df_combined, correlations_by_cluster, groupes_variables)
    df_with_risk_score.to_excel(args.output, index=False)
    
    print(f"Results saved to {args.output}")

    all_correlations_df = pd.DataFrame(correlations_by_cluster).T
    heatmap_path = os.path.join(args.folder, "heatmap_correlation.png")
    plot_heatmap(all_correlations_df, "Heatmap of Correlations to Yields", heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")

if __name__ == "__main__":
    main()
