import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from src.analysis.Co_occurrence import prepare_heatmap_data, calculate_co_occurrence  
from src.visualization.heatmap import create_and_save_heatmap 
from src.analysis.random_forest import run_random_forest_analysis
from src.data_processing.general_preprocessing import load_and_preprocess
import matplotlib.pyplot as plt

# File paths and settings
INPUT_FILE = "C:/Users/vigne/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Mitigation_EntryPoints_CodeRepo/data/raw/REWindSolar.xlsx"
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsRE_"
HEATMAP_OUTPUT_PREFIX = "Renewables:Wind&Solar_Co_occurrence_heatmap_"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def calculate_cluster_centers(co_occurrence_matrices):
    # Convert CSR matrix to dense array, then flatten
    centers = np.array([matrix.toarray().flatten() for matrix in co_occurrence_matrices.values()])
    return centers

def bisect_clusters(distance_matrix):
    n_clusters = len(distance_matrix)
    best_split = None
    min_difference = float('inf')
    
    for i in range(1, 2**n_clusters):
        group1 = [j for j in range(n_clusters) if (i & (1 << j))]
        group2 = [j for j in range(n_clusters) if not (i & (1 << j))]
        
        if len(group1) == 0 or len(group2) == 0:
            continue
        
        mean_dist1 = np.mean([distance_matrix[a][b] for a, b in combinations(group1, 2)]) if len(group1) > 1 else 0
        mean_dist2 = np.mean([distance_matrix[a][b] for a, b in combinations(group2, 2)]) if len(group2) > 1 else 0
        
        difference = abs(mean_dist1 - mean_dist2)
        
        if difference < min_difference:
            min_difference = difference
            best_split = (group1, group2)
    
    return best_split

def main():
    # Load and Preprocess Data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)
    
    # Calculate full co-occurrences
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, CLUSTER_COLUMN)
    
    # Determine clusters and batches
    clusters = df[CLUSTER_COLUMN].unique()
    if len(clusters) > 4:
        # Calculate cluster centers
        center_vectors = calculate_cluster_centers(co_occurrence_matrices)
        
        # Calculate distances between cluster centers
        distances = pdist(center_vectors)
        distance_matrix = squareform(distances)
        
        # Bisect clusters
        group1, group2 = bisect_clusters(distance_matrix)
        cluster_batches = [
            [clusters[i] for i in group1],
            [clusters[i] for i in group2]
        ]
    else:
        cluster_batches = [clusters]
    # Color palettes
    color_palettes = [plt.cm.viridis, plt.cm.plasma]

    # Batch Analysis
    for batch_idx, batch_clusters in enumerate(cluster_batches):
        df_batch = df[df[CLUSTER_COLUMN].isin(batch_clusters)]

        # Run Random Forest for the batch
        rf_results_file = f"{RF_RESULTS_FILE_PREFIX}batch_{batch_idx + 1}.joblib"
        results = run_random_forest_analysis(
            INPUT_FILE,  # file_path
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            15,  # n_enablers (you may want to define this as a constant)
            10,  # n_entries (you may want to define this as a constant)
            rf_results_file,
            detailed=False  # or True, depending on your preference
        )
        df_batch = results['df']
        top_enablers = results['top_enablers']
        top_entries = results['top_entries']
        feature_imp = results['feature_imp']

        # Prepare Heatmap Data for the batch
        co_occurrence_matrix = prepare_heatmap_data(df_batch, top_enablers, top_entries, feature_imp)

        # Create and Save Heatmap for the batch
        heatmap_output = f"{HEATMAP_OUTPUT_PREFIX}batch_{batch_idx + 1}.png"
        create_and_save_heatmap(co_occurrence_matrix, batch_clusters, heatmap_output, color_palette=color_palettes[batch_idx % len(color_palettes)])

if __name__ == "__main__":
    main()