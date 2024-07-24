
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from src.analysis.Co_occurrence import run_co_occurrence_analysis
from src.analysis.Co_occurrence import calculate_co_occurrence
from src.data_processing.general_preprocessing import load_and_preprocess
from src.visualization.heatmap import create_and_save_heatmap
from src.analysis.random_forest import run_random_forest_analysis
import matplotlib.pyplot as plt

# File paths and settings
INPUT_FILE = "C:\\Users\\vigneshr\\OneDrive - Wageningen University & Research\\Internship\\Literature Review\\Final Data Processing\\Mitigation_EntryPoints_CodeRepo\\data\\raw\\REWindSolar.xlsx"
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsRE_"
HEATMAP_OUTPUT_PREFIX = "WindSolar3_Co_occurrence_heatmap_true"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def calculate_cluster_centers(co_occurrence_matrices):
    centers = np.array([matrix.toarray().flatten() for matrix in co_occurrence_matrices.values()])
    return centers

def bisect_clusters(distance_matrix):
    n_clusters = len(distance_matrix)
    best_split = None
    min_difference = float('inf')
    
    # If n_clusters is even, we only consider equal splits
    if n_clusters % 2 == 0:
        target_group_size = n_clusters // 2
    else:
        target_group_size = None

    for i in range(1, 2**n_clusters):
        group1 = [j for j in range(n_clusters) if (i & (1 << j))]
        group2 = [j for j in range(n_clusters) if not (i & (1 << j))]
        
        # Skip if either group is empty
        if len(group1) == 0 or len(group2) == 0:
            continue
        
        # For even n_clusters, only consider equal splits
        if target_group_size is not None and (len(group1) != target_group_size or len(group2) != target_group_size):
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
        # Print cluster batches
        print("Cluster Batches:")
        for i, batch in enumerate(cluster_batches):
            print(f"Batch {i+1}: {', '.join(batch)}")
    else:
        cluster_batches = [clusters]
        print("Cluster Batch:")
        print(f"Single Batch: {', '.join(clusters)}")
    # Define two distinct color palettes
    color_palette1 = plt.cm.Set1(np.linspace(0, 1, 9))
    color_palette2 = plt.cm.Set2(np.linspace(0, 1, 8))
    # Batch Analysis
    for batch_idx, batch_clusters in enumerate(cluster_batches):
        df_batch = df[df[CLUSTER_COLUMN].isin(batch_clusters)].copy()
           # Run Random Forest for the batch
        results= run_random_forest_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            10,  # n_enablers
            10,  # n_entries
            RF_RESULTS_FILE_PREFIX,
            detailed=True,
            df=df_batch  # Pass the subset dataframe to the function
        )
        top_enablers = results['top_enablers']
        top_entries = results['top_entries']

        # Run Co-occurrence Analysis for the batch
        co_occurrence_data = run_co_occurrence_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            top_enablers,
            top_entries
           # Pass the subset dataframe to the function
        )

        # Create and Save Heatmap for the batch
        output_dir = os.path.join(os.path.dirname(os.path.dirname(INPUT_FILE)), "output")
        os.makedirs(output_dir, exist_ok=True)
        heatmap_output = os.path.join(output_dir, f"{HEATMAP_OUTPUT_PREFIX}batch_{batch_idx + 1}.png")
        # Use different color palette for each batch
        color_palette = color_palette1 if batch_idx == 0 else color_palette2
        create_and_save_heatmap(co_occurrence_data, batch_clusters, heatmap_output, color_palette=color_palette)
        plt.close('all')
if __name__ == "__main__":
    main()