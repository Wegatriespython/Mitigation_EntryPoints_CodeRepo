import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from src.analysis.Co_occurrence import run_co_occurrence_analysis
from src.analysis.Co_occurrence import calculate_co_occurrence
from src.data_processing.general_preprocessing import load_and_preprocess
from src.visualization.heatmap import create_and_save_heatmap
from src.analysis.random_forest import run_random_forest_analysis
from src.analysis.Co_occurrence import get_bisection_data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# File paths and settings
INPUT_FILE = r"C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\REWindSolar.xlsx"
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsRE_"
HEATMAP_OUTPUT_PREFIX = "WindSolar3_Co_occurrence_heatmap_final15"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def bisect_clusters(distance_matrix, bisection_data):
    n_clusters = len(distance_matrix)
    co_occurrence_matrices = bisection_data['co_occurrence_matrices']
    clusters = list(co_occurrence_matrices.keys())
    
    def calculate_group_score(group):
        if len(group) < 2:
            return 0
        
        group_matrices = [co_occurrence_matrices[clusters[i]] for i in group]
        
        # Sum co-occurrence matrices for the group
        group_sum = sum(group_matrices)
        
        # Calculate entry and enabler similarities
        entry_sim = group_sum.sum(axis=0).mean()  # Average similarity across entries
        enabler_sim = group_sum.sum(axis=1).mean()  # Average similarity across enablers
        
        return entry_sim - enabler_sim  # High entry similarity, low enabler similarity
    
    best_split = None
    best_score = float('-inf')
    
    for i in range(1, 2**n_clusters - 1):
        group1 = [j for j in range(n_clusters) if (i & (1 << j))]
        group2 = [j for j in range(n_clusters) if not (i & (1 << j))]
        
        if len(group1) == 0 or len(group2) == 0 or abs(len(group1) - len(group2)) > 1:
            continue
        
        score = calculate_group_score(group1) + calculate_group_score(group2)
        
        if score > best_score:
            best_score = score
            best_split = (group1, group2)
    
    return best_split

def main():
    # Load and Preprocess Data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)
    
    # Calculate full co-occurrences
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, CLUSTER_COLUMN)
    
    # Determine clusters
    clusters = df[CLUSTER_COLUMN].unique()
    print(f"Clusters: {clusters}")
    print(f"Number of clusters: {len(clusters)}")

    # Get bisection data
    bisection_data = get_bisection_data(df, vectorized_data, CLUSTER_COLUMN)

    # Calculate distances between cluster centers
    center_vectors = np.array(list(bisection_data['center_vectors'].values()))
    distances = np.zeros((len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distances[i, j] = distances[j, i] = np.linalg.norm(center_vectors[i] - center_vectors[j])
    
    print("Distance matrix:")
    print(distances)

    # Bisect clusters
    if len(clusters) > 4:
        print("Bisecting clusters...")
        group1, group2 = bisect_clusters(distances, bisection_data)
        cluster_batches = [
            [clusters[i] for i in group1],
            [clusters[i] for i in group2]
        ]
    else:
        cluster_batches = [clusters]

    # Print cluster batches
    print("\nCluster Batches:")
    for i, batch in enumerate(cluster_batches):
        print(f"Batch {i+1}: {', '.join(map(str, batch))}")

    # Define two distinct color palettes
    color_palette1 = plt.cm.Set1(np.linspace(0, 1, 9))
    color_palette2 = plt.cm.Set2(np.linspace(0, 1, 8))

    # Batch Analysis
    for batch_idx, batch_clusters in enumerate(cluster_batches):
        print(f"\nProcessing Batch {batch_idx + 1}")
        df_batch = df[df[CLUSTER_COLUMN].isin(batch_clusters)].copy()
        
        if batch_idx == 0:
            n_enablers = 9
            n_entries = 12
        else : 
            n_enablers = 12
            n_entries = 9   
        # Run Random Forest for the batch
        results = run_random_forest_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            n_enablers,  # n_enablers
            n_entries,  # n_entries
            RF_RESULTS_FILE_PREFIX,
            detailed= False,
            cluster_specific= True,
            df=df_batch
        )
        top_enablers = results['top_enablers']
        top_entries = results['top_entries']

        # Run Co-occurrence Analysis for the batch
        co_occurrence_data, enabler_importance, secular_enablers = run_co_occurrence_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            top_enablers,
            top_entries,
            df=df_batch  # Pass the batch-specific dataframe
        )

        # Create and Save Heatmap for the batch
        output_dir = os.path.join(os.path.dirname(os.path.dirname(INPUT_FILE)), "output")
        os.makedirs(output_dir, exist_ok=True)
        heatmap_output = os.path.join(output_dir, f"{HEATMAP_OUTPUT_PREFIX}batch_{batch_idx + 1}.png")
        
        # Use different color palette for each batch
        title = f"Wind & Solar Entry Points for Unlocks Part {batch_idx + 1}"
        color_palette = color_palette1 if batch_idx == 0 else color_palette2
        
        print(f"Creating heatmap for Batch {batch_idx + 1}")
        print(f"Clusters in this batch: {batch_clusters}")
        create_and_save_heatmap(co_occurrence_data, batch_clusters, heatmap_output, 
                                color_palette=color_palette, title=title)

        # Print secular enablers for this batch
        print(f"\nSecular Enablers for Batch {batch_idx + 1}:")
        for enabler in secular_enablers:
            print(f"- {enabler}")

if __name__ == "__main__":
    main()