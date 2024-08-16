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
HEATMAP_OUTPUT_PREFIX = "WindSolar3_Co_occurrence_heatmap_final156"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def main():
    # Load and Preprocess Data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)

    # Calculate full co-occurrences
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, CLUSTER_COLUMN)

    # Hardcoded cluster batches
    cluster_batches = [
        ["Market_Based", "Distributed_Industrial_Policy ", "Regional_Autonomy"],
        ["Centralized_Industrial_Policy", "Financial Cross-Cutters", "Adaptive_Pragmatists"]
    ]

    # Print cluster batches
    print("\nCluster Batches:")
    for i, batch in enumerate(cluster_batches):
        print(f"Batch {i+1}: {', '.join(batch)}")

    # Define two distinct color palettes
    color_palette1 = plt.cm.Set1(np.linspace(0, 1, 9))
    color_palette2 = plt.cm.Set2(np.linspace(0, 1, 8))

    # Batch Analysis
    for batch_idx, batch_clusters in enumerate(cluster_batches):
        print(f"\nProcessing Batch {batch_idx + 1}")
        df_batch = df[df[CLUSTER_COLUMN].isin(batch_clusters)].copy()

        if batch_idx == 0:
            n_enablers = 12
            n_entries = 12
            detailed2 = True
            specific = False
        else:
            n_enablers = 11
            n_entries = 10
            detailed2 = True
            specific = False

        # Run Random Forest for the batch
        results = run_random_forest_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            n_enablers,
            n_entries,
            RF_RESULTS_FILE_PREFIX,
            detailed=detailed2,
            cluster_specific=specific,
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
            df=df_batch
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
