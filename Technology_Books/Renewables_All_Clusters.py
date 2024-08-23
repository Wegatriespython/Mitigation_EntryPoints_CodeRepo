import numpy as np
import os

import sys
# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.Co_occurrence import run_co_occurrence_analysis
from src.analysis.Co_occurrence import calculate_co_occurrence
from src.data_processing.general_preprocessing import load_and_preprocess
from src.visualization.heatmap_big import create_and_save_heatmap
from src.analysis.random_forest import run_random_forest_analysis
from src.analysis.Co_occurrence import get_bisection_data
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# File paths and settings
INPUT_FILE = r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\REWindSolar.xlsx"
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsRE_ALL_S"
HEATMAP_OUTPUT_PREFIX = "WindSolar_Co_occurrence_heatmap_all_clusters96"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def main():
    # Load and Preprocess Data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)

    # Calculate full co-occurrences
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, CLUSTER_COLUMN)

    # Determine clusters
    clusters = df[CLUSTER_COLUMN].unique()
    print(f"Clusters: {clusters}")
    print(f"Number of clusters: {len(clusters)}")

    # Get bisection data (not used for bisection in this case, but kept for consistency)
    bisection_data = get_bisection_data(df, vectorized_data, CLUSTER_COLUMN)

    # Define color palette for all 6 clusters
    color_palette = plt.cm.Set1(np.linspace(0, 1, 6))

    # Define cluster order to match the legend in the image
    cluster_order = [
        "Centralized_Industrial_Policy",
        "Financial Cross-Cutters",
        "Market_Based",
        "Distributed_Industrial_Policy ",
        "Regional_Autonomy",
        "Adaptive_Pragmatists"
    ]

    # Create a dictionary mapping cluster names to colors
    cluster_colors = dict(zip(cluster_order, color_palette))

    # Run Random Forest Analysis
    results = run_random_forest_analysis(
        INPUT_FILE,
        ENABLER_COLUMN,
        ENTRY_COLUMN,
        CLUSTER_COLUMN,
        20,  # n_enablers
        20,  # n_entries
        RF_RESULTS_FILE_PREFIX,
        detailed=False,
        cluster_specific=True,
        df=df
    )
    top_enablers = results['top_enablers']
    top_entries = results['top_entries']

    # Run Co-occurrence Analysis
    co_occurrence_data, enabler_importance, secular_enablers = run_co_occurrence_analysis(
        INPUT_FILE,
        ENABLER_COLUMN,
        ENTRY_COLUMN,
        CLUSTER_COLUMN,
        top_enablers,
        top_entries
    )
    print(top_enablers, top_entries)
    # Print secular enablers
    print(f"\nSecular Enablers:")
    for enabler in secular_enablers:
        print(f"- {enabler}")

    # Create and Save Heatmap
    output_dir = os.path.join(os.path.dirname(os.path.dirname(INPUT_FILE)), "output")
    os.makedirs(output_dir, exist_ok=True)
    heatmap_output = os.path.join(output_dir, f"{HEATMAP_OUTPUT_PREFIX}.png")

    # Define the title for the heatmap
    title = "Wind & Solar Entry Points for Unlocks - All Clusters"
    create_and_save_heatmap(co_occurrence_data, cluster_order, heatmap_output,
                            color_palette=color_palette, title=title, threshold=4)

if __name__ == "__main__":
    main()
