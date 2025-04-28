import os
import numpy as np
import matplotlib.pyplot as plt
import sys
# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.Co_occurrence import run_co_occurrence_analysis, calculate_co_occurrence, get_bisection_data
from src.data_processing.general_preprocessing import load_and_preprocess
from src.visualization.heatmap import create_and_save_heatmap
from src.analysis.random_forest import run_random_forest_analysis

# File paths and settings
file_name = "Codebook_BuildingsV1.xlsm"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the parent directory
parent_dir = os.path.dirname(script_dir)
INPUT_FILE = os.path.join(parent_dir, "data", "raw", file_name)
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsBuildings_detS"
HEATMAP_OUTPUT_PREFIX = "Buildings_Co_occurrence_heatmap_finalS22"
CLUSTER_COLUMN = "technology_mitigation_option"
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

    # Define color palette
    color_palette = plt.cm.Set1(np.linspace(0, 1, 9))

    # Run Random Forest Analysis
    results = run_random_forest_analysis(
        INPUT_FILE,
        ENABLER_COLUMN,
        ENTRY_COLUMN,
        CLUSTER_COLUMN,
        10,  # n_enablers
        10,  # n_entries
        RF_RESULTS_FILE_PREFIX,
        detailed= True,
        cluster_specific= False,
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

    # Print secular enablers
    print(f"\nSecular Enablers:")
    for enabler in secular_enablers:
        print(f"- {enabler}")

    # Create and Save Heatmap
    output_dir = os.path.join(os.path.dirname(os.path.dirname(INPUT_FILE)), "output")
    os.makedirs(output_dir, exist_ok=True)
    heatmap_output = os.path.join(output_dir, f"{HEATMAP_OUTPUT_PREFIX}.png")

    # Define the title for the heatmap
    title = "Buildings Entry Points for Unlocks"
    create_and_save_heatmap(co_occurrence_data, clusters, heatmap_output,
                            color_palette=color_palette, title=title)

if __name__ == "__main__":
    main()
