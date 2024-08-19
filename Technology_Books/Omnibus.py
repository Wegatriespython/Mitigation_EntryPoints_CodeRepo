import numpy as np
import pandas as pd
import os
import sys
import joblib

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.Co_occurrence import run_co_occurrence_analysis, calculate_co_occurrence
from src.data_processing.general_preprocessing import load_and_preprocess
from visualization.heatmap_space_alt import create_and_save_heatmap
from src.analysis.random_forest import run_random_forest_analysis

# File paths and settings
INPUT_FILE = r"C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\Codebook_Omnibus_Extractions_Select_Columns_W_Duplicates.xlsx"
HEATMAP_OUTPUT_PREFIX = "Omnibus_Co_occurrence_heatmap_Final4"
CLUSTER_COLUMN = "Sector\Scope "
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

n_entries = 9
n_enablers = 12

def get_top_features(df, n_enablers, n_entries):
    rf_results_file = "rf_results_globalFinal.joblib"

    if os.path.exists(rf_results_file):
        print(f"Loading existing Random Forest results from {rf_results_file}")
        rf_results = joblib.load(rf_results_file)
    else:
        print("Running Random Forest analysis...")
        rf_results = run_random_forest_analysis(
            INPUT_FILE,
            ENABLER_COLUMN,
            ENTRY_COLUMN,
            CLUSTER_COLUMN,
            n_enablers,  # n_enablers
            n_entries,  # n_entries
            rf_results_file,
            detailed=False,
            cluster_specific=True,
            df=df
        )

    return rf_results['top_enablers'], rf_results['top_entries']

def main():
    # Load and Preprocess Data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)

    # Get unique clusters (sectors/scopes)
    clusters = df[CLUSTER_COLUMN].unique()
    print(f"Clusters: {clusters}")

    # Get top 15 enablers and entries for all data
    top_enablers, top_entries = get_top_features(df, n_enablers, n_entries)

    # Run Co-occurrence Analysis for all data
    co_occurrence_data, enabler_importance, secular_enablers = run_co_occurrence_analysis(
        INPUT_FILE,
        ENABLER_COLUMN,
        ENTRY_COLUMN,
        CLUSTER_COLUMN,
        top_enablers,
        top_entries,
        df=df
    )

    # Create and Save Heatmap
    output_dir = os.path.join(os.path.dirname(os.path.dirname(INPUT_FILE)), "output")
    os.makedirs(output_dir, exist_ok=True)
    heatmap_output = os.path.join(output_dir, f"{HEATMAP_OUTPUT_PREFIX}_global.png")

    create_and_save_heatmap(co_occurrence_data, clusters, heatmap_output,
                            title="Global Co-occurrence Heatmap")

    # Print secular enablers
    print("\nSecular Enablers:")
    for enabler in secular_enablers:
        print(f"- {enabler}")

if __name__ == "__main__":
    main()
