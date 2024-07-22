import pandas as pd
import joblib
from src.analysis.Co_occurrence import prepare_heatmap_data  
from src.visualization.heatmap import create_and_save_heatmap 
from src.analysis.random_forest import run_random_forest_analysis
from src.data_processing.general_preprocessing import load_and_preprocess
import matplotlib.pyplot as plt

# File paths and settings
INPUT_FILE = "C:\\Users\\vigneshr\\OneDrive - Wageningen University & Research\\Internship\\Literature Review\\Final Data Processing\\Mitigation_EntryPoints_CodeRepo\\data\\raw\\REWindSolar.xlsx"
RF_RESULTS_FILE_PREFIX = "rf_analysis_resultsRE_"
HEATMAP_OUTPUT_PREFIX = "Renewables:Wind&Solar_Co_occurrence_heatmap_"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

# Define cluster batches and color palettes
cluster_batches = [
    ["Centralized_Industrial_Policy", "Decentralized_Industrial_Policy", "Market_Based"],
    ["Adaptive_Pragmatists", "Financial_Cross-Cutters", "Regional_Autonomy"] 
]

color_palettes = [
    plt.cm.viridis,  # Colormap for the first batch
    plt.cm.plasma    # Colormap for the second batch
]

# Load and Preprocess Data
df = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)

# Batch Analysis
for batch_idx, batch_clusters in enumerate(cluster_batches):
    df_batch = df[df[CLUSTER_COLUMN].isin(batch_clusters)]

    # Run Random Forest for the batch
    rf_results_file = f"{RF_RESULTS_FILE_PREFIX}batch_{batch_idx + 1}.joblib"
    results = run_random_forest_analysis(df_batch, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN, rf_results_file)
    df_batch = results['df']
    top_enablers = results['top_enablers']
    top_entries = results['top_entries']
    feature_imp = results['feature_imp'] 

    # Prepare Heatmap Data for the batch
    co_occurrence_matrix = prepare_heatmap_data(df_batch, top_enablers, top_entries, feature_imp)

    # Create and Save Heatmap for the batch
    heatmap_output = f"{HEATMAP_OUTPUT_PREFIX}batch_{batch_idx + 1}.png"
    create_and_save_heatmap(co_occurrence_matrix, batch_clusters, heatmap_output, color_palette=color_palettes[batch_idx]) 