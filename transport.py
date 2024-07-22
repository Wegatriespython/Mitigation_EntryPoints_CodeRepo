import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.analysis.random_forest import load_or_run_random_forest_analysis
from src.analysis.Co_occurrence import run_co_occurrence_analysis
from src.visualization.heatmap import create_and_save_heatmap

# File paths and settings
INPUT_FILE = "C:/Users/vigne/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Mitigation_EntryPoints_CodeRepo/data/raw/Codebook_Transport.xlsm"
RF_RESULTS_FILE = "rf_analysis_results.joblib"
HEATMAP_OUTPUT = "transport_co_occurrence_heatmap.png"

# Column names
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"
CLUSTER_COLUMN = "Cluster"

# Analysis parameters
N_ENABLERS = 15
N_ENTRIES = 10
DETAILED_RF = False  # Set to True for detailed analysis, False for vanilla

def main():
    print("Starting analysis pipeline...")

    # Run Random Forest Analysis
    print("Running Random Forest analysis...")
    rf_results = load_or_run_random_forest_analysis(
        INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN, 
        N_ENABLERS, N_ENTRIES, RF_RESULTS_FILE, DETAILED_RF
    )
    
    # Extract top enablers and entries from RF results
    top_enablers = rf_results['top_enablers']
    top_entries = rf_results['top_entries']
    
    print(f"Top {N_ENABLERS} Enablers:")
    for enabler in top_enablers:
        importance = rf_results['feature_imp'][rf_results['feature_imp']['feature'] == enabler]['importance'].values[0]
        print(f"{enabler}: {importance}")

    print(f"\nTop {N_ENTRIES} Entries:")
    for entry in top_entries:
        importance = rf_results['feature_imp'][rf_results['feature_imp']['feature'] == entry]['importance'].values[0]
        print(f"{entry}: {importance}")

    # Run Co-occurrence Analysis
    print("\nRunning Co-occurrence analysis...")
    co_occurrence_data = run_co_occurrence_analysis(
        INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN, 
        top_enablers, top_entries
    )
    
    # Create and save heatmap
    print("Creating and saving heatmap...")
    clusters = rf_results['df'][CLUSTER_COLUMN].unique()
    create_and_save_heatmap(co_occurrence_data, clusters, HEATMAP_OUTPUT)
    
    print(f"Analysis complete. Heatmap saved as {HEATMAP_OUTPUT}")

if __name__ == "__main__":
    main()