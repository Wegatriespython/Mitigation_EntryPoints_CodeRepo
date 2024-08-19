from src.analysis.random_forest import run_random_forest_analysis
from src.analysis.Co_occurrence import run_co_occurrence_analysis
from visualization.heatmap_space_alt import create_and_save_heatmap

def main():
    # Configuration
    file_path = "path/to/your/data.xlsx"
    enabler_column = "Enabler"
    entry_column = "Entry (policy intervention)"
    cluster_column = "Cluster"
    n_enablers = 15
    n_entries = 10
    
    # Run Random Forest Analysis
    rf_results = run_random_forest_analysis(file_path, enabler_column, entry_column, cluster_column, n_enablers, n_entries)
    
    # Extract top enablers and entries from RF results
    top_enablers = rf_results['top_enablers']
    top_entries = rf_results['top_entries']
    
    # Run Co-occurrence Analysis
    co_occurrence_data = run_co_occurrence_analysis(file_path, enabler_column, entry_column, cluster_column, top_enablers, top_entries)
    
    # Create and save heatmap
    create_and_save_heatmap(co_occurrence_data, rf_results['df'][cluster_column].unique(), "co_occurrence_heatmap.png")

if __name__ == "__main__":
    main()