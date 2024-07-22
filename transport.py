import pandas as pd
import joblib
from src.analysis.Co_occurrence import prepare_heatmap_data  
from src.visualization.heatmap import create_and_save_heatmap 
from src.analysis.random_forest import run_random_forest_analysis
from src.data_processing.general_preprocessing import load_and_preprocess
# File paths and settings

INPUT_FILE = "C:/Users/vigneshr/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Mitigation_EntryPoints_CodeRepo/data/raw/Codebook_Transport.xlsm"
RF_RESULTS_FILE = "rf_analysis_results.joblib"
HEATMAP_OUTPUT = "transport_co_occurrence_heatmap.png"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

# Load Random Forest results 
df = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)
results = run_random_forest_analysis(df, ENABLER_COLUMN,ENTRY_COLUMN,CLUSTER_COLUMN, RF_RESULTS_FILE)
df = results['df']
top_enablers = results['top_enablers']
top_entries = results['top_entries']
feature_imp = results['feature_imp'] 

# Prepare data for heatmap
co_occurrence_matrix = prepare_heatmap_data(df, top_enablers, top_entries, feature_imp)

# Create and save heatmap
clusters = df[CLUSTER_COLUMN].unique()
create_and_save_heatmap(co_occurrence_matrix, clusters, HEATMAP_OUTPUT)