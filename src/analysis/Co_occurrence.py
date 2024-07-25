import pandas as pd
import numpy as np
from src.data_processing.general_preprocessing import load_and_preprocess
from src.data_processing.general_preprocessing import vectorize_data

def calculate_co_occurrence(df, vectorized_data, cluster_column):
    enabler_matrix = vectorized_data['enabler_matrix']
    entry_matrix = vectorized_data['entry_matrix']
    clusters = df[cluster_column].unique()
    
    co_occurrence_matrices = {}
    for cluster in clusters:
        cluster_mask = (df[cluster_column] == cluster).values
        cluster_enablers = enabler_matrix[cluster_mask]
        cluster_entries = entry_matrix[cluster_mask]
        
        co_occurrence = (cluster_enablers.T @ cluster_entries).tocsr()
        co_occurrence_matrices[cluster] = co_occurrence
    
    return co_occurrence_matrices

def prepare_heatmap_data(co_occurrence_matrices, vectorized_data, top_enablers, top_entries):
    enabler_features = vectorized_data['enabler_features']
    entry_features = vectorized_data['entry_features']
    
    # Get indices of top enablers and entries
    top_enabler_indices = [np.where(enabler_features == e)[0][0] for e in top_enablers if e in enabler_features]
    top_entry_indices = [np.where(entry_features == e)[0][0] for e in top_entries if e in entry_features]
    
    # Create the combined matrix
    combined_matrix = np.zeros((len(top_enabler_indices), len(top_entry_indices), len(co_occurrence_matrices)))
    
    for i, (cluster, matrix) in enumerate(co_occurrence_matrices.items()):
        submatrix = matrix[top_enabler_indices, :][:, top_entry_indices].toarray()
        combined_matrix[:, :, i] = submatrix
    
    # Convert to pandas DataFrame
    heatmap_data = pd.DataFrame(
        data=[tuple(row) for row in combined_matrix.reshape(-1, len(co_occurrence_matrices))],
        index=pd.MultiIndex.from_product([top_enablers, top_entries]),
        columns=co_occurrence_matrices.keys()
    )
    
    # Calculate enabler importance across clusters
    enabler_importance = {}
    for enabler in top_enablers:
        enabler_importance[enabler] = heatmap_data.loc[enabler].sum()
    
    return heatmap_data, enabler_importance

def identify_secular_enablers(enabler_importance, threshold=0.8):
    """
    Identify secular enablers that are important across all clusters.
    
    Args:
    enabler_importance (dict): Dictionary of enabler importance across clusters.
    threshold (float): Minimum importance threshold (as a fraction of max importance).
    
    Returns:
    list: List of secular enablers.
    """
    secular_enablers = []
    for enabler, importances in enabler_importance.items():
        min_importance = importances.min()
        max_importance = importances.max()
        if min_importance >= threshold * max_importance:
            secular_enablers.append(enabler)
    return secular_enablers

def get_bisection_data(df, vectorized_data, cluster_column):
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, cluster_column)
    
    # Calculate center vectors for each cluster
    center_vectors = {}
    for cluster, matrix in co_occurrence_matrices.items():
        center_vectors[cluster] = matrix.toarray().flatten()
    
    return {
        'co_occurrence_matrices': co_occurrence_matrices,
        'center_vectors': center_vectors,
        'enabler_features': vectorized_data['enabler_features'],
        'entry_features': vectorized_data['entry_features']
    }
def run_co_occurrence_analysis(file_path, enabler_column, entry_column, cluster_column, top_enablers, top_entries, df=None):
    if df is None:
        # Load data from file if df is not provided
        df, vectorized_data = load_and_preprocess(file_path, enabler_column, entry_column, cluster_column)
    else:
        # Use the provided df and vectorize it
        vectorized_data = vectorize_data(df, enabler_column, entry_column)
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, cluster_column)
    heatmap_data, enabler_importance = prepare_heatmap_data(co_occurrence_matrices, vectorized_data, top_enablers, top_entries)
    secular_enablers = identify_secular_enablers(enabler_importance)
    return heatmap_data, enabler_importance, secular_enablers

# If you want to be able to run this script independently for testing:
if __name__ == "__main__":
    # Add some test code here
    pass