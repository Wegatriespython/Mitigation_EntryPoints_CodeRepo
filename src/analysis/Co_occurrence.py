import pandas as pd
import re
import numpy as np
from scipy.cluster import hierarchy
from fuzzywuzzy import fuzz, process

def calculate_co_occurrence(cluster_data, enablers, entries):
    """Calculates the co-occurrence of enablers and entries for a specific cluster."""
    co_occurrence_matrix = pd.DataFrame(0, index=enablers, columns=entries, dtype=int)
    for _, row in cluster_data.iterrows():
        row_enablers = row["Enabler"]
        row_entries = row["Entry (policy intervention)"]
        for enabler in row_enablers:
            for entry in row_entries:
                if enabler in co_occurrence_matrix.index and entry in co_occurrence_matrix.columns:
                    co_occurrence_matrix.loc[enabler, entry] += 1
    return co_occurrence_matrix

def create_combined_matrix(df, enablers, entries):
    """Creates a combined co-occurrence matrix with counts for each cluster as tuples."""
    clusters = df["Cluster"].unique()
    co_occurrence_matrices = {}
    for cluster in clusters:
        cluster_data = df[df["Cluster"] == cluster]
        co_occurrence_matrices[cluster] = calculate_co_occurrence(cluster_data, enablers, entries)

    all_enablers = list(set().union(*[cm.index for cm in co_occurrence_matrices.values()]))
    combined_matrix = pd.DataFrame(index=all_enablers, columns=entries, dtype=object)
    for enabler in all_enablers:
        for entry in entries:
            combined_matrix.loc[enabler, entry] = tuple(
                cm.loc[enabler, entry] if enabler in cm.index else 0
                for cm in co_occurrence_matrices.values()
            )
    return combined_matrix

def apply_fuzzy_matching_and_sorting(matrix, top_enablers, top_entries, feature_imp):
    """Applies fuzzy matching, filtering, and sorting based on feature importance."""
    feature_imp_dict = dict(zip(feature_imp["feature"], feature_imp["importance"]))

    def fuzzy_match(needle, haystack):
        """Checks if needle is present in haystack (case-insensitive, partial match)."""
        pattern = r".*".join([re.escape(char) for char in needle])
        return bool(re.search(pattern, haystack, re.IGNORECASE))

    filtered_rows = [r for r in matrix.index if any(fuzzy_match(e, r) for e in top_enablers)]
    filtered_rows = sorted(filtered_rows, key=lambda x: feature_imp_dict.get(x, float('-inf')), reverse=True)

    filtered_cols_indices = []
    for entry in top_entries:
        for i, col in enumerate(matrix.columns):
            if fuzzy_match(entry, col) and i not in filtered_cols_indices:
                filtered_cols_indices.append(i)
                break
    matrix = matrix.iloc[:, filtered_cols_indices]
    matrix = matrix.reindex(index=filtered_rows)
    return matrix

def prepare_heatmap_data(df, top_enablers, top_entries, feature_imp):
    """Prepares the co-occurrence matrix for heatmap visualization."""
    enablers = df["Enabler"].explode().unique().tolist()
    entries = df["Entry (policy intervention)"].explode().unique().tolist()
    combined_matrix = create_combined_matrix(df, enablers, entries)

    # Cluster rows and columns 
    avg_matrix = combined_matrix.applymap(lambda x: np.mean(x) if isinstance(x, tuple) else 0)
    row_linkage = hierarchy.linkage(avg_matrix, method="ward")
    col_linkage = hierarchy.linkage(avg_matrix.T, method="ward")
    row_order = hierarchy.dendrogram(row_linkage, no_plot=True)["leaves"]
    col_order = hierarchy.dendrogram(col_linkage, no_plot=True)["leaves"]
    combined_matrix = combined_matrix.iloc[row_order, col_order]

    combined_matrix_filtered = apply_fuzzy_matching_and_sorting(combined_matrix, top_enablers, top_entries, feature_imp)
    return combined_matrix_filtered 