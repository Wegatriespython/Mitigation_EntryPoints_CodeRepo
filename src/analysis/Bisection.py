import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_processing.general_preprocessing import load_and_preprocess
from src.analysis.Co_occurrence import calculate_co_occurrence

def calculate_cluster_centers(co_occurrence_matrices):
    centers = []
    for matrix in co_occurrence_matrices.values():
        centers.append(matrix.toarray().flatten())
    return np.array(centers)

def get_bisection_data(df, vectorized_data, cluster_column):
    co_occurrence_matrices = calculate_co_occurrence(df, vectorized_data, cluster_column)
    center_vectors = {cluster: matrix.toarray().flatten() for cluster, matrix in co_occurrence_matrices.items()}
    return {
        'center_vectors': center_vectors,
        'enabler_features': vectorized_data['enabler_features'],
        'entry_features': vectorized_data['entry_features']
    }

def bisect_clusters(distance_matrix, bisection_data):
    n_clusters = len(distance_matrix)
    center_vectors = np.array(list(bisection_data['center_vectors'].values()))
    n_enablers = len(bisection_data['enabler_features'])
    
    def calculate_group_score(group):
        if len(group) < 2:
            return 0
        group_vectors = center_vectors[group]
        enabler_sim = np.mean(cosine_similarity(group_vectors[:, :n_enablers]))
        entry_sim = np.mean(cosine_similarity(group_vectors[:, n_enablers:]))
        return entry_sim - enabler_sim  # High entry similarity, low enabler similarity
    
    best_split = None
    best_score = float('-inf')
    
    for i in range(1, 2**n_clusters - 1):
        group1 = [j for j in range(n_clusters) if (i & (1 << j))]
        group2 = [j for j in range(n_clusters) if not (i & (1 << j))]
        
        if len(group1) == 0 or len(group2) == 0 or abs(len(group1) - len(group2)) > 1:
            continue
        
        score = calculate_group_score(group1) + calculate_group_score(group2)
        
        if score > best_score:
            best_score = score
            best_split = (group1, group2)
    
    return best_split

def main():
    # File paths and settings
    INPUT_FILE = r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\REWindSolar.xlsx"
    CLUSTER_COLUMN = "Cluster"
    ENABLER_COLUMN = "Enabler"
    ENTRY_COLUMN = "Entry (policy intervention)"

    # Load and Preprocess Data
    print("Loading and preprocessing data...")
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)
    print(f"Loaded data shape: {df.shape}")

    # Determine clusters
    clusters = df[CLUSTER_COLUMN].unique()
    print(f"Clusters: {clusters}")
    print(f"Number of clusters: {len(clusters)}")

    # Get bisection data
    bisection_data = get_bisection_data(df, vectorized_data, CLUSTER_COLUMN)

    # Calculate distances between cluster centers
    center_vectors = np.array(list(bisection_data['center_vectors'].values()))
    distances = np.zeros((len(clusters), len(clusters)))
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distances[i, j] = distances[j, i] = np.linalg.norm(center_vectors[i] - center_vectors[j])
    
    print("Distance matrix:")
    print(distances)

    # Bisect clusters
    if len(clusters) > 4:
        print("Bisecting clusters...")
        group1, group2 = bisect_clusters(distances, bisection_data)
        cluster_batches = [
            [clusters[i] for i in group1],
            [clusters[i] for i in group2]
        ]
    else:
        cluster_batches = [clusters]

    # Print cluster batches
    print("\nCluster Batches:")
    for i, batch in enumerate(cluster_batches):
        print(f"Batch {i+1}: {', '.join(map(str, batch))}")

    # Here you would normally proceed with the Random Forest analysis
    # But we'll stop here as requested

if __name__ == "__main__":
    main()