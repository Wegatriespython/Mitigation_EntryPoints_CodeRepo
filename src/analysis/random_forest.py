import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter
from typing import Dict, List, Tuple, Union
from src.data_processing.general_preprocessing import load_and_preprocess

def create_feature_matrix(df: pd.DataFrame, enabler_column: str, entry_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    enabler_vectorizer = CountVectorizer(binary=True)
    enabler_matrix = enabler_vectorizer.fit_transform(df[enabler_column].apply(lambda x: ' '.join(x)))
    enabler_features = enabler_vectorizer.get_feature_names_out()

    entry_vectorizer = CountVectorizer(binary=True)
    entry_matrix = entry_vectorizer.fit_transform(df[entry_column].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
    entry_features = entry_vectorizer.get_feature_names_out()

    feature_matrix = np.hstack((enabler_matrix.toarray(), entry_matrix.toarray()))
    feature_names = np.concatenate((enabler_features, entry_features))

    return feature_matrix, feature_names, enabler_features

def train_random_forest(feature_matrix: np.ndarray, y: np.ndarray, detailed: bool) -> RandomForestClassifier:
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }
    
    if detailed:
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        rf_params['class_weight'] = [class_weight_dict, 'balanced', 'balanced_subsample']
    else:
        rf_params['class_weight'] = ['balanced', 'balanced_subsample', None]

    rf = RandomForestClassifier(random_state=42)
    min_class_size = np.min(np.bincount(y))
    n_splits = min(5, min_class_size)
    cv = LeaveOneOut() if min_class_size == 1 else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, 
                                       n_iter=100, cv=cv, verbose=1, random_state=42, n_jobs=-1)
    random_search.fit(feature_matrix, y)
    return random_search

def get_top_features(feature_imp: pd.DataFrame, feature_type: str, N: int) -> List[str]:
    """
    Select top N features based on importance.
    
    Args:
    feature_imp (pd.DataFrame): DataFrame with columns 'feature', 'importance', and 'type'
    feature_type (str): Type of feature to select ('Enabler' or 'Entry')
    N (int): Number of features to select

    Returns:
    List[str]: Top N features of the specified type
    """
    # Filter for the specified feature type and sort by importance
    type_features = feature_imp[feature_imp['type'] == feature_type].sort_values('importance', ascending=False)
    
    # Select top N unique features
    top_features = []
    for feature in type_features['feature']:
        if feature not in top_features:
            top_features.append(feature)
            if len(top_features) == N:
                break
    
    return top_features

def get_top_features_proportional(feature_imp: pd.DataFrame, feature_type: str, N: int, y: Union[np.ndarray, None] = None) -> List[str]:
    print(f"\nSelecting top {N} features for {feature_type} ({'proportional' if y is not None else 'aggregated'} method)")
    
    type_feature_imp = feature_imp[feature_imp['type'] == feature_type].copy()
    
    if y is not None:
        # Cluster-specific analysis
        class_sizes = np.bincount(y)
        class_proportions = class_sizes / np.sum(class_sizes)
        
        print(f"Class sizes: {class_sizes}")
        print(f"Class proportions: {class_proportions}")
        
        top_features = []
        for class_label, proportion in enumerate(class_proportions):
            n_features = max(1, int(np.ceil(N * proportion)))
            print(f"\nClass {class_label}: Selecting {n_features} features")
            
            class_importance = type_feature_imp['importance'] * (y == class_label).mean()
            type_feature_imp['class_importance'] = class_importance
            class_top_features = type_feature_imp.nlargest(n_features, 'class_importance')['feature'].tolist()
            
            print(f"Top features for class {class_label}: {class_top_features}")
            top_features.extend(class_top_features)
        
        # Remove duplicates while preserving order
        top_features = list(dict.fromkeys(top_features))
    else:
        # Aggregated analysis
        top_features = type_feature_imp.nlargest(N, 'importance')['feature'].tolist()
    
    # If we don't have enough features, add the most important remaining ones
    if len(top_features) < N:
        print(f"Not enough features selected. Adding {N - len(top_features)} more.")
        remaining_features = type_feature_imp[~type_feature_imp['feature'].isin(top_features)].nlargest(N - len(top_features), 'importance')['feature'].tolist()
        top_features.extend(remaining_features)
    
    result = top_features[:N]
    print(f"\nFinal {N} features selected: {result}")
    return result


def run_cluster_specific_random_forest(df: pd.DataFrame, feature_matrix: np.ndarray, y: np.ndarray, 
                                       feature_names: np.ndarray, enabler_features: np.ndarray,
                                       cluster_names: List[str]) -> Dict:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    
    cluster_importances = {}
    
    for i, cluster in enumerate(cluster_names):
        print(f"\nAnalyzing cluster: {cluster}")
        
        # Create binary target for current cluster
        y_cluster = y_bin[:, i]
        
        # Train random forest for current cluster
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, 
                                           n_iter=100, cv=cv, verbose=1, random_state=42, n_jobs=-1)
        random_search.fit(feature_matrix, y_cluster)
        
        # Store feature importances for this cluster
        importances = random_search.best_estimator_.feature_importances_
        cluster_importances[cluster] = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'type': ['Enabler' if i < len(enabler_features) else 'Entry' for i in range(len(feature_names))]
        })
    print ("Cluster importances: ", cluster_importances)

    return cluster_importances

def aggregate_cluster_results(cluster_importances: Dict[str, pd.DataFrame], 
                              n_enablers: int, n_entries: int,
                              cluster_sizes: Dict[str, int]) -> Dict:
    print("\nStarting aggregate_cluster_results function")
    print(f"Number of enablers requested: {n_enablers}")
    print(f"Number of entries requested: {n_entries}")
    
    # Sort clusters by size (descending order)
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    print(f"Clusters sorted by size: {sorted_clusters}")
    
    # Calculate the number of features to select per cluster
    n_clusters = len(cluster_importances)
    enablers_per_cluster = max(1, n_enablers // n_clusters)
    entries_per_cluster = max(1, n_entries // n_clusters)
    print(f"Number of enablers per cluster: {enablers_per_cluster}")
    print(f"Number of entries per cluster: {entries_per_cluster}")
    
    top_enablers = []
    top_entries = []
    used_enablers = set()
    used_entries = set()
    
    # Select top features for each cluster, in order of cluster size
    for cluster, size in sorted_clusters:
        print(f"\nProcessing cluster: {cluster} (size: {size})")
        importance_df = cluster_importances[cluster]
        print(f"Importance DataFrame shape: {importance_df}") 
        # Select top enablers for this cluster
        cluster_top_enablers = get_top_features_proportional(
            importance_df[importance_df['type'] == 'Enabler'], 
            'Enabler', 
            enablers_per_cluster
        )
        print(f"Top enablers for {cluster}: {cluster_top_enablers}")
        
        # Select top entries for this cluster
        cluster_top_entries = get_top_features_proportional(
            importance_df[importance_df['type'] == 'Entry'], 
            'Entry', 
            entries_per_cluster
        )
        print(f"Top entries for {cluster}: {cluster_top_entries}")
        
        # Add unique features to the lists
        for feature in cluster_top_enablers:
            if feature not in used_enablers:
                top_enablers.append(feature)
                used_enablers.add(feature)
        
        for feature in cluster_top_entries:
            if feature not in used_entries:
                top_entries.append(feature)
                used_entries.add(feature)
        
        print(f"Current top enablers: {top_enablers}")
        print(f"Current top entries: {top_entries}")
    
    print("\nChecking if more features are needed")
    # If we need more features, add them based on global importance
    if len(top_enablers) < n_enablers or len(top_entries) < n_entries:
        print("Combining importances for global selection")
        # Now combine importances for global selection
        combined_importances = pd.concat(cluster_importances.values())
        combined_importances = combined_importances.groupby(['feature', 'type']).agg({
            'importance': 'mean'
        }).reset_index()
        
        while len(top_enablers) < n_enablers:
            print(f"Selecting additional enablers. Current count: {len(top_enablers)}")
            global_top_enablers = get_top_features_proportional(
                combined_importances[combined_importances['type'] == 'Enabler'], 
                'Enabler', 
                n_enablers
            )
            for feature in global_top_enablers:
                if feature not in used_enablers:
                    top_enablers.append(feature)
                    used_enablers.add(feature)
                    if len(top_enablers) == n_enablers:
                        break
        
        while len(top_entries) < n_entries:
            print(f"Selecting additional entries. Current count: {len(top_entries)}")
            global_top_entries = get_top_features_proportional(
                combined_importances[combined_importances['type'] == 'Entry'], 
                'Entry', 
                n_entries
            )
            for feature in global_top_entries:
                if feature not in used_entries:
                    top_entries.append(feature)
                    used_entries.add(feature)
                    if len(top_entries) == n_entries:
                        break
    
    print(f"\nFinal top enablers: {top_enablers}")
    print(f"Final top entries: {top_entries}")
    
    results = {
        'top_enablers': top_enablers,
        'top_entries': top_entries,
        'feature_imp': pd.concat(cluster_importances.values())  # Keep all importances
    }
    
    print("Finished aggregate_cluster_results function")
    return results

def run_random_forest_analysis(file_path: str, enabler_column: str, entry_column: str, 
                               cluster_column: str, n_enablers: int, n_entries: int, 
                               output_file: str, detailed: bool = False, df: pd.DataFrame = None,
                               cluster_specific: bool = False) -> Dict:
    print("Starting run_random_forest_analysis")
    if df is None:
        print("Loading and preprocessing data")
        df, _ = load_and_preprocess(file_path, enabler_column, entry_column, cluster_column)
    
    print("Creating feature matrix")
    feature_matrix, feature_names, enabler_features = create_feature_matrix(df, enabler_column, entry_column)

    le = LabelEncoder()
    y = le.fit_transform(df[cluster_column])
    cluster_names = le.classes_
    print(f"Clusters: {cluster_names}")

    results = {}  # Initialize results dictionary

    if cluster_specific:
        print("Running cluster-specific random forest")
        cluster_importances = run_cluster_specific_random_forest(df, feature_matrix, y, feature_names, enabler_features, cluster_names)
        print("Aggregating cluster results")
        cluster_sizes = {cluster: np.sum(y == i) for i, cluster in enumerate(cluster_names)}
        results = aggregate_cluster_results(cluster_importances,n_enablers, n_entries, cluster_sizes)
    else:
        print("Running regular random forest")
        random_search = train_random_forest(feature_matrix, y, detailed)
        importances = random_search.best_estimator_.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'type': ['Enabler' if i < len(enabler_features) else 'Entry' for i in range(len(feature_names))]
        })
        feature_imp = feature_imp.sort_values(by='importance', ascending=False)

        print("Selecting top features")
        top_enablers = get_top_features(feature_imp, 'Enabler', n_enablers)
        top_entries = get_top_features(feature_imp, 'Entry', n_entries)

        results = {
            'top_enablers': top_enablers,
            'top_entries': top_entries,
            'feature_imp': feature_imp
        }

    print("Saving results")
    joblib.dump(results, output_file)
    
    print("Results structure:")
    for key, value in results.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
        elif isinstance(value, pd.DataFrame):
            print(f"  Shape: {value.shape}")
    
    if not results:
        print("Warning: results dictionary is empty")
        results = {'top_enablers': [], 'top_entries': [], 'feature_imp': pd.DataFrame()}
    
    print("Finished run_random_forest_analysis")
    return results


if __name__ == "__main__":
    # Example usage
    file_path = "C:/Users/vigneshr/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Omnibus Generator/Codebook_Transport.xlsm".replace("\\", "/")
    enabler_column = "Enabler" 
    entry_column = "Entry (policy intervention)"
    cluster_column = "Cluster"
    output_file = "rf_analysis_results.joblib"
    n_enablers = 15
    n_entries = 10
    detailed = False  # Set to True for detailed analysis, False for vanilla

    results = run_random_forest_analysis(file_path, enabler_column, entry_column, cluster_column, 
                                                 n_enablers, n_entries, output_file, detailed)