import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
from collections import Counter
from typing import Dict, List, Tuple
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

def get_top_features(feature_imp: pd.DataFrame, feature_type: str, N: int, df: pd.DataFrame, y: np.ndarray, detailed: bool) -> List[str]:
    if detailed:
        return get_top_features_proportional(feature_imp, feature_type, N, y)
    else:
        return get_top_features_vanilla(feature_imp, feature_type, N, df)

def get_top_features_vanilla(feature_imp: pd.DataFrame, feature_type: str, N: int, df: pd.DataFrame) -> List[str]:
    top_features = feature_imp[(feature_imp['type'] == feature_type) & (feature_imp['importance'] > 0)]['feature'].tolist()
    
    if len(top_features) < N:
        if feature_type == 'Enabler':
            all_features = [item for sublist in df['Enabler'] for item in sublist]
        else:
            all_features = [item for sublist in df['Entry (policy intervention)'] for item in sublist]
        
        feature_counts = Counter(all_features)
        additional_features = [f for f, _ in feature_counts.most_common() if f not in top_features]
        top_features.extend(additional_features[:N - len(top_features)])
    
    return top_features[:N]

def get_top_features_proportional(feature_imp: pd.DataFrame, feature_type: str, N: int, y: np.ndarray) -> List[str]:
    print(f"\nDebugging get_top_features_proportional for {feature_type}")
    class_sizes = np.bincount(y)
    class_proportions = class_sizes / np.sum(class_sizes)
    
    print(f"Class sizes: {class_sizes}")
    print(f"Class proportions: {class_proportions}")
    
    type_feature_imp = feature_imp[feature_imp['type'] == feature_type].copy()
    
    top_features = []
    for class_label, proportion in enumerate(class_proportions):
        n_features = max(1, int(np.ceil(N * proportion)))
        print(f"\nClass {class_label}: Selecting {n_features} features")
        
        class_importance = type_feature_imp['importance'] * (y == class_label).mean()
        type_feature_imp['class_importance'] = class_importance
        class_top_features = type_feature_imp.nlargest(n_features, 'class_importance')['feature'].tolist()
        
        print(f"Top features for class {class_label}: {class_top_features}")
        top_features.extend(class_top_features)
    
    # Simple fix for duplicates: use a set to remove duplicates, then convert back to list
    top_features = list(dict.fromkeys(top_features))
    
    print(f"\nTotal features selected before truncation: {len(top_features)}")
    print(f"Features: {top_features}")
    
    # If we don't have enough features, add the most important remaining ones
    if len(top_features) < N:
        print(f"Not enough features selected. Adding {N - len(top_features)} more.")
        remaining_features = type_feature_imp[~type_feature_imp['feature'].isin(top_features)].nlargest(N - len(top_features), 'importance')['feature'].tolist()
        top_features.extend(remaining_features)
    
    result = top_features[:N]
    print(f"\nFinal {N} features selected: {result}")
    return result

def run_random_forest_analysis(file_path: str, enabler_column: str, entry_column: str, 
                               cluster_column: str, n_enablers: int, n_entries: int, 
                               output_file: str, detailed: bool = False, df: pd.DataFrame = None) -> Dict:
    if df is None:
        df, _ = load_and_preprocess(file_path, enabler_column, entry_column, cluster_column)
    
    feature_matrix, feature_names, enabler_features = create_feature_matrix(df, enabler_column, entry_column)

    le = LabelEncoder()
    y = le.fit_transform(df[cluster_column])

    # Check class distribution and remove classes with only one member
    class_counts = np.bincount(y)
    if np.min(class_counts) == 1:
        print("Removing classes with only one member...")
        valid_classes = np.where(class_counts > 1)[0]
        mask = np.isin(y, valid_classes)
        feature_matrix = feature_matrix[mask]
        y = y[mask]
        y = le.fit_transform(y)
        print("New class distribution:", np.bincount(y))
        print("New minimum class size:", np.min(np.bincount(y)))

    random_search = train_random_forest(feature_matrix, y, detailed)

    print("\nRandom Forest optimization complete.")
    print("Best parameters:", random_search.best_params_)
    print("Best cross-validation score:", random_search.best_score_)

    importances = random_search.best_estimator_.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'type': ['Enabler' if i < len(enabler_features) else 'Entry' for i in range(len(feature_names))]
    })
    feature_imp = feature_imp.sort_values(by='importance', ascending=False)

    top_enablers = get_top_features(feature_imp, 'Enabler', n_enablers, df, y, detailed)
    top_entries = get_top_features(feature_imp, 'Entry', n_entries, df, y, detailed)

    print(f"\nTop {n_enablers} Enablers:")
    for enabler in top_enablers:
        importance = feature_imp[feature_imp['feature'] == enabler]['importance'].values
        if len(importance) > 0 and importance[0] > 0:
            print(f"{enabler}: {importance[0]}")
        else:
            count = sum(enabler in enablers for enablers in df['Enabler'])
            print(f"{enabler}: Added based on frequency (count: {count})")

    print(f"\nTop {n_entries} Entries:")
    for entry in top_entries:
        importance = feature_imp[feature_imp['feature'] == entry]['importance'].values
        if len(importance) > 0 and importance[0] > 0:
            print(f"{entry}: {importance[0]}")
        else:
            count = df['Entry (policy intervention)'].value_counts().get(entry, 0)
            print(f"{entry}: Added based on frequency (count: {count})")

    results = {
        'df': df,
        'top_enablers': top_enablers,
        'top_entries': top_entries,
        'feature_imp': feature_imp
    }

    if detailed:
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        results['class_weights'] = class_weight_dict

    joblib.dump(results, output_file)
    return results


def load_or_run_random_forest_analysis(file_path: str, enabler_column: str, entry_column: str, 
                                       cluster_column: str, n_enablers: int, n_entries: int, 
                                       output_file: str, detailed: bool = False, df: pd.DataFrame = None):
    # Extract the codebook name from the file path
    codebook_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1]
    
    # Determine the method (vanilla or detailed)
    method = "detailed" if detailed else "vanilla"
    
    # Create the new output filename
    new_output_file = f"RF_{codebook_name}_{method}.joblib"
    
    # Ensure the output is saved in the data/processed directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "processed")
    os.makedirs(output_dir, exist_ok=True)
    new_output_file = os.path.join(output_dir, new_output_file)
    
    if os.path.exists(new_output_file) and df is None:
        print(f"Loading existing results from {new_output_file}")
        results = joblib.load(new_output_file)
    else:
        print(f"Running Random Forest analysis and saving results to {new_output_file}")
        results = run_random_forest_analysis(file_path, enabler_column, entry_column, cluster_column, 
                                             n_enablers, n_entries, new_output_file, detailed, df)
        
        # Add metadata to the results
        results['metadata'] = {
            'codebook': codebook_name,
            'method': method
        }
        
        joblib.dump(results, new_output_file)
    
    return results, new_output_file

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

    results = load_or_run_random_forest_analysis(file_path, enabler_column, entry_column, cluster_column, 
                                                 n_enablers, n_entries, output_file, detailed)