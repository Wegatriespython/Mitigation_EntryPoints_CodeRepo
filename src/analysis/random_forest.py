import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
from ..data_processing.general_preprocessing import load_and_preprocess


def create_feature_matrix(df: pd.DataFrame, enabler_column: str, entry_column: str) -> tuple:
    """Creates a feature matrix from the preprocessed 'Enabler' and 'Entry' columns."""

    enabler_vectorizer = CountVectorizer(binary=True)
    enabler_matrix = enabler_vectorizer.fit_transform(df[enabler_column].apply(lambda x: ' '.join(x)))
    enabler_features = enabler_vectorizer.get_feature_names_out()

    entry_vectorizer = CountVectorizer(binary=True)
    entry_matrix = entry_vectorizer.fit_transform(df[entry_column].apply(lambda x: ' '.join(x)))
    entry_features = entry_vectorizer.get_feature_names_out()

    feature_matrix = np.hstack((enabler_matrix.toarray(), entry_matrix.toarray()))
    feature_names = np.concatenate((enabler_features, entry_features))

    return feature_matrix, feature_names, enabler_features

def train_random_forest(feature_matrix, y):
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }

    rf = RandomForestClassifier(random_state=42)
    n_splits = min(5, np.min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, 
                                       n_iter=100, cv=cv, verbose=1, random_state=42, n_jobs=-1)

    random_search.fit(feature_matrix, y)
    return random_search

def get_top_features(feature_imp, feature_type, N, df_filtered):
    top_features = feature_imp[(feature_imp['type'] == feature_type) & (feature_imp['importance'] > 0)]['feature'].tolist()
    
    if len(top_features) < N:
        if feature_type == 'Enabler':
            all_features = [item for sublist in df_filtered['Enabler'] for item in sublist]
        else:
            all_features = [item for sublist in df_filtered['Entry (policy intervention)'] for item in sublist]
        
        feature_counts = Counter(all_features)
        additional_features = [f for f, _ in feature_counts.most_common() if f not in top_features]
        top_features.extend(additional_features[:N - len(top_features)])
    
    return top_features[:N]

def run_random_forest_analysis(df: pd.DataFrame, enabler_column: str, entry_column: str, 
                              cluster_column: str, output_file: str):
    """Runs the complete Random Forest analysis pipeline."""

    feature_matrix, feature_names, enabler_features = create_feature_matrix(
        df, enabler_column, entry_column
    )

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

    random_search = train_random_forest(feature_matrix, y)

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

    N = 10
    top_enablers = get_top_features(feature_imp, 'Enabler', N, df)
    top_entries = get_top_features(feature_imp, 'Entry', N, df)

    print(f"\nTop {N} Enablers:")
    for enabler in top_enablers:
        importance = feature_imp[feature_imp['feature'] == enabler]['importance'].values
        if len(importance) > 0 and importance[0] > 0:
            print(f"{enabler}: {importance[0]}")
        else:
            count = sum(enabler in enablers for enablers in df['Enabler'])
            print(f"{enabler}: Added based on frequency (count: {count})")

    print(f"\nTop {N} Entries:")
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
    joblib.dump(results, output_file)

    return results

def load_or_run_random_forest_analysis(df, output_file):
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}")
        return joblib.load(output_file)
    else:
        print(f"Running Random Forest analysis and saving results to {output_file}")
        return run_random_forest_analysis(df, output_file)

if __name__ == "__main__":
    # Example usage
    # Load the data
    file_path = "C:/Users/vigneshr/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Omnibus Generator/Codebook_Transport.xlsm".replace("\\", "/")
    enabler_column = "Enabler" 
    entry_column = "Entry (policy intervention)"
    cluster_column = "Cluster"
    output_file = "rf_analysis_results.joblib"

    df = load_and_preprocess(file_path, enabler_column, entry_column, cluster_column)
    results = run_random_forest_analysis(df, enabler_column, entry_column, cluster_column, output_file)