import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from src.data_processing.general_preprocessing import load_and_preprocess
from src.analysis.random_forest import create_feature_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier

# File paths and settings
INPUT_FILE = r"C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\Copy of Codebook_Coal_Clean.xlsm"
OUTPUT_FILE = "coal_cluster_analysis_results.xlsx"
CLUSTER_COLUMN = "Cluster"
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def run_coal_cluster_analysis():
    # Load and preprocess data
    df, vectorized_data = load_and_preprocess(INPUT_FILE, ENABLER_COLUMN, ENTRY_COLUMN, CLUSTER_COLUMN)

    # Create feature matrix
    feature_matrix, feature_names, enabler_features = create_feature_matrix(df, ENABLER_COLUMN, ENTRY_COLUMN)

    # Encode cluster labels
    le = LabelEncoder()
    y = le.fit_transform(df[CLUSTER_COLUMN])
    cluster_names = le.classes_

    # Initialize results dictionary
    results = {
        'Cluster': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'ROC AUC': []
    }

    # Run analysis for each cluster
    for i, cluster in enumerate(cluster_names):
        print(f"Analyzing cluster: {cluster}")

        # Create binary target for current cluster
        y_binary = (y == i).astype(int)

        # Determine appropriate sampling method and cross-validation strategy
        min_samples = np.min(np.bincount(y_binary))
        if min_samples == 1:
            sampler = RandomOverSampler(random_state=42)
            cv = LeaveOneOut()
        else:
            k_neighbors = min(5, min_samples - 1)
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        X_resampled, y_resampled = sampler.fit_resample(feature_matrix, y_binary)

        # Initialize and train Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')

        # Perform cross-validation
        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        cv_roc_auc = []

        for train_index, val_index in cv.split(X_resampled, y_resampled):
            X_train, X_val = X_resampled[train_index], X_resampled[val_index]
            y_train, y_val = y_resampled[train_index], y_resampled[val_index]

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            y_pred_proba = rf.predict_proba(X_val)[:, 1]

            cv_accuracy.append(accuracy_score(y_val, y_pred))
            cv_precision.append(precision_score(y_val, y_pred, zero_division=0))
            cv_recall.append(recall_score(y_val, y_pred, zero_division=0))
            cv_f1.append(f1_score(y_val, y_pred, zero_division=0))
            cv_roc_auc.append(roc_auc_score(y_val, y_pred_proba))

        # Store results
        results['Cluster'].append(cluster)
        results['Accuracy'].append(np.mean(cv_accuracy))
        results['Precision'].append(np.mean(cv_precision))
        results['Recall'].append(np.mean(cv_recall))
        results['F1 Score'].append(np.mean(cv_f1))
        results['ROC AUC'].append(np.mean(cv_roc_auc))

        print(f"ROC AUC: {results['ROC AUC'][-1]:.4f}")
        print("-----------------------------")

    # Create DataFrame from results
    results_df = pd.DataFrame(results)

    # Calculate mean scores
    mean_scores = results_df.mean(numeric_only=True)
    mean_scores_df = pd.DataFrame(mean_scores).T
    mean_scores_df['Cluster'] = 'Mean'

    # Combine results and mean scores
    final_results = pd.concat([results_df, mean_scores_df], ignore_index=True)

    # Save results to Excel
    final_results.to_excel(OUTPUT_FILE, index=False, float_format='%.3f')
    print(f"Results saved to {OUTPUT_FILE}")

    # Print results table
    print("\n Coal Cluster Analysis Results:")
    print(final_results.to_string(index=False))

if __name__ == "__main__":
    run_coal_cluster_analysis()