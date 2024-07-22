import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter

def clean_term(text):
    if not isinstance(text, str):
        return ''
    
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Remove stop words
    stop_words = set(['and', 'be', 'in', 'only', 'out', 'rather', 'should', 'so', 'take', 'than', 'the', 'to', 'add', 'consider', 'is', 'whether', 'foreign'])
    words = [word for word in text.split() if word not in stop_words]
    
    return ' '.join(words)

def preprocess_data(df):
    df_filtered = df.dropna(subset=['Cluster'])
    df_filtered['Enabler_clean'] = df_filtered['Enabler'].apply(lambda x: [clean_term(term.strip()) for term in x.split(',')])
    df_filtered['Entry_clean'] = df_filtered['Entry (policy intervention)'].apply(clean_term)
    return df_filtered

def create_feature_matrix(df_filtered):
    enabler_vectorizer = CountVectorizer(binary=True)
    enabler_matrix = enabler_vectorizer.fit_transform(df_filtered['Enabler_clean'].apply(lambda x: ' '.join(x)))
    enabler_features = enabler_vectorizer.get_feature_names_out()

    entry_vectorizer = CountVectorizer(binary=True)
    entry_matrix = entry_vectorizer.fit_transform(df_filtered['Entry_clean'])
    entry_features = entry_vectorizer.get_feature_names_out()

    feature_matrix = np.hstack((enabler_matrix.toarray(), entry_matrix.toarray()))
    feature_names = np.concatenate((enabler_features, entry_features))

    return feature_matrix, feature_names

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
            all_features = [item for sublist in df_filtered['Enabler'].str.split(',') for item in sublist]
        else:
            all_features = df_filtered['Entry (policy intervention)'].tolist()
        
        feature_counts = Counter(all_features)
        additional_features = [f for f, _ in feature_counts.most_common() if f not in top_features]
        top_features.extend(additional_features[:N - len(top_features)])
    
    return top_features[:N]

def run_random_forest_analysis(df, output_file):
    df_filtered = preprocess_data(df)
    feature_matrix, feature_names = create_feature_matrix(df_filtered)

    le = LabelEncoder()
    y = le.fit_transform(df_filtered['Cluster'])

    random_search = train_random_forest(feature_matrix, y)

    importances = random_search.best_estimator_.feature_importances_
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'type': ['Enabler' if i < len(enabler_features) else 'Entry' for i in range(len(feature_names))]
    })
    feature_imp = feature_imp.sort_values(by='importance', ascending=False)

    N = 10
    top_enablers = get_top_features(feature_imp, 'Enabler', N, df_filtered)
    top_entries = get_top_features(feature_imp, 'Entry', N, df_filtered)

    results = {
        'df': df,
        'top_enablers': top_enablers,
        'top_entries': top_entries,
        'feature_imp': feature_imp
    }
    joblib.dump(results, output_file)

    return results

if __name__ == "__main__":
    # Example usage
    file_path = "path/to/your/codebook.xlsx"
    df = pd.read_excel(file_path)
    results = run_random_forest_analysis(df, 'rf_analysis_results.joblib')
    print("Analysis complete. Results saved to 'rf_analysis_results.joblib'")