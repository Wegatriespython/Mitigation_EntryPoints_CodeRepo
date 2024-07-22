import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Dict
import numpy as np


def clean_term(text: str) -> str:
    """Clean and standardize a single term from the CSV string."""
    if not isinstance(text, str):
        return ''

    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Remove stop words (customize this list as needed)
    stop_words = set(
        ['and', 'be', 'in', 'only', 'out', 'rather', 'should', 'so', 'take', 
         'than', 'the', 'to', 'add', 'consider', 'is', 'whether', 'foreign']
    )
    words = [word for word in text.split() if word not in stop_words]

    return ' '.join(words)
def create_vectorizers() -> Tuple[CountVectorizer, CountVectorizer]:
    enabler_vectorizer = CountVectorizer(binary=True)
    entry_vectorizer = CountVectorizer(binary=True)
    return enabler_vectorizer, entry_vectorizer

def vectorize_data(df: pd.DataFrame, enabler_column: str, entry_column: str) -> Dict[str, np.ndarray]:
    enabler_vectorizer, entry_vectorizer = create_vectorizers()
    
    enabler_matrix = enabler_vectorizer.fit_transform(df[enabler_column].apply(lambda x: ' '.join(x)))
    entry_matrix = entry_vectorizer.fit_transform(df[entry_column].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)))
    
    return {
        'enabler_matrix': enabler_matrix,
        'entry_matrix': entry_matrix,
        'enabler_features': enabler_vectorizer.get_feature_names_out(),
        'entry_features': entry_vectorizer.get_feature_names_out(),
        'enabler_vectorizer': enabler_vectorizer,
        'entry_vectorizer': entry_vectorizer
    }

def load_and_preprocess(file_path: str, enabler_column: str, entry_column: str, cluster_column: str) -> pd.DataFrame:
    """
    Load data from an Excel file and preprocess the 'Enabler' and 'Entry' columns.

    Args:
        file_path (str): Path to the Excel file.
        enabler_column (str): Name of the column containing Enabler data.
        entry_column (str): Name of the column containing Entry data.
        cluster_column (str): Name of the column containing Cluster data. 

    Returns:
        pd.DataFrame: DataFrame with preprocessed 'Enabler' and 'Entry' columns.
    """
    df = pd.read_excel(file_path)

    # Apply clean_term to each element in 'Enabler' and 'Entry' columns
    df[enabler_column] = df[enabler_column].apply(lambda x: [clean_term(term.strip()) for term in str(x).split(',')])
    df[entry_column] = df[entry_column].apply(lambda x: [clean_term(term.strip()) for term in str(x).split(',')])

    # Filter out rows with NaN values in the Cluster column
    df = df.dropna(subset=[cluster_column])
        # Vectorize the data
    vectorized_data = vectorize_data(df, enabler_column, entry_column)

    return df, vectorized_data