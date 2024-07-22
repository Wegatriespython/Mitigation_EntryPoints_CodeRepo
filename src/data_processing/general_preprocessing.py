import pandas as pd
import re

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
    return df