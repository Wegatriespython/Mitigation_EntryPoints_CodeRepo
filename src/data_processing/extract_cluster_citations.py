import pandas as pd
import os
from fuzzywuzzy import fuzz
import bibtexparser

def load_bibtex(file_path):
    with open(file_path, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    return bib_database.entries

def fuzzy_match_citation(article_name, bib_entries, threshold=90):
    for entry in bib_entries:
        if 'title' in entry:
            if fuzz.ratio(article_name.lower(), entry['title'].lower()) >= threshold:
                return entry.get('ID', '')
    return ''

def extract_cluster_citations(input_file, bib_file):
    # Read the Excel file
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # Load BibTeX entries
    bib_entries = load_bibtex(bib_file)
    
    # Group by cluster and aggregate articles and citation keys
    cluster_data = df.groupby('Cluster').agg({
        'Article': lambda x: list(x.dropna()),
    }).reset_index()
    
    # Function to get citation keys for a list of articles
    def get_citation_keys(articles):
        citation_keys = []
        unmatched_articles = []
        for article in articles:
            key = fuzzy_match_citation(article, bib_entries)
            citation_keys.append(key)
            if not key:
                unmatched_articles.append(article)
        return citation_keys, unmatched_articles
    
    # Apply fuzzy matching to get citation keys and unmatched articles
    cluster_data['Citation Keys'], cluster_data['Unmatched Articles'] = zip(*cluster_data['Article'].apply(get_citation_keys))
    
    # Convert lists to comma-separated strings
    cluster_data['Articles'] = cluster_data['Article'].apply(lambda x: ', '.join(x))
    cluster_data['Citation Keys'] = cluster_data['Citation Keys'].apply(lambda x: ', '.join(filter(None, x)))
    
    # Keep only the required columns
    result = cluster_data[['Cluster', 'Articles', 'Citation Keys']]
    
    # Print unmatched articles
    print("\nArticles without matching citation keys:")
    for _, row in cluster_data.iterrows():
        if row['Unmatched Articles']:
            print(f"Cluster: {row['Cluster']}")
            for article in row['Unmatched Articles']:
                print(f"  - {article}")
    
    return result

def main():
    # Input file paths
    input_file = "C:/Users/vigneshr/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Mitigation_EntryPoints_CodeRepo/data/raw/REWindSolar.xlsx"
    bib_file = "C:/Users/vigneshr/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Mitigation_EntryPoints_CodeRepo/data/raw/results_11608.bib"
    
    # Extract the filename without extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Output file path
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{file_name}_cluster_citations.csv")
    
    # Extract cluster citations
    cluster_citations = extract_cluster_citations(input_file, bib_file)
    
    # Save to CSV
    cluster_citations.to_csv(output_file, index=False)
    
    print(f"\nCluster citations have been saved to: {output_file}")

if __name__ == "__main__":
    main()