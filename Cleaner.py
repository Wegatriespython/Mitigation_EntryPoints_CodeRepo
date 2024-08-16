import pandas as pd
import numpy as np
from src.data_processing.general_preprocessing import clean_term

# File path
INPUT_FILE = r"C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\Codebook_Omnibus_Extractions_Select_Columns_W_Duplicates.xlsx"

# Column names
ENABLER_COLUMN = "Enabler"
ENTRY_COLUMN = "Entry (policy intervention)"

def get_unique_values(df, column_name):
    # Flatten the list of lists and get unique values
    unique_values = set()
    for item in df[column_name]:
        if isinstance(item, list):
            unique_values.update(item)
        elif isinstance(item, str):
            unique_values.add(item)
    return sorted(list(unique_values))

def main():
    # Read the Excel file
    df = pd.read_excel(INPUT_FILE)

    # Clean and preprocess the Enabler and Entry columns
    for column in [ENABLER_COLUMN, ENTRY_COLUMN]:
        df[column] = df[column].apply(lambda x: [clean_term(term.strip()) for term in str(x).split(',')])

    # Get unique values
    unique_enablers = get_unique_values(df, ENABLER_COLUMN)
    unique_entries = get_unique_values(df, ENTRY_COLUMN)

    # Print results
    print("Unique Enablers:")
    for enabler in unique_enablers:
        print(f"- {enabler}")

    print("\nUnique Entries:")
    for entry in unique_entries:
        print(f"- {entry}")

    # Optional: Save to file
    with open("unique_enablers_entries.txt", "w") as f:
        f.write("Unique Enablers:\n")
        for enabler in unique_enablers:
            f.write(f"- {enabler}\n")
        f.write("\nUnique Entries:\n")
        for entry in unique_entries:
            f.write(f"- {entry}\n")

    print(f"\nResults also saved to unique_enablers_entries.txt")

if __name__ == "__main__":
    main()