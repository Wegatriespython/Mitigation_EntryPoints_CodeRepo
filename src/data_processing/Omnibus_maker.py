import pandas as pd
import os
import numpy as np

def combine_excel_files(directory):
    # List of specific Excel files to process with .xlsm extension
    excel_files = [
        "Codebook_energy_cleaned.xlsm",
        "Codebook_Coal_Clean.xlsm",
        "Codebook_Renewables_Clean.xlsm",
        "Codebook_Wind_Solar_Cleaned.xlsm"
    ]
    
    combined_df = pd.DataFrame()
    all_columns = []

    for file in excel_files:
        df = pd.read_excel(os.path.join(directory, file), engine='openpyxl')
        
        # Check if the first column has unique non-NaN values
        first_column = df.iloc[:, 0]
        non_nan_values = first_column.dropna()
        if non_nan_values.duplicated().any():
            raise ValueError(f"The first column in {file} contains duplicate non-NaN values.")
        
        # If it's the first file, use its columns as the base
        if combined_df.empty:
            base_columns = df.columns.tolist()
            combined_df = df
            all_columns = base_columns
        else:
            # Find unmatched columns
            unmatched_columns = [col for col in df.columns if col not in all_columns]
            
            # Add unmatched columns to all_columns
            all_columns.extend(unmatched_columns)
            
            # Create a new dataframe with all columns, fill with "Missing" for new columns
            new_df = pd.DataFrame(index=df.index, columns=all_columns)
            for col in all_columns:
                if col in df.columns:
                    new_df[col] = df[col]
                else:
                    new_df[col] = "Missing"
            
            # Concatenate with the combined dataframe
            combined_df = pd.concat([combined_df, new_df], ignore_index=True)

    # Remove rows where the first column is NaN
    combined_df = combined_df.dropna(subset=[combined_df.columns[0]])

    # Check for duplicate non-NaN values in the first column of the combined dataframe
    first_column = combined_df.iloc[:, 0]
    non_nan_values = first_column.dropna()
    if non_nan_values.duplicated().any():
        raise ValueError("The combined dataframe contains duplicate non-NaN values in the first column.")

    return combined_df

# Usage
try:
    directory = r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Omnibus Generator"
    result = combine_excel_files(directory)
    print("Combined dataframe created successfully.")
    print(result.head())  # Display the first few rows of the combined dataframe
    
    # Save the combined dataframe to a new Excel file
    output_path = os.path.join(directory, "Combined_Codebook.xlsx")
    result.to_excel(output_path, index=False)
    print(f"Combined data saved to {output_path}")

    # Print column order
    print("\nFinal column order:")
    for i, col in enumerate(result.columns, 1):
        print(f"{i}. {col}")

except Exception as e:
    print(f"An error occurred: {str(e)}")