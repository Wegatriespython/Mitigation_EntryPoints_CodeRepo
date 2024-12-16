import pandas as pd
from collections import Counter

# File path
file_path = r"C:\Users\rahelma\Documents\Elevate\EntryPointCode\Mitigation_EntryPoints_CodeRepo\Data\raw\Codebook_Omnibus_Global_Technologies.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)
#df = pd.read_csv(file_path)

# Function to handle multi-country entries and return a list
def split_countries(entry):
    if isinstance(entry, str):
        return [country.strip() for country in entry.split(',')]
    else:
        return [entry]

# Apply the function to split multi-country entries and flatten the list
all_regions = [region for regions in df['research_geography'].apply(split_countries) for region in regions]

# Count occurrences of each region
region_counts = Counter(all_regions)

# Create a DataFrame from the counts
region_counts_df = pd.DataFrame.from_dict(region_counts, orient='index', columns=['count']).reset_index()
region_counts_df.columns = ['research_geography', 'count']
region_counts_df = region_counts_df.sort_values('count', ascending=False)

# Calculate and print the total number of unique regions
total_regions = len(region_counts)
print(f"Total number of unique regions: {total_regions}")

# Print the top 20 regions by count
print("\nTop 20 regions by count:")
print(region_counts_df.head(20))

# Optional: Save the full results to a CSV file
output_file = "region_counts.csv"
region_counts_df.to_csv(output_file, index=False)
print(f"\nFull results saved to {output_file}")