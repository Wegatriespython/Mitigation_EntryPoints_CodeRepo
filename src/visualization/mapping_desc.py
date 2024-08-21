import pandas as pd
from collections import Counter
from openpyxl import Workbook
# Read the Excel file into a DataFram
file_path = r"C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Codebook_Omnibus_Extractions_Select_Columns_W_Duplicates.xlsx"
df = pd.read_excel(file_path)
output_file = "C:\\Users\\vigne\\OneDrive - Wageningen University & Research\\Internship\\Literature Review\\Final Data Processing\\Mapping_Description.xlsx"

# Define subnational regions and their corresponding countries
subnational_mapping = {
    "California": ["California"],  # State in USA
    "Ontario": ["Ontario"],  # Province in Canada
    "Québec": ["Québec"],  # Province in Canada
    "Tokyo": ["Tokyo"],  # Prefecture in Japan
    "Scotland": ["Scotland"],  # Council areas in Scotland
    "Apulia": ["Apulia"],  # Provinces in Apulia, Italy
    "Tuscany": ["Tuscany"],  # Provinces in Tuscany, Italy
    "Shanghai": ["Shanghai"],  # Municipality in China
    "Shenzhen": ["Shenzhen"],  # City in China
    "Anhui Province": ["Anhui"],  # Province in China
    "Jiangxi Province": ["Jiangxi"],  # Province in China
}

# Define supranational entities and their member countries
supranational_entities = {
    "European Union": ["EU"],
    "ASEAN": ["ASEAN"]
}

# Function to handle multi-country entries and return a list
def map_multi_country(entry):
    if isinstance(entry, str):
        countries = [country.strip() for country in entry.split(',')]
        mapped_countries = []
        for country in countries:
            if country in subnational_mapping:
                mapped_countries.extend(subnational_mapping[country])
            else:
                mapped_countries.append(country)
        return mapped_countries
    else:
        return [subnational_mapping.get(entry, entry)]

# Apply mapping and flatten the list of countries
all_countries = [country for countries in df['research_geography'].apply(map_multi_country) for country in countries]

# Count occurrences of each geography
region_counts = Counter(all_countries)
region_freq = {k: v / len(all_countries) for k, v in region_counts.items()}
region_counts_df = pd.DataFrame.from_dict(region_counts, orient='index', columns=['count']).reset_index()
region_counts_df.columns = ['research_geography', 'count']
region_counts_df = region_counts_df.sort_values('count', ascending=False)

wb = Workbook()
ws = wb.active
ws.title = "Region Distribution"
ws.append(region_counts_df.columns.tolist())

for row in region_counts_df.iterrows():
    ws.append(row[1].tolist())

wb.save(output_file)
