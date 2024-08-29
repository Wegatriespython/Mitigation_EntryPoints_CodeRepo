import geopandas as gpd
import sys

# Ensure proper encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Load the states map data from the local file
states = gpd.read_file(r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\ne_10m_admin_1_states_provinces\ne_10m_admin_1_states_provinces.shp")

# List of countries we're interested in
countries = ['Canada', 'Brazil', 'Japan', 'Italy', 'United Kingdom', 'China']

# Dictionary to store the states for each country
country_states = {country: [] for country in countries}

# Iterate through the states and categorize them by country
for index, state in states.iterrows():
    if state['admin'] in countries:
        country_states[state['admin']].append(state['name'])

# Print the results
for country, state_list in country_states.items():
    print(f"\nStates/Provinces in {country}:")
    for state in sorted(state_list):
        print(f"- {state}")

print("\nNote: The United Kingdom may be listed as 'Great Britain' in some datasets.")
print("Note: Some countries might use different terms for their subnational divisions (e.g., provinces, regions, etc.)")