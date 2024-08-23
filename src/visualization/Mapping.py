import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import plotly.io as pio

# Read the Excel file into a DataFrame
# Load the CSV file
file_path = r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Codebook_Omnibus_Extractions_Select_Columns_W_Duplicates.xlsx"
df = pd.read_excel(file_path)


# Define subnational regions and their corresponding countries
subnational_mapping = {
    "California": ["California"],  # State in USA
    "Ontario": ["Ontario"],  # Province in Canada
    "Québec": ["Québec"],  # Province in Canada
    "Tokyo": ["Tokyo"],  # Prefecture in Japan
    "Scotland": ["Aberdeen", "Aberdeenshire", "Angus", "Argyll and Bute", "Clackmannanshire", 
                 "Dumfries and Galloway", "Dundee", "East Ayrshire", "East Dunbartonshire", 
                 "East Lothian", "East Renfrewshire", "Edinburgh", "Falkirk", "Fife", 
                 "Glasgow", "Highland", "Inverclyde", "Midlothian", "Moray", "Eilean Siar", 
                 "North Ayshire", "North Lanarkshire", "Orkney", "Perthshire and Kinross", 
                 "Renfrewshire", "Scottish Borders", "Shetland Islands", "South Ayrshire", 
                 "South Lanarkshire", "Stirling", "West Dunbartonshire", "West Lothian"],
    "Apulia": ["Bari", "Barletta-Andria Trani", "Brindisi", "Foggia", "Lecce", "Taranto"],  # Provinces in Apulia, Italy
    "Tuscany": ["Arezzo", "Firenze", "Grosseto", "Livorno", "Lucca", "Massa-Carrara", 
                "Pisa", "Pistoia", "Prato", "Siena"],  # Provinces in Tuscany, Italy
    "Shanghai": ["Shanghai"],  # Municipality in China
    "Shenzhen": ["Guangdong"],  # City in Guangdong province, China
    "Anhui Province": ["Anhui"],  # Province in China
    "Jiangxi Province": ["Jiangxi"],  # Province in China
}

# Define supranational entities and their member countries
supranational_entities = {
    "European Union": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", 
                       "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", 
                       "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", 
                       "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"],
    "ASEAN": ["Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia", "Myanmar", 
              "Philippines", "Singapore", "Thailand", "Vietnam"]
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
region_counts_df = pd.DataFrame.from_dict(region_counts, orient='index', columns=['count']).reset_index()
region_counts_df.columns = ['research_geography', 'count']
region_counts_df = region_counts_df.sort_values('count', ascending=False)

# Load the world map data from the local file
world = gpd.read_file(r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp")

# Ensure country names match between your data and the shapefile
world['NAME'] = world['NAME'].replace({
    "United States of America": "United States",
    "United Kingdom": "United Kingdom",
    "Korea, Republic of": "South Korea",
    "Czech Republic": "Czechia"
})

# Merge the frequency data with world map data
world['count'] = world['NAME'].map(region_counts).fillna(0)

# Load the states map data from the local file
states = gpd.read_file(r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\ne_10m_admin_1_states_provinces\ne_10m_admin_1_states_provinces.shp")

# Add count column to states for subnational regions
states['count'] = states['name'].map(lambda x: region_counts.get(x, 0))

# Filter states with non-zero counts
states_filtered = states[states['count'] > 0]

# Create the figure
fig = go.Figure()

# Add supranational entities (EU and ASEAN) with a duller color
for entity, countries in supranational_entities.items():
    entity_geom = world[world['NAME'].isin(countries)].dissolve()
    entity_count = sum(region_counts.get(country, 0) for country in countries)
    
    fig.add_trace(go.Choropleth(
        geojson=entity_geom.__geo_interface__,
        locations=[0],  # Dummy location
        z=[entity_count],
        colorscale=[[0, 'rgba(255,215,0,0.2)'], [1, 'rgba(255,215,0,0.2)']],
        showscale=False,
        hoverinfo='text',
        text=f"{entity}: {entity_count}",
        marker_line_color='rgba(255,215,0,0.5)',
        marker_line_width=2,
    ))

# Create a custom color scale with white for 0
colors = ['#FFFFFF'] + px.colors.sequential.YlOrRd[1:]  # White for 0, then yellow to red

# Add national regions
fig.add_trace(go.Choropleth(
    geojson=world.geometry.__geo_interface__,
    locations=world.index,
    z=world['count'],
    colorscale=colors,
    zmin=0,
    zmax=world['count'].max(),
    marker_opacity=0.8,
    marker_line_width=0.5,
    colorbar_title="Research Frequency",
))

# Add subnational regions overlay
fig.add_trace(go.Choropleth(
    geojson=states_filtered.geometry.__geo_interface__,
    locations=states_filtered.index,
    z=states_filtered['count'],
    colorscale=[[0, 'rgba(128,128,128,0)'], [1, 'rgba(128,128,128,0.5)']],
    zmin=0,
    zmax=states_filtered['count'].max(),
    marker_opacity=0.5,
    marker_line_width=0.5,
    showscale=False,
))

fig.update_layout(
    title_text='Frequency of Research by Region',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular',
        oceancolor='rgb(173, 216, 230)',  # Light blue color for the ocean
        showocean=True,  # Ensure the ocean is visible
        showcountries=True,
        countrycolor='rgb(204, 204, 204)',  # Light grey for country borders
    ),
    height=600,
    margin={"r":0,"t":30,"l":0,"b":0},
)
# Save the figure as an interactive HTML file
fig.write_html("final_global_research_frequency_map4.html")

print("\nInteractive map has been saved as 'final_global_research_frequency_map.html'")
