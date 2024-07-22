import pandas as pd
import plotly.express as px
from collections import defaultdict

# Load the CSV file

file_path = "C:/Users/vigne/OneDrive - Wageningen University & Research/Internship/Literature Review/Final Data Processing/Omnibus Generator/Updated_Merged_Codebook.xlsx"
data = pd.read_excel(file_path)

def clean_label(text):
    # Remove specific acronyms and parentheses
    replacements = {
        "electric_vehicles_(BEV)": "Electric Vehicles",
        "fuel_cell_cars_(FCEV)": "Fuel Cell Cars",
        "Behind-the-meter (BTM) energy storage": "Behind-the-meter Energy Storage",
        "Behind-the-meter energy storage": "Behind-the-meter Energy Storage",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # General cleaning
    text = text.replace('_', ' ').replace('(', '').replace(')', '')
    words = text.split()
    formatted_words = [word.capitalize() for word in words]
    return ' '.join(formatted_words)

def manual_count(df):
    sector_count = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        sectors = row['technology_sector'].split(',') if pd.notna(row['technology_sector']) else []
        options = row['technology_mitigation_option'].split(',') if pd.notna(row['technology_mitigation_option']) else []
        
        if sectors:
            sector = clean_label(sectors[0].strip())
            for option in options:
                option = clean_label(option.strip())
                sector_count[sector][option] += 1
    
    return sector_count

# Perform manual counting
counted_data = manual_count(data)

# Prepare data for sunburst chart
sunburst_data = []
for sector, options in counted_data.items():
    for option, count in options.items():
        sunburst_data.append({'sector': sector, 'mitigation_option': option, 'count': count})

nested_freq_table = pd.DataFrame(sunburst_data)

# Save the nested frequency table as a CSV file to inspect the results
nested_freq_table.to_csv('Filtered_Nested_Frequency_Table.csv', index=False)

print("Filtered nested frequency table saved as 'Filtered_Nested_Frequency_Table.csv'")

# Create the sunburst chart
fig = px.sunburst(
    nested_freq_table,
    path=['sector', 'mitigation_option'],
    values='count',
    color='count',
    color_continuous_scale='RdBu',
    color_continuous_midpoint=nested_freq_table['count'].mean()
)

fig.update_layout(
    margin=dict(t=80, l=0, r=0, b=0),
    title=dict(
        text='Technology Sectors and Mitigation Options',
        font=dict(size=24, family="Arial Black, sans-serif"),
        x=0.5,
        y=0.95
    ),
    coloraxis_colorbar=dict(
        title="Count",
        tickvals=[nested_freq_table['count'].min(), nested_freq_table['count'].mean(), nested_freq_table['count'].max()],
        ticktext=["Low", "Medium", "High"]
    )
)

fig.update_traces(
    textfont=dict(size=14, family="Arial Black, sans-serif"),
    insidetextorientation='radial',
    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}',
    marker=dict(line=dict(color='white', width=2)),
    domain=dict(x=[0.1, 0.9], y=[0.1, 0.9])
)

fig.show()