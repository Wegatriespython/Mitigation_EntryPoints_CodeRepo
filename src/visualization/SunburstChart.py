import pandas as pd
import plotly.express as px
from collections import defaultdict
import logging
from typing import Dict, List
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_label(text: str) -> str:
    """
    Clean and format text labels.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned and formatted text.
    """
    replacements = {
        "electric_vehicles_(BEV)": "Electric vehicles",
        "fuel_cell_cars_(FCEV)": "Fuel cell",
        "Behind-the-meter (BTM) energy storage": "Behind-the-meter Energy Storage",
        "Behind-the-meter energy storage": "Behind-the-meter Energy Storage",
        "Rewind-solar" : "RE",
        "carbon_management": "C.Mgt",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)


    text = text.replace('_', ' ').replace('(', '').replace(')', '')
    if text.lower() == "c.mgt":
        return "C.Mgt"
    return ' '.join(word.capitalize() for word in text.split())

def manual_count(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Count occurrences of technology sectors and mitigation options.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        Dict[str, Dict[str, int]]: A nested dictionary with counts.
    """
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

def prepare_sunburst_data(counted_data: Dict[str, Dict[str, int]]) -> List[Dict[str, str]]:
    """
    Prepare data for the sunburst chart.

    Args:
        counted_data (Dict[str, Dict[str, int]]): The counted data.

    Returns:
        List[Dict[str, str]]: Data ready for the sunburst chart.
    """
    sunburst_data = []
    total_count = sum(sum(options.values()) for options in counted_data.values())

    for sector, options in counted_data.items():
        print(sector, options)
        sector_total = sum(options.values())
        sector_percentage = (sector_total / total_count) * 100


        sector_label = f"{sector} ({sector_percentage:.1f}%)"


        for option, count in options.items():
            # Standardize labels
            if option.lower() == "ccs":
                option = "CCS"
            elif option == "C.Mgmt":
                option = "C.Mgt"
            elif option.lower() in ["rewind-solar", "RE(Wind-Solar)"]:
                option = "RE"

            sunburst_data.append({
                'sector': sector_label,
                'mitigation_option': f"{option} ({count})",
                'count': count
            })
    return sunburst_data

def create_sunburst_chart(nested_freq_table: pd.DataFrame) -> px.sunburst:
    """
    Create a sunburst chart with automatically positioned external annotations for small segments.
    """
    # Create the base sunburst chart
    fig = px.sunburst(
        nested_freq_table,
        path=['sector', 'mitigation_option'],
        values='count',
        color='count',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=nested_freq_table['count'].mean()
    )

    # Base layout updates
    fig.update_layout(
        width=1200,
        height=800,
        margin=dict(t=80, l=50, r=150, b=100),  # Changed right margin from 250 to 50 for better centering
        title=dict(
            text='Sectoral and technological coverage',
            font=dict(size=28, family="Arial Black, sans-serif"),
            x=0.5,  # This already centers the title horizontally
            y=0.95
        ),
        coloraxis_colorbar=dict(
            title="Frequency",
            tickvals=[nested_freq_table['count'].min(), nested_freq_table['count'].mean(), nested_freq_table['count'].max()],
            ticktext=["low", "medium", "high"],
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            len=0.6
        ),
        showlegend=False,
    )

    # Update trace properties
    fig.update_traces(
        textfont=dict(size=16, family="Arial Black, sans-serif"),
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent:.1%}',
        marker=dict(line=dict(color='white', width=2))
    )

    return fig

def main():
    try:
        # Load the CSV file
        file_path = r"V:\Paper_rahel\Files\Codebook_Omnibus_Global_Technologies.xlsx"
        data = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}")

        # Perform manual counting
        counted_data = manual_count(data)
        logging.info("Manual counting completed")

        # Prepare data for sunburst chart
        sunburst_data = prepare_sunburst_data(counted_data)
        nested_freq_table = pd.DataFrame(sunburst_data)

        # Save the nested frequency table as a CSV file
        output_file = 'Filtered_Nested_Frequency_Table.csv'
        nested_freq_table.to_csv(output_file, index=False)
        logging.info(f"Filtered nested frequency table saved as '{output_file}'")

        # Create the sunburst chart
        fig = create_sunburst_chart(nested_freq_table)
        fig.show()
        logging.info("Sunburst chart created and displayed")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
