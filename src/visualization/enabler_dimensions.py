import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

# Configuration
CLEAN_ENABLER = {
    'b_reduced_regulatory_overhead': 'Regulatory streamlining',
    'b_removal_of_fossilfuel_subsidies': 'Fossil fuel subsidy removal',
    'cl_coordinating_body_for_climate_strategy': 'Climate strategy coordination',
    'com_narrative': 'Communication narrative',
    'e_carbon_price': 'Carbon price',
    'e_co2_taxes': 'CO2 taxes',
    'e_coal_taxes': 'Coal taxes',
    'e_cost_consumer': 'Technology costs',
    'e_cost_of_capital': 'Capital costs',
    'e_cost_producer': 'Technology costs',
    'e_cost_state': 'State cost',
    'e_cost': 'Technology costs',
    'e_costs': 'Technology costs',
    'e_costs_state': 'State cost',
    'e_economic_growth_objective': 'Economic growth objective',
    'e_electricity_mix': 'Electricity mix',
    'e_employment_effect': 'Employment effect',
    'e_employment_effects': 'Employment effect',
    'e_energy_mix': 'Energy mix',
    'e_eu_ets': 'EU ETS',
    'e_feedin_premiums': 'Feed-in premiums',
    'e_feedin_tariffs': 'Feed-in tariffs',
    'e_financing_options': 'Financing options',
    'e_fiscal_latitude': 'Fiscal latitude',
    'e_funding_through_excess_co2_quotas': 'Extraordinary funding',
    'e_GDP_current_level': 'GDP',
    'e_gdp_per_capita': 'GDP',
    # 'e_GDP_per_capita': 'GDP',  # Duplicate key, handled below.  Best practice is to avoid.
    'e_GDP': 'GDP',
    'e_grants_and_subsidies': 'Grants and subsidies',
    'e_green_certificates': 'Green certificates',
    'e_income_per_capita': 'Income per capita',
    'e_infrastructure_investments': 'Infrastructure investments',
    'e_interest_group_support': 'Interest group support',
    'e_interest_rate': 'Interest rates',
    'e_interest_rates': 'Interest rates',
    'e_investor_risk': 'Investment risk',
    'e_market_creation': 'Market formation',
    'e_market_financial': 'Financial markets',
    'e_market_incentives': 'Market incentives',
    'e_market_structure': 'Market structure',
    'e_oil_prices': 'Oil price',
    'e_other': 'Other',
    'e_procurement_rules': 'Procurement rules',
    'e_production_subsidy': 'Production subsidies',
    'e_rdd_funding': 'R&D funding',
    'e_retirement_premium': 'Retirement premiums',
    'e_share_electricity_coal': 'Coal share of electricity',
    'e_share_energy_coal': 'Coal share of energy',
    'e_share_oil_gas_gdp': 'Oil and gas share of GDP',
    # 'e_share_oil_gas_GDP': 'Oil and gas share of GDP', # Duplicate, handled below
    'e_state_guarantees': 'State guarantees',
    'e_state_loans': 'State loans',
    'e_structural_reform': 'Structural reform',
    'e_tax_relief_demand_side': 'Demand-side tax relief',
    'e_tax_relief_supply_side': 'Supply-side tax relief',
    'e_tendering_schemes': 'Tendering schemes',
    'env_climate_conditions': 'Climate conditions',
    'env_impact': 'Environmental impact',
    'g_resources_available': 'Resource availability',
    'i_advice_and_aid_in_implementation': 'Implementation support',
    'i_envi_party': 'Environmental party',
    'i_institutional_capacity': 'Institutional capacity',
    'i_international_agreement': 'International agreements',
    'i_international_cooperation': 'International cooperation',
    'i_learning': 'Learning',
    'i_legal_and_administrative_capacity': 'Legal and administrative capacity',
    'i_policy_design': 'Policy design',
    'i_policy_environment': 'Policy environment',
    'i_policy_surveillance': 'Policy surveillance',
    'i_policy_transparency': 'Policy transparency',
    'i_policy': 'Policy environment',
    'i_political_acceptance': 'Political acceptance',
    'i_political_stability': 'Political stability',
    'i_political_support_international_competition': 'Global competition',
    'i_political_support_national_competition': 'Global competition',
    'i_political_system': 'Political system',
    'i_professional_training_and_qualification': 'Professional training',
    'p_institutional_creation': 'Institutional creation',
    'p_strategic_planning': 'Strategic planning',
    'r_auditing': 'Auditing',
    'r_ban_moratorium_fossil_alternative': 'Ban/Moratorium on fossil fuels',
    'r_biofuel_blending_mandate': 'Biofuel blend Mandate',
    'r_mandatory_biofuel_share': 'Mandatory biofuel Share',
    'r_non_financial_incentives': 'Non-financial incentives',
    'r_obligation_schemes': 'Obligation schemes',
    'r_other_mandatory_requirements': 'Other mandatory requirements',
    'r_procurement_rules_general': 'Renewable portfolio standards',
    'r_procurement_rules_govt_consumption': 'State procurement',
    'r_product_standards': 'Product standards',
    'rdd_demonstration_project': 'Demonstration projects',
    's_climate_targets': 'Climate targets',
    's_distributional_effects': 'Distributional effects',
    's_domestic_industry': 'Domestic industry',
    's_education': 'Education',
    's_effects_on_health_and_wellbeing': 'Health and wellbeing effects',
    's_effects_on_health_and_well-being': 'Health and wellbeing effects',
    'eco_effects_on_health_and_well-being': 'Health and wellbeing effects', #from original R code
    's_energy_security': 'Energy security',
    's_history_of_sustained_support': 'Sustained support',
    's_industry_of_strategic_political_interest': 'Strategic industry',
    's_interest_group_support': 'Interest group support',
    's_land_use_consenting': 'Land use consenting',
    's_nuclearphase-out': 'Nuclear phase-out',
    's_path_dependency': 'Path dependency',
    's_population_density': 'Population density',
    's_public_acceptance': 'Public acceptance',
    's_rural_development': 'Rural development objectieve',
    's_secondary_education': 'Education',
    's_structural_reform': 'Structural reform',
    'e_share_oil_gas_GDP': 'GDP Share of Oil and Gas',
    'e_GDP_per_capita': 'GDP per Capita', 
    't_formal_re_target': 'RE targets',
    'i_other': 'Other Institutional Factors',
    't_formal_target': 'Formal targets',
    't_maturity': 'Technological maturity',
    't_network_dependency': 'Network dependency',
    't_other': 'Technology specific factors',
    't_political_renewable_energy_target': 'RE targets',
    't_political_target': 'Coal phase out target',
    't_risk': 'Technology risk',
    't_scalability': 'Scalability',
    't_simplicity': 'Simplicity',
    't_skill_and_accessibility': 'Skill accessibility',
    't_substitutability': 'Technological substitutability',
    # 't_substitutability': 'Technological substitutability',  # Triple duplicate, handled below.
    # 't_substitutability': 'Technological substitutability',
    't_transfer': 'Technology transfer',
    'v_negotiated_agreements_publicprivate_sector': 'Public-private agreements',
    'e_national_development_bank': 'National development bank',
    'eco_env_impact': 'Environmental impact',
    None: None  # Handle NULL, maps to Python's None
}



DIMENSION_MAP = {
    'e_': 'economic',
    's_': 'social',
    't_': 'technological',
    'g_': 'geographical',
    'eco_': 'ecological',
    'i_': 'institutional'
}

def categorize_dimension(enabler):
    if not isinstance(enabler, str):  # Handle non-string values
        return 'other'
    for prefix, dimension in DIMENSION_MAP.items():
        if enabler.startswith(prefix):
            return dimension
    return 'other'

def process_data(filepath):
    # Load and clean data
    df = pd.read_excel(filepath, usecols=['Enabler', 'Entry (policy intervention)'])
    
    # Split and clean enablers
    enablers = df['Enabler'].str.split(',').explode()
    enablers = enablers.str.strip().str.replace(r'[\"()]', '', regex=True)
    
    # Convert to string and handle missing values
    enablers = enablers.astype(str).replace('nan', '')
    
    # Apply cleaning and categorization
    enablers_df = pd.DataFrame({'enabler': enablers})
    enablers_df['clean_enabler'] = enablers_df['enabler'].map(CLEAN_ENABLER).fillna(enablers_df['enabler'])
    enablers_df['dimension'] = enablers_df['enabler'].apply(categorize_dimension)
    
    # Filter and count
    counts = enablers_df.groupby(['clean_enabler', 'dimension']).size().reset_index(name='count')
    return counts[counts['count'] > 2]


def generate_dimension_tables(counts_df):
    """Generate dimension-specific tables with frequency-based coloring"""
    dimensions = counts_df['dimension'].unique()
    max_count = counts_df['count'].max()
    
    for dimension in dimensions:
        subset = counts_df[counts_df['dimension'] == dimension].sort_values('count', ascending=False)
        
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        ax.axis('off')
        
        # Create table with colored background - FIXED COLOR ARRAY
        cell_colors = []
        for count in subset['count']:
            norm_count = count / max_count
            # Create color for both columns in row
            cell_colors.append([plt.cm.viridis(norm_count), plt.cm.viridis(norm_count)])
            
        table = plt.table(
            cellText=subset[['clean_enabler', 'count']].values,
            colLabels=['Enabler', 'Count'],
            cellLoc='center',
            cellColours=cell_colors,  # Now matches 2 columns per row
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title(f"Enabler Frequency: {dimension.capitalize()}", pad=20)
        plt.savefig(f"enabler_{dimension}.png", bbox_inches='tight')
        plt.close()

def generate_circular_plot(counts_df):
    """Create circular bar plot visualization"""
    # Add 'other' to color palette
    color_palette = {
        'economic': 'blue',
        'social': 'red',
        'technological': 'green',
        'geographical': 'purple',
        'ecological': 'orange',
        'institutional': 'brown',
        'other': 'gray'  # Add default color for uncategorized items
    }
    
    # Filter out any remaining 'other' category if desired
    # counts_df = counts_df[counts_df['dimension'] != 'other']
    
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Calculate angles and ordering
    categories = counts_df['clean_enabler'].values
    values = counts_df['count'].values
    dimensions = counts_df['dimension'].unique()
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the circle
    
    # Plot bars
    for idx, (category, value, dimension) in enumerate(zip(categories, values, counts_df['dimension'])):
        ax.bar(angles[idx], value, width=0.5, color=color_palette[dimension], alpha=0.7)
    
    # Customize axis
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.tick_params(axis='x', pad=25)  # Adjust label padding
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_palette[d]) for d in dimensions]
    plt.legend(handles, dimensions, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Circular Enabler Frequency", y=1.08)
    plt.savefig("circular_plot.png", bbox_inches='tight')
    plt.close()

def generate_faceted_barchart(counts_df, top_n=10):
    """Generate 2x2 grid of horizontal bar charts showing top enablers by dimension"""
    # Filter for main dimensions and get top n enablers
    main_dimensions = ['economic', 'institutional', 'social', 'technological']
    top_enablers = (counts_df[counts_df['dimension'].isin(main_dimensions)]
                   .groupby('dimension')
                   .apply(lambda x: x.nlargest(top_n, 'count'))
                   .reset_index(drop=True))
    
    # Set paper-style formatting
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'grid.color': '.9',
        'axes.facecolor': 'white',
        'axes.edgecolor': '.4',
        'axes.labelcolor': '.3'
    })
    
    # Create plot figure
    fig = plt.figure(figsize=(12, 8))
    
    # Set up 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()  # Flatten to make iteration easier
    
    # Create a subplot for each dimension
    for idx, dimension in enumerate(main_dimensions):
        ax = axes[idx]
        dim_data = top_enablers[top_enablers['dimension'] == dimension]
        
        # Sort by count for consistent visualization
        dim_data = dim_data.sort_values('count')
        
        # Create horizontal bars
        bars = ax.barh(dim_data['clean_enabler'], dim_data['count'], 
                      color=sns.color_palette()[idx], alpha=0.8)
        
        # Customize subplot
        ax.set_title(dimension.capitalize(), pad=10, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(0, dim_data['count'].max() * 1.1)  # Add 10% padding
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            # Add small offset (0.5) to x-coordinate instead of using padding
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', 
                   fontsize=12, fontweight='bold')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add light grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.suptitle('Top Enablers by Dimension', fontsize=15, fontweight ='bold', y=1.02)
    
    # Modify footer position and add padding
    plt.subplots_adjust(bottom=0.15)  # Increase bottom margin
    fig.text(0.5, 0.04,  # Move footer up slightly
            "*Geographic Dimension omitted (single enabler: Resource Availability, 20 occurrences)",
            ha='center', fontsize=12, color='.4', style='italic')

    # Add padding between title and plots
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Add 5% padding at bottom

    plt.savefig("faceted_barchart.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    counts_df = process_data(r"V:\Paper_rahel\Files\OverviewEnablerEntry.xlsx")
    counts_df.to_csv("enabler_counts.csv", index=False)
    
    generate_faceted_barchart(counts_df)
    generate_dimension_tables(counts_df)
