import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from typing import List, Tuple
import os
import textwrap


def clean_labels(labels):
    """Clean up labels for better readability."""
    cleanup_dict = {
        'e_national_development_bank': 'National Development Bank',
        't_network_dependency': 'Network Dependency',
        'i_political_stability': 'Political Stability',
        'e_other': 'Other',
        'e_structural_reform': 'Structural Reform',
        'e_cost_producer': 'Producer Costs',
        'e_cost_consumer': 'Consumer Costs',
        'i_policy_transparency': 'Policy Transparency',
        'i_policy_surveillance': 'Policy Surveillance',
        't_other': 'Technology Specific Factors',
        'e_coal_taxes': 'Coal Taxes',
        's_effects_on_health_and_wellbeing': 'Health and Wellbeing Effects',
        'e_state_guarantees': 'State Guarantees',
        'e_electricity_mix': 'Electricity Mix',
        't_skill_and_accessibility': 'Skill Accessibility',
        'g_resources_available': 'Resource Availability',
        'e_economic_growth_objective': 'Economic Growth Objective',
        'e_market_structure': 'Market Structure',
        's_path_dependency': 'Path Dependency',
        's_population_density' : 'Population Density',
        's_distributional_effects': 'Distributional Effects',
        's_energy_security': 'Energy Security',
        'e_gdp_per_capita': 'GDP per Capita',
        'e_cost_of_capital': 'Capital Costs',
        'i_policy_environment': 'Policy Environment',
        't_risk': 'Risk',
        'e_feedin_premiums': 'Feed-in Premiums',
        'e_tax_relief_supply_side': 'Supply-side Tax Relief',
        'e_market_financial': 'Financial Markets',
        'e_interest_rates': 'Interest Rates',
        'e_procurement_rules': 'Procurement Rules',
        'i_learning': 'Learning',
        'e_share_oil_gas_gdp': 'Oil and Gas Share of GDP',
        'e_market_creation': 'Market Formation',
        'e_rdd_funding': 'R&D Funding',
        'cl_coordinating_body_for_climate_strategy': 'Climate Strategy Coordination',
        't_political_renewable_energy_target': 'RE Targets',
        'i_political_system': 'Political System',
        's_interest_group_support': 'Interest Group Support',
        'i_institutional_capacity': 'Institutional Capacity',
        's_public_acceptance': 'Public Acceptance',
        't_transfer': 'Transfer',
        'i_policy': 'Policy Environment',
        'e_fiscal_latitude': 'Fiscal Latitude',
        'e_feedin_tariffs': 'Feed-in Tariffs',
        'e_costs' : 'Costs',
        'e_eu_ets': 'EU ETS',
        't_substitutability': 'Substitutability',
        'i_professional_training_and_qualification': 'Professional Training',
        's_land_use_consenting': 'Land Use Consenting',
        'e_investor_risk': 'Investment Risk',
        'i_legal_and_administrative_capacity': 'Legal and Administrative Capacity',
        'p_institutional_creation': 'Institutional Creation',
        'b_reduced_regulatory_overhead': 'Regulatory Streamlining',
        'e_financing_options': 'Financing Options',
        'e_tendering_schemes': 'Tendering Schemes',
        't_formal_re_target': 'Formal RE Targets',
        'e_production_subsidy': 'Production Subsidies',
        'v_negotiated_agreements_publicprivate_sector': 'Public-Private Agreements',
        's_domestic_industry': 'Domestic Industry',
        's_industry_of_strategic_political_interest': 'Strategic Industry',
        'e_income_per_capita': 'Income per Capita',
        'r_ban_moratorium_fossil_alternative': 'Ban/Moratorium on Fossil Fuels',
        'e_funding_through_excess_co2_quotas': 'Extraordinary Funding',
        'rdd_demonstration_project': 'Demonstration Projects',
        'r_non_financial_incentives': 'Non-financial Incentives',
        't_formal_target': 'Formal Targets',
        'r_procurement_rules_govt_consumption': 'State Procurement',
        'e_tax_relief_demand_side': 'Demand-side Tax Relief',
        'r_product_standards': 'Product Standards',
        'i_international_agreement': 'International Agreements',
        's_climate_targets': 'Climate Targets',
        's_history_of_sustained_support': 'Sustained Support',
        's_rural_development': 'Rural Development Objectieve',
        'r_other_mandatory_requirements': 'Other Mandatory Requirements',
        'e_market_incentives': 'Market Incentives',
        'env_climate_conditions': 'Climate Conditions',
        'r_mandatory_biofuel_share': 'Mandatory Biofuel Share',
        'r_biofuel_blending_mandate': 'Biofuel Blend Mandate',
        'e_co2_taxes': 'CO2 Taxes',
        'e_grants_and_subsidies': 'Grants and Subsidies',
        'r_obligation_schemes': 'Obligation Schemes',
        'e_carbon_price': 'Carbon Price',
        'p_strategic_planning': 'Strategic Planning',
        'e_state_loans': 'State Loans',
        'r_auditing': 'Auditing',
        'e_employment_effects': 'Employment Effect',
        'i_political_support_international_competition': 'International Competition',
        'e_energy_mix': 'Energy Mix',
        't_substituability': 'Substitutability',
        'i_envi_party': 'Environmental Party',
        'env_impact': 'Environmental Impact',
        'e_interest_group_support': 'Economic Interest Backing',
        's_education': 'Education',
        'e_costs_state':'Cost to State',
        's_structural_reform': 'Structural Reform',
        'e_cost': 'Cost',
        't_substitutability': 'Technological Substitutability',
        'e_employment_effect': 'Employment Effect',
        't_maturity': 'Technological Maturity',
        'e_share_electricity_coal': 'Coal Share of Electricity',
        'e_share_energy_coal': 'Coal Share of Energy',
        'i_policy_design': 'Policy Design',
        't_political_target': 'Coal Phase Out Target',
        'e_retirement_premium': 'Retirement Premiums',
        'b_removal_of_fossilfuel_subsidies': 'Fossil Fuel Subsidy Removal',
        'r_procurement_rules_general': 'Renewable Portfolio Standards',
        'e_green_certificates': 'Green Certificates',
        'e_infrastructure_investments': 'Infrastructure Investments',
        'com_narrative': 'Communication Narrative',
        'i_advice_and_aid_in_implementation': 'Implementation Support'
    }
    return [cleanup_dict.get(label, label) for label in labels]



def create_and_save_heatmap(co_occurrence_data, clusters, output_path, color_palette=None, title=None, threshold=1):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create and save dummy grid visualization
    dummy_fig, dummy_ax = create_dummy_grid(co_occurrence_data, title="Raw Co-occurrence Counts")
    dummy_output = output_path.replace(".png", "_dummy_grid.png")
    dummy_fig.savefig(dummy_output, bbox_inches='tight', dpi=300)
    plt.close(dummy_fig)
    
    # Adjust figure proportions and spacing
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 0.8], hspace=0.6)  # Increased hspace for x-axis label
    
    heatmap_ax = fig.add_subplot(gs[0])
    legend_ax = fig.add_subplot(gs[1])

    n_clusters = len(clusters)

    # Use specified color palette or default
    if color_palette is None:
        color_palette = plt.cm.Set1(np.linspace(0, 1, max(9, n_clusters)))

    base_colors = color_palette[:n_clusters]

    cluster_cmaps = [LinearSegmentedColormap.from_list("", ["white", color]) for color in base_colors]

    max_co_occurrence = co_occurrence_data.max().max()

    # Modified filtering logic
    total_co_occurrence = co_occurrence_data.sum(axis=1)
    co_occurrence_data = co_occurrence_data[total_co_occurrence >= threshold]
    mask = (co_occurrence_data >= threshold).any(axis=1)
    co_occurrence_data = co_occurrence_data[mask]
    
    # Get updated enablers/entries after filtering
    enablers = co_occurrence_data.index.get_level_values(0).unique()
    entries = co_occurrence_data.index.get_level_values(1).unique()

    # Calculate the highest co-occurrence for each cluster
    cluster_max_values = co_occurrence_data.max()
    for i, enabler in enumerate(enablers):
        for j, entry in enumerate(entries):
            if (enabler, entry) in co_occurrence_data.index:
                values = co_occurrence_data.loc[(enabler, entry)].values

                # Only plot if at least one value is >= threshold
                if any(v >= threshold for v in values):
                    for k, value in enumerate(values):
                        if value >= threshold:
                            # Normalize the value relative to the cluster's highest co-occurrence
                            normalized_value = value / cluster_max_values[clusters[k]]

                            color = cluster_cmaps[k](normalized_value)
                            size = max(100, min(1000, 1000 * (value / max_co_occurrence)))
                            # Adjust the x-position to be closer to the center
                            heatmap_ax.scatter(
                                j + (k + 1) / (n_clusters + 1),
                                i + 0.5,
                                s=size,
                                color=color,
                                edgecolor="black",
                                linewidth=0.5,
                            )
    # Clean and set labels
    cleaned_x_labels = clean_labels(entries)
    cleaned_y_labels = clean_labels(enablers)
    # Add faint dotted gridlines
    heatmap_ax.grid(which='both', color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    # Ensure gridlines are behind the data
    heatmap_ax.set_axisbelow(True)

    heatmap_ax.set_xticks(np.arange(len(cleaned_x_labels))+0.5)
    heatmap_ax.set_yticks(np.arange(len(cleaned_y_labels))+0.5)
    heatmap_ax.set_xticklabels(cleaned_x_labels, rotation=45, ha="right", fontsize=18)
    heatmap_ax.set_yticklabels(cleaned_y_labels, fontsize=18)
    heatmap_ax.set_title(title, fontsize=20, fontweight='bold')
    heatmap_ax.set_xlabel("Policy Intervention", fontsize=20)
    heatmap_ax.set_ylabel("Enablers", fontsize=20)

    # Configure legend axis
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.grid(False)
    legend_ax.set_frame_on(False)

    # Add grey background matching main plot width
    legend_bg = plt.Rectangle(
        (heatmap_ax.get_xlim()[0], 0.2),  # Start at main plot's left edge
        heatmap_ax.get_xlim()[1] - heatmap_ax.get_xlim()[0],  # Match main plot width
        0.6,  # Height
        transform=legend_ax.get_xaxis_transform(),  # Use blended transform
        color='#f0f0f0',
        zorder=1
    )
    legend_ax.add_patch(legend_bg)

    # Calculate positions relative to main plot's x-axis
    x_start = heatmap_ax.get_xlim()[0]
    total_width = heatmap_ax.get_xlim()[1] - x_start
    spacing = total_width / len(clusters)
    
    for i, (cluster, color) in enumerate(zip(clusters, base_colors)):
        x_pos = x_start + (i * spacing) + (spacing * 0.3)
        
        # Draw marker shifted left by diameter
        legend_ax.scatter(
            x_pos - 0.15,  # Shift left by circle diameter
            0.5, 
            s=600, 
            c=[color], 
            edgecolor="black", 
            zorder=2,
            transform=legend_ax.get_xaxis_transform()
        )
        
        # Text position remains the same
        wrapped_text = '\n'.join(textwrap.wrap(cluster, width=20))
        legend_ax.text(
            x_pos + 0.02, 
            0.5, 
            wrapped_text, 
            fontsize=16,
            va='center', 
            ha='left', 
            zorder=2,
            transform=legend_ax.get_xaxis_transform()
        )

    # Set matching x-limits with main plot
    legend_ax.set_xlim(heatmap_ax.get_xlim())
    legend_ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved as {output_path}")

def create_dummy_grid(co_occurrence_data: pd.DataFrame, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grid visualization showing raw co-occurrence counts.
    """
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111)

    # Get raw counts before any filtering
    enablers = co_occurrence_data.index.get_level_values(0).unique()
    entries = co_occurrence_data.index.get_level_values(1).unique()
    
    # Create empty matrix filled with zeros
    matrix = pd.DataFrame(0, index=enablers, columns=entries)
    
    # Fill matrix with actual co-occurrence counts
    for (enabler, entry), row in co_occurrence_data.iterrows():
        matrix.loc[enabler, entry] = row.sum()

    # Plot background colors based on co-occurrence values
    ax.imshow(matrix, cmap='RdBu', aspect='auto', 
              vmin=0, vmax=matrix.max().max())

    # Add text annotations
    for i, enabler in enumerate(enablers):
        for j, entry in enumerate(entries):
            value = matrix.loc[enabler, entry]
            ax.text(j, i, int(value), ha="center", va="center", 
                   color="white" if value > matrix.values.max()/2 else "black",
                   fontsize=14)
    # Clean and set labels
    cleaned_x_labels = clean_labels(entries)
    cleaned_y_labels = clean_labels(enablers)
    
    ax.set_xticks(np.arange(len(cleaned_x_labels)))
    ax.set_yticks(np.arange(len(cleaned_y_labels)))
    ax.set_xticklabels(cleaned_x_labels, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(cleaned_y_labels, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Policy Intervention", fontsize=18)
    ax.set_ylabel("Enablers", fontsize=18)
    
    plt.tight_layout()
    return fig, ax
