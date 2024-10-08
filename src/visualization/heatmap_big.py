import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from typing import List, Tuple

def clean_labels(labels):
    """Clean up labels for better readability."""
    cleanup_dict = {
        'e_national_development_bank': 'National Development Bank',
        't_network_dependency': 'Network Dependency',
        'i_political_stability': 'Political Stability',
        'e_other': 'Other',
        'e_consumption_subsidy': 'Consumption Subsidies',
        'e_oil_prices': 'Oil Prices',
        't_simplicity': 'Technological Simplicity',
        'i_stakeholder_engagement': 'Stakeholder Engagement',
        'p_technology_transfer': 'Technology Transfer',
        'e_land_supply': 'Land Supply',
        't_formal_ghg_reduction_target': 'Formal GHG Targets',
        'e_structural_reform': 'Structural Reform',
        'e_cost_producer': 'Producer Costs',
        'e_cost_consumer': 'Consumer Costs',
        'e_coal_taxes': 'Coal Taxes',
        's_effects_on_health_and_wellbeing': 'Health and Wellbeing Effects',
        'e_state_guarantees': 'State Guarantees',
        'e_electricity_mix': 'Electricity Mix',
        't_skill_and_accessibility': 'Skill Accessibility',
        'g_resources_available': 'Resource Availability',
        'e_economic_growth_objective': 'Economic Growth Objective',
        'e_market_structure': 'Market Structure',
        's_path_dependency': 'Path Dependency',
        's_distributional_effects': 'Distributional Effects',
        's_energy_security': 'Energy Security',
        'e_gdp_per_capita': 'GDP per Capita',
        'e_cost_of_capital': 'Capital Costs',
        'e_gdp': 'GDP',
        's_nuclear': 'Nuclear Sentiment',
        't_simplicity' : 'Simplicity of Technology',
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
        'r_monitoring': 'Monitoring',
        's_public_acceptance': 'Public Acceptance',
        't_transfer': 'Transfer',
        'i_policy': 'Policy Environment',
        'e_fiscal_latitude': 'Fiscal Latitude',
        'e_feedin_tariffs': 'Feed-in Tariffs',
        'e_costs' : 'Costs',
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
        's_population_density': 'Population Density',
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
        'e_pilot_project': 'Pilot Projects',
        'e_ghg_emission_reduction_crediting_and_offsetting_mechanism': 'Credits and Offsets',
        'e_costs_state':'Cost to State',
        'e_energy_and_other_taxes': 'Energy and Other Taxes',
        'i_policy_narrative': 'Policy Narrative',
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

def create_heatmap(co_occurrence_data: pd.DataFrame, clusters: List[str], color_palette=None,  title: str = None, threshold: int = 1) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a custom heatmap from co-occurrence data.
    """
    # Filter co_occurrence_data to include only the specified clusters
    co_occurrence_data = co_occurrence_data[clusters]

    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])

    legend_ax = fig.add_subplot(gs[0])
    heatmap_ax = fig.add_subplot(gs[1])

    n_clusters = len(clusters)

    # Use specified color palette or create a new one
    if color_palette is None:
        color_palette = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))

    base_colors = color_palette[:n_clusters]

    cluster_cmaps = [LinearSegmentedColormap.from_list("", ["white", color]) for color in base_colors]

    max_co_occurrence = co_occurrence_data.max().max()

    enablers = co_occurrence_data.index.get_level_values(0).unique()
    entries = co_occurrence_data.index.get_level_values(1).unique()

    # Calculate the highest co-occurrence for each cluster
    cluster_max_values = co_occurrence_data.max()

    for i, enabler in enumerate(enablers):
        for j, entry in enumerate(entries):
            values = co_occurrence_data.loc[(enabler, entry)].values
            if any(v >= threshold for v in values):
                if sum(v > 0 for v in values) >= 5:
                    heatmap_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor="lightgray", edgecolor="none"))

                for k, value in enumerate(values):
                    if value > 0:
                        # Normalize the value relative to the cluster's highest co-occurrence
                        normalized_value = value / cluster_max_values[clusters[k]]

                        color = cluster_cmaps[k](normalized_value)
                        size = max(100, min(1000, 1000 * (value / max_co_occurrence)))
                        heatmap_ax.scatter(
                            j + (k + 1) / (n_clusters + 1),
                            i + 0.5,
                            s=size,
                            color=color,
                            edgecolor="black",
                            linewidth=0.5,
                        )
    legend_ax.text(0.05, 0.3, f"Only co-occurrences ≥ {threshold} are plotted", fontsize=12, fontweight='bold')
    # Clean and set labels
    cleaned_x_labels = clean_labels(entries)
    cleaned_y_labels = clean_labels(enablers)

    heatmap_ax.set_xticks(np.arange(len(cleaned_x_labels))+0.5)
    heatmap_ax.set_yticks(np.arange(len(cleaned_y_labels))+0.5)
    heatmap_ax.set_xticklabels(cleaned_x_labels, rotation=45, ha="right")
    heatmap_ax.set_yticklabels(cleaned_y_labels)
    heatmap_ax.set_title(title, fontsize=16)
    heatmap_ax.set_xlabel("Entries", fontsize=12)
    heatmap_ax.set_ylabel("Enablers", fontsize=12)

    # --- Legend ---
    legend_ax.axis("off")
    legend_ax.text(0.05, 0.98, "Legend", fontsize=16, fontweight="bold", va='top')

    # Cluster colors
    legend_ax.text(0.05, 0.90, "Cluster Colors:", fontsize=12, va='top', fontweight='bold')
    for i, (cluster, color) in enumerate(zip(clusters, base_colors)):
        y_pos = 0.85 - i * 0.06  # Adjust spacing as needed
        legend_ax.scatter(0.1, y_pos, s=600, c=[color], edgecolor="black")
        legend_ax.text(0.2, y_pos, cluster.replace('_', ' '), fontsize=10, va="center")


    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig, heatmap_ax, legend_ax

def create_and_save_heatmap(co_occurrence_data: pd.DataFrame, clusters: List[str],
                            output_file: str, color_palette=None, title: str = None, threshold = 1) -> None:
    """
    Create, customize, and save the heatmap.
    """
    fig, heatmap_ax, legend_ax = create_heatmap(co_occurrence_data, clusters, color_palette, title=title, threshold=threshold)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved as {output_file}")
