import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from typing import List, Tuple


def clean_labels(labels):
    """Clean up labels for better readability."""
    cleanup_dict = {
        'e_national_development_bank': 'National Development Bank',
        't_network_dependency': 'Network Dependency',
        'i_political_stability': 'Political Stability',
        'e_cost_producer': 'Producer Costs',
        'e_cost_consumer': 'Consumer Costs',
        'e_electricity_mix': 'Electricity Mix',
        't_skill_and_accessibility': 'Skill Accessibility',
        'g_resources_available': 'Resource Availability',
        'e_economic_growth_objective': 'Economic Growth Objective',
        'e_market_structure': 'Market Structure',
        's_path_dependency': 'Path Dependency',
        'e_cost_of_capital': 'Capital Costs',
        'i_policy_environment': 'Policy Environment',
        't_risk': 'Risk',
        'i_political_system': 'Political System',
        's_interest_group_support': 'Interest Group Support',
        'i_institutional_capacity': 'Institutional Capacity',
        's_public_acceptance': 'Public Acceptance',
        't_transfer': 'Transfer',
        'e_fiscal_latitude': 'Fiscal Latitude',
        'e_feedin_tariffs': 'Feed-in Tariffs',
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
        'p_strategic_planning': 'Strategic Planning',
        'e_state_loans': 'State Loans',
        'r_auditing': 'Auditing',
        'r_procurement_rules_general': 'Renewable Portfolio Standards',  
        'e_green_certificates': 'Green Certificates',
        'e_infrastructure_investments': 'Infrastructure Investments',
        'com_narrative': 'Communication Narrative',
        'i_advice_and_aid_in_implementation': 'Implementation Support'
    }
    return [cleanup_dict.get(label, label) for label in labels]

def create_heatmap(co_occurrence_data: pd.DataFrame, clusters: List[str], color_palette=None) -> Tuple[plt.Figure, plt.Axes]:
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

    # Use specified color palette or default
    if color_palette is None:
        color_palette = plt.cm.Set1(np.linspace(0, 1, max(9, n_clusters)))
    
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

            if sum(v > 0 for v in values) >= 2:
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

    # Clean and set labels
    cleaned_x_labels = clean_labels(entries)
    cleaned_y_labels = clean_labels(enablers)

    heatmap_ax.set_xticks(np.arange(len(cleaned_x_labels))+0.5)
    heatmap_ax.set_yticks(np.arange(len(cleaned_y_labels))+0.5)
    heatmap_ax.set_xticklabels(cleaned_x_labels, rotation=45, ha="right")
    heatmap_ax.set_yticklabels(cleaned_y_labels)
    heatmap_ax.set_title("Co-occurrence of Enablers and Entries for Unlocks", fontsize=16)
    heatmap_ax.set_xlabel("Entries", fontsize=12)
    heatmap_ax.set_ylabel("Enablers", fontsize=12)
 # --- Legend --- 
    legend_ax.axis("off")
    legend_ax.text(0.05, 0.95, "Legend", fontsize=16, fontweight="bold")
    legend_ax.text(
        0.05, 0.85, "Circle size increases with co-occurrence count", fontsize=10
    )
    arrow = FancyArrowPatch(
        (0.1, 0.75), (0.9, 0.75), arrowstyle="->", mutation_scale=20
    )
    legend_ax.add_patch(arrow)
    legend_ax.text(0.1, 0.7, "Low", ha="left", va="top", fontsize=12)
    legend_ax.text(0.9, 0.7, "High", ha="right", va="top", fontsize=12)

    sizes = [100, 400, 1000]
    positions = [0.2, 0.5, 0.8]
    for size, pos in zip(sizes, positions):
        legend_ax.scatter(pos, 0.75, s=size, c="gray", edgecolor="black")

    legend_ax.text(0.05, 0.5, "Color: Cluster", fontsize=14)
   
    for i, (cluster, color) in enumerate(zip(clusters, base_colors)):
        legend_ax.scatter(0.2, 0.4 - i * 0.1, s=900, c=[color], edgecolor="black")
        legend_ax.text(0.3, 0.4 - i * 0.1, cluster, fontsize=12, va="center")

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.close('all')
    return fig, heatmap_ax, legend_ax

def create_and_save_heatmap(co_occurrence_data: pd.DataFrame, clusters: List[str], 
                            title: str, color_palette=None) -> None:
    """
    Create, customize, and save the heatmap.
    """
    fig, heatmap_ax, legend_ax = create_heatmap(co_occurrence_data, clusters, color_palette)
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved as {title}")