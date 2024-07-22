import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np

def create_heatmap(co_occurrence_data, clusters):
    """Creates and customizes the heatmap figure and axes."""
    fig = plt.figure(figsize=(24, 12))  # Create a Figure
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])  # Define a grid

    legend_ax = fig.add_subplot(gs[0])  # Legend axis
    heatmap_ax = fig.add_subplot(gs[1])  # Heatmap axis

    n_clusters = len(clusters)
    base_colors = plt.cm.Set1(np.linspace(0, 1, 9))[:n_clusters]
    cluster_cmaps = [LinearSegmentedColormap.from_list("", ["white", color]) for color in base_colors]

    max_co_occurrence = co_occurrence_data.applymap(
        lambda x: max(x) if isinstance(x, tuple) else 0
    ).values.max()

    for i, enabler in enumerate(co_occurrence_data.index):
        for j, entry in enumerate(co_occurrence_data.columns):
            values = co_occurrence_data.loc[enabler, entry]
            if sum(v > 0 for v in values) >= 2:
                heatmap_ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=True, facecolor="lightgray", edgecolor="none")
                )

            for k, value in enumerate(values):
                if value > 0:
                    normalized_value = value / max_co_occurrence
                    color = cluster_cmaps[k](normalized_value)
                    size = max(100, min(1000, 1000 * normalized_value))
                    heatmap_ax.scatter(
                        j + (k + 1) / (n_clusters + 1),
                        i + 0.5,
                        s=size,
                        color=color,
                        edgecolor="black",
                        linewidth=0.5,
                    )

    heatmap_ax.set_xticks(np.arange(len(co_occurrence_data.columns)) + 0.5)
    heatmap_ax.set_yticks(np.arange(len(co_occurrence_data.index)) + 0.5)
    heatmap_ax.set_xticklabels(co_occurrence_data.columns, rotation=45, ha="right")
    heatmap_ax.set_yticklabels(co_occurrence_data.index)
    heatmap_ax.set_title("Transport: Co-occurrence of Enablers and Entries for Unlocks", fontsize=16)
    heatmap_ax.set_xlabel("Entries", fontsize=12)
    heatmap_ax.set_ylabel("Enablers", fontsize=12)

    # --- Legend --- 
    legend_ax.axis("off")
    legend_ax.text(0.05, 0.95, "Legend", fontsize=14, fontweight="bold")
    legend_ax.text(
        0.05, 0.85, "Circle size increases with co-occurrence count", fontsize=10
    )
    arrow = FancyArrowPatch(
        (0.1, 0.75), (0.9, 0.75), arrowstyle="->", mutation_scale=20
    )
    legend_ax.add_patch(arrow)
    legend_ax.text(0.1, 0.7, "Low", ha="left", va="top", fontsize=8)
    legend_ax.text(0.9, 0.7, "High", ha="right", va="top", fontsize=8)

    sizes = [100, 400, 1000]
    positions = [0.2, 0.5, 0.8]
    for size, pos in zip(sizes, positions):
        legend_ax.scatter(pos, 0.75, s=size, c="gray", edgecolor="black")

    legend_ax.text(0.05, 0.5, "Color: Cluster", fontsize=10)
    base_colors = plt.cm.Set1(np.linspace(0, 1, 9))[:len(clusters)]
    for i, (cluster, color) in enumerate(zip(clusters, base_colors)):
        legend_ax.scatter(0.2, 0.4 - i * 0.1, s=100, c=[color], edgecolor="black")
        legend_ax.text(0.3, 0.4 - i * 0.1, cluster, fontsize=8, va="center")

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)



    plt.tight_layout()
    plt.show()

def create_and_save_heatmap(co_occurrence_data, clusters, output_file):
    """Creates, customizes, and saves the heatmap."""
    create_heatmap(co_occurrence_data, clusters)  
    plt.savefig(output_file, dpi=300, bbox_inches="tight")