import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors

def load_timeline_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        return df
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        raise

def curved_path(start, end, height):
    """Create a curved connector path between two points"""
    verts = [
        (start[0], start[1]),
        (start[0] + height, start[1]),
        (end[0] - height, end[1]),
        (end[0], end[1]),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)

def plot_renewables_timeline(df, color_map):
    """Create a timeline visualization for renewables"""
    # Filter data for renewables
    tech_df = df[df['Technology'] == 'Coal'].sort_values('Year')
    
    if tech_df.empty:
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the main timeline
    ax.axvline(x=0, color='black', linewidth=1)
    
    # Group policies by year
    grouped = tech_df.groupby('Year')
    
    # Plot points and labels
    y_offset = 0
    max_x = 2
    
    for year, group in grouped:
        # Plot point on timeline
        ax.scatter(0, y_offset, color='black', s=20, zorder=3)
        
        # Add year label
        ax.text(-0.5, y_offset, str(int(year)), ha='right', va='center', 
                fontsize=10, fontweight='bold')
        
        # Calculate maximum width for labels
        max_width = max(len(policy['Policy Name']) for _, policy in group.iterrows())
        label_width = max_width * 0.12
        
        # Plot policies for this year
        for i, (_, policy) in enumerate(group.iterrows()):
            label_y = y_offset - i * 1.5
            
            # Get color based on cluster, with handling for NaN values
            cluster = policy['Cluster']
            if pd.isna(cluster):
                color = '#808080'  # Gray for NaN values
            else:
                # Remove any trailing whitespace from cluster names
                cluster = cluster.strip()
                color = color_map.get(cluster, '#808080')
            
            # Draw curved connector
            path = curved_path((0, y_offset), (max_x, label_y), 1.5)
            patch = PathPatch(path, facecolor='none', edgecolor=color, linewidth=0.5)
            ax.add_patch(patch)
            
            # Add region and policy labels
            ax.text(max_x + 0.1, label_y + 0.4, policy['Region'], 
                   ha='left', va='bottom', fontsize=8, color='grey', alpha=0.7)
            ax.text(max_x + 0.1, label_y, policy['Policy Name'], 
                   ha='left', va='center', fontsize=9, color=color)
        
        y_offset -= max(2, len(group) * 1.5 + 1)
        max_x = label_width/4
    
    # Customize plot
    ax.axis('off')
    ax.set_title('Coal Policy Timeline', fontsize=16, pad=20)
    ax.set_ylim(y_offset - 3, 3)
    ax.set_xlim(-1, max_x + 3)
    
    # Add legend for clusters (excluding NaN)
    unique_clusters = [c for c in tech_df['Cluster'].unique() if pd.notna(c)]
    legend_elements = [plt.Line2D([0], [0], color=color_map[c.strip()], lw=4, label=c.strip())
                      for c in unique_clusters]
    if legend_elements:  # Only add legend if there are valid clusters
        ax.legend(handles=legend_elements, 
                 loc='center right',
                 bbox_to_anchor=(1.25, 0.5),
                 title='Clusters', 
                 fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    file_path = r"V:\Paper_rahel\Timeline Policies.csv"
    df = load_timeline_data(file_path)
    
    # Print data info for verification
    print("DataFrame columns:", df.columns.tolist())
    print("DataFrame shape:", df.shape)
    
    # Get unique clusters for renewables only
    renewables_clusters = df[df['Technology'] == 'Renewables']['Cluster'].unique()
    print("\nClusters found in Renewables:", [c for c in renewables_clusters if pd.notna(c)])
    
    # Define color scheme for all clusters found in the data
    cluster_colors = {
        'Strategists': '#1f77b4',           # Blue
        'Adaptive Pragmatists': '#2ca02c',  # Green
        'Planners': '#ff7f0e',              # Orange
        'Instrumentalists': '#9467bd',       # Purple
        'Regional Autonomy': '#e377c2',      # Pink
        'Coal Dependent & Climate Advocate': '#8c564b',  # Brown
        'Liberalized & Coal Independent': '#d62728',     # Red
        'Green Innovators': '#17becf',                   # Cyan
        'Heavily State-Subsidized Transition': '#bcbd22', # Yellow-green
        'Starters and Niche-Testers': '#7f7f7f'         # Gray
    }
    
    # Create plot for renewables
    print("\nCreating timeline for Renewables...")
    fig = plot_renewables_timeline(df, cluster_colors)
    plt.show()
    
    print("\nPlotting complete.")

if __name__ == "__main__":
    main() 
