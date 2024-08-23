import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.colors as mcolors

# Read the CSV file
file_path = r"C:\Users\vigneshr\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Mitigation_EntryPoints_CodeRepo\data\raw\Timeline Policies.csv"
df = pd.read_csv(file_path)

# Sort the dataframe by year
df = df.sort_values('Year')

# Create the plot
fig, ax = plt.subplots(figsize=(12, 24))  # Increased figure size

# Plot the main timeline
ax.axvline(x=0, color='black', linewidth=1)

# Define color scheme
color_map = {
    'Coal': '#000000',  # Black
    'Transport': '#B8860B',  # Dark yellow
    'Renewables': '#006400',  # Dark green
}

# Function to determine color based on technology
def get_color(tech):
    if 'Coal' in tech:
        return color_map['Coal']
    elif any(t in tech for t in ['EV', 'Vehicle', 'Transport']):
        return color_map['Transport']
    else:
        return color_map['Renewables']

# Function to create a curved path
def curved_path(start, end, height):
    verts = [
        (start[0], start[1]),
        (start[0] + height, start[1]),
        (end[0] - height, end[1]),
        (end[0], end[1]),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)

# Group policies by year
grouped = df.groupby('Year')

# Plot points and labels for each policy
y_offset = 0
max_x = 2  # Starting x-position for labels

for year, group in grouped:
    # Plot point on the timeline
    ax.scatter(0, y_offset, color='black', s=20, zorder=3)
    
    # Add year label
    ax.text(-0.5, y_offset, str(int(year)), ha='right', va='center', fontsize=10, fontweight='bold')
    
    # Calculate the maximum width needed for this year's labels
    max_width = max(len(policy['Policy Name']) for _, policy in group.iterrows())
    label_width = max_width * 0.12  # Increased factor for more spacing
    
    # Plot policies for this year
    for i, (_, policy) in enumerate(group.iterrows()):
        # Calculate position for policy label
        label_y = y_offset - i * 2.5  # Increased vertical spacing
        
        # Determine color based on technology
        color = get_color(policy['Technology'])
        
        # Draw curved connector
        path = curved_path((0, y_offset), (max_x, label_y), 1.5)  # Increased curve height
        patch = PathPatch(path, facecolor='none', edgecolor=color, linewidth=0.5)
        ax.add_patch(patch)
        
        # Add region label in grey above the policy name
        ax.text(max_x + 0.1, label_y + 0.7, policy['Region'], ha='left', va='bottom', fontsize=8, color='grey', alpha=0.7)
        
        # Add policy label
        ax.text(max_x + 0.1, label_y, policy['Policy Name'], ha='left', va='center', fontsize=9, color=color)

    # Update y_offset and max_x for the next year
    y_offset -= max(4, len(group) * 2.5 + 1)  # Increased spacing between years
    max_x = label_width/4  # Increased spacing between labels

# Remove axes
ax.axis('off')

# Set title
ax.set_title('Policy Timeline', fontsize=24, pad=20)

# Set y-axis limits with some padding
ax.set_ylim(y_offset - 3, 3)
ax.set_xlim(-1, max_x + 3)

# Add legend
legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=tech)
                   for tech, color in color_map.items()]
ax.legend(handles=legend_elements, loc='upper right', title='Technologies', fontsize=12)

# Adjust layout and display
plt.tight_layout()
plt.show()