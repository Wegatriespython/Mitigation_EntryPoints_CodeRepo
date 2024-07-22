import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = r'C:\Users\vigne\OneDrive - Wageningen University & Research\Internship\Literature Review\Final Data Processing\Omnibus_Unique_10th.xlsx'
data = pd.read_excel(file_path)

# Count the frequency of each research method
method_frequencies = data['research_method'].value_counts().reset_index()
method_frequencies.columns = ['method', 'frequency']

# Set up the plot
plt.figure(figsize=(12, 8))
barplot = sns.barplot(
    x='frequency', 
    y='method', 
    data=method_frequencies, 
    palette='Blues_r'
)

# Add title and labels
plt.title('Frequency of Methodologies Used', fontsize=16)
plt.xlabel('Number of Articles', fontsize=14)
plt.ylabel('Methodology', fontsize=14)

# Style adjustments to match the provided image
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# Add a color scale on the side
norm = plt.Normalize(method_frequencies['frequency'].min(), method_frequencies['frequency'].max())
sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=barplot, label='Frequency')

# Save the plot as an image file
plot_path = 'methodology_frequencies_with_scale.png'
plt.savefig(plot_path)
