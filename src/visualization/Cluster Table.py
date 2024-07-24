import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

# Create the DataFrame
data = {
    'Cluster Name': ['Centralized Industrial Policy', 'Distributed Industrial Policy', 'Market Based', 'Adaptive Pragmatists', 'Financial Cross-Cutters', 'Regional Autonomy'],
    'Description': [
        'A very centralized and targeted approach with RE, usually seen with rapid progress at the expense of high industry concentration, higher deployment than generation, heavy state dependence and uneven investment in RE technologies and Grid',
        'Developed industrial economies with history of RE leadership and environmental concerns. Balancing industrial capacity with increasing RE share of electricity mix. Balanced policies with stimulus and investment incentives with emphasis on distributed positive spillover rather than targeted interventions',
        'Market First Economies seeking to avoid market distoritions. Focus on consumer oriented RE policy, with considerable differences within sub-national level. Financial instruments, through credit markets and tax credits designed to promote investment and market collaboration.',
        'Large lower middle income states with resource abundance and developing prospects. Institutional quality and market development is mixed with higher costs and investment risk arising from non-technological factors. Fiscal capacity is limited and cost-reduction is a key goal, enabled by auctions.',
        'Financial interventions focusing on structuring capital flows, involving transnational arrangements and where lowering cost of capital was the key entry.',
        'Sub-national regions dealing with local opposition and resistance to renewable infrastrcuture. Entry points include policy entrepreneurship, narrative construction and land use consenting.'
    ],
    'Entry Points': [
        '(C. Kim 2021; Dent 2015; Andrews-Speed 2015; Fang Zhang 2020; Sreenath et al. 2022; Do et al. 2021; Gao and Yuan 2020; Do et al. 2020; S. Zhang and Chen 2022; Mirzania, Balta-Ozkan, and Marais 2020; Ydersbond and Korsnes 2016; Karltorp, Guo, and Sandén 2017; S. Kim et al. 2018)',
        '(Rechsteiner 2020; Flores-Fernández 2020; Guild 2019; F. Zhang 2023; Kruger and Eberhard 2018; Hafeznia et al. 2017; Gaffney, Deane, and Gallachóir 2017; Schmieder et al. 2023; Strunz, Gawel, and Lehmann 2016; S. Wurster and Hagemann 2018; Stefan Wurster and Hagemann 2019)',
        '(Andrews-Speed 2015; Cowell et al. 2017; De Laurentis and Pearson; F. Zhang 2023; Hsu 2018; Ameli, Pisu, and Kammen 2017; Garcia Padron 2016; Shrimali, Lynes, and Indvik 2015; Zhou and Solomon 2020; OShaughnessy 2022)',
        '(Guliyev 2023; Costa et al. 2022; Bradshaw and de Martino Jannuzzi 2019; Shidore and Busby; Kruger and Eberhard 2018; Amankwah-Amoah and Sarpong 2016; Chen et al. 2022; Matsuo and Schmidt 2019; Eberhard and Kåberger 2016; Thapar, Sharma, and Verma 2018)',
        '(Crago and Chernyakhovskiy 2017; F. Zhang 2023; J. Kim and Park 2018; Isah et al. 2023; van den Bold 2022; Fang Zhang 2020; Geddes, Schmidt, and Steffen 2018)',
        '(OHanlon and Cummins 2020; Bradshaw and de Martino Jannuzzi 2019; Giest 2015)'
    ]
}

df = pd.DataFrame(data)

# Function to wrap text
def wrap_text(text, width=40):
    return '\n'.join(wrap(text, width))

# Apply text wrapping to 'Description' and 'Entry Points' columns
df['Description'] = df['Description'].apply(wrap_text)
df['Entry Points'] = df['Entry Points'].apply(wrap_text)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 12))

# Hide axes
ax.axis('off')

# Create the table
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')

# Set font size
table.auto_set_font_size(False)
table.set_fontsize(8)

# Adjust cell heights and widths
table.auto_set_column_width([0, 1, 2])
for i in range(len(df)):
    table[(i+1, 0)].set_height(0.15)
    table[(i+1, 1)].set_height(0.15)
    table[(i+1, 2)].set_height(0.15)

# Color the header row
for i in range(3):
    table[(0, i)].set_facecolor("#FFA500")
    table[(0, i)].set_text_props(color='black')

# Add alternating row colors
for i in range(1, len(df) + 1):
    if i % 2 == 0:
        for j in range(3):
            table[(i, j)].set_facecolor("#FFFFCC")

# Adjust layout and save
plt.tight_layout()
plt.savefig('renewable_energy_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print("Table has been created and saved as 'renewable_energy_clusters.png'")