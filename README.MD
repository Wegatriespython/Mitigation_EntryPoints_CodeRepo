# Mitigation Entry Points Analysis

Based on data extraction from our literature review, this project performs data_analysis and visualistion. The main pipeline involves identifying key features for enablers and entry points and making co-occurrance visualisationss. 

The pipeline includes data preprocessing, random forest analysis, co-occurrence analysis, and visualization of results through heatmaps.

## Project Structure

```
root/
├── src/
│   ├── data_processing/
│   │   └── general_preprocessing.py
│   ├── analysis/
│   │   ├── random_forest.py
│   │   └── Co_occurrence.py
│   └── visualization/
│       └── heatmap.py
├── Technology_Books/
│   ├── Transport.py
│   ├── Coal.py
│   └── Renewables.py
└── README.md
```

## Pipeline Overview

1. **Data Preprocessing**: The `general_preprocessing.py` module loads and preprocesses the input data, cleaning and vectorizing the 'Enabler' and 'Entry' columns.

2. **Random Forest Analysis**: The `random_forest.py` module performs random forest analysis to identify the most important features (enablers and entries) for each technology sector.

3. **Co-occurrence Analysis**: The `Co_occurrence.py` module calculates co-occurrence matrices and identifies secular enablers across clusters.

4. **Visualization**: The `heatmap.py` module creates and saves heatmaps to visualize the co-occurrence data.

5. **Sector-Specific Analysis**: The `Transport.py`, `Coal.py`, and `Renewables.py` files in the `Technology_Books` folder run the complete analysis pipeline for each respective sector.

## How to Use

1. Ensure you have all the required dependencies installed (pandas, numpy, matplotlib, scikit-learn, etc.).

2. Place your input data files in the appropriate location and update the `INPUT_FILE` path in the respective sector-specific Python files.

3. Run the sector-specific analysis by executing the corresponding Python file:

   ```
   python Technology_Books/Transport.py
   python Technology_Books/Coal.py
   python Technology_Books/Renewables.py
   ```

4. The results, including heatmaps and analysis outputs, will be saved in the `output` directory.

## Customization

- You can adjust the number of top enablers and entries to analyze by modifying the `n_enablers` and `n_entries` parameters in the sector-specific files.
- The `detailed` and `cluster_specific` parameters in the `run_random_forest_analysis` function can be toggled to change the analysis approach.
- Color palettes for heatmaps can be customized in the sector-specific files.

## Output

The pipeline generates the following outputs:

1. Random Forest analysis results (saved as joblib files)
2. Co-occurrence heatmaps (saved as PNG files)
3. Lists of top enablers and entries
4. Secular enablers across clusters

## Notes

- The Renewables analysis (Wind & Solar) is split into two batches with different cluster groups.
- The Coal and Transport analyses process all clusters together.
- Make sure to update file paths and adjust parameters as needed for your specific use case.

For more detailed information on each module and its functions, please refer to the docstrings and comments within the individual Python files.