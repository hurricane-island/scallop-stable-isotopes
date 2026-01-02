# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a scallop stable isotope analysis project that examines environmental and biological data from different gear types (Cages, Nets, Wild) collected across multiple months in 2023. The analysis focuses on stable isotope signatures (δ13C, δ15N), gonadosomatic index (GSI), and environmental factors like temperature to understand differences between farmed and wild scallops.

## Environment Management

This project uses **pixi** for environment and dependency management.

### Setup
```bash
# Install dependencies (creates .pixi environment)
pixi install

# Run Python scripts in the pixi environment
pixi run python <script_name>.py
```

### Key Dependencies
- Python 3.13.5
- Data analysis: pandas, numpy, scipy, statsmodels
- Visualization: matplotlib, seaborn, plotly
- Statistical methods: scikit-learn, prince (FAMD analysis)
- File handling: openpyxl

## Code Architecture

### Module Structure

**stats_analysis.py** - Core statistical analysis module
- Contains reusable statistical functions and the `Dimension` enum for standardized column names
- Key functions:
  - `analysis_of_variance()` - ANOVA using OLS models
  - `levenes_test_month()` / `levenes_test_gear()` - Homogeneity of variance tests
  - `quantize_categorical_column()` - Convert categorical strings to integers for analysis
  - `return_column_to_categorical()` - Reverse quantization for plotting
- Performs PCA and FAMD (Factor Analysis of Mixed Data) on stable isotope data
- Can be imported as a module or run directly (`python stats_analysis.py`)

**boxplots_tables.py** - GSI visualization
- Creates boxplots comparing GSI across gear types and months
- Pre-processes and groups data by gear type and collection month

**gsi_boxplots.py** - GSI spawning analysis
- Analyzes GSI data to determine spawning timing across gear types
- Creates comparative boxplots for nets, cages, and wild scallops

**rawdata_scatterplots.py** - Raw isotope data visualization
- Imports functions from `stats_analysis.py`
- Creates pairplots and scatterplots of isotope data colored by gear type
- Generates monthly time-series plots

**temp.py** - Environmental data analysis
- Processes temperature and light data from environmental loggers
- Calculates monthly temperature averages and degree-hours above threshold
- Creates time-series visualizations

### Data Flow

1. Raw CSV data loaded from `data/` directory
2. Contaminated samples filtered out (Date Run: 9/6/23)
3. Data processed through quantization for statistical analysis
4. Results visualized and saved to `figures/` directory

### Key Data Files

Located in `data/`:
- `2023IsotopeDataReport-no-outliers.csv` - Primary stable isotope data
- `2023_StableIsotope_GSI_data.csv` - GSI (gonadosomatic index) data
- `2023_2022_GSI_Environmental_Data_2023_Temperature_and_Light.csv` - Environmental logger data

### Categorical Variable Conventions

**Gear Types:**
- `C` = Cage
- `N` = Net  
- `W` = Wild
- `CF`, `NF`, `WF` = Filter samples from respective gear types

**Tissue Types:**
- `M` = Muscle
- `G` = Gonad

**Sex:**
- `F` = Female
- `M` = Male

**Collection Dates:**
- Stored as month integers: 6=June, 7=July, 8=August, 9=September, 10=October

### Standard Color Scheme

The codebase uses consistent colors across visualizations:
- Black = Cages/Nets
- Blue = Cages
- Red = Wild
- Green = Nets (in some plots)

### Dimension Enum Pattern

Scripts use the `Dimension` enum from `stats_analysis.py` to avoid string typos in column names. When working with stable isotope data columns, import and use these enums:

```python
from stats_analysis import Dimension

# Access columns using enum values
df[Dimension.CARBON_FRACTIONATION.value]  # "d13C"
df[Dimension.NITROGEN_FRACTIONATION.value]  # "d15N"
df[Dimension.MOLAR_RATIO.value]  # "C/N (Molar)"
```

## Common Development Workflows

### Running Analysis Scripts

Scripts are designed to be run directly and output figures to `figures/`:

```bash
pixi run python stats_analysis.py
pixi run python gsi_boxplots.py
pixi run python rawdata_scatterplots.py
pixi run python boxplots_tables.py
pixi run python temp.py
```

### Data Quality Controls

1. **Contaminated samples**: Date Run = "9/6/23" are automatically filtered out
2. **Missing data**: Scripts use `.dropna()` on relevant columns before analysis
3. **Gear type filtering**: Only scallop samples (with Gear Type values) are included in most analyses

### Adding New Visualizations

1. Import necessary functions from `stats_analysis.py`
2. Use the `Dimension` enum for column access
3. Apply `quantize_categorical_column()` before statistical analysis
4. Use `return_column_to_categorical()` before plotting for readable labels
5. Save figures to the `figures/` directory using the established `Path` pattern

### Statistical Analysis Pattern

The codebase follows this pattern for statistical analysis:
1. Load and filter data (remove bad runs, handle NA values)
2. Separate by tissue type (muscle vs gonad)
3. Test ANOVA assumptions (normality, homogeneity of variance)
4. If assumptions fail, use dimensionality reduction (PCA/FAMD)
5. Visualize with consistent color schemes

## Output Management

- All figures are saved to `figures/` (gitignored)
- Figure paths are constructed using `Path(__file__).parent / "figures"`
- Figure filenames use descriptive names with underscores (e.g., `GSI_gear_boxplot.png`)
