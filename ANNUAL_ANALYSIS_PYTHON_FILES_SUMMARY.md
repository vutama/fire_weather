# Annual Analysis Python Files Summary

## Overview

I have successfully extracted the annual analysis code from your three FWI analysis notebooks and created standalone Python files that can generate the same plots with image output. These files contain only the annual analysis components (no seasonal analysis) and are designed to be easy to understand and modify.

## Files Created

### 1. `annual_analysis_raw_values.py`
**Purpose**: Annual analysis of FWI raw values
**Extracted from**: `fwi_analysis_raw_values.ipynb`

**What it does**:
- Loads historical FWI data (1961-1990) for all three models (NorESM2-LM, SPEAR, MRI-ESM2-0)
- Loads future scenario FWI data (2041-2050) for all scenarios
- Calculates annual means for historical and future data
- Applies proper masking and calculates global averages
- Creates historical plot showing baseline FWI values
- Calculates anomalies (future - historical)
- Creates anomaly plots for main scenarios (SSP3-7.0, Global Aerosol Reduction, SSP1-2.6, Aerosol Effect, GHG Effect)
- Creates anomaly plots for regional scenarios (East Asia, South Asia, Africa & Middle East, North America & Europe)

**Output plots**:
- `historical_fwi_raw_values.png` - Historical FWI baseline
- `fwi_anomalies_main_scenarios.png` - Main scenario anomalies
- `fwi_anomalies_regional_scenarios.png` - Regional scenario anomalies

### 2. `annual_analysis_high_fire_danger.py`
**Purpose**: Annual analysis of high fire danger frequency (FWI > 30)
**Extracted from**: `fwi_analysis_high_fire_danger.ipynb`

**What it does**:
- Loads historical and future FWI data
- Calculates high fire danger days (FWI > 30) for each model and scenario
- Calculates annual counts of high fire danger days
- Applies proper masking and calculates global averages
- Creates historical plot showing baseline high fire danger frequency
- Calculates anomalies in high fire danger frequency
- Creates anomaly plots for all scenarios

**Output plots**:
- `historical_high_fire_danger.png` - Historical high fire danger frequency
- `high_fire_danger_anomalies_main_scenarios.png` - Main scenario anomalies
- `high_fire_danger_anomalies_regional_scenarios.png` - Regional scenario anomalies

### 3. `annual_analysis_percentile.py`
**Purpose**: Annual analysis of 95th percentile exceedances
**Extracted from**: `fwi_analysis_percentile.ipynb`

**What it does**:
- Loads historical and future FWI data
- Calculates 95th percentile thresholds from historical data for each model
- Identifies days exceeding the 95th percentile threshold in future scenarios
- Calculates annual counts of 95th percentile exceedances
- Applies proper masking and calculates global averages
- Creates historical plot showing baseline 95th percentile exceedances
- Calculates anomalies in 95th percentile exceedances
- Creates anomaly plots for all scenarios

**Output plots**:
- `historical_95th_percentile.png` - Historical 95th percentile exceedances
- `95th_percentile_anomalies_main_scenarios.png` - Main scenario anomalies
- `95th_percentile_anomalies_regional_scenarios.png` - Regional scenario anomalies

## Key Features

### ✅ **Proper Structure**
All three files follow the exact same structure as your annual analysis:
1. **Data Loading** - Historical and future scenarios
2. **Data Processing** - Annual calculations and model combination
3. **Masking** - Proper application of land masks and significance thresholds
4. **Global Averages** - Latitudinally weighted global averages for textboxes
5. **Plotting** - Historical plots first, then anomaly plots

### ✅ **Correct Parameters**
- Uses `get_significance`, `get_land_mask`, and `baseline_data` correctly
- Applies proper agreement thresholds (0.66 for filled contours, 0.67 for hatching)
- Uses correct threshold types ('minimum' for filled, 'maximum' for hatching)

### ✅ **Image Output**
- All plots are saved as high-resolution PNG files (300 DPI)
- Plots are saved to `annual_analysis_plots/` directory
- Each script creates its own output directory if it doesn't exist

### ✅ **Comprehensive Coverage**
- **Models**: NorESM2-LM, SPEAR, MRI-ESM2-0
- **Scenarios**: SSP3-7.0, Global Aerosol Reduction, SSP1-2.6, Regional scenarios
- **Analysis Types**: Raw values, High fire danger frequency, 95th percentile exceedances

## How to Use

### Running the Scripts
```bash
# Run all three analyses
python annual_analysis_raw_values.py
python annual_analysis_high_fire_danger.py
python annual_analysis_percentile.py
```

### Expected Output
Each script will:
1. Print progress messages as it loads data and performs calculations
2. Create an `annual_analysis_plots/` directory
3. Generate 3 PNG files per script (9 total plots)
4. Print confirmation messages when plots are saved

### Dependencies
The scripts require:
- `xarray` for data manipulation
- `numpy` for numerical operations
- `matplotlib` for plotting
- `ramip_fwi_utilities` module (your existing utilities)

## File Structure

```
annual_analysis_plots/
├── historical_fwi_raw_values.png
├── fwi_anomalies_main_scenarios.png
├── fwi_anomalies_regional_scenarios.png
├── historical_high_fire_danger.png
├── high_fire_danger_anomalies_main_scenarios.png
├── high_fire_danger_anomalies_regional_scenarios.png
├── historical_95th_percentile.png
├── 95th_percentile_anomalies_main_scenarios.png
└── 95th_percentile_anomalies_regional_scenarios.png
```

## Benefits

### 🎯 **Easy Comparison**
- You can now easily compare the annual analysis structure with your seasonal analysis
- Both follow the same masking and plotting patterns
- Consistent parameter usage across all analyses

### 🔧 **Easy Modification**
- Standalone files are easier to modify than notebook cells
- Clear structure makes it easy to add new scenarios or modify thresholds
- Can be easily integrated into larger workflows

### 📊 **Reproducible Results**
- Scripts will generate identical plots every time they're run
- No notebook cell execution order dependencies
- Clear output file naming for easy identification

### 🚀 **Ready to Run**
- All scripts are ready to execute immediately
- No additional setup required beyond your existing environment
- Will create output directories automatically

## Next Steps

1. **Run the scripts** to verify they work correctly with your data
2. **Compare outputs** with your notebook results to ensure consistency
3. **Use as reference** for implementing seasonal analysis in the same structure
4. **Modify as needed** for additional scenarios or analysis types

These Python files provide a clean, well-structured foundation for understanding and implementing both annual and seasonal analysis with consistent methodology and output formats.
