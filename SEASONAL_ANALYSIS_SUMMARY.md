# Seasonal Analysis Additions to FWI Analysis Notebooks

## Overview
This document summarizes the seasonal analysis components that have been added to the three FWI analysis notebooks:

1. `fwi_analysis_raw_values.ipynb` - FWI value analysis
2. `fwi_analysis_high_fire_danger.ipynb` - Frequency of high fire danger days
3. `fwi_analysis_percentile.ipynb` - Frequency of 95th percentile exceedances

## Seasons Analyzed
The seasonal analysis includes four standard meteorological seasons:
- **JJA**: June-July-August (Northern Hemisphere summer)
- **SON**: September-October-November (Northern Hemisphere autumn)
- **DJF**: December-January-February (Northern Hemisphere winter)
- **MAM**: March-April-May (Northern Hemisphere spring)

## Analysis Periods
- **Historical**: 1961-1990 (baseline period)
- **Future**: 2041-2050 (projection period)
- **Comparison**: Future scenarios relative to historical baseline

## Scenarios Included
All seven scenarios are analyzed for each season:
1. **SSP3-7.0** (ssp370) - High emissions scenario
2. **Global Aerosol Reduction** (global) - Global aerosol emission reduction
3. **SSP1-2.6** (ssp126) - Low emissions scenario
4. **East Asia Aerosol Reduction** (eas) - Regional aerosol reduction
5. **South Asia Aerosol Reduction** (sas) - Regional aerosol reduction
6. **Africa & Middle East Aerosol Reduction** (afr) - Regional aerosol reduction
7. **North America & Europe Aerosol Reduction** (nae) - Regional aerosol reduction

## Models Included
All three climate models are used:
- **NorESM2-LM**
- **SPEAR**
- **MRI-ESM2-0**

## What Was Added

### 1. Seasonal Calculation Cells
Each notebook now includes a new cell that:
- Defines the four seasons (JJA, SON, DJF, MAM)
- Creates functions to calculate seasonal statistics
- Processes historical data for each season
- Processes future scenario data for each season
- Combines results across all three models

### 2. Seasonal Plotting Cells (Following Annual Anomaly Structure)
Each notebook now includes a comprehensive seasonal plotting cell that follows the exact structure of the annual anomaly analysis:

#### **A. Seasonal Anomaly Calculations**
- Calculates seasonal anomalies (future minus historical) for each scenario and season
- Uses the same anomaly calculation approach as annual analysis

#### **B. Three-Tier Masking System**
Following the annual anomaly analysis structure, the seasonal analysis creates three types of masked DataArrays:

1. **Global Average DataArrays** (`seasonal_anomalies_masked_textbox`)
   - Masked for non-barren land only
   - Used for calculating latitudinally weighted global averages
   - No model agreement threshold applied
   - Used for textbox values on maps

2. **Filled Contour DataArrays** (`seasonal_anomalies_masked_filled`)
   - Masked for non-barren land AND minimum model agreement threshold (0.66)
   - Used for filled contour plots on maps
   - Shows areas with sufficient model agreement for color filling

3. **Hatching DataArrays** (`seasonal_anomalies_masked_hatching`)
   - Masked for non-barren land AND maximum model agreement threshold (0.67)
   - Used for hatching patterns on maps
   - Shows areas with high model agreement for statistical significance

#### **C. Global Average Calculations**
- Uses `weighted_horizontal_avg()` function for latitudinally weighted global averages
- Calculates global averages for each season and scenario
- Provides textbox values showing global mean changes

#### **D. Comprehensive Plotting**
- **Historical Maps**: Individual seasonal maps with global averages in textboxes
- **Anomaly Grids**: 2x2 grid plots for each scenario showing all four seasons
- **Proper Masking**: Uses appropriate masked data for filled contours and hatching
- **Global Averages**: Displays global average changes in textboxes for each season

## Specific Analysis Types

### Raw FWI Values (`fwi_analysis_raw_values.ipynb`)
- **Historical**: Seasonal mean FWI values
- **Future**: Seasonal mean FWI values
- **Anomalies**: Future minus historical seasonal means
- **Units**: FWI index values
- **Colormap**: Viridis for historical, RdBu_r for anomalies
- **Global Average Format**: 2 decimal places (e.g., "1.23")

### High Fire Danger Frequency (`fwi_analysis_high_fire_danger.ipynb`)
- **Historical**: Seasonal frequency of high fire danger days
- **Future**: Seasonal frequency of high fire danger days
- **Anomalies**: Change in seasonal frequency
- **Units**: Days per season
- **Colormap**: Viridis for historical, RdBu_r for anomalies
- **Global Average Format**: 1 decimal place (e.g., "5.2")

### 95th Percentile Exceedances (`fwi_analysis_percentile.ipynb`)
- **Historical**: Seasonal frequency of 95th percentile exceedances
- **Future**: Seasonal frequency of 95th percentile exceedances
- **Anomalies**: Change in seasonal frequency
- **Units**: Days per season
- **Colormap**: Viridis for historical, RdBu_r for anomalies
- **Global Average Format**: 1 decimal place (e.g., "3.1")

## Output Figures

### Individual Seasonal Maps
For each season (JJA, SON, DJF, MAM):
- Historical baseline maps showing seasonal conditions
- Clear titles indicating the season and analysis type
- Appropriate color scales and units
- Global average values displayed in textboxes

### Seasonal Anomaly Grids
For each scenario:
- 2x2 grid showing all four seasons
- Anomaly maps (future minus historical) with proper masking
- Filled contours using minimum model agreement threshold
- Hatching patterns using maximum model agreement threshold
- Global average changes displayed in textboxes for each season
- Consistent color scales across all seasons
- Clear scenario titles and season labels

## Technical Implementation

### Data Processing
- Uses xarray's time-based filtering with `data.time.dt.month.isin(season_months)`
- Maintains the same masking and processing pipeline as annual analysis
- Computes ensemble means across all three models
- Applies appropriate masks for land, ocean, Antarctica, and Arctic regions

### Masking System
- **Non-barren land masking**: Applied to all data arrays
- **Model agreement thresholds**: 
  - Minimum threshold (0.66) for filled contours
  - Maximum threshold (0.67) for hatching
- **Consistent with annual analysis**: Uses same threshold values and masking approach

### Global Average Calculations
- Uses `weighted_horizontal_avg()` function for proper latitudinal weighting
- Calculates averages over non-barren land only
- Provides consistent formatting for textbox display

### Plotting
- Uses existing `create_global_map()` and `create_global_map_grid()` functions
- Maintains consistent styling with existing annual analysis
- Includes proper titles, colorbars, and annotations
- Uses appropriate colormaps for different data types
- Incorporates hatching patterns for statistical significance

## Usage Instructions

1. **Run the notebooks** in the same order as before
2. **Execute all cells** including the new seasonal analysis cells
3. **Review the seasonal maps** to understand seasonal patterns
4. **Compare seasonal anomalies** across different scenarios
5. **Note seasonal differences** in fire weather impacts
6. **Interpret the masking**: 
   - Filled areas show regions with sufficient model agreement
   - Hatched areas show regions with high statistical significance
   - Textbox values show global average changes

## Expected Runtime
The seasonal analysis will add computational time due to:
- Processing four seasons instead of annual means
- Additional plotting for seasonal maps
- Grid plots for each scenario
- Multiple masking operations for each season/scenario combination

However, the analysis uses the same efficient xarray operations as the annual analysis.

## Benefits of Seasonal Analysis

1. **Seasonal Patterns**: Reveals how fire weather changes vary by season
2. **Regional Insights**: Shows which regions are most affected in different seasons
3. **Policy Relevance**: Helps identify seasonal windows of vulnerability
4. **Model Comparison**: Allows comparison of seasonal responses across models
5. **Scenario Assessment**: Evaluates how different scenarios affect seasonal patterns
6. **Statistical Rigor**: Uses proper masking and agreement thresholds for robust results
7. **Global Context**: Provides global average changes for each season

## File Modifications
The following files were modified:
- `fwi_analysis_raw_values.ipynb` - Added seasonal FWI value analysis with proper masking
- `fwi_analysis_high_fire_danger.ipynb` - Added seasonal high fire danger frequency analysis with proper masking
- `fwi_analysis_percentile.ipynb` - Added seasonal 95th percentile exceedance analysis with proper masking
- `add_seasonal_analysis.py` - Initial script used to make the modifications
- `update_seasonal_analysis.py` - Updated script to follow annual anomaly structure
- `SEASONAL_ANALYSIS_SUMMARY.md` - This summary document

## Key Features of the Updated Structure

### **Consistency with Annual Analysis**
- Follows exact same masking approach as annual anomaly analysis
- Uses same threshold values (0.66 for filled contours, 0.67 for hatching)
- Implements same global average calculation method
- Maintains consistent variable naming conventions

### **Robust Statistical Approach**
- Proper masking for non-barren land
- Model agreement thresholds for statistical significance
- Latitudinally weighted global averages
- Clear distinction between different types of masked data

### **Comprehensive Visualization**
- Historical baseline maps with global averages
- Anomaly maps with proper masking and hatching
- Grid plots showing all seasons for each scenario
- Textbox values showing global average changes

## Next Steps
1. Run the modified notebooks to generate seasonal analysis results
2. Review and interpret the seasonal patterns
3. Compare seasonal results with annual analysis
4. Consider additional seasonal metrics if needed
5. Document key findings in your research
6. Use the masked data appropriately for statistical interpretation
