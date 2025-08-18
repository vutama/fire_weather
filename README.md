# Fire Weather Research Project

## Overview

This repository contains a comprehensive fire weather research project focused on analyzing **Fire Weather Index (FWI)** data from climate models as part of the **RAMIP (Regional Aerosol Model Intercomparison Project)** initiative. The project investigates how fire weather conditions change under different climate scenarios and emission pathways.

## Research Objectives

The project aims to:
- Analyze fire weather conditions under different climate scenarios (historical, SSP370, SSP370-126aer)
- Investigate regional variations in fire weather risk across multiple geographic regions
- Assess the impact of aerosol reductions on fire weather patterns
- Compare fire weather projections across multiple climate models
- Understand the relationship between climate change and fire weather extremes

## Climate Models and Scenarios

### Supported Climate Models
- **CESM2** (Community Earth System Model 2)
- **NorESM2-LM** (Norwegian Earth System Model 2)
- **SPEAR** (Seamless System for Prediction and EArth System Research)
- **MRI-ESM2-0** (Meteorological Research Institute Earth System Model)

### Climate Scenarios
- **Historical**: Baseline conditions (reference period)
- **SSP370**: High emissions pathway (business-as-usual)
- **SSP370-126aer**: High emissions with aerosol reductions (aerosol mitigation scenario)

## Geographic Regions

The analysis covers multiple fire-prone regions worldwide:
- **Western North America** (30°N-45°N, 235°E-244.5°E)
- **Mediterranean** (36°N-47°N, 349.5°E-48°E)
- **Australia** (multiple sub-regions including Northern, Southeastern, Interior)
- **South America/Amazon** (-27.64°S-0.48°S, 276°E-325.73°E)
- **Africa & Middle East** (-35°S-35°N, 340°E-60°E)
- **East Asia** (20°N-53°N, 95°E-133°E)
- **South Asia** (5°N-35°N, 65°E-95°E)
- **Europe** (35°N-70°N, 340°E-45°E)
- **Western Steppe** (47°N-55°N, 30°E-80°E)
- **Inner Mongolia Steppe** (33°N-50°N, 97°E-120°E)

## Project Components

### Core Modules

#### `fwdp.py` - Fire Weather Diagnostics Package
Contains the primary functions for computing fire weather variables:
- **FFMC** (Fine Fuel Moisture Code): Surface fuel moisture
- **DMC** (Duff Moisture Code): Sub-surface fuel moisture
- **DC** (Drought Code): Deep fuel moisture
- **ISI** (Initial Spread Index): Fire spread potential
- **BUI** (Buildup Index): Available fuel
- **FWI** (Fire Weather Index): Overall fire danger rating

Key features:
- Optimized with Numba for high-performance calculations
- Handles unit conversions automatically
- Supports parallel processing for large datasets

#### `ramip_fwi_utilities.py` - Data Processing and Analysis Utilities
Comprehensive utilities for:
- **Data Loading**: Reading zarr-formatted climate model data
- **FWI Calculation**: Computing fire weather indices from meteorological variables
- **Statistical Analysis**: Temporal and spatial averaging, exceedance calculations
- **Visualization**: Global mapping and plotting functions
- **Regional Analysis**: Weighted averages and regional statistics

### Analysis Notebooks

#### `FWI_HFD_MultimodelMean_Significance.ipynb`
- Multi-model analysis of fire weather changes
- Statistical significance testing
- Regional comparison across models
- High fire danger day analysis

#### `PDF_Decomposition.ipynb`
- Probability distribution analysis
- Statistical decomposition of fire weather changes
- Attribution analysis for different climate drivers

## Data Requirements

### Input Variables
The FWI calculation requires four meteorological variables:
- **tasmax**: Daily maximum temperature (°C)
- **hurs**: Relative humidity (%)
- **pr**: Precipitation (mm/day)
- **sfcWind**: Surface wind speed (km/hr)

### Data Format
- Climate model data stored in Zarr format
- Time series data with lat/lon coordinates
- Support for ensemble members and multiple models

## Installation and Usage

### Dependencies
```python
numpy
pandas
xarray
cartopy
matplotlib
cftime
numba
```

### Basic Usage

```python
from ramip_fwi_utilities import read_zarr, output_FWI_data
from fwdp import computeFireWeatherIndices

# Read climate model data
tasmax = read_zarr('CESM2', 'historical', 'tasmax')
hurs = read_zarr('CESM2', 'historical', 'hurs')
pr = read_zarr('CESM2', 'historical', 'pr')
sfcWind = read_zarr('CESM2', 'historical', 'sfcWind')

# Calculate FWI
fwi_data = computeFireWeatherIndices(tasmax, pr, hurs, sfcWind)

# Generate FWI outputs
output_FWI_data('CESM2', 'historical', output_type='FWI')
```

## Key Features

### Performance Optimizations
- Numba-accelerated calculations for speed
- Efficient memory management for large datasets
- Parallel processing support

### Analysis Capabilities
- Multi-model ensemble analysis
- Regional and global statistics
- Temporal aggregation (daily, monthly, seasonal, annual)
- Exceedance probability calculations
- Statistical significance testing

### Visualization Tools
- Global mapping functions
- Regional comparison plots
- Statistical distribution analysis
- Customizable plotting options

## Research Applications

This project supports research on:
- Climate change impacts on fire weather
- Regional fire risk assessment
- Aerosol-climate-fire interactions
- Extreme fire weather events
- Climate model intercomparison studies

## Contributing

This is a research project focused on fire weather analysis. For questions or contributions, please contact the project maintainers.

## References

The project builds on:
- Canadian Fire Weather Index System
- RAMIP (Regional Aerosol Model Intercomparison Project)
- CMIP6 climate model outputs
- Regional climate change studies

## License

This project is for research purposes. Please cite appropriately when using the code or results in publications.