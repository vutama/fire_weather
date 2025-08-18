#!/usr/bin/env python3
"""
FWI Raw Values Analysis - Annual and Seasonal
=============================================

This script performs comprehensive analysis of Fire Weather Index (FWI) raw values
from climate model simulations, examining both annual and seasonal changes under
different emission scenarios.

Analysis includes:
- Historical baseline (1961-1990)
- Future scenarios (2041-2050): SSP3-7.0, SSP1-2.6, Global aerosol reduction
- Regional aerosol reduction effects: East Asia, North America & Europe, South Asia, Africa & Middle East
- Statistical significance testing and model agreement assessment
- Both annual and seasonal (JJA, SON, DJF, MAM) analyses

Output: Publication-quality maps showing FWI changes and statistical significance.
"""

import xarray as xr # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cartopy.crs as ccrs # type: ignore
import os # type: ignore
import cftime # type: ignore
from pathlib import Path # type: ignore

# Import utility functions
from ramip_fwi_utilities import (
read_zarr, 
apply_masks, 
weighted_horizontal_avg, 
create_global_map, 
create_global_map_grid,
season_mean
)

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================
print("=" * 80)
print("FWI RAW VALUES ANALYSIS - ANNUAL AND SEASONAL")
print("=" * 80)

# Create output directory for plots
output_dir = Path("mm_fwi_plots")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

# Analysis parameters
HISTORICAL_START = cftime.DatetimeNoLeap(1961, 1, 1, 12, 0, 0, 0, has_year_zero=True)
HISTORICAL_END = cftime.DatetimeNoLeap(1990, 12, 31, 12, 0, 0, 0, has_year_zero=True)
FUTURE_START = cftime.DatetimeNoLeap(2041, 1, 1, 12, 0, 0, 0, has_year_zero=True)
FUTURE_END = cftime.DatetimeNoLeap(2050, 12, 31, 12, 0, 0, 0, has_year_zero=True)

# Statistical significance thresholds
AGREEMENT_THRESHOLD_PLOT = 0.66  # For filled contours
AGREEMENT_THRESHOLD_HATCHING = 0.67  # For hatching patterns

# Regional scenario order (consistent throughout analysis)
REGIONAL_SCENARIOS = ['eas', 'nae', 'sas', 'afr']
REGIONAL_NAMES = ['East Asia', 'North America & Europe', 'South Asia', 'Africa & Middle East']

print(f"Analysis periods: Historical ({HISTORICAL_START.year}-{HISTORICAL_END.year}), Future ({FUTURE_START.year}-{FUTURE_END.year})")
print(f"Statistical thresholds: Plot={AGREEMENT_THRESHOLD_PLOT}, Hatching={AGREEMENT_THRESHOLD_HATCHING}")
print()

# =============================================================================
# DATA LOADING
# =============================================================================
print("Loading FWI data from climate model simulations...")
print("-" * 60)

# Historical baseline (1961-1990)
print("Loading historical baseline data...")
noresm2_historical_fwi = read_zarr('NorESM2-LM', 'historical', 'FWI',
                                    start_analysis=HISTORICAL_START,
                                    end_analysis=HISTORICAL_END)
spear_historical_fwi = read_zarr('SPEAR', 'historical', 'FWI',
                                    start_analysis=HISTORICAL_START,
                                    end_analysis=HISTORICAL_END)
mri_historical_fwi = read_zarr('MRI-ESM2-0', 'historical', 'FWI',
                                start_analysis=HISTORICAL_START,
                                end_analysis=HISTORICAL_END)
print("✓ Historical data loaded")

# Future scenarios (2041-2050)
print("\nLoading future scenario data...")

# Main emission scenarios
print("  - Main emission scenarios...")
noresm2_ssp370_fwi = read_zarr('NorESM2-LM', 'ssp370', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
spear_ssp370_fwi = read_zarr('SPEAR', 'ssp370', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
mri_ssp370_fwi = read_zarr('MRI-ESM2-0', 'ssp370', 'FWI',
                            start_analysis=FUTURE_START,
                            end_analysis=FUTURE_END)

noresm2_global_fwi = read_zarr('NorESM2-LM', 'ssp370-126aer', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
spear_global_fwi = read_zarr('SPEAR', 'ssp370-126aer', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
mri_global_fwi = read_zarr('MRI-ESM2-0', 'ssp370-126aer', 'FWI',
                            start_analysis=FUTURE_START,
                            end_analysis=FUTURE_END)

noresm2_ssp126_fwi = read_zarr('NorESM2-LM', 'ssp126', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
spear_ssp126_fwi = read_zarr('SPEAR', 'ssp126', 'FWI',
                                start_analysis=FUTURE_START,
                                end_analysis=FUTURE_END)
mri_ssp126_fwi = read_zarr('MRI-ESM2-0', 'ssp126', 'FWI',
                            start_analysis=FUTURE_START,
                            end_analysis=FUTURE_END)
print("  ✓ Main emission scenarios loaded")

# Regional aerosol reduction scenarios
print("  - Regional aerosol reduction scenarios...")

# Load regional scenarios in consistent order: EAS → NAE → SAS → AFR
for i, (scenario, name) in enumerate(zip(REGIONAL_SCENARIOS, REGIONAL_NAMES)):
    print(f"    {i+1}. {name} ({scenario})")
    
    # Load data for each model
    if scenario == 'eas':
        noresm2_eas_fwi = read_zarr('NorESM2-LM', 'ssp370-eas126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        spear_eas_fwi = read_zarr('SPEAR', 'ssp370-eas126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        mri_eas_fwi = read_zarr('MRI-ESM2-0', 'ssp370-eas126aer', 'FWI',
                                start_analysis=FUTURE_START, end_analysis=FUTURE_END)
    elif scenario == 'nae':
        noresm2_nae_fwi = read_zarr('NorESM2-LM', 'ssp370-nae126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        spear_nae_fwi = read_zarr('SPEAR', 'ssp370-nae126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        mri_nae_fwi = read_zarr('MRI-ESM2-0', 'ssp370-nae126aer', 'FWI',
                                start_analysis=FUTURE_START, end_analysis=FUTURE_END)
    elif scenario == 'sas':
        noresm2_sas_fwi = read_zarr('NorESM2-LM', 'ssp370-sas126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        spear_sas_fwi = read_zarr('SPEAR', 'ssp370-sas126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        mri_sas_fwi = read_zarr('MRI-ESM2-0', 'ssp370-sas126aer', 'FWI',
                                start_analysis=FUTURE_START, end_analysis=FUTURE_END)
    elif scenario == 'afr':
        noresm2_afr_fwi = read_zarr('NorESM2-LM', 'ssp370-afr126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        spear_afr_fwi = read_zarr('SPEAR', 'ssp370-afr126aer', 'FWI',
                                    start_analysis=FUTURE_START, end_analysis=FUTURE_END)
        mri_afr_fwi = read_zarr('MRI-ESM2-0', 'ssp370-afr126aer', 'FWI',
                                start_analysis=FUTURE_START, end_analysis=FUTURE_END)

print("  ✓ Regional aerosol reduction scenarios loaded")
print("✓ All data loading complete")
print()

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
print("Preprocessing data...")
print("-" * 60)

# Regrid all data to NorESM2-LM grid for consistent analysis
print("Regridding all data to NorESM2-LM grid...")
reference_grid = noresm2_global_fwi

# Regrid historical data
print("  - Historical data...")
spear_historical_fwi = spear_historical_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
mri_historical_fwi = mri_historical_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)

# Regrid main emission scenarios
print("  - Main emission scenarios...")
spear_ssp370_fwi = spear_ssp370_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
mri_ssp370_fwi = mri_ssp370_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)

spear_global_fwi = spear_global_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
mri_global_fwi = mri_global_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)

spear_ssp126_fwi = spear_ssp126_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
mri_ssp126_fwi = mri_ssp126_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)

# Regrid regional scenarios in consistent order
print("  - Regional scenarios...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        spear_eas_fwi = spear_eas_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
        mri_eas_fwi = mri_eas_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
    elif scenario == 'nae':
        spear_nae_fwi = spear_nae_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
        mri_nae_fwi = mri_nae_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
    elif scenario == 'sas':
        spear_sas_fwi = spear_sas_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
        mri_sas_fwi = mri_sas_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
    elif scenario == 'afr':
        spear_afr_fwi = spear_afr_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)
        mri_afr_fwi = mri_afr_fwi.interp(lat=reference_grid.lat, lon=reference_grid.lon)

print("✓ Regridding complete")
print()

# =============================================================================
# ANNUAL ANALYSIS
# =============================================================================
print("ANNUAL FWI ANALYSIS")
print("=" * 80)
print("Calculating annual means and anomalies...")
print("-" * 60)

# Step 1: Calculate annual means for all scenarios
print("Step 1: Calculating annual means...")

# Historical baseline
print("  - Historical baseline...")
noresm2_historical_fwi_mean = noresm2_historical_fwi.mean(dim=['time', 'member'], skipna=True)
spear_historical_fwi_mean = spear_historical_fwi.mean(dim=['time', 'member'], skipna=True)
mri_historical_fwi_mean = mri_historical_fwi.mean(dim=['time', 'member'], skipna=True)

# Combine historical models
multi_historical_fwi_mean = xr.concat([
    noresm2_historical_fwi_mean, 
    spear_historical_fwi_mean, 
    mri_historical_fwi_mean
], dim="model", coords='minimal')
historical_fwi_mean = multi_historical_fwi_mean.compute()
print("  ✓ Historical annual means calculated")

# Main emission scenarios
print("  - Main emission scenarios...")
noresm2_ssp370_fwi_mean = noresm2_ssp370_fwi.mean(dim=['time', 'member'], skipna=True)
spear_ssp370_fwi_mean = spear_ssp370_fwi.mean(dim=['time', 'member'], skipna=True)
mri_ssp370_fwi_mean = mri_ssp370_fwi.mean(dim=['time', 'member'], skipna=True)

noresm2_global_fwi_mean = noresm2_global_fwi.mean(dim=['time', 'member'], skipna=True)
spear_global_fwi_mean = spear_global_fwi.mean(dim=['time', 'member'], skipna=True)
mri_global_fwi_mean = mri_global_fwi.mean(dim=['time', 'member'], skipna=True)

noresm2_ssp126_fwi_mean = noresm2_ssp126_fwi.mean(dim=['time', 'member'], skipna=True)
spear_ssp126_fwi_mean = spear_ssp126_fwi.mean(dim=['time', 'member'], skipna=True)
mri_ssp126_fwi_mean = mri_ssp126_fwi.mean(dim=['time', 'member'], skipna=True)

# Regional scenarios (in consistent order)
print("  - Regional scenarios...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        noresm2_eas_fwi_mean = noresm2_eas_fwi.mean(dim=['time', 'member'], skipna=True)
        spear_eas_fwi_mean = spear_eas_fwi.mean(dim=['time', 'member'], skipna=True)
        mri_eas_fwi_mean = mri_eas_fwi.mean(dim=['time', 'member'], skipna=True)
    elif scenario == 'nae':
        noresm2_nae_fwi_mean = noresm2_nae_fwi.mean(dim=['time', 'member'], skipna=True)
        spear_nae_fwi_mean = spear_nae_fwi.mean(dim=['time', 'member'], skipna=True)
        mri_nae_fwi_mean = mri_nae_fwi.mean(dim=['time', 'member'], skipna=True)
    elif scenario == 'sas':
        noresm2_sas_fwi_mean = noresm2_sas_fwi.mean(dim=['time', 'member'], skipna=True)
        spear_sas_fwi_mean = spear_sas_fwi.mean(dim=['time', 'member'], skipna=True)
        mri_sas_fwi_mean = mri_sas_fwi.mean(dim=['time', 'member'], skipna=True)
    elif scenario == 'afr':
        noresm2_afr_fwi_mean = noresm2_afr_fwi.mean(dim=['time', 'member'], skipna=True)
        spear_afr_fwi_mean = spear_afr_fwi.mean(dim=['time', 'member'], skipna=True)
        mri_afr_fwi_mean = mri_afr_fwi.mean(dim=['time', 'member'], skipna=True)

print("  ✓ All annual means calculated")

# Step 2: Combine models for each scenario
print("\nStep 2: Combining models...")

# Main emission scenarios
print("  - Main emission scenarios...")
multi_ssp370_fwi_mean = xr.concat([
    noresm2_ssp370_fwi_mean, spear_ssp370_fwi_mean, mri_ssp370_fwi_mean
], dim="model", coords='minimal')
ssp370_fwi_mean = multi_ssp370_fwi_mean.compute()

multi_global_fwi_mean = xr.concat([
    noresm2_global_fwi_mean, spear_global_fwi_mean, mri_global_fwi_mean
], dim="model", coords='minimal')
global_fwi_mean = multi_global_fwi_mean.compute()

multi_ssp126_fwi_mean = xr.concat([
    noresm2_ssp126_fwi_mean, spear_ssp126_fwi_mean, mri_ssp126_fwi_mean
], dim="model", coords='minimal')
ssp126_fwi_mean = multi_ssp126_fwi_mean.compute()

# Regional scenarios (in consistent order)
print("  - Regional scenarios...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        multi_eas_fwi_mean = xr.concat([
            noresm2_eas_fwi_mean, spear_eas_fwi_mean, mri_eas_fwi_mean
        ], dim="model", coords='minimal')
        eas_fwi_mean = multi_eas_fwi_mean.compute()
    elif scenario == 'nae':
        multi_nae_fwi_mean = xr.concat([
            noresm2_nae_fwi_mean, spear_nae_fwi_mean, mri_nae_fwi_mean
        ], dim="model", coords='minimal')
        nae_fwi_mean = multi_nae_fwi_mean.compute()
    elif scenario == 'sas':
        multi_sas_fwi_mean = xr.concat([
            noresm2_sas_fwi_mean, spear_sas_fwi_mean, mri_sas_fwi_mean
        ], dim="model", coords='minimal')
        sas_fwi_mean = multi_sas_fwi_mean.compute()
    elif scenario == 'afr':
        multi_afr_fwi_mean = xr.concat([
            noresm2_afr_fwi_mean, spear_afr_fwi_mean, mri_afr_fwi_mean
        ], dim="model", coords='minimal')
        afr_fwi_mean = multi_afr_fwi_mean.compute()

print("  ✓ All models combined")

# Step 3: Apply masks and calculate global averages
print("\nStep 3: Applying masks and calculating global averages...")

# Historical baseline
print("  - Historical baseline...")
historical_fwi_mean_masked, _ = apply_masks(historical_fwi_mean, 
                                            get_significance=False,    
                                            get_land_mask=True,
                                            baseline_data=None)

historical_fwi_mean_masked_globalavg = weighted_horizontal_avg(
    historical_fwi_mean_masked.mean('model'), 
    ensemble=False, 
    time=False
)
print(f"  ✓ Historical global average: {historical_fwi_mean_masked_globalavg.values.item():.2f}")

# Step 4: Create historical baseline plot
print("\nStep 4: Creating historical baseline plot...")

fig, ax = create_global_map(
    historical_fwi_mean_masked.mean('model'), 
    projection=ccrs.Robinson(),
    title="Historical (1961-1990)",
    colormap='Reds',
    colorbar_title="Fire Weather Index",
    textbox_text=f"{historical_fwi_mean_masked_globalavg.values.item():.2f}",
    figsize=(10.5, 6),
    vmin=0,
    vmax=40,
    extend='max',
    colorbar_levels=np.arange(0, 40.1, 4),
    contour_levels=None,
    hatching='///',
    regional_boundaries='ar6',
    hatching_style='overlay',
    hatching_data=None,
    show_gridlines=False
)

# Save historical plot
plt.savefig(output_dir / "mm_historical_fwi_annual.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Historical plot saved: {output_dir / 'mm_historical_fwi_annual.png'}")

# Step 5: Calculate anomalies
print("\nStep 5: Calculating anomalies...")

# Main scenario anomalies (vs. historical baseline)
print("  - Main scenario anomalies (vs. historical)...")
ssp370_fwi_mean_anomaly = ssp370_fwi_mean - historical_fwi_mean
global_fwi_mean_anomaly = global_fwi_mean - historical_fwi_mean
ssp126_fwi_mean_anomaly = ssp126_fwi_mean - historical_fwi_mean

# Effect calculations
print("  - Effect calculations...")
aer126eff_fwi_mean_anomaly = global_fwi_mean_anomaly - ssp370_fwi_mean_anomaly
ghg126eff_fwi_mean_anomaly = ssp126_fwi_mean_anomaly - global_fwi_mean_anomaly

# Regional scenario anomalies (vs. SSP3-7.0 baseline)
print("  - Regional scenario anomalies (vs. SSP3-7.0)...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_mean_anomaly = eas_fwi_mean - ssp370_fwi_mean
    elif scenario == 'nae':
        nae_fwi_mean_anomaly = nae_fwi_mean - ssp370_fwi_mean
    elif scenario == 'sas':
        sas_fwi_mean_anomaly = sas_fwi_mean - ssp370_fwi_mean
    elif scenario == 'afr':
        afr_fwi_mean_anomaly = afr_fwi_mean - ssp370_fwi_mean

print("  ✓ All anomalies calculated")

# Step 6: Apply masks for anomaly analysis
print("\nStep 6: Applying masks for anomaly analysis...")

# Apply masks for global average calculations (textbox values)
print("  - Applying masks for global average calculations...")

# Main scenarios
ssp370_fwi_mean_anomaly_masked_textbox, _ = apply_masks(ssp370_fwi_mean_anomaly, 
                                                        get_significance=False, 
                                                        agreement_threshold=0.0,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=None)
global_fwi_mean_anomaly_masked_textbox, _ = apply_masks(global_fwi_mean_anomaly, 
                                                        get_significance=False,
                                                        agreement_threshold=0.0,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=None)
ssp126_fwi_mean_anomaly_masked_textbox, _ = apply_masks(ssp126_fwi_mean_anomaly, 
                                                        get_significance=False,
                                                        agreement_threshold=0.0,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=None)

# Effect scenarios
aer126eff_fwi_mean_anomaly_masked_textbox, _ = apply_masks(aer126eff_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)
ghg126eff_fwi_mean_anomaly_masked_textbox, _ = apply_masks(ghg126eff_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)

# Regional scenarios (in consistent order)
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_mean_anomaly_masked_textbox, _ = apply_masks(eas_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)
    elif scenario == 'nae':
        nae_fwi_mean_anomaly_masked_textbox, _ = apply_masks(nae_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)
    elif scenario == 'sas':
        sas_fwi_mean_anomaly_masked_textbox, _ = apply_masks(sas_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)
    elif scenario == 'afr':
        afr_fwi_mean_anomaly_masked_textbox, _ = apply_masks(afr_fwi_mean_anomaly, 
                                                            get_significance=False,
                                                            agreement_threshold=0.0,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=None)

print("  ✓ Textbox masks applied")

# Calculate global averages for textbox values
print("  - Calculating global averages...")

# Main scenarios
ssp370_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
    ssp370_fwi_mean_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
global_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
    global_fwi_mean_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
ssp126_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
    ssp126_fwi_mean_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)

# Effect scenarios
aer126eff_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
    aer126eff_fwi_mean_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
ghg126eff_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
    ghg126eff_fwi_mean_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)

# Regional scenarios (in consistent order)
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
            eas_fwi_mean_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'nae':
        nae_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
            nae_fwi_mean_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'sas':
        sas_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
            sas_fwi_mean_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'afr':
        afr_fwi_mean_anomaly_masked_globalavg = weighted_horizontal_avg(
            afr_fwi_mean_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )

print("  ✓ Global averages calculated")

# Apply masks for filled contour plots
print("  - Applying masks for filled contour plots...")

# Main scenarios
ssp370_fwi_mean_anomaly_masked_plot, _ = apply_masks(ssp370_fwi_mean_anomaly, 
                                                    get_significance=True,
                                                    agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                    threshold_type='minimum',
                                                    get_land_mask=True,
                                                    baseline_data=ssp370_fwi_mean_anomaly)
global_fwi_mean_anomaly_masked_plot, _ = apply_masks(global_fwi_mean_anomaly, 
                                                    get_significance=True,
                                                    agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                    threshold_type='minimum',
                                                    get_land_mask=True,
                                                    baseline_data=global_fwi_mean_anomaly)
ssp126_fwi_mean_anomaly_masked_plot, _ = apply_masks(ssp126_fwi_mean_anomaly, 
                                                    get_significance=True,
                                                    agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                    threshold_type='minimum',
                                                    get_land_mask=True,
                                                    baseline_data=ssp126_fwi_mean_anomaly)

# Effect scenarios
aer126eff_fwi_mean_anomaly_masked_plot, _ = apply_masks(aer126eff_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=aer126eff_fwi_mean_anomaly)
ghg126eff_fwi_mean_anomaly_masked_plot, _ = apply_masks(ghg126eff_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=ghg126eff_fwi_mean_anomaly)

# Regional scenarios (in consistent order)
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_mean_anomaly_masked_plot, _ = apply_masks(eas_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=eas_fwi_mean_anomaly)
    elif scenario == 'nae':
        nae_fwi_mean_anomaly_masked_plot, _ = apply_masks(nae_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=nae_fwi_mean_anomaly)
    elif scenario == 'sas':
        sas_fwi_mean_anomaly_masked_plot, _ = apply_masks(sas_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=sas_fwi_mean_anomaly)
    elif scenario == 'afr':
        afr_fwi_mean_anomaly_masked_plot, _ = apply_masks(afr_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=afr_fwi_mean_anomaly)

print("  ✓ Plot masks applied")

# Apply masks for hatching patterns
print("  - Applying masks for hatching patterns...")

# Main scenarios
ssp370_fwi_mean_anomaly_masked_hatching, _ = apply_masks(ssp370_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                        threshold_type='maximum',
                                                        get_land_mask=True,
                                                        baseline_data=ssp370_fwi_mean_anomaly)
global_fwi_mean_anomaly_masked_hatching, _ = apply_masks(global_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                        threshold_type='maximum',
                                                        get_land_mask=True,
                                                        baseline_data=global_fwi_mean_anomaly)
ssp126_fwi_mean_anomaly_masked_hatching, _ = apply_masks(ssp126_fwi_mean_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                        threshold_type='maximum',
                                                        get_land_mask=True,
                                                        baseline_data=ssp126_fwi_mean_anomaly)

# Effect scenarios
aer126eff_fwi_mean_anomaly_masked_hatching, _ = apply_masks(aer126eff_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=aer126eff_fwi_mean_anomaly)
ghg126eff_fwi_mean_anomaly_masked_hatching, _ = apply_masks(ghg126eff_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=ghg126eff_fwi_mean_anomaly)

# Regional scenarios (in consistent order)
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_mean_anomaly_masked_hatching, _ = apply_masks(eas_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=eas_fwi_mean_anomaly)
    elif scenario == 'nae':
        nae_fwi_mean_anomaly_masked_hatching, _ = apply_masks(nae_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=nae_fwi_mean_anomaly)
    elif scenario == 'sas':
        sas_fwi_mean_anomaly_masked_hatching, _ = apply_masks(sas_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=sas_fwi_mean_anomaly)
    elif scenario == 'afr':
        afr_fwi_mean_anomaly_masked_hatching, _ = apply_masks(afr_fwi_mean_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=afr_fwi_mean_anomaly)

print("  ✓ Hatching masks applied")
print("✓ Annual analysis masking complete")
print()

# Step 7: Create annual anomaly plots
print("\nStep 7: Creating annual anomaly plots...")

# Main scenarios grid
print("  - Main scenarios grid...")
data_list = [
    ssp370_fwi_mean_anomaly_masked_plot.mean('model'), 
    global_fwi_mean_anomaly_masked_plot.mean('model'), 
    ssp126_fwi_mean_anomaly_masked_plot.mean('model'),
    aer126eff_fwi_mean_anomaly_masked_plot.mean('model'),
    ghg126eff_fwi_mean_anomaly_masked_plot.mean('model')
]

title_list = ['SSP3-7.0', 
                'SSP1-2.6', 
                'SSP3-7.0 with Global Aerosol Reduction', 
                'Effect of Aerosol Emission Reduction', 
                'Effect of GHG Emission Reduction'
                ]

textbox_text_list = [
    f"{ssp370_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
    f"{global_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
    f"{ssp126_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
    f"{aer126eff_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
    f"{ghg126eff_fwi_mean_anomaly_masked_globalavg.values.item():.2f}"
]

hatching_data_list = [
    ssp370_fwi_mean_anomaly_masked_hatching.isel(model=0),
    global_fwi_mean_anomaly_masked_hatching.isel(model=0),
    ssp126_fwi_mean_anomaly_masked_hatching.isel(model=0),
    aer126eff_fwi_mean_anomaly_masked_hatching.isel(model=0),
    ghg126eff_fwi_mean_anomaly_masked_hatching.isel(model=0)
]

vmins_list = [-3, -3, -3, -1, -1]
vmaxs_list = [3, 3, 3, 1, 1]

colorbar_levels_list = [np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3), 
                        np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1)
                        ]

fig, axes = create_global_map_grid(
    data_list, 
    rows=2, 
    cols=3,
    main_title="Annual FWI Changes",
    projection=ccrs.Robinson(),
    titles=title_list,
    colormaps='RdBu_r',
    colorbar_titles="Δ Fire Weather Index",
    textbox_texts=textbox_text_list,
    vmins=vmins_list,
    vmaxs=vmaxs_list,
    extends='both',
    colorbar_levels=colorbar_levels_list,
    hatchings='///',
    regional_boundaries='ar6',
    hatching_styles='overlay',
    hatching_data=hatching_data_list,
    show_gridlines=False,
    ramip_regions=False,
)

# Save main scenarios plot
plt.savefig(output_dir / "mm_main_scenarios_fwi_annual.png", dpi=600, bbox_inches='tight')
plt.close()
print(f"  ✓ Main scenarios plot saved: {output_dir / 'mm_main_scenarios_fwi_annual.png'}")

# Regional scenarios grid
print("  - Regional scenarios grid...")
regional_data_list = [eas_fwi_mean_anomaly_masked_plot.mean('model'), nae_fwi_mean_anomaly_masked_plot.mean('model'), 
                        sas_fwi_mean_anomaly_masked_plot.mean('model'), afr_fwi_mean_anomaly_masked_plot.mean('model'),
                        ]

regional_title_list = ['Effect of East Asian Aerosol Reduction', 
                        'Effect of North American & European Aerosol Reduction', 
                        'Effect of South Asian Aerosol Reduction', 
                        'Effect of African & Middle Eastern Aerosol Reduction']

regional_textbox_text_list = [f"{eas_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
                                f"{nae_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
                                f"{sas_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
                                f"{afr_fwi_mean_anomaly_masked_globalavg.values.item():.2f}",
                    ]

regional_hatching_data_list = [eas_fwi_mean_anomaly_masked_hatching.isel(model=0),
                                nae_fwi_mean_anomaly_masked_hatching.isel(model=0),
                                sas_fwi_mean_anomaly_masked_hatching.isel(model=0),
                                afr_fwi_mean_anomaly_masked_hatching.isel(model=0),
                    ]

regional_vmins_list = [-0.5, -0.5, -0.5, -0.5]
regional_vmaxs_list = [0.5, 0.5, 0.5, 0.5]

regional_colorbar_levels_list = [np.arange(-0.5, 0.51, 0.05), np.arange(-0.5, 0.51, 0.05),
                                np.arange(-0.5, 0.51, 0.05), np.arange(-0.5, 0.51, 0.05)]

fig, axes = create_global_map_grid(
    regional_data_list, 
    rows=2, 
    cols=2,
    main_title="Regional Aerosol Reduction Effects on Annual FWI",
    projection=ccrs.Robinson(),
    titles=regional_title_list,
    colormaps='RdBu_r',
    colorbar_titles="Δ Fire Weather Index",
    textbox_texts=regional_textbox_text_list,
    vmins=regional_vmins_list,
    vmaxs=regional_vmaxs_list,
    extends='both',
    colorbar_levels=regional_colorbar_levels_list,
    hatchings='///',
    regional_boundaries='ar6',
    hatching_styles='overlay',
    hatching_data=regional_hatching_data_list,
    show_gridlines=False,
    ramip_regions=['east_asia', 'north_america_europe', 'south_asia', 'africa_mideast'],
)

# Save regional scenarios plot
plt.savefig(output_dir / "mm_regional_scenarios_fwi_annual.png", dpi=600, bbox_inches='tight')
plt.close()
print(f"  ✓ Regional scenarios plot saved: {output_dir / 'mm_regional_scenarios_fwi_annual.png'}")

print("✓ Annual analysis complete")
print()

# =============================================================================
# SEASONAL ANALYSIS
# =============================================================================
print("SEASONAL FWI ANALYSIS")
print("=" * 80)
print("Calculating seasonal means and anomalies...")
print("-" * 60)

# Step 1: Calculate seasonal means for all scenarios
print("Step 1: Calculating seasonal means...")

# Historical baseline
print("  - Historical baseline...")
noresm2_historical_fwi_seasonal = season_mean(noresm2_historical_fwi).mean('member', skipna=True)
spear_historical_fwi_seasonal = season_mean(spear_historical_fwi).mean('member', skipna=True)
mri_historical_fwi_seasonal = season_mean(mri_historical_fwi).mean('member', skipna=True)

# Combine historical models
multi_historical_fwi_seasonal = xr.concat([
    noresm2_historical_fwi_seasonal, 
    spear_historical_fwi_seasonal, 
    mri_historical_fwi_seasonal
], dim="model", coords='minimal')
historical_fwi_seasonal = multi_historical_fwi_seasonal.compute()
print("  ✓ Historical seasonal means calculated")

# Main emission scenarios
print("  - Main emission scenarios...")
noresm2_ssp370_fwi_seasonal = season_mean(noresm2_ssp370_fwi).mean('member', skipna=True)
spear_ssp370_fwi_seasonal = season_mean(spear_ssp370_fwi).mean('member', skipna=True)
mri_ssp370_fwi_seasonal = season_mean(mri_ssp370_fwi).mean('member', skipna=True)

noresm2_global_fwi_seasonal = season_mean(noresm2_global_fwi).mean('member', skipna=True)
spear_global_fwi_seasonal = season_mean(spear_global_fwi).mean('member', skipna=True)
mri_global_fwi_seasonal = season_mean(mri_global_fwi).mean('member', skipna=True)

noresm2_ssp126_fwi_seasonal = season_mean(noresm2_ssp126_fwi).mean('member', skipna=True)
spear_ssp126_fwi_seasonal = season_mean(spear_ssp126_fwi).mean('member', skipna=True)
mri_ssp126_fwi_seasonal = season_mean(mri_ssp126_fwi).mean('member', skipna=True)

# Regional scenarios (in consistent order)
print("  - Regional scenarios...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        noresm2_eas_fwi_seasonal = season_mean(noresm2_eas_fwi).mean('member', skipna=True)
        spear_eas_fwi_seasonal = season_mean(spear_eas_fwi).mean('member', skipna=True)
        mri_eas_fwi_seasonal = season_mean(mri_eas_fwi).mean('member', skipna=True)
    elif scenario == 'nae':
        noresm2_nae_fwi_seasonal = season_mean(noresm2_nae_fwi).mean('member', skipna=True)
        spear_nae_fwi_seasonal = season_mean(spear_nae_fwi).mean('member', skipna=True)
        mri_nae_fwi_seasonal = season_mean(mri_nae_fwi).mean('member', skipna=True)
    elif scenario == 'sas':
        noresm2_sas_fwi_seasonal = season_mean(noresm2_sas_fwi).mean('member', skipna=True)
        spear_sas_fwi_seasonal = season_mean(spear_sas_fwi).mean('member', skipna=True)
        mri_sas_fwi_seasonal = season_mean(mri_sas_fwi).mean('member', skipna=True)
    elif scenario == 'afr':
        noresm2_afr_fwi_seasonal = season_mean(noresm2_afr_fwi).mean('member', skipna=True)
        spear_afr_fwi_seasonal = season_mean(spear_afr_fwi).mean('member', skipna=True)
        mri_afr_fwi_seasonal = season_mean(mri_afr_fwi).mean('member', skipna=True)

print("  ✓ All seasonal means calculated")

# Step 2: Combine models for each scenario
print("\nStep 2: Combining models...")

# Main emission scenarios
print("  - Main emission scenarios...")
multi_ssp370_fwi_seasonal = xr.concat([
    noresm2_ssp370_fwi_seasonal, spear_ssp370_fwi_seasonal, mri_ssp370_fwi_seasonal
], dim="model", coords='minimal')
ssp370_fwi_seasonal = multi_ssp370_fwi_seasonal.compute()

multi_global_fwi_seasonal = xr.concat([
    noresm2_global_fwi_seasonal, spear_global_fwi_seasonal, mri_global_fwi_seasonal
], dim="model", coords='minimal')
global_fwi_seasonal = multi_global_fwi_seasonal.compute()

multi_ssp126_fwi_seasonal = xr.concat([
    noresm2_ssp126_fwi_seasonal, spear_ssp126_fwi_seasonal, mri_ssp126_fwi_seasonal
], dim="model", coords='minimal')
ssp126_fwi_seasonal = multi_ssp126_fwi_seasonal.compute()

# Regional scenarios (in consistent order)
print("  - Regional scenarios...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        multi_eas_fwi_seasonal = xr.concat([
            noresm2_eas_fwi_seasonal, spear_eas_fwi_seasonal, mri_eas_fwi_seasonal
        ], dim="model", coords='minimal')
        eas_fwi_seasonal = multi_eas_fwi_seasonal.compute()
    elif scenario == 'nae':
        multi_nae_fwi_seasonal = xr.concat([
            noresm2_nae_fwi_seasonal, spear_nae_fwi_seasonal, mri_nae_fwi_seasonal
        ], dim="model", coords='minimal')
        nae_fwi_seasonal = multi_nae_fwi_seasonal.compute()
    elif scenario == 'sas':
        multi_sas_fwi_seasonal = xr.concat([
            noresm2_sas_fwi_seasonal, spear_sas_fwi_seasonal, mri_sas_fwi_seasonal
        ], dim="model", coords='minimal')
        sas_fwi_seasonal = multi_sas_fwi_seasonal.compute()
    elif scenario == 'afr':
        multi_afr_fwi_seasonal = xr.concat([
            noresm2_afr_fwi_seasonal, spear_afr_fwi_seasonal, mri_afr_fwi_seasonal
        ], dim="model", coords='minimal')
        afr_fwi_seasonal = multi_afr_fwi_seasonal.compute()

print("  ✓ All models combined")

# Step 3: Apply masks and calculate global averages
print("\nStep 3: Applying masks and calculating global averages...")

# Historical baseline
print("  - Historical baseline...")
historical_fwi_seasonal_masked, _ = apply_masks(historical_fwi_seasonal, 
                                                get_significance=False,
                                                get_land_mask=True)

historical_fwi_seasonal_masked_globalavg = weighted_horizontal_avg(
    historical_fwi_seasonal_masked.mean('model'), 
    ensemble=False, 
    time=False
)
print("  ✓ Historical seasonal masks applied")

# Step 4: Create historical seasonal baseline plot
print("\nStep 4: Creating historical seasonal baseline plot...")

# Create 2x2 grid for historical seasonal maps
historical_seasonal_data_list = [
    historical_fwi_seasonal_masked.sel(season='JJA').mean('model'),
    historical_fwi_seasonal_masked.sel(season='SON').mean('model'),
    historical_fwi_seasonal_masked.sel(season='DJF').mean('model'),
    historical_fwi_seasonal_masked.sel(season='MAM').mean('model')
]

historical_seasonal_titles = ['JJA (Jun-Jul-Aug)', 'SON (Sep-Oct-Nov)', 
                                'DJF (Dec-Jan-Feb)', 'MAM (Mar-Apr-May)']

historical_seasonal_textbox_texts = [
    f"{historical_fwi_seasonal_masked_globalavg.sel(season='JJA').values.item():.2f}",
    f"{historical_fwi_seasonal_masked_globalavg.sel(season='SON').values.item():.2f}",
    f"{historical_fwi_seasonal_masked_globalavg.sel(season='DJF').values.item():.2f}",
    f"{historical_fwi_seasonal_masked_globalavg.sel(season='MAM').values.item():.2f}"
]

fig, axes = create_global_map_grid(
    data_list=historical_seasonal_data_list,
    rows=2, cols=2,
    main_title="Historical (1961-1990)",
    titles=historical_seasonal_titles,
    colormaps='Reds',
    colorbar_titles="Fire Weather Index",
    textbox_texts=historical_seasonal_textbox_texts,
    vmins=[0, 0, 0, 0],
    vmaxs=[40, 40, 40, 40],
    extends='max',
    colorbar_levels=[np.arange(0, 40.1, 4), np.arange(0, 40.1, 4), 
                    np.arange(0, 40.1, 4), np.arange(0, 40.1, 4)],
    regional_boundaries='ar6',
    show_gridlines=False
)

# Save historical seasonal plot
plt.savefig(output_dir / "mm_historical_fwi_seasonal.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Historical seasonal plot saved: {output_dir / 'mm_historical_fwi_seasonal.png'}")

# Step 5: Calculate seasonal anomalies
print("\nStep 5: Calculating seasonal anomalies...")

# Main scenario anomalies (vs. historical baseline)
print("  - Main scenario anomalies (vs. historical)...")
ssp370_fwi_seasonal_anomaly = ssp370_fwi_seasonal - historical_fwi_seasonal
global_fwi_seasonal_anomaly = global_fwi_seasonal - historical_fwi_seasonal
ssp126_fwi_seasonal_anomaly = ssp126_fwi_seasonal - historical_fwi_seasonal

# Effect calculations
print("  - Effect calculations...")
aer126eff_fwi_seasonal_anomaly = global_fwi_seasonal_anomaly - ssp370_fwi_seasonal_anomaly
ghg126eff_fwi_seasonal_anomaly = ssp126_fwi_seasonal_anomaly - global_fwi_seasonal_anomaly

# Regional scenario anomalies (vs. SSP3-7.0 baseline)
print("  - Regional scenario anomalies (vs. SSP3-7.0)...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_seasonal_anomaly = eas_fwi_seasonal - ssp370_fwi_seasonal
    elif scenario == 'nae':
        nae_fwi_seasonal_anomaly = nae_fwi_seasonal - ssp370_fwi_seasonal
    elif scenario == 'sas':
        sas_fwi_seasonal_anomaly = sas_fwi_seasonal - ssp370_fwi_seasonal
    elif scenario == 'afr':
        afr_fwi_seasonal_anomaly = afr_fwi_seasonal - ssp370_fwi_seasonal

print("  ✓ All seasonal anomalies calculated")

# Step 6: Apply masks for seasonal anomaly analysis
print("\nStep 6: Applying masks for seasonal anomaly analysis...")

# Apply masks for global average calculations (textbox values)
print("  - Applying masks for global average calculations...")

# Main scenarios
ssp370_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(ssp370_fwi_seasonal_anomaly, 
                                                            get_significance=False,
                                                            get_land_mask=True)
global_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(global_fwi_seasonal_anomaly, 
                                                            get_significance=False,
                                                            get_land_mask=True)
ssp126_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(ssp126_fwi_seasonal_anomaly, 
                                                            get_significance=False,
                                                            get_land_mask=True)

# Effect scenarios
aer126eff_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(aer126eff_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)
ghg126eff_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(ghg126eff_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)

print("  ✓ Textbox masks applied")

# Calculate global averages for textbox values
print("  - Calculating global averages...")

# Main scenarios
ssp370_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
    ssp370_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
global_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
    global_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
ssp126_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
    ssp126_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)

# Effect scenarios
aer126eff_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
    aer126eff_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)
ghg126eff_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
    ghg126eff_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
    ensemble=False, 
    time=False
)

print("  ✓ Global averages calculated")

# Apply masks for filled contour plots
print("  - Applying masks for filled contour plots...")

# Main scenarios
ssp370_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(ssp370_fwi_seasonal_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=ssp370_fwi_seasonal_anomaly)
global_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(global_fwi_seasonal_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=global_fwi_seasonal_anomaly)
ssp126_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(ssp126_fwi_seasonal_anomaly, 
                                                        get_significance=True,
                                                        agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                        threshold_type='minimum',
                                                        get_land_mask=True,
                                                        baseline_data=ssp126_fwi_seasonal_anomaly)

# Effect scenarios
aer126eff_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(aer126eff_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=aer126eff_fwi_seasonal_anomaly)
ghg126eff_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(ghg126eff_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=ghg126eff_fwi_seasonal_anomaly)

print("  ✓ Plot masks applied")

# Apply masks for hatching patterns
print("  - Applying masks for hatching patterns...")

# Main scenarios
ssp370_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(ssp370_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=ssp370_fwi_seasonal_anomaly)
global_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(global_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=global_fwi_seasonal_anomaly)
ssp126_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(ssp126_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                            threshold_type='maximum',
                                                            get_land_mask=True,
                                                            baseline_data=ssp126_fwi_seasonal_anomaly)

# Effect scenarios
aer126eff_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(aer126eff_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=aer126eff_fwi_seasonal_anomaly)
ghg126eff_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(ghg126eff_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=ghg126eff_fwi_seasonal_anomaly)

print("  ✓ Hatching masks applied")
print("✓ Seasonal analysis masking complete")
print()

# Step 7: Apply masks for regional scenarios seasonal analysis
print("\nStep 7: Applying masks for regional scenarios seasonal analysis...")

# Apply masks for global average calculations (textbox values)
print("  - Applying masks for global average calculations...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(eas_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)
    elif scenario == 'nae':
        nae_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(nae_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)
    elif scenario == 'sas':
        sas_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(sas_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)
    elif scenario == 'afr':
        afr_fwi_seasonal_anomaly_masked_textbox, _ = apply_masks(afr_fwi_seasonal_anomaly, 
                                                                get_significance=False,
                                                                get_land_mask=True)

print("  ✓ Textbox masks applied")

# Calculate global averages for textbox values
print("  - Calculating global averages...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
            eas_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'nae':
        nae_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
            nae_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'sas':
        sas_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
            sas_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )
    elif scenario == 'afr':
        afr_fwi_seasonal_anomaly_masked_globalavg = weighted_horizontal_avg(
            afr_fwi_seasonal_anomaly_masked_textbox.mean('model'), 
            ensemble=False, 
            time=False
        )

print("  ✓ Global averages calculated")

# Apply masks for filled contour plots
print("  - Applying masks for filled contour plots...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(eas_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=eas_fwi_seasonal_anomaly)
    elif scenario == 'nae':
        nae_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(nae_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=nae_fwi_seasonal_anomaly)
    elif scenario == 'sas':
        sas_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(sas_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=sas_fwi_seasonal_anomaly)
    elif scenario == 'afr':
        afr_fwi_seasonal_anomaly_masked_plot, _ = apply_masks(afr_fwi_seasonal_anomaly, 
                                                            get_significance=True,
                                                            agreement_threshold=AGREEMENT_THRESHOLD_PLOT,
                                                            threshold_type='minimum',
                                                            get_land_mask=True,
                                                            baseline_data=afr_fwi_seasonal_anomaly)

print("  ✓ Plot masks applied")

# Apply masks for hatching patterns
print("  - Applying masks for hatching patterns...")
for scenario in REGIONAL_SCENARIOS:
    if scenario == 'eas':
        eas_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(eas_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=eas_fwi_seasonal_anomaly)
    elif scenario == 'nae':
        nae_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(nae_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=nae_fwi_seasonal_anomaly)
    elif scenario == 'sas':
        sas_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(sas_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=sas_fwi_seasonal_anomaly)
    elif scenario == 'afr':
        afr_fwi_seasonal_anomaly_masked_hatching, _ = apply_masks(afr_fwi_seasonal_anomaly, 
                                                                get_significance=True,
                                                                agreement_threshold=AGREEMENT_THRESHOLD_HATCHING,
                                                                threshold_type='maximum',
                                                                get_land_mask=True,
                                                                baseline_data=afr_fwi_seasonal_anomaly)

print("  ✓ Hatching masks applied")
print("✓ Regional scenarios seasonal analysis masking complete")
print()

# Step 8: Create seasonal anomaly plots for main scenarios
print("\nStep 8: Creating seasonal anomaly plots for main scenarios...")

# Create seasonal plots for each main scenario
scenarios = ['ssp370', 'global', 'ssp126', 'aer126eff', 'ghg126eff']
scenario_names = ['SSP3-7.0', 'SSP3-7.0 with Global Aerosol Reduction', 'SSP1-2.6', 
                    'Effect of Aerosol Emission Reduction', 'Effect of GHG Emission Reduction']

for i, (scenario, scenario_name) in enumerate(zip(scenarios, scenario_names)):
    # Get the appropriate data for this scenario
    if scenario == 'ssp370':
        data_masked = ssp370_fwi_seasonal_anomaly_masked_plot
        hatching_data = ssp370_fwi_seasonal_anomaly_masked_hatching
        global_avg = ssp370_fwi_seasonal_anomaly_masked_globalavg
    elif scenario == 'global':
        data_masked = global_fwi_seasonal_anomaly_masked_plot
        hatching_data = global_fwi_seasonal_anomaly_masked_hatching
        global_avg = global_fwi_seasonal_anomaly_masked_globalavg
    elif scenario == 'ssp126':
        data_masked = ssp126_fwi_seasonal_anomaly_masked_plot
        hatching_data = ssp126_fwi_seasonal_anomaly_masked_hatching
        global_avg = ssp126_fwi_seasonal_anomaly_masked_globalavg
    elif scenario == 'aer126eff':
        data_masked = aer126eff_fwi_seasonal_anomaly_masked_plot
        hatching_data = aer126eff_fwi_seasonal_anomaly_masked_hatching
        global_avg = aer126eff_fwi_seasonal_anomaly_masked_globalavg
    elif scenario == 'ghg126eff':
        data_masked = ghg126eff_fwi_seasonal_anomaly_masked_plot
        hatching_data = ghg126eff_fwi_seasonal_anomaly_masked_hatching
        global_avg = ghg126eff_fwi_seasonal_anomaly_masked_globalavg
    
    # Create seasonal data list
    seasonal_data_list = [
        data_masked.sel(season='JJA').mean('model'),
        data_masked.sel(season='SON').mean('model'),
        data_masked.sel(season='DJF').mean('model'),
        data_masked.sel(season='MAM').mean('model')
    ]
    
    seasonal_titles = ['JJA (Jun-Jul-Aug)', 'SON (Sep-Oct-Nov)', 
                        'DJF (Dec-Jan-Feb)', 'MAM (Mar-Apr-May)']
    
    seasonal_textbox_texts = [
        f"{global_avg.sel(season='JJA').values.item():.2f}",
        f"{global_avg.sel(season='SON').values.item():.2f}",
        f"{global_avg.sel(season='DJF').values.item():.2f}",
        f"{global_avg.sel(season='MAM').values.item():.2f}"
    ]
    
    seasonal_hatching_data_list = [
        hatching_data.sel(season='JJA').isel(model=0),
        hatching_data.sel(season='SON').isel(model=0),
        hatching_data.sel(season='DJF').isel(model=0),
        hatching_data.sel(season='MAM').isel(model=0)
    ]
    
    # Set color limits based on scenario
    if scenario in ['ssp370', 'global', 'ssp126']:
        vmins_list = [-3, -3, -3, -3]
        vmaxs_list = [3, 3, 3, 3]
        colorbar_levels_list = [np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3), 
                                np.arange(-3, 3.1, 0.3), np.arange(-3, 3.1, 0.3)]
    else:  # aer126eff, ghg126eff
        vmins_list = [-1, -1, -1, -1]
        vmaxs_list = [1, 1, 1, 1]
        colorbar_levels_list = [np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1),
                                np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1)]
    
    fig, axes = create_global_map_grid(
        data_list=seasonal_data_list,
        rows=2, cols=2,
        main_title=f"Seasonal FWI Changes: {scenario_name}",
        titles=seasonal_titles,
        colormaps='RdBu_r',
        colorbar_titles="Δ Fire Weather Index",
        textbox_texts=seasonal_textbox_texts,
        vmins=vmins_list,
        vmaxs=vmaxs_list,
        extends='both',
        colorbar_levels=colorbar_levels_list,
        hatchings='///',
        regional_boundaries='ar6',
        hatching_styles='overlay',
        hatching_data=seasonal_hatching_data_list,
        show_gridlines=False
    )
    
    # Save seasonal plot
    plt.savefig(output_dir / f"mm_{scenario}_fwi_seasonal.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Seasonal plot for {scenario_name} saved: {output_dir / f'mm_{scenario}_fwi_seasonal.png'}")

print("  ✓ Main scenarios seasonal plots complete")

# Step 9: Create seasonal anomaly plots for regional scenarios
print("\nStep 9: Creating seasonal anomaly plots for regional scenarios...")

# Create seasonal plots for each regional scenario (in consistent order)
regional_scenarios = ['eas', 'nae', 'sas', 'afr']
regional_scenario_names = ['East Asian Aerosol Reduction', 'North American & European Aerosol Reduction',
                          'South Asian Aerosol Reduction', 'African & Middle Eastern Aerosol Reduction']

for i, (scenario, scenario_name) in enumerate(zip(regional_scenarios, regional_scenario_names)):
    # Get the appropriate data for this scenario
    if scenario == 'eas':
        data_masked = eas_fwi_seasonal_anomaly_masked_plot
        hatching_data = eas_fwi_seasonal_anomaly_masked_hatching
        global_avg = eas_fwi_seasonal_anomaly_masked_globalavg
        ramip_regions_val = 'east_asia'
    elif scenario == 'nae':
        data_masked = nae_fwi_seasonal_anomaly_masked_plot
        hatching_data = nae_fwi_seasonal_anomaly_masked_hatching
        global_avg = nae_fwi_seasonal_anomaly_masked_globalavg
        ramip_regions_val = 'north_america_europe'
    elif scenario == 'sas':
        data_masked = sas_fwi_seasonal_anomaly_masked_plot
        hatching_data = sas_fwi_seasonal_anomaly_masked_hatching
        global_avg = sas_fwi_seasonal_anomaly_masked_globalavg
        ramip_regions_val = 'south_asia'
    elif scenario == 'afr':
        data_masked = afr_fwi_seasonal_anomaly_masked_plot
        hatching_data = afr_fwi_seasonal_anomaly_masked_hatching
        global_avg = afr_fwi_seasonal_anomaly_masked_globalavg
        ramip_regions_val = 'africa_mideast'
    
    # Create seasonal data list
    seasonal_data_list = [
        data_masked.sel(season='JJA').mean('model'),
        data_masked.sel(season='SON').mean('model'),
        data_masked.sel(season='DJF').mean('model'),
        data_masked.sel(season='MAM').mean('model')
    ]
    
    seasonal_titles = ['JJA (Jun-Jul-Aug)', 'SON (Sep-Oct-Nov)', 
                        'DJF (Dec-Jan-Feb)', 'MAM (Mar-Apr-May)']
    
    seasonal_textbox_texts = [
        f"{global_avg.sel(season='JJA').values.item():.2f}",
        f"{global_avg.sel(season='SON').values.item():.2f}",
        f"{global_avg.sel(season='DJF').values.item():.2f}",
        f"{global_avg.sel(season='MAM').values.item():.2f}"
    ]
    
    seasonal_hatching_data_list = [
        hatching_data.sel(season='JJA').isel(model=0),
        hatching_data.sel(season='SON').isel(model=0),
        hatching_data.sel(season='DJF').isel(model=0),
        hatching_data.sel(season='MAM').isel(model=0)
    ]
    
    # Set color limits for regional scenarios (smaller range)
    vmins_list = [-0.5, -0.5, -0.5, -0.5]
    vmaxs_list = [0.5, 0.5, 0.5, 0.5]
    colorbar_levels_list = [np.arange(-0.5, 0.51, 0.05), np.arange(-0.5, 0.51, 0.05),
                            np.arange(-0.5, 0.51, 0.05), np.arange(-0.5, 0.51, 0.05)]
    
    fig, axes = create_global_map_grid(
        data_list=seasonal_data_list,
        rows=2, cols=2,
        main_title=f"Seasonal FWI Changes: {scenario_name}",
        titles=seasonal_titles,
        colormaps='RdBu_r',
        colorbar_titles="Δ Fire Weather Index",
        textbox_texts=seasonal_textbox_texts,
        vmins=vmins_list,
        vmaxs=vmaxs_list,
        extends='both',
        colorbar_levels=colorbar_levels_list,
        hatchings='///',
        regional_boundaries='ar6',
        hatching_styles='overlay',
        hatching_data=seasonal_hatching_data_list,
        show_gridlines=False,
        ramip_regions=ramip_regions_val
    )
    
    # Save seasonal plot
    plt.savefig(output_dir / f"mm_{scenario}_fwi_seasonal.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Seasonal plot for {scenario_name} saved: {output_dir / f'mm_{scenario}_fwi_seasonal.png'}")

print("  ✓ Regional scenarios seasonal plots complete")

print("✓ Seasonal analysis complete")
print()

# =============================================================================
# ANALYSIS COMPLETE
# =============================================================================
print("=" * 80)
print("FWI RAW VALUES ANALYSIS - COMPLETE")
print("=" * 80)
print(f"✓ Annual analysis: {len(list(output_dir.glob('*annual*')))} plots generated")
print(f"✓ Seasonal analysis: {len(list(output_dir.glob('*seasonal*')))} plots generated")
print(f"✓ Total plots: {len(list(output_dir.glob('*.png')))} plots generated")
print(f"✓ All plots saved to: {output_dir}")
print("=" * 80)

